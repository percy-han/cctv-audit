# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ADK entry point. Turns a chat message into an audit run.

This is a thin shell: it parses the target out of the user's message, starts
the pipeline, and relays progress. All the work lives in `pipeline.py`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from .analyzer import load_rules
from .capture.types import Clip
from .config import config  # noqa: F401  -- imported for its .env / SSL side effects
from .intent import (  # noqa: F401  -- parse_request re-exported for the tests
    Intent,
    UnreadableTimeSpan,
    interpret_request,
    parse_request,
)
from .monitor import monitor
from .navigator import HumanGate
from .navigator.base import TargetUnavailable
from .pipeline import AuditPipeline
from .store import AuditStore

logger = logging.getLogger("cctv_audit.agent")

_SEVERITY_LABEL = {"RED_LINE": "致命缺陷", "NORMAL": "一般缺陷", "NONE": "—"}

# A chat reply is not the place to render a three-hour audit. The JSONL is.
_MAX_TABLE_ROWS = 50

USAGE = (
    "请提供待稽核的视频地址，例如：\n"
    "  `https://www.bilibili.com/video/BV1URpRzCEXK/`\n"
    "  （请粘贴能在浏览器里正常打开的完整地址，`BVxxx` 这类占位符会直接 404）\n\n"
    "时间段直接用大白话写在同一句话里就行，例如：\n"
    "  • `第1分钟到第5分钟` / `01:00 到 05:00` / `从 12:30 开始看十分钟`\n"
    "  • `前三分钟` / `开头五分钟`\n"
    "  • 不写时间 = 整段视频\n"
    "  （这句话是交给模型读的，不是关键词匹配；读不准会**停下来问你**，"
    "不会默默按整段跑）\n\n"
    "⚠️ 这句话里**只有** URL、起始时间、时长会被读取。稽核标准不在这里写 —— "
    f"它的唯一来源是 `{config.sop_rules_path}`，改那个 YAML 才会生效"
    "（这样每一条判定才可追溯到某一版规则）。\n\n"
    f"当前配置：窗口 {config.window_seconds}s（重叠 {config.window_overlap_seconds}s）、"
    f"采集模式 {config.capture_mode}、并发 {config.analysis_concurrency}、"
    f"分辨率档位 {config.media_resolution}。"
)


def _preview_caption(mode: str) -> tuple:
    """What the dashboard's live picture is, and is not, under each plan.

    Under Plan B the picture *is* the evidence -- same page, same moment.
    Under Plan A ffmpeg downloads and seeks the media on its own, so the
    browser is only keeping the media token alive and its playhead has nothing
    to do with the window being judged. Saying so beats an operator watching
    minute one while the report scrolls past minute five.
    """
    if mode == "stream":
        return ("抓流中（Plan A）",
                "画面仅供参考：视频正由 ffmpeg 直接下载分析，进度与这里的播放位置无关")
    return ("录屏中（Plan B）", "这就是送去判定的画面")


def _explain_failure(exc: Exception) -> str:
    """Turns an exception into something a store manager can act on.

    Two things had to be fixed here. The chat UI renders markdown, so a message
    containing a literal HTML tag had the tag parsed away and the rest of the
    sentence with it -- the reader saw "稽核中断：No" and nothing else. And a
    raw Playwright timeout dumps its own internal wait log, which says a lot
    about locators and nothing about what to do next.
    """
    detail = str(exc).strip() or exc.__class__.__name__
    # Playwright appends "\nwaiting for locator(...)" style diagnostics.
    headline = detail.split("\n", 1)[0].strip()
    lines = [f"❌ 稽核中断：{_neutralise_markup(headline)}"]

    if isinstance(exc, TargetUnavailable):
        lines.append(
            "\n请确认链接能在浏览器里正常打开。注意 `BVxxx` 之类的示例占位符不是真实视频号，"
            "需要换成完整的真实地址。"
        )
    rest = detail[len(headline):].strip()
    if rest:
        # Kept, but demoted: it is diagnostic material, not the message.
        lines.append(f"\n<details><summary>详细信息</summary>\n\n```\n{rest[:800]}\n```\n</details>")
    return "\n".join(lines)


def _neutralise_markup(text: str) -> str:
    """Keeps angle brackets visible instead of letting the UI eat them."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _span(summary: dict) -> str:
    """What was audited, next to what was asked for, in video timeline terms."""
    fmt = Clip.format_offset
    start, end = summary.get("covered_from_seconds"), summary.get("covered_to_seconds")
    covered = f"{fmt(start)} - {fmt(end)}" if start is not None and end is not None else "无"
    requested_end = summary.get("requested_end_seconds")
    if requested_end is None:
        return covered
    requested_start = summary.get("requested_start_seconds") or 0.0
    return f"{covered}（请求 {fmt(requested_start)} - {fmt(requested_end)}）"


class CctvAuditAgent(BaseAgent):
    """Runs the capture -> analyse -> store pipeline and streams progress."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        user_text = ""
        if ctx.user_content and ctx.user_content.parts:
            user_text = "".join(p.text or "" for p in ctx.user_content.parts if hasattr(p, "text"))

        try:
            intent = await interpret_request(user_text)
        except UnreadableTimeSpan as exc:
            yield self._say(ctx, (
                f"⚠️ 时间段没读准，**没有开跑**：{exc}\n"
                "默认按「整段视频、从头开始」跑是最贵也最可能错的一种猜法，所以直接停下来问你。\n\n"
                "换个说法就行，例如：\n"
                "  • `第1分钟到第5分钟` / `01:00 到 05:00`\n"
                "  • `从 01:00 开始，看 4 分钟`\n"
                "  • 不写时间 = 整段视频"
            ))
            return
        if intent is None:
            yield self._say(ctx, USAGE)
            return
        request = intent.request

        problems = config.validate()
        if problems:
            yield self._say(ctx, "⚠️ 配置有误，无法启动：\n" + "\n".join(f"  • {p}" for p in problems))
            return

        # The monitor server usually runs as a separate process (start_web.sh),
        # so the gate reads its state over HTTP rather than relying on an
        # in-process callback that would not exist there.
        gate = HumanGate(notify=monitor.request_human, read_intervention=monitor.read_intervention)
        store = AuditStore(on_record=monitor.push_record)
        events: asyncio.Queue = asyncio.Queue()

        pipeline = AuditPipeline(
            store=store,
            gate=gate,
            on_preview_frame=monitor.update_frame_b64,
            on_status=lambda event, payload: events.put_nowait((event, payload)),
        )

        # The dashboard used to announce "1分钟视频抽检进行中" whatever the run
        # actually was -- a leftover from a design where the window really was
        # fixed at a minute. Tell it the settings in force instead.
        monitor.start_session(
            f"稽核 {request.target}",
            settings=(f"窗口 {config.window_seconds:g}s / 重叠 "
                      f"{config.window_overlap_seconds:g}s / 抽帧 "
                      f"{config.analysis_fps:g} FPS / 画质 {config.media_resolution}"),
        )
        yield self._say(ctx, self._start_banner(intent))

        task = asyncio.create_task(pipeline.run(request))
        try:
            # Relay status events as they happen; the audit itself may run for
            # hours, so waiting for the final summary is not an option.
            while True:
                drain = asyncio.create_task(events.get())
                done, _ = await asyncio.wait({task, drain}, return_when=asyncio.FIRST_COMPLETED)
                if drain in done:
                    event, payload = drain.result()
                    if event == "capture_mode":
                        monitor.note(*_preview_caption(payload["mode"]))
                    message = self._format(event, payload)
                    if message:
                        yield self._say(ctx, message)
                    continue

                drain.cancel()
                while not events.empty():
                    event, payload = events.get_nowait()
                    message = self._format(event, payload)
                    if message:
                        yield self._say(ctx, message)
                break

            summary = await task
        except Exception as exc:
            logger.exception("Audit failed")
            monitor.fail_session(str(exc))
            yield self._say(ctx, _explain_failure(exc))
            return
        finally:
            with_close = getattr(monitor, "close", None)
            if with_close:
                try:
                    await monitor.close()
                except Exception as exc:
                    logger.debug("Monitor cleanup failed: %s", exc)

        report = self._report(summary, store)
        monitor.finish_session(report)
        yield self._say(ctx, report)

    # -- presentation ------------------------------------------------------

    def _say(self, ctx: InvocationContext, text: str) -> Event:
        return Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
        )

    @staticmethod
    def _start_banner(intent: Intent) -> str:
        request = intent.request
        # Spelled out as timeline marks, not as "0s + 240s". This line is the
        # only chance to catch a misread request before the run bills for it.
        fmt = Clip.format_offset
        if request.duration_seconds:
            end = request.start_seconds + request.duration_seconds
            span = (f"{fmt(request.start_seconds)} → {fmt(end)}"
                    f"，共 {int(request.duration_seconds)}s")
        else:
            span = f"{fmt(request.start_seconds)} → 录像结束"
        # The model's own restatement, next to the numbers it produced. If the
        # two disagree the reading was wrong, and this is where to catch it.
        # A plain line, not a blockquote: markdown's lazy continuation would
        # pull the settings line below it into the quote.
        if intent.source == "regex":
            note = ("\n⚠️ 没能连上模型，这句话是用旧的关键词规则读的——它只认得少数几种写法，"
                    "上面的区间请自己核一眼。")
        elif intent.reading:
            note = f"\n🧾 我的理解：{intent.reading}。不对就打断我，重说一遍。"
        else:
            note = ""
        # Naming the rules up front is the cheapest way to catch the mistake of
        # pasting a standard into the chat box and assuming it took effect.
        rules = load_rules()
        return (
            f"🎬 开始稽核 `{request.target}`（{span}）{note}\n\n"
            f"窗口 {config.window_seconds}s / 重叠 {config.window_overlap_seconds}s，"
            f"抽帧 {config.analysis_fps} FPS，画质档 {config.media_resolution}，"
            f"并发 {config.analysis_concurrency}。\n"
            f"稽核标准：`{config.sop_rules_path.name}` v{rules.version} 共 {len(rules.rules)} 条"
            f"（{', '.join(rules.ids)}）\n"
            f"实时画面与逐条判定请看监控大屏：http://127.0.0.1:{config.monitor_port}/"
        )

    @staticmethod
    def _format(event: str, payload: dict) -> Optional[str]:
        if event == "capture_mode":
            label = "抓流（Plan A）" if payload["mode"] == "stream" else "录屏（Plan B）"
            return f"📡 采集方式：{label} — {payload['reason']}"
        if event == "window":
            mark = {"VIOLATION": "❌", "CANNOT_DETERMINE": "❔"}.get(payload["status"], "✅")
            suffix = f"，{payload['violations']} 项违规（{payload['severity']}）" if payload["violations"] else ""
            index = payload.get("window_index", payload["id"])
            return f"{mark} 窗口 #{index + 1} [{payload['time_range']}]{suffix}"
        if event == "window_skipped":
            return f"⏭️ 跳过窗口 [{payload['time_range']}]：{payload['reason']}"
        if event == "video_ended":
            return f"🏁 录像播放结束（{payload['at_seconds']:.0f}s），收尾中…"
        if event == "player_geometry_restored":
            return "🔲 播放器被弹窗挤出了全屏，已恢复（画面裁剪范围保持不变）"
        if event == "player_geometry_lost":
            return (
                "⚠️ 播放器离开了全屏且恢复失败，后续画面的裁剪范围已经对不上，"
                "判定结果仅供参考"
            )
        if event == "playback_recovering":
            return (
                f"⏯️ 播放在 {Clip.format_offset(payload['at_seconds'])} 处卡住，"
                f"尝试恢复播放（第 {payload['attempt']} 次）…"
            )
        if event == "playback_stalled":
            tries = payload.get("recovery_attempts") or 0
            tail = f"（已尝试 {tries} 次恢复）" if tries else ""
            return (
                f"⏸️ 播放在 {Clip.format_offset(payload['at_seconds'])} 处停止推进已 "
                f"{payload['for_seconds']:.0f}s{tail}，停止采集"
            )
        if event == "budget_stop":
            return f"🛑 停止采集：{payload['reason']}"
        if event == "capture_error":
            return f"⚠️ 采集异常：{payload['error']}"
        return None

    @staticmethod
    def _report(summary: dict, store: AuditStore) -> str:
        complete = summary.get("complete", True)
        if complete:
            headline = "### 📊 稽核完成"
        elif summary.get("stopped_kind") == "video_ended":
            # The footage running out is not the pipeline failing; flagging it
            # with the same warning as a stall teaches people to ignore both.
            headline = "### 📊 稽核完成（视频比请求的时间段短）"
        else:
            headline = "### ⚠️ 稽核提前结束（未覆盖完请求的时间段）"
        lines = [headline]
        if not complete:
            # Above the numbers, not below them: the first thing a reader takes
            # from a clean-looking report is that the footage was clean.
            lines.append(
                f"> **以下结论只代表已看过的片段**：{summary.get('incomplete_reason')}。"
                f"未覆盖的时间段既没有通过，也没有不通过。"
            )
        lines += [
            f"- 分析窗口：{summary['windows_analyzed']} 个"
            + (f"（{summary['windows_failed']} 个失败）" if summary["windows_failed"] else ""),
            f"- 稽核区间：{_span(summary)}",
            f"- 发现违规：{summary['violations']} 项，其中红线 {summary['red_line_violations']} 项",
            f"- 采集方式：{summary['capture_mode']}，耗时 {summary['elapsed_seconds']}s",
            f"- 结束原因：{summary['stopped_because']}",
            f"- Token：输入 {summary['input_tokens']}，输出 {summary['output_tokens']}",
            f"- 明细：`{summary['records_path']}`，证据帧：`{summary['evidence_dir']}`",
        ]
        # Sorted by video time, not by which window's analysis finished first.
        # With ANALYSIS_CONCURRENCY > 1 those differ, and a table that jumps
        # 01:12 → 01:00 → 01:24 is unreadable however correct each row is.
        shown = store.violations(limit=_MAX_TABLE_ROWS)
        total = len(store.violations())
        if shown:
            # One table across the whole run, not one per window. A three-hour
            # audit is hundreds of windows; the per-window visual scan lives in
            # the JSONL and on the dashboard, where it can be paged through.
            lines += [
                "\n### ⚠️ 违规判定结果",
                "| 违规规则 | 相关时间范围 | 判定依据 | 严重程度 | 证据帧 |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
            for record in shown:
                # Within a window too: the model reports findings in whatever
                # order it thought of them.
                findings = sorted(record["findings"], key=lambda f: f.get("offset_seconds", 0.0))
                for finding in findings:
                    if finding["status"] != "VIOLATION":
                        continue
                    lines.append(
                        f"| {finding['rule_id']} {finding.get('rule_name', '')} "
                        # `timestamp` is already an absolute video offset, not a
                        # position inside the clip file -- labelling it "本片段"
                        # made 03:14 read as 3 minutes into a 15-second window.
                        f"| {record['time_range']}（定位 {finding['timestamp']}）"
                        f"| {finding['evidence'].replace('|', '/')} "
                        f"| {_SEVERITY_LABEL.get(finding['severity'], finding['severity'])}"
                        f"（置信度 {finding['confidence']:.0%}）"
                        f"| {finding['evidence_frame'] or '—'} |"
                    )
            if total > len(shown):
                # A silently truncated table reads as the complete finding
                # list, which on a three-hour audit is the difference between
                # "12 violations" and "12 violations that we bothered to print".
                lines.append(
                    f"\n> 表里只列了时间最早的 {len(shown)} 个窗口，本次共 {total} 个窗口有违规。"
                    f"完整清单在 `{summary['records_path']}`。"
                )
        else:
            lines.append("\n✅ 未发现违反 SOP 的行为。")
        lines.append(
            "\n> 每个窗口的「视觉证据锚定」与完整动作描述记录在 JSONL 的 "
            "`visual_scan` / `action_narrative` 字段中，可用于复核判定是否有据。"
        )
        return "\n".join(lines)


root_agent = CctvAuditAgent(
    name="cctv_audit_agent",
    description=(
        "霸王茶姬门店 CCTV 视频智能稽核 Agent。打开监控/视频页面，按时间窗口连续采集视频，"
        "用 Gemini 原生视频理解逐窗口比对 SOP，输出带证据帧的违规清单。"
    ),
)
