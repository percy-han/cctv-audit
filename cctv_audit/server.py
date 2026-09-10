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

"""The Agent Runtime container: what Gemini Enterprise actually talks to.

The contract is not documented in prose; every shape below was read off the
wire or out of the SDK during Phase 0, and `deploy/phase0/README.md` keeps the
evidence. The parts that cost the most to find:

  * GE calls **one** method, `streaming_agent_run_with_events`, on the
    **streaming** route. Whatever else `classMethods` declares is reachable
    only by a direct `:query` -- useful for debugging, invisible to customers.
  * Each line of the stream must be the envelope
    `{"events": [<ADK Event>], "session_id": ...}`. Bare events are dropped in
    silence, with no diagnostic beyond "produced no events".
  * `request_json` is a JSON string inside the JSON body. Double-encoded.

And the constraint that shapes everything: work awaited inside the streaming
request is cancelled at 900s, while GE has already given up at 602s. So no
handler here may await an audit. The turn that starts one returns a job number
in seconds; a later turn collects the result. `AuditService` owns that split --
this module is only the translation layer between a chat sentence and those
three calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .audit_service import AuditService, audit_service
from .config import config
from .intent import UnreadableTimeSpan, from_fields
from .jobs import Job
from .logsetup import setup_logging
from .turn import ModelUnavailable, Turn, read_intent, read_turn

# Before the first logger is used, and at import time rather than on startup:
# uvicorn imports this module, so anything logged while the app is being built
# is already inside the window this covers. Without it the root logger has no
# handler and every INFO in the audit -- the whole diagnostic trail -- is
# discarded before it reaches Cloud Logging. See `logsetup.py`.
setup_logging()

logger = logging.getLogger("cctv_audit.server")

app = FastAPI(title="CCTV audit agent")

AUTHOR = "cctv-audit"
GE_METHOD = "streaming_agent_run_with_events"

INSTANCE_ID = (
    os.environ.get("K_REVISION") or os.environ.get("HOSTNAME") or "local"
)

# One line, at import, saying what this container is actually configured to do.
# Written because on 2026-09-09 the only way to answer "is the deployment on
# agentic?" was to read the engine's env vars out of the API -- the logs said
# nothing, and the per-window token counts of static/60s and agentic/60s are
# close enough to be ambiguous. A deployment that cannot state its own settings
# forces every question about them into a deploy-time archaeology exercise.
logger.info(
    "Config: instance=%s processing=%s window=%ss overlap=%ss model=%s "
    "capture=%s concurrency=%s resolution=%s fps=%s",
    INSTANCE_ID,
    config.media_processing,
    config.window_seconds,
    config.window_overlap_seconds,
    config.analysis_model,
    config.capture_mode,
    config.analysis_concurrency,
    config.media_resolution,
    config.analysis_fps,
)


# ----------------------------------------------------------------------
# The ADK wire shapes. Copied field for field from a real
# `google.adk.events.Event` dump -- see phase0/README.md. snake_case, and
# `exclude_none=True`, so optional fields are omitted rather than sent null.
# ----------------------------------------------------------------------
def _adk_event(text: str, invocation: str, *, partial: bool = False) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "content": {"parts": [{"text": text}], "role": "model"},
        "invocation_id": invocation,
        "author": AUTHOR,
        "actions": {
            "state_delta": {},
            "artifact_delta": {},
            "requested_auth_configs": {},
            "requested_tool_confirmations": {},
        },
        "id": uuid.uuid4().hex[:8],
        "timestamp": time.time(),
    }
    if partial:
        event["partial"] = True
    return event


def _envelope(event: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"events": [event]}
    if session_id:
        out["session_id"] = session_id
    return out


# ----------------------------------------------------------------------
# One GE turn
# ----------------------------------------------------------------------
async def serve_turn(
    payload: Dict[str, Any],
    service: Optional[AuditService] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Times `_serve_turn` and says so when the stream ends.

    Only the *arrival* of a turn used to be logged, and that gap cost an
    afternoon. "Gemini Enterprise answers slowly while a job is running" could
    not be answered from the logs at all, because nothing recorded when we
    finished -- it took starting two real audits and timing the turns from
    outside to establish that this container answers in about three seconds
    whether or not an audit is running. With this line that is a log query.

    `first` is reported separately from the total because this is a stream and
    the customer starts reading at the first sentence. A turn that takes twenty
    seconds to finish but speaks within two is not the same complaint.

    A wrapper rather than a `finally` inside the body: the body has several
    early returns, and the interesting moment is when the last chunk leaves,
    which only the thing doing the iterating can see.
    """
    began = time.monotonic()
    said = 0
    first_at = -1.0
    try:
        async for chunk in _serve_turn(payload, service):
            said += 1
            if first_at < 0:
                first_at = time.monotonic() - began
            yield chunk
    finally:
        # `session` so two turns in flight at once stay tellable apart --
        # container_concurrency is 2.
        session = ""
        try:
            session = read_turn(payload).session_id
        except Exception:  # pragma: no cover - logging must not break a turn
            pass
        logger.info(
            "turn done session=%s in %.2fs (first sentence %.2fs, %d sentences)",
            session, time.monotonic() - began, first_at, said,
        )


async def _serve_turn(
    payload: Dict[str, Any],
    service: Optional[AuditService] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Reads one turn, does one thing, yields one envelope per sentence."""
    svc = service or audit_service()
    turn = read_turn(payload)
    invocation = uuid.uuid4().hex[:12]

    def say(text: str) -> Dict[str, Any]:
        return _envelope(_adk_event(text, invocation), turn.session_id)

    logger.info(
        "turn user=%s session=%s instance=%s text=%r",
        turn.user_id, turn.session_id, INSTANCE_ID, turn.text[:120],
    )

    if not turn.user_id:
        # Every stored job is keyed by the caller. Without one there is nothing
        # safe to read or write, and inventing a placeholder would put several
        # customers' audits in the same bucket.
        yield say("没拿到调用者身份，无法为你建单。请从 Gemini Enterprise 里发起。")
        return

    try:
        decision = await read_intent(turn)
    except ModelUnavailable as exc:
        # Deliberately not a keyword fallback. A confident wrong reading is the
        # failure this path exists to avoid; saying "say that again" is cheap.
        logger.warning("Could not read the turn: %s", exc)
        # Says which of the two it was. "读不懂" for a call that never came back
        # sends the customer off rewriting a sentence that was fine -- measured:
        # the same words worked on the retry.
        timed_out = "超时" in str(exc)
        yield say(
            "我这边读你这句话的时候超时了，**不是你说得不清楚**。原样再发一遍就行。"
            if timed_out else
            "我这边一时读不懂这句话，麻烦再说一遍要稽核哪个视频、哪个时间段。"
        )
        return

    try:
        if decision.action == "unclear":
            yield say(decision.question or "没太听懂，能再说一遍吗？")
        elif decision.action == "audit":
            async for line in _do_audit(svc, turn, decision, say):
                yield line
        elif decision.action == "confirm":
            async for line in _do_confirm(svc, turn, say):
                yield line
        else:
            async for line in _do_status(svc, turn, decision, say):
                yield line
    except Exception as exc:
        # A GE turn that raises shows the customer a platform error with no
        # clue what went wrong. A sentence is always better.
        logger.exception("Turn failed")
        yield say(f"处理这句话时出错了：{str(exc)[:200]}")


async def _do_audit(svc, turn: Turn, decision, say):
    """New request: look before spending, then ask for a yes."""
    try:
        intent = from_fields(
            decision.target_url,
            start=decision.start_seconds,
            end=None if decision.end_seconds < 0 else decision.end_seconds,
        )
    except UnreadableTimeSpan as exc:
        yield say(f"时间段没读准，没有开跑：{exc}")
        return
    except ValueError as exc:
        yield say(f"这个请求我没法执行：{exc}")
        return

    yield say("收到，我先去看一眼这个视频……")

    job = await svc.preflight(
        turn.user_id, intent.request,
        session_id=turn.session_id,
        sop_id=decision.sop_id,
    )
    yield say(_describe_preflight(job))


def _describe_preflight(job: Job) -> str:
    """What preflight found, in the words a customer needs to decide."""
    info = job.preflight or {}
    if job.state != "ready":
        return f"❌ 这段查不了：{info.get('problem') or job.error or '未知原因'}"

    lines = [f"找到视频了{'：' + info['title'] if info.get('title') else ''}"]
    lines.append(f"采集方式：{_CAPTURE_LABELS.get(info.get('capture_mode'), '录屏')}")

    analysis = _describe_analysis(info)
    if analysis:
        lines.append(analysis)

    duration = info.get("video_duration_seconds")
    if duration:
        lines.append(f"视频总长：{_hms(duration)}")
    start, end = info.get("requested_start_seconds"), info.get("requested_end_seconds")
    if _is_whole_file(info):
        # The time span is not honoured on this path, so it must not be echoed
        # back as if it were. Saying it out loud is the point: a customer who
        # asked for 05:00-07:00 and gets a report covering half an hour would
        # otherwise conclude the audit ignored them, which is exactly what it
        # did -- the difference is whether they were told before confirming.
        asked = (
            f"（你说的 {_hms(start or 0)} - {_hms(end)} 这次用不上）"
            if end is not None else ""
        )
        lines.append(f"要稽核：整段视频，从头看到尾{asked}")
    elif end is not None:
        lines.append(f"要稽核：{_hms(start or 0)} - {_hms(end)}")
    if info.get("span_available") is False and info.get("problem"):
        lines.append(f"⚠️ {info['problem']}")

    lines.append("")
    lines.append(f"单号 `{job.job_id}`。确认开始稽核吗？回「确认」我就开跑。")
    return "\n".join(lines)


# The default is "录屏" rather than "不知道" because that is what an unknown
# mode used to render as, and Plan B is the fallback the pipeline actually
# picks when it cannot decide. A new mode showing up here as 录屏 is a wrong
# label; showing up as blank would be a broken-looking reply.
_CAPTURE_LABELS = {
    "stream": "抓流",
    "screen": "录屏",
    "file": "直接读文件（GCS）",
}


def _describe_analysis(info: dict) -> str:
    """Which way the footage gets read, in one line.

    Two very different runs hide behind the same "开始稽核": agentic costs 3.4x
    static and takes 45-155s a window instead of ~20s. The customer is being
    asked to confirm one of them, so it should say which. The mode's name and
    nothing else -- what it costs and how careful it is were our editorial and
    do not belong in a line the customer reads as fact.

    The second half is the shape of the run, and it differs by plan: a `gs://`
    object goes to the model whole, everything else is cut into windows.

    Falls back to the running config for jobs whose preflight predates this
    field -- and says nothing at all if even that is unreadable, rather than
    naming a mode that might not be the one about to run.
    """
    mode = info.get("analysis_mode") or config.media_processing
    window = info.get("analysis_window_seconds") or config.window_seconds
    if mode not in ("agentic", "static"):
        return ""
    if _is_whole_file(info):
        return f"分析方式：{mode}，整段视频一次看完，不切片"
    return f"分析方式：{mode}，{window} 秒一段"


def _is_whole_file(info: dict) -> bool:
    """Whether this job reads one whole object instead of a run of windows.

    Reads `analysis_scope` when preflight recorded one and falls back to the
    capture mode, which has meant the same thing since Plan C existed. Two
    sources because jobs created before the field was added are still in
    Firestore, and answering "windows" for one of them would promise a live
    picture that this path has never had.
    """
    scope = info.get("analysis_scope")
    if scope:
        return scope == "whole_file"
    return info.get("capture_mode") == "file"


async def _do_confirm(svc, turn: Turn, say):
    """The customer said yes. Start, and get out of the request fast."""
    job = await svc.get_status(turn.user_id, session_id=turn.session_id)
    if job is None:
        yield say("没找到待确认的稽核单。先告诉我要看哪个视频、哪个时间段。")
        return
    if job.state == "running":
        yield say(f"单号 `{job.job_id}` 已经在跑了，回「好了吗」可以查进度。")
        return
    if job.state != "ready":
        yield say(
            f"单号 `{job.job_id}` 现在是「{job.state}」，不能开始。"
            + (f"原因：{job.error}" if job.error else "")
        )
        return

    started = await svc.start_audit(turn.user_id, job.job_id)
    # No live picture on a whole-file run: there is no page to show and no
    # window-by-window progress to watch, so the link would open an empty
    # dashboard. An empty dashboard is how the last two demos went wrong --
    # the operator reads it as a freeze. Better not to offer it.
    watch = (
        "" if _is_whole_file(job.preflight or {})
        else f"实时画面：{_dashboard_link(started.job_id)}\n"
    )
    yield say(
        f"好，开始稽核了，单号 `{started.job_id}`。\n"
        f"{watch}"
        "这一步要跑一段时间，你随时回「好了吗」查进度，跑完我把报告给你。"
    )


def _dashboard_link(job_id: str) -> str:
    """Where to watch this particular audit.

    Handed over at the moment the audit starts, which is the only moment it is
    any use -- before that there is nothing to see, and afterwards the report
    has already arrived. The `?job=` is what puts the viewer in this audit's
    room rather than in whatever else the service is showing.
    """
    from .config import config

    base = config.monitor_url or f"http://127.0.0.1:{config.monitor_port}"
    return f"{base}/?job={job_id}"


async def _do_status(svc, turn: Turn, decision, say):
    """How is it going, or the finished report."""
    job = await svc.get_status(
        turn.user_id, job_id=decision.job_id, session_id=turn.session_id
    )
    if job is None:
        yield say(
            "没找到你的稽核单。" +
            (f"单号 `{decision.job_id}` 查不到，" if decision.job_id else "") +
            "要不要现在开始一单？给我视频地址和时间段就行。"
        )
        return

    if job.state == "done":
        report = (job.result or {}).get("report") or "跑完了，但报告是空的。"
        yield say(report)
        return
    if job.state == "failed":
        yield say(f"❌ 单号 `{job.job_id}` 跑失败了：{job.error or '未知原因'}")
        return
    if job.state in ("rejected",):
        yield say(f"单号 `{job.job_id}` 没能开始：{job.error or '未知原因'}")
        return
    if job.state == "ready":
        yield say(f"单号 `{job.job_id}` 还等你确认呢，回「确认」我就开跑。")
        return

    progress = job.progress or {}
    windows = progress.get("windows_analyzed", 0)
    violations = progress.get("violations_so_far", 0)
    last = progress.get("last_window_returned", "")
    line = f"单号 `{job.job_id}` 正在跑：已分析 {windows} 个窗口，发现 {violations} 处违规。"
    if last:
        # "最近返回的"，不是"跑到哪了"：窗口是并发分析的，回来的顺序不等于时间顺序。
        line += f"\n最近返回的一个窗口是 {last}。"
    yield say(line)


def _hms(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ----------------------------------------------------------------------
# The two routes the platform requires
# ----------------------------------------------------------------------
def _read_call(body: Dict[str, Any]) -> tuple:
    method = body.get("class_method") or body.get("classMethod") or ""
    payload = body.get("input") or {}
    return str(method), payload if isinstance(payload, dict) else {}


@app.post("/api/stream_reasoning_engine")
async def stream_reasoning_engine(request: Request):
    """The route GE uses. ndjson, one JSON object per line."""
    body = await request.json()
    method, payload = _read_call(body)

    async def lines() -> AsyncIterator[bytes]:
        if method == GE_METHOD:
            async for chunk in serve_turn(payload):
                yield (json.dumps(chunk, ensure_ascii=False) + "\n").encode()
            return
        # Anything else on the streaming route is a direct caller, not GE.
        result = await _call_named(method, payload)
        yield (json.dumps(result, ensure_ascii=False) + "\n").encode()

    # `application/json`, not ndjson -- that is what ADK's own server sends and
    # what the platform accepts.
    return StreamingResponse(lines(), media_type="application/json")


@app.post("/api/reasoning_engine")
async def reasoning_engine(request: Request):
    """The unary route. Not used by GE; kept for `:query` and for debugging."""
    body = await request.json()
    method, payload = _read_call(body)
    return JSONResponse({"output": await _call_named(method, payload)})


async def _call_named(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """The three operations, reachable directly. Same code GE reaches."""
    svc = audit_service()
    user_id = str(payload.get("user_id") or "")
    if not user_id:
        return {"error": "user_id is required"}

    if method == "preflight":
        intent = from_fields(
            payload.get("target") or payload.get("url") or "",
            start=payload.get("start"),
            end=payload.get("end"),
            duration=payload.get("duration"),
        )
        job = await svc.preflight(
            user_id, intent.request,
            session_id=str(payload.get("session_id") or ""),
            sop_id=str(payload.get("sop_id") or ""),
        )
        return job.to_dict()
    if method == "start_audit":
        job = await svc.start_audit(user_id, str(payload.get("job_id") or ""))
        return job.to_dict()
    if method == "get_status":
        job = await svc.get_status(
            user_id,
            job_id=str(payload.get("job_id") or ""),
            session_id=str(payload.get("session_id") or ""),
        )
        return job.to_dict() if job else {"error": "no such job"}
    return {"error": f"unknown method: {method}"}


@app.get("/")
async def index():
    return {"service": "cctv-audit", "instance": INSTANCE_ID, "ge_method": GE_METHOD}


# How long `/is_busy` holds the response open while an audit is running.
# Zero is a plain, instant answer. Anything above zero turns the probe into a
# long poll, and the reason is measured rather than stylistic -- see the
# endpoint below. `update-spec` can change this without a rebuild, which is the
# whole point of it being an environment variable.
KEEPALIVE_HOLD_SECONDS = float(os.environ.get("KEEPALIVE_HOLD_SECONDS", "0") or 0)
_HOLD_STEP_SECONDS = 1.0


@app.get("/is_busy")
async def is_busy():
    """Says whether an audit is running, and optionally waits while it is.

    This exists because of a measurement, not a guess. An audit runs on a task
    deliberately detached from the request that started it -- the only shape
    that survives Gemini Enterprise cancelling in-request work at 900s. But an
    instance with no request in flight has its CPU taken away, and the kernel
    says so plainly: `/proc/self/schedstat` reported this process runnable but
    unscheduled for **79-85% of every window** while the audit ran unattended,
    with Chromium's screencast down to 0.0 fps and the operator's dashboard a
    slideshow. Holding requests in flight against the same instance dropped
    that to 30% and the picture went to 6.6 fps immediately.

    So the audit is not slow. It is not being given a processor, and the
    decision that keeps it alive is what starves it.

    `keepAliveProbe` in the deployment spec is the platform's own lever here,
    and its example path in the API schema is literally `/is_busy`. What is
    *not* documented anywhere -- the runtime-contract and optimize-and-scale
    pages say nothing about probes or CPU -- is whether being probed restores
    the CPU, or merely stops the instance being reclaimed. A probe answered in
    two milliseconds every few seconds would do nothing for a starved audit.

    Hence the hold. With `KEEPALIVE_HOLD_SECONDS` above zero the probe becomes
    a long poll: the response is kept open while there is work, so there is
    always a request in flight, which is the condition that was measured to
    work. It costs one concurrency slot for as long as an audit runs, so
    `container_concurrency` must be at least two or the held probe will crowd
    out the real traffic -- that is not a tuning preference, it is arithmetic.

    Answered with 200 either way, never a 5xx. "Nothing to do" is a truthful
    answer, not a failure, and a probe that reports an idle container as
    unhealthy invites the platform to do something about it.
    """
    service = audit_service()
    running = service.runner.running
    if running and KEEPALIVE_HOLD_SECONDS > 0:
        deadline = time.monotonic() + KEEPALIVE_HOLD_SECONDS
        # Stepwise, so a job that finishes early releases the slot instead of
        # holding it for the rest of the window.
        while time.monotonic() < deadline and service.runner.running:
            await asyncio.sleep(
                min(_HOLD_STEP_SECONDS, deadline - time.monotonic()))
    return {
        "busy": bool(running),
        "jobs": running,
        "instance": INSTANCE_ID,
        "held_seconds": KEEPALIVE_HOLD_SECONDS if running else 0,
    }
