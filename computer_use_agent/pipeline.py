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

"""Wires the layers together: navigate -> capture -> analyse -> store.

One producer coroutine turns the playing page into a stream of MP4 windows and
pushes them onto a bounded queue; `ANALYSIS_CONCURRENCY` workers pull from it,
call Gemini once per window, and hand the verdict to the store.

The queue bound is the back-pressure mechanism. If analysis falls behind, the
producer blocks rather than filling the disk with clips nobody has looked at.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from playwright.async_api import async_playwright

from .analyzer import VideoAnalyzer, load_rules
from .capture.preview import LivePreview
from .capture.probe import StreamProbe
from .capture.screen_recorder import ScreenRecorder, content_box
from .capture.stream_grabber import StreamGrabber
from .capture.types import CaptureSource, Clip
from .config import config
from .navigator import ComputerUseFallback, HumanGate, for_target, load_state, platform_of, save_state
from .store import AuditStore

logger = logging.getLogger("cctv_audit.pipeline")

_CHROMIUM_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--autoplay-policy=no-user-gesture-required",
    "--mute-audio",
    "--disable-background-networking",
    "--disable-breakpad",
    "--disable-component-update",
]


def _same_box(a: dict, b: dict, tolerance: float = 4.0) -> bool:
    """Is the player still where it was? Sub-pixel layout jitter does not count."""
    return all(abs(a[k] - b[k]) <= tolerance for k in ("x", "y", "width", "height"))


class BudgetExceeded(Exception):
    """A hard stop was reached. Not an error -- the run did what it was told."""


@dataclass
class Budget:
    """Hard stops. Every one of these is a spend limit, so none of them are soft."""

    max_wall_clock_seconds: int = field(default_factory=lambda: config.max_wall_clock_seconds)
    max_tokens: Optional[int] = field(default_factory=lambda: config.max_cost_tokens)
    max_windows: Optional[int] = field(default_factory=lambda: config.max_windows)
    started_at: float = field(default_factory=time.monotonic)
    tokens_used: int = 0
    windows_started: int = 0

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def reason_to_stop(self) -> Optional[str]:
        if self.max_wall_clock_seconds and self.elapsed() >= self.max_wall_clock_seconds:
            return f"wall-clock budget of {self.max_wall_clock_seconds}s reached"
        if self.max_tokens and self.tokens_used >= self.max_tokens:
            return f"token budget of {self.max_tokens} reached ({self.tokens_used} used)"
        if self.max_windows and self.windows_started >= self.max_windows:
            return f"window budget of {self.max_windows} reached"
        return None


@dataclass
class AuditRequest:
    target: str
    start_seconds: float = 0.0
    duration_seconds: Optional[float] = None  # None -> run until a budget stops us


class AuditPipeline:
    def __init__(
        self,
        store: Optional[AuditStore] = None,
        gate: Optional[HumanGate] = None,
        on_preview_frame: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str, dict], None]] = None,
    ):
        problems = config.validate()
        if problems:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(problems))
        config.ensure_dirs()

        self.store = store or AuditStore()
        self.gate = gate or HumanGate()
        self.on_preview_frame = on_preview_frame
        self.on_status = on_status
        self.analyzer = VideoAnalyzer(load_rules())
        self.budget = Budget()
        self.source: Optional[CaptureSource] = None
        self._stop = asyncio.Event()
        self._stop_reason: Optional[str] = None
        # Why we stopped, in a form code can branch on: "video_ended",
        # "playback_stalled", "requested". The reason string is for humans.
        self._stop_kind: Optional[str] = None
        # Where the player was when the recorder started. The capture geometry
        # -- the viewport, and ffmpeg's crop if there is one -- is fixed for the
        # life of the recording, so if the page moves the player afterwards the
        # recording silently starts framing the wrong pixels.
        self._player_rect: Optional[dict] = None
        self._geometry_warned = False
        # Video offset past which there is no more footage, only the player's
        # frozen last frame. Set by the watchdog; honoured by the workers.
        self._footage_ends_at: Optional[float] = None

    def stop(self, reason: str = "stop requested", kind: str = "requested") -> None:
        """Requests a graceful shutdown from outside the pipeline."""
        if self._stop_reason is None:
            self._stop_reason = reason
            self._stop_kind = kind
        self._stop.set()

    def _emit(self, event: str, payload: dict) -> None:
        logger.info("[%s] %s", event, payload)
        if self.on_status is not None:
            try:
                self.on_status(event, payload)
            except Exception as exc:
                logger.debug("Status hook failed: %s", exc)

    async def run(self, request: AuditRequest) -> dict:
        work_dir = config.work_dir / f"run_{int(time.time())}"
        work_dir.mkdir(parents=True, exist_ok=True)
        platform = platform_of(request.target)

        playwright = await async_playwright().start()
        browser = context = preview = None
        try:
            browser = await playwright.chromium.launch(headless=config.headless, args=_CHROMIUM_ARGS)
            context = await browser.new_context(
                viewport={"width": config.screen_width, "height": config.screen_height},
                storage_state=load_state(platform),
            )
            page = await context.new_page()

            # Started before navigation, not after: the login wall, the CAPTCHA
            # and the "video unavailable" page are exactly the moments an
            # operator needs to see, and they all happen before playback.
            if self.on_preview_frame is not None:
                preview = LivePreview(
                    page=page,
                    on_frame=self.on_preview_frame,
                    fps=config.preview_fps,
                    width=config.preview_width,
                    height=config.preview_height,
                    quality=config.preview_quality,
                )
                await preview.start()

            navigator = for_target(request.target, gate=self.gate, fallback=ComputerUseFallback())

            # Attach before navigating: the player fetches its manifest within
            # the first second, and a probe attached afterwards misses it.
            probe = StreamProbe(page)
            probe.attach()

            self._emit("navigating", {"target": request.target, "platform": platform})
            await navigator.login(page)
            await navigator.open_target(page, request.target)
            await navigator.ensure_playing(page)

            # Persist the session now that we are past any challenge, so the
            # next run starts already logged in.
            with contextlib.suppress(Exception):
                await save_state(context, platform)

            if request.start_seconds > 0:
                await navigator.seek_to(page, request.start_seconds)

            source = await probe.decide(config.capture_mode, wait_seconds=config.stream_probe_seconds)
            probe.detach()
            self.source = source
            self._emit("capture_mode", {"mode": source.mode, "reason": source.reason})

            producer = await self._build_producer(
                source, page, navigator, work_dir, request,
            )

            queue: asyncio.Queue = asyncio.Queue(maxsize=config.clip_queue_size)
            capture_task = asyncio.create_task(self._capture(producer, queue, request))
            workers = [
                asyncio.create_task(self._analyse(queue, i))
                for i in range(config.analysis_concurrency)
            ]
            # Plan A pulls the media independently and stops at EOF by itself;
            # only the page-bound recorder needs minding.
            watchdog = (
                asyncio.create_task(self._watch_page(page, navigator))
                if source.mode == "screen" else None
            )

            try:
                await capture_task
            finally:
                if watchdog is not None:
                    watchdog.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await watchdog
                # Sentinel per worker so each one exits after draining.
                for _ in workers:
                    await queue.put(None)
                await asyncio.gather(*workers, return_exceptions=True)

        finally:
            for closer in (
                lambda: preview.aclose() if preview else None,
                lambda: context.close() if context else None,
                lambda: browser.close() if browser else None,
                playwright.stop,
            ):
                try:
                    result = closer()
                    if result is not None:
                        await result
                except Exception as exc:
                    logger.debug("Shutdown step failed: %s", exc)
            if not config.keep_clips:
                shutil.rmtree(work_dir, ignore_errors=True)

        summary = {
            **self.store.summary(),
            "capture_mode": self.source.mode if self.source else "none",
            "elapsed_seconds": round(self.budget.elapsed(), 1),
            "stopped_because": (
                self.budget.reason_to_stop() or self._stop_reason or "采集结束"
            ),
            **self._coverage(request),
        }
        self._emit("finished", summary)
        return summary

    def _coverage(self, request: AuditRequest) -> dict:
        """States plainly how much of the requested span was actually audited.

        Without this the report says "📊 稽核完成" whether it watched the ten
        minutes it was asked for or the first forty-eight seconds of them --
        and a quality report that overstates its own coverage is worse than no
        report, because nobody goes back to check.
        """
        span = self.store.covered_span()
        requested_end = (
            request.start_seconds + request.duration_seconds
            if request.duration_seconds
            else None
        )
        out = {
            "requested_start_seconds": round(request.start_seconds, 1),
            "requested_end_seconds": round(requested_end, 1) if requested_end else None,
            "covered_from_seconds": round(span[0], 1) if span else None,
            "covered_to_seconds": round(span[1], 1) if span else None,
            "stopped_kind": self._stop_kind,
            "complete": True,
            "incomplete_reason": None,
        }

        # One window of slack: the last clip is cut on a window boundary, so
        # landing a few seconds short of the request is not a shortfall.
        tolerance = max(float(config.window_seconds), 5.0)
        fmt = Clip.format_offset
        if requested_end is not None:
            if span is None:
                out["complete"] = False
                out["incomplete_reason"] = "没有采集到任何可分析的片段"
            elif span[1] < requested_end - tolerance:
                missing = int(requested_end - span[1])
                out["complete"] = False
                # "Asked for ten minutes of a video that only has fifteen
                # seconds left" is a different thing from "gave up early", and
                # flagging both the same way trains people to ignore the flag.
                if self._stop_kind == "video_ended" and self._footage_ends_at:
                    out["incomplete_reason"] = (
                        f"视频本身在 {fmt(self._footage_ends_at)} 就结束了，"
                        f"请求的 {fmt(requested_end)} 超出了视频长度，"
                        f"实际稽核到 {fmt(span[1])}"
                    )
                else:
                    out["incomplete_reason"] = (
                        f"只覆盖到 {fmt(span[1])}，请求的是到 {fmt(requested_end)}，"
                        f"还差约 {missing} 秒未稽核"
                    )
        elif self._stop_kind == "playback_stalled":
            out["complete"] = False
            out["incomplete_reason"] = (
                f"播放中途卡住，只覆盖到 {fmt(span[1])}，视频后段未稽核"
                if span else "播放中途卡住，没有采集到任何可分析的片段"
            )
        return out

    async def _build_producer(self, source, page, navigator, work_dir: Path, request: AuditRequest):
        if source.mode == "stream":
            # Purely for the operator: nothing here is recorded, but a
            # dashboard showing a postage-stamp player inside a page of
            # sidebars and comments is not worth watching. Costs one call.
            if config.fullscreen_player:
                await navigator.enter_fullscreen(page)

            # Plan A pulls the media independently of the page, so the playhead
            # is irrelevant; ffmpeg seeks with -ss instead. The page is left
            # playing on purpose -- some platforms invalidate the media token
            # when the player stops.
            return StreamGrabber(
                source=source,
                work_dir=work_dir,
                window_seconds=config.window_seconds,
                overlap_seconds=config.window_overlap_seconds,
                start_offset=request.start_seconds,
            )

        # Plan B is bound to the page, so the playhead is the ground truth for
        # where the first window starts.
        if config.playback_rate != 1.0:
            await navigator.set_playback_rate(page, config.playback_rate)

        # Fullscreen first, crop second. Fullscreen makes the *player* render
        # at full size, so the detail is really there rather than merely
        # un-cropped -- but it does not remove letterbox bars, it enlarges
        # them. So crop afterwards either way: to the player when fullscreen
        # was refused, to the footage inside the player when it worked.
        fullscreen = False
        if config.fullscreen_player:
            fullscreen = await navigator.enter_fullscreen(page)
            self._emit("player_fullscreen", {"ok": fullscreen})

        # Measured whether or not we end up cropping: the watchdog needs a
        # baseline to notice the page moving the player, and a video whose
        # aspect matches the viewport needs no crop yet is hurt just as badly
        # by falling out of fullscreen -- more so, because then the recording
        # is the whole page, sidebar and all, with the footage in one corner.
        rect = await navigator.video_rect(page)
        self._player_rect = dict(rect) if rect else None

        crop = None
        if config.crop_to_player:
            if not rect:
                logger.info("Could not locate the player; recording the whole viewport.")
            elif fullscreen:
                crop = content_box(rect)
            else:
                crop = content_box(rect) or (rect["width"], rect["height"], rect["x"], rect["y"])

        # Setting up -- probing for a stream, going fullscreen, measuring the
        # player -- takes ten-odd seconds, and the video plays throughout. Seek
        # back now that the recorder is about to start, or an audit asked to
        # begin at 01:40 quietly begins at 01:52 and nobody ever sees 01:40.
        #
        # This has to include start=0. "From the beginning" is the most common
        # request there is, and the old `> 0` guard skipped exactly that case:
        # the first window came out at 00:14, and the opening fourteen seconds
        # -- which for a store audit is the whole hand-washing step -- were
        # never looked at, with nothing in the report to say so.
        playhead = await navigator.read_player_time(page)
        if abs((playhead or 0.0) - request.start_seconds) > 1.0:
            # A live stream has no meaningful zero to go back to, and seeking
            # one lands you at the start of the DVR window instead.
            state = await navigator.read_playback_state(page)
            if (state or {}).get("duration") is not None:
                with contextlib.suppress(Exception):
                    await navigator.seek_to(page, request.start_seconds)
                playhead = await navigator.read_player_time(page)
        start_offset = playhead if playhead is not None else request.start_seconds
        if playhead is None:
            logger.warning(
                "Could not read the playhead; timestamps will be relative to %.1fs.",
                start_offset,
            )

        return ScreenRecorder(
            page=page,
            work_dir=work_dir,
            window_seconds=config.window_seconds,
            overlap_seconds=config.window_overlap_seconds,
            capture_fps=config.capture_fps,
            width=config.screen_width,
            height=config.screen_height,
            playback_rate=config.playback_rate,
            start_offset=start_offset,
            crop=crop,
        )

    async def _hold_geometry(self, page, navigator) -> None:
        """Puts the player back where the recorder expects to find it.

        The capture geometry is fixed when the recorder starts and can never
        follow the player. Meanwhile the page moves it: measured on bilibili,
        the login modal that appears a minute in drops the player out of web
        fullscreen, from (0,0) 1920x1080 to (62,172) 1354x762. Dismissing the
        modal does not put it back.

        Nothing errors when that happens, which is what makes it bad. The
        recording continues and the frames are the right size, but they are now
        the wrong pixels. With a crop -- computed for a fullscreen 9:16 video --
        it frames the page header, the title bar and the left half of the
        picture. Without one it is worse: the frame becomes the entire bilibili
        page, and the footage that is actually being audited is a third of it.
        Either way the model goes on grading SOP compliance against a scene it
        can barely see, and reports it with full confidence.
        """
        if self._player_rect is None:
            return
        rect = await navigator.video_rect(page)
        if not rect or _same_box(rect, self._player_rect):
            return

        was = self._player_rect
        if config.fullscreen_player and await navigator.enter_fullscreen(page):
            rect = await navigator.video_rect(page)
            if rect and _same_box(rect, was):
                self._emit("player_geometry_restored", {
                    "from": [round(was["x"]), round(was["y"]),
                             round(was["width"]), round(was["height"])],
                })
                logger.info("Player had left fullscreen; restored it.")
                return

        # Could not restore. Say so once -- the alternative is a clean-looking
        # report built on mis-framed footage.
        if not self._geometry_warned:
            self._geometry_warned = True
            self._emit("player_geometry_lost", {
                "expected": [round(was["x"]), round(was["y"]),
                             round(was["width"]), round(was["height"])],
                "actual": ([round(rect["x"]), round(rect["y"]),
                            round(rect["width"]), round(rect["height"])] if rect else None),
            })
            logger.warning(
                "Player moved from %sx%s@(%s,%s) and could not be restored; the "
                "recording is now framing the wrong region.",
                round(was["width"]), round(was["height"]), round(was["x"]), round(was["y"]),
            )

    async def _watch_page(self, page, navigator) -> None:
        """Housekeeping for Plan B, which is the only mode tied to the page.

        Two jobs the recorder cannot do for itself:

        * Keep the player visible. Login nags and session-expiry prompts get
          re-raised while the audit runs, and the recorder would happily
          capture a dialog box sitting on top of the footage.
        * Notice the end. A finished <video> renders its last frame forever, so
          without this a 15-minute recording keeps producing identical windows
          until the wall-clock budget expires -- billing for every one of them.

        Stalls are treated as end-of-video only after several consecutive
        polls, so buffering on a slow link does not abort a live audit -- but a
        stall in the last few seconds of a known duration is confirmed sooner,
        because there is nothing left it could be buffering.

        A stall in the *middle* of a known duration is a different animal: at
        179s of a 916s video nothing has ended, the player has been paused by a
        login nag or has run out of buffer. So mid-video we try to restart
        playback before writing the run off -- an audit that was asked for ten
        minutes and quietly delivered forty-eight seconds is worse than one
        that takes a moment longer.
        """
        stalled = 0
        recoveries = 0
        poll = max(config.page_watch_seconds, 1.0)
        # 30s of no movement mid-recording; 10s once we are within TAIL of the
        # end. Observed: bilibili reports duration 916s but playback actually
        # stops at 910.5s, so "current >= duration" alone never fires.
        stall_limit = max(3, int(30.0 / poll))
        tail_stall_limit = max(2, int(10.0 / poll))
        _TAIL_SECONDS = 15.0
        # Nudge the player every 10s of stall, up to three times, before
        # concluding the footage is over.
        recover_every = max(2, int(10.0 / poll))
        max_recoveries = 3
        last_time: Optional[float] = None

        while not self._stop.is_set():
            await asyncio.sleep(config.page_watch_seconds)
            try:
                await navigator.keep_clear(page)
                await self._hold_geometry(page, navigator)
                state = await navigator.read_playback_state(page)
            except Exception as exc:
                logger.debug("Page watchdog poll failed: %s", exc)
                continue

            if not config.stop_on_video_end or not state:
                continue

            if state.get("ended"):
                at = round(state.get("current_time") or 0.0, 1)
                self._footage_ends_at = at
                self._emit("video_ended", {"at_seconds": at})
                self.stop(f"视频已播放完毕（{Clip.format_offset(at)}）", kind="video_ended")
                return

            current = state.get("current_time")
            duration = state.get("duration")
            if duration and current is not None and current >= duration - 1.0:
                self._footage_ends_at = round(current, 1)
                self._emit("video_ended", {"at_seconds": round(current, 1)})
                self.stop(
                    f"视频已播放完毕（{Clip.format_offset(current)}）", kind="video_ended"
                )
                return

            # A paused player is not necessarily finished -- a human may be
            # working the gate -- so only a playhead that stops moving counts.
            near_end = bool(duration and current is not None and current >= duration - _TAIL_SECONDS)
            if current is not None and last_time is not None and current <= last_time + 0.05:
                stalled += 1
                if (
                    not near_end
                    and recoveries < max_recoveries
                    and stalled % recover_every == 0
                ):
                    recoveries += 1
                    self._emit("playback_recovering", {
                        "at_seconds": round(current, 1), "attempt": recoveries,
                    })
                    try:
                        await navigator.keep_clear(page)
                        await navigator.ensure_playing(page)
                    except Exception as exc:
                        logger.debug("Could not restart playback: %s", exc)
                exhausted = near_end or recoveries >= max_recoveries
                if exhausted and stalled >= (tail_stall_limit if near_end else stall_limit):
                    held = round(stalled * poll, 1)
                    self._footage_ends_at = round(current, 1)
                    self._emit("playback_stalled", {
                        "at_seconds": round(current, 1),
                        "for_seconds": held,
                        "recovery_attempts": recoveries,
                    })
                    tail = "" if near_end else f"，已尝试 {recoveries} 次恢复播放"
                    self.stop(
                        f"播放在 {Clip.format_offset(current)} 卡住 {held:.0f} 秒不再前进{tail}",
                        kind="playback_stalled",
                    )
                    return
            else:
                if stalled:
                    logger.info("Playback resumed at %.1fs.", current or 0.0)
                # Playback moved, so whatever we did worked (or it recovered on
                # its own). Give the next stall a fresh set of attempts.
                stalled = 0
                recoveries = 0
            last_time = current

    async def _capture(self, producer, queue: asyncio.Queue, request: AuditRequest) -> None:
        end_offset = (
            request.start_seconds + request.duration_seconds
            if request.duration_seconds
            else None
        )
        clips = producer.clips()
        try:
            async for clip in clips:
                reason = self.budget.reason_to_stop()
                if reason:
                    self._emit("budget_stop", {"reason": reason})
                    break
                if self._stop.is_set():
                    self._emit("budget_stop", {"reason": self._stop_reason or "stop requested"})
                    break
                if end_offset is not None and clip.start_offset >= end_offset:
                    reason = f"已采集到请求的终点 {Clip.format_offset(end_offset)}"
                    self.stop(reason, kind="requested")
                    self._emit("budget_stop", {"reason": reason})
                    break

                self.budget.windows_started += 1
                # Blocks when the queue is full: this is the back-pressure that
                # stops capture outrunning analysis.
                await queue.put(clip)
        except Exception as exc:
            logger.exception("Capture failed: %s", exc)
            self._emit("capture_error", {"error": str(exc)[:300]})
        finally:
            # Breaking out of `async for` does not run the generator's own
            # cleanup; closing it explicitly is what stops ffmpeg.
            with contextlib.suppress(Exception):
                await clips.aclose()
            with contextlib.suppress(Exception):
                await producer.aclose()

    async def _analyse(self, queue: asyncio.Queue, worker_id: int) -> None:
        while True:
            clip: Optional[Clip] = await queue.get()
            try:
                if clip is None:
                    return
                # Windows queued before the watchdog noticed the end hold
                # nothing but the player's frozen last frame. Analysing them
                # costs a full request each and can only produce a verdict
                # about a still image that no longer represents the shop.
                if (
                    self._footage_ends_at is not None
                    and clip.start_offset >= self._footage_ends_at
                ):
                    self._emit("window_skipped", {
                        "time_range": clip.time_range,
                        "reason": f"starts after the footage ended at "
                                  f"{self._footage_ends_at:.0f}s",
                    })
                    continue
                outcome = await self.analyzer.analyze(clip)
                self.budget.tokens_used += outcome.input_tokens + outcome.output_tokens
                record = await self.store.record(outcome)
                if record is not None:
                    self._emit("window", {
                        "id": record["id"],
                        # `id` counts every window ever archived, across every
                        # run; `window_index` counts this one. Showing the
                        # former as "窗口 #131" next to a 00:14 timestamp reads
                        # as "we skipped 130 windows".
                        "window_index": record["window_index"],
                        "time_range": record["time_range"],
                        "status": record["sop_status"],
                        "severity": record["severity"],
                        "violations": record["violation_count"],
                    })
            except Exception as exc:
                logger.exception("Worker %d failed on a window: %s", worker_id, exc)
            finally:
                # Evidence frames are already extracted, so the clip has served
                # its purpose. Multi-hour runs would otherwise fill the disk.
                if clip is not None and not config.keep_clips:
                    with contextlib.suppress(OSError):
                        clip.path.unlink(missing_ok=True)
                queue.task_done()
