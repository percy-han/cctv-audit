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
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

from playwright.async_api import async_playwright

from .analyzer import SopRuleSet, VideoAnalyzer, load_rules
from .artifacts import artifact_sink
from .capture import gcs_video
from .capture.preview import LivePreview
from .capture.probe import StreamProbe
from .capture.screen_recorder import ScreenRecorder, content_box
from .capture.stream_grabber import StreamGrabber
from .capture.types import CaptureSource, Clip
from .config import config
from .gcp import id_token_for
from .navigator import ComputerUseFallback, HumanGate, for_target, load_state, platform_of, save_state
from .navigator.base import NavigationError
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


async def _attach_oidc_headers(context) -> None:
    """Adds a Google ID token to requests bound for IAM-protected origins.

    One route per configured origin, never a catch-all `**/*`. The pattern is
    what enforces the scoping: Playwright only invokes the handler for URLs
    that match it, so a page on a protected origin that pulls a thumbnail from
    a CDN cannot pick the token up on the way past. A single handler that
    inspected `route.request.url` itself would work too, and would put the one
    check that matters inside a function instead of in the router where it
    cannot be skipped by an early return.

    Does nothing when `OIDC_ORIGINS` is empty, which is the default and the
    only configuration that has ever run against bilibili.
    """
    for origin in config.oidc_origins:
        token = id_token_for(origin)
        if not token:  # origin is listed but minting declined -- nothing to add
            continue
        await context.route(f"{origin}/**", _authorising_handler(f"Bearer {token}"))
        logger.info("Requests to %s will carry a Google ID token.", origin)


def _authorising_handler(header: str):
    """A one-argument route handler that adds `Authorization: <header>`.

    A closure and not a default argument (`async def h(route, _hdr=header)`),
    which is the obvious way to write it and is wrong: Playwright inspects the
    handler's arity and calls a two-parameter handler as `(route, request)`.
    The default silently became a Request object and every navigation died with

        TypeError: Route.continue_: Object of type Request is not JSON
        serializable

    sixty seconds later, as a page-load timeout with no mention of routing.
    """
    async def _add_auth(route):
        await route.continue_(
            headers={**route.request.headers, "authorization": header}
        )

    return _add_auth


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

    @property
    def end_seconds(self) -> Optional[float]:
        if self.duration_seconds is None:
            return None
        return self.start_seconds + self.duration_seconds


@dataclass(frozen=True)
class PreflightResult:
    """What a look at the video tells us, before committing to an audit.

    This is the thing a customer is asked to confirm, so every field here has
    to be answerable in seconds and readable by someone who does not know how
    the capture works. "Plan A" and "Plan B" are our words; `capture_reason`
    is what gets shown.

    `ok=False` is a normal outcome, not an exception: "that video is not there"
    and "the hour you asked for is past the end of the recording" are answers,
    and turning them into stack traces loses the part the customer needs.
    """

    ok: bool
    target: str
    platform: str
    capture_mode: str = ""          # "stream" | "screen"
    capture_reason: str = ""
    video_duration_seconds: Optional[float] = None
    requested_start_seconds: float = 0.0
    requested_end_seconds: Optional[float] = None
    # False when the recording plainly does not reach the requested end. Note
    # this is a *prediction*; `_coverage` reports what was actually watched.
    span_available: bool = True
    problem: Optional[str] = None
    # Locator for a still from the moment we opened the page, so the customer
    # can see we are looking at their shop and not somebody else's.
    cover_frame: Optional[str] = None
    title: str = ""
    # How this deployment will read the footage. Not a property of the video --
    # it is what the container is configured to do -- but it belongs in the
    # answer the customer confirms, because it is what the run will cost and
    # how long it will take. There is no startup banner anywhere, so without
    # this the only way to know which mode is live is to read the engine's env
    # vars, which is what had to be done on 2026-09-09 to answer the question.
    # `default_factory` rather than a value passed at each construction site:
    # preflight returns from four places, and a field that can be forgotten in
    # one of them would report "static" on the error path of an agentic run.
    analysis_mode: str = field(default_factory=lambda: config.media_processing)
    analysis_window_seconds: int = field(default_factory=lambda: config.window_seconds)
    analysis_model: str = field(default_factory=lambda: config.analysis_model)

    @property
    def analysis_scope(self) -> str:
        """"whole_file" when the model reads the recording in one piece.

        Derived from `capture_mode` rather than stored, because the two can
        never disagree and a stored copy could: this is the same decision seen
        from the other end. Plan C hands Vertex a `gs://` address and Vertex
        reads all of it -- there is nothing to slice, and asking it to look at
        one window of the object is a documented no-op under agentic
        (see `WholeFileProducer`).
        """
        return "whole_file" if self.capture_mode == "file" else "windows"

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "target": self.target,
            "platform": self.platform,
            "capture_mode": self.capture_mode,
            "capture_reason": self.capture_reason,
            "video_duration_seconds": self.video_duration_seconds,
            "requested_start_seconds": self.requested_start_seconds,
            "requested_end_seconds": self.requested_end_seconds,
            "span_available": self.span_available,
            "problem": self.problem,
            "cover_frame": self.cover_frame,
            "title": self.title,
            "analysis_mode": self.analysis_mode,
            "analysis_window_seconds": self.analysis_window_seconds,
            "analysis_model": self.analysis_model,
            "analysis_scope": self.analysis_scope,
        }


@dataclass
class _Session:
    """A video we are ready to capture from, and whatever is holding it open.

    Usually that is a live browser sitting on a playing page. On the `gs://`
    path there is no browser at all, so `page`, `context` and `navigator` are
    None -- which is the one thing every consumer has to check before reaching
    for them. `has_page` is that check, spelled out, because `if session.page`
    reads like a truthiness bug even when it is not.
    """

    page: object
    context: object
    navigator: object
    source: CaptureSource
    work_dir: Path
    platform: str

    @property
    def has_page(self) -> bool:
        return self.page is not None


class AuditPipeline:
    def __init__(
        self,
        store: Optional[AuditStore] = None,
        gate: Optional[HumanGate] = None,
        on_preview_frame: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str, dict], None]] = None,
        rules: Optional[SopRuleSet] = None,
        has_viewers: Optional[Callable[[], Awaitable[bool]]] = None,
    ):
        problems = config.validate()
        if problems:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(problems))
        config.ensure_dirs()

        # Resolved by the caller when a customer picked a version -- see
        # `load_rules_for`. The local file is the standard only when nobody
        # named one, which is the `adk web` case.
        self.rules = rules or load_rules()
        self.store = store or AuditStore(rules=self.rules)
        self.gate = gate or HumanGate()
        self.on_preview_frame = on_preview_frame
        # Left None, the preview streams unconditionally -- correct locally,
        # where the frames never leave the machine. The cloud path supplies a
        # probe so an unwatched audit costs no egress at all.
        self.has_viewers = has_viewers
        self.on_status = on_status
        self.analyzer = VideoAnalyzer(self.rules)
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
        # Whether we have already told the operator the picture stopped moving.
        # Latched, because the watchdog polls every few seconds and the one
        # thing worse than a silent freeze is the same warning forty times.
        self._picture_frozen = False
        # Whether the *repaint* detector is the one currently complaining, so
        # that it only ever clears its own warning. The watchdog latches the
        # same flag for reasons this detector cannot see (a video that ended
        # while the page still animates), and one detector cancelling the
        # other's warning is how you end up with a freeze nobody reports.
        self._repaint_stalled = False

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

    def _note_picture_frozen(self, reason: str) -> None:
        """Says once, out loud, that the dashboard has stopped moving.

        This exists because of how the bilibili run failed: the picture froze
        for two minutes and every other signal said the job was healthy. The
        job *was* healthy -- Plan A does not need the page -- but nobody
        watching could tell the difference between "the browser paused" and
        "the whole thing died", and the logs did not say either.

        A warning that fires when nothing is wrong gets ignored, so this is
        deliberately narrow: it means the picture, and only the picture.
        """
        if self._picture_frozen:
            return
        self._picture_frozen = True
        logger.warning("Dashboard picture frozen: %s", reason)
        self._emit("picture_frozen", {"reason": reason})

    def _note_repaint(self, alive: bool, seconds: float) -> None:
        """The preview's verdict on whether Chromium is still painting.

        Two detectors, and they fail in different directions, which is why
        both exist. The watchdog reads the playhead: precise about *why*, but
        blind whenever the page cannot be read at all. This one counts
        screencast frames: it knows nothing about the cause, and it is the only
        thing that can see a page that reports itself as playing while
        rendering nothing.

        On job a8c236 this was the only detector that fired, and all it did was
        write a line to the container log while the operator sat looking at a
        still.
        """
        if alive:
            if self._repaint_stalled:
                self._repaint_stalled = False
                self._picture_frozen = False
            return
        self._repaint_stalled = True
        self._note_picture_frozen(
            f"网页已经 {seconds:.0f} 秒没有重绘了，大屏显示的是一张静止画面"
            f"（弹窗、暂停或者页面卡住）。稽核不受影响，仍在继续。"
        )

    @contextlib.asynccontextmanager
    async def _capture_session(self, request: AuditRequest, *, live_preview: bool = True):
        """Gets the video ready to capture, whichever kind of target it is.

        The one place the `gs://` path forks off. Everything downstream --
        windowing, analysis, evidence, the report -- sees the same `_Session`
        and the same MP4 clips, so this is the only branch either plan needs.
        """
        if gcs_video.is_gcs_uri(request.target):
            async with self._file_session(request) as session:
                yield session
        else:
            async with self._browser_session(request, live_preview=live_preview) as session:
                yield session

    @contextlib.asynccontextmanager
    async def _file_session(self, request: AuditRequest):
        """Plan C: a `gs://` object. No browser, no login, no player to drive.

        Deliberately short next to `_browser_session`. Nearly everything in
        that method exists to deal with a page -- the login dance, the probe
        race, the fullscreen fight, the CAPTCHA gate -- and none of it has an
        analogue here. A file either reads or it does not.

        No navigation budget either, for the same reason: the wall clock in
        `_browser_session` guards a sequence of four page calls that can each
        hang forever. Here there is one subprocess, and `open_source` already
        bounds it.
        """
        work_dir = config.work_dir / f"run_{int(time.time() * 1000)}"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._emit("navigating", {"target": request.target, "platform": "gcs"})
            source = await gcs_video.open_source(request.target)
            self.source = source
            self._emit("capture_mode", {"mode": source.mode, "reason": source.reason})
            yield _Session(
                page=None,
                context=None,
                navigator=None,
                source=source,
                work_dir=work_dir,
                platform="gcs",
            )
        finally:
            if not config.keep_clips:
                shutil.rmtree(work_dir, ignore_errors=True)

    @contextlib.asynccontextmanager
    async def _browser_session(self, request: AuditRequest, *, live_preview: bool = True):
        """Opens the page, gets it playing, and decides how to capture it.

        Everything up to and including `probe.decide()` is the same work for a
        preflight and for a real audit -- and it is the part that touches the
        network, so it is the part that breaks. Keeping one copy means a fix to
        the login dance or the probe timing lands on both paths at once.

        Yields a `_Session`, and tears the browser and the work directory down
        on the way out whichever path raised.

        The session deliberately does not outlive the `async with`. Preflight
        and start_audit arrive as two separate GE turns and are not guaranteed
        to land on the same container instance, so a browser held open between
        them would be a browser the second turn cannot see.
        """
        # Milliseconds, not seconds: a preflight and the audit it approves can
        # start within the same second, and sharing a directory means the
        # preflight's cleanup deletes the audit's clips out from under it.
        work_dir = config.work_dir / f"run_{int(time.time() * 1000)}"
        work_dir.mkdir(parents=True, exist_ok=True)
        platform = platform_of(request.target)

        playwright = await async_playwright().start()
        browser = context = preview = None

        async def _open() -> _Session:
            nonlocal browser, context, preview
            browser = await playwright.chromium.launch(headless=config.headless, args=_CHROMIUM_ARGS)
            context = await browser.new_context(
                viewport={"width": config.screen_width, "height": config.screen_height},
                storage_state=load_state(platform),
            )
            await _attach_oidc_headers(context)
            page = await context.new_page()

            # Started before navigation, not after: the login wall, the CAPTCHA
            # and the "video unavailable" page are exactly the moments an
            # operator needs to see, and they all happen before playback.
            if live_preview and self.on_preview_frame is not None:
                preview = LivePreview(
                    page=page,
                    on_frame=self.on_preview_frame,
                    fps=config.preview_fps,
                    width=config.preview_width,
                    height=config.preview_height,
                    quality=config.preview_quality,
                    has_viewers=self.has_viewers,
                    on_repaint=self._note_repaint,
                )
                await preview.start()

            navigator = for_target(request.target, gate=self.gate, fallback=ComputerUseFallback())

            # Attach before navigating: the player fetches its manifest within
            # the first second, and a probe attached afterwards misses it.
            probe = StreamProbe(page)
            probe.attach()

            self._emit("navigating", {"target": request.target, "platform": platform})
            # One line per step, at INFO because DEBUG does not reach Cloud
            # Logging. Without these the container went silent between "Opening"
            # and "capture_mode" -- fifteen minutes of nothing, on a run that
            # was hung in the third of these four calls, with no way to tell
            # which one from the outside.
            await navigator.login(page)
            logger.info("Login step done; opening the target.")
            await navigator.open_target(page, request.target)
            logger.info("Target open; starting playback.")
            await navigator.ensure_playing(page)
            logger.info("Playback confirmed advancing.")

            # Persist the session now that we are past any challenge, so the
            # next run starts already logged in.
            with contextlib.suppress(Exception):
                await save_state(context, platform)

            if request.start_seconds > 0:
                await navigator.seek_to(page, request.start_seconds)
                logger.info("Seeked to %.1fs.", request.start_seconds)

            source = await probe.decide(config.capture_mode, wait_seconds=config.stream_probe_seconds)
            probe.detach()
            self.source = source
            self._emit("capture_mode", {"mode": source.mode, "reason": source.reason})

            return _Session(
                page=page,
                context=context,
                navigator=navigator,
                source=source,
                work_dir=work_dir,
                platform=platform,
            )

        try:
            # A wall clock around the whole opening sequence, on top of the
            # per-step timeouts. Those were all in place when job fcfd3c hung
            # for 900 seconds (2026-09-08, a YouTube link): `page.evaluate` has
            # no timeout, so the one call nobody had bounded was the one that
            # hung. A guard that has to name the failing step in advance is a
            # guard against the failures we already know about.
            #
            # It belongs here rather than in `preflight` because `run()` opens
            # the page the same way, and `run()` is a detached background task
            # -- a hang there is a job that says `running` and never stops
            # saying it, with nobody on the other end of a request to notice.
            budget = config.navigation_budget_seconds
            try:
                session = (
                    await asyncio.wait_for(_open(), timeout=budget) if budget > 0
                    else await _open()
                )
            except asyncio.TimeoutError as exc:
                raise NavigationError(
                    f"{budget:.0f} 秒内没能打开这个视频并确认它在播放。"
                    f"页面可能一直在加载，或者播放器起不来。"
                    f"如果这是个我们还没适配过的网站，多半需要先做一个适配器。"
                ) from exc
            yield session
        finally:
            cancelled: Optional[BaseException] = None
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
                except asyncio.CancelledError as exc:
                    # A cancelled cleanup step must not skip the ones after it.
                    # That is how a Chromium process outlives the request that
                    # started it, and on a container that survives for hours
                    # they accumulate. The budget guard around `preflight`
                    # cancels this scope by design, so this is the normal path
                    # on a timeout, not an edge case. Re-raised below, once
                    # everything else has been closed.
                    cancelled = exc
                except Exception as exc:
                    logger.debug("Shutdown step failed: %s", exc)
            if not config.keep_clips:
                shutil.rmtree(work_dir, ignore_errors=True)
            if cancelled is not None:
                raise cancelled

    async def preflight(self, request: AuditRequest, job_id: str = "preflight") -> PreflightResult:
        """Looks at the video and reports back, without auditing anything.

        Takes seconds to a minute. That matters: this is the call that has to
        answer inside one GE turn, and the audit it describes cannot.

        Failures here are *returned*, not raised. "That link does not open" and
        "your hour is past the end of the recording" are the two answers a
        customer is most likely to get, and they are the two an exception
        would turn into a stack trace with the useful part missing.

        And "we could not tell inside N seconds" is one of them, which is why
        `_browser_session` has a wall clock: every step in it already had its
        own timeout when job fcfd3c hung anyway, silently, for 900 seconds.
        """
        try:
            result = await self._preflight_inner(request, job_id)
        except Exception as exc:
            logger.exception("Preflight failed for %s: %s", request.target, exc)
            result = PreflightResult(
                ok=False,
                target=request.target,
                platform=platform_of(request.target),
                requested_start_seconds=request.start_seconds,
                requested_end_seconds=request.end_seconds,
                span_available=False,
                # 400, not 200. The message that matters here is often two
                # sentences -- what the site said, then what the fallback tried
                # -- with the target URL between them, and a bilibili link with
                # its tracking parameters is 150 characters on its own. At 200
                # the second sentence was always the one that got cut.
                problem=f"打不开这个视频：{str(exc)[:400]}",
            )
        self._emit("preflight", result.as_dict())
        return result

    async def _preflight_inner(self, request: AuditRequest, job_id: str) -> PreflightResult:
        """The body of `preflight`. Raises; the caller turns that into a result."""
        async with self._capture_session(request, live_preview=False) as session:
            duration = await self._video_duration(session)
            cover = await self._cover_frame(session, job_id, request.start_seconds)
            title = ""
            if session.has_page:
                with contextlib.suppress(Exception):
                    title = await session.page.title()
            elif gcs_video.is_gcs_uri(request.target):
                # The object name, not the whole URI: it is the part the
                # customer recognises, and the bucket is the same for all of
                # them anyway.
                with contextlib.suppress(Exception):
                    title = gcs_video.parse_gs_uri(request.target)[1]

            requested_end = request.end_seconds
            # One window of slack, matching `_coverage`: the last clip is cut on
            # a window boundary, so landing seconds short is not a shortfall and
            # should not be reported to the customer as one.
            tolerance = max(float(config.window_seconds), 5.0)
            fmt = Clip.format_offset

            # Unknown duration means live, or a player that will not say.
            # Neither is a reason to refuse -- it is a reason not to claim.
            span_available = True
            problem: Optional[str] = None
            # A whole-file run has no span to check. Both tests below ask "does
            # the recording reach the moment you asked for", and on this path
            # nobody asked for a moment: the whole object is going to the model
            # whatever the customer typed. Running them anyway would reject a
            # perfectly auditable video for a time range that is about to be
            # ignored -- "视频只有 03:00 长，请求的起点 05:00 已经超出了视频末尾"
            # on a video we are about to audit end to end.
            whole_file = session.source.mode == "file"
            if duration is not None and not whole_file:
                if request.start_seconds >= duration:
                    # Nothing to audit at all. This is the one preflight outcome
                    # that has to be ok=False on a page that opened fine:
                    # starting the audit would produce an empty report rather
                    # than an error.
                    return PreflightResult(
                        ok=False,
                        target=request.target,
                        platform=session.platform,
                        capture_mode=session.source.mode,
                        capture_reason=session.source.reason,
                        video_duration_seconds=round(duration, 1),
                        requested_start_seconds=request.start_seconds,
                        requested_end_seconds=requested_end,
                        span_available=False,
                        problem=(
                            f"视频只有 {fmt(duration)} 长，"
                            f"请求的起点 {fmt(request.start_seconds)} 已经超出了视频末尾"
                        ),
                        cover_frame=cover,
                        title=title,
                    )
                if requested_end is not None and requested_end > duration + tolerance:
                    span_available = False
                    problem = (
                        f"视频只有 {fmt(duration)} 长，请求的是到 {fmt(requested_end)}，"
                        f"最多只能稽核到 {fmt(duration)}"
                    )

            return PreflightResult(
                ok=True,
                target=request.target,
                platform=session.platform,
                capture_mode=session.source.mode,
                capture_reason=session.source.reason,
                video_duration_seconds=round(duration, 1) if duration is not None else None,
                requested_start_seconds=request.start_seconds,
                requested_end_seconds=requested_end,
                span_available=span_available,
                problem=problem,
                cover_frame=cover,
                title=title,
            )

    async def _video_duration(self, session: "_Session") -> Optional[float]:
        """How long the recording is, from whichever source can say.

        ffprobe read the container, so it wins when Plan A applies. Plan B has
        no container to read, and the `<video>` element's own `duration` is
        then the only number available -- it is also the number the platform's
        own timeline is drawn from, so it is the one the customer would quote.
        """
        if session.source.duration_seconds:
            return session.source.duration_seconds
        if not session.has_page:
            # Plan C's only source of a duration is the probe above. If ffprobe
            # could not read one, nothing else can, and there is no player to
            # ask -- so say so rather than reaching through a None navigator
            # and catching the AttributeError as if it meant something.
            return None
        try:
            state = await session.navigator.read_playback_state(session.page)
        except Exception as exc:
            logger.debug("Could not read playback state for duration: %s", exc)
            return None
        raw = (state or {}).get("duration")
        try:
            value = float(raw) if raw else None
        except (TypeError, ValueError):
            return None
        # A live stream reports Infinity, and `float("inf")` compares larger
        # than every requested end -- which would silently pass every span
        # check rather than admitting we do not know.
        return value if value and value != float("inf") else None

    async def _cover_frame(
        self, session: "_Session", job_id: str, start_seconds: float = 0.0
    ) -> Optional[str]:
        """A still of what we opened, so the customer can see it is their shop.

        With a page that is a screenshot. With a file it is a frame decoded at
        the requested start -- which is arguably the better picture of the two,
        since it shows the moment the audit begins rather than whatever the
        player happened to be displaying when the probe finished.

        Best effort on purpose: a preflight that can otherwise answer every
        question is not worth failing over a thumbnail.
        """
        try:
            if session.has_page:
                shot = await session.page.screenshot(
                    type="jpeg", quality=config.preview_quality
                )
            else:
                shot = await gcs_video.grab_frame(session.source, start_seconds)
            if not shot:
                return None
            return await artifact_sink().put(f"{job_id}/cover.jpg", shot, "image/jpeg")
        except Exception as exc:
            logger.warning("Could not capture a cover frame: %s", str(exc)[:200])
            return None

    async def run(self, request: AuditRequest) -> dict:
        async with self._capture_session(request) as session:
            page, navigator = session.page, session.navigator
            source, work_dir = session.source, session.work_dir

            producer = await self._build_producer(
                source, page, navigator, work_dir, request,
            )

            queue: asyncio.Queue = asyncio.Queue(maxsize=config.clip_queue_size)
            capture_task = asyncio.create_task(self._capture(producer, queue, request))
            workers = [
                asyncio.create_task(self._analyse(queue, i))
                for i in range(config.analysis_concurrency)
            ]
            # Both plans need the page minded, for different reasons.
            #
            # Plan B is bound to the page, so the page decides when the run
            # ends. Plan A pulls the media independently and stops at EOF by
            # itself -- but the *picture on the dashboard* still comes from
            # this page, and that is not a detail we get to ignore. Measured
            # on bilibili (job 692b53): the login nag appears about 70 seconds
            # in, pauses the video, and Chromium then has nothing to repaint,
            # so the screencast delivers 0 fps for the rest of the run. The
            # pump kept forwarding at 11.9 fps and every frame was byte-for-byte
            # the same 84.3 KB still. The audit was fine and the operator was
            # staring at a frozen screen -- which is how you lose a demo while
            # every number on the dashboard says success.
            # Nothing to mind on Plan C. The watchdog's whole job is the page:
            # is it still playing, has a nag covered it, is Chromium still
            # painting. With no page, a task that polls a None navigator would
            # only produce a steady trickle of caught exceptions that look like
            # a fault and are not.
            watchdog = (
                asyncio.create_task(
                    self._watch_page(page, navigator, page_is_source=source.mode == "screen")
                )
                if session.has_page else None
            )

            try:
                await capture_task
            finally:
                # Sentinel per worker so each one exits after draining.
                for _ in workers:
                    await queue.put(None)
                try:
                    await asyncio.gather(*workers, return_exceptions=True)
                finally:
                    # The watchdog outlives capture on purpose. Capture stops
                    # the moment ffmpeg reaches the requested end, but the
                    # analysis queue still has a minute of work in it and the
                    # preview keeps feeding the dashboard for all of it -- it
                    # is closed with the browser, further down. Cancelling the
                    # watchdog here used to leave that whole tail with nobody
                    # minding the page. Measured on job a8c236: capture ended
                    # at 15:17:47, the screencast fell to 0 fps two seconds
                    # later when bilibili raised its login nag, and the
                    # operator watched a frozen still for the remaining 56
                    # seconds while the windows finished. The audit was
                    # perfect; the demo was not.
                    if watchdog is not None:
                        watchdog.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await watchdog

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
        if self.source is not None and self.source.mode == "file":
            # A whole-file run has no requested span to fall short of: the
            # entire recording went to the model in one piece. Comparing the
            # covered span against what the customer typed would report
            # "只覆盖到 05:00，请求的是到 07:00" about a run that watched all
            # thirty minutes -- an understatement, which is the one direction
            # a coverage report must never err in.
            return {
                "requested_start_seconds": 0.0,
                "requested_end_seconds": (
                    round(self.source.duration_seconds, 1)
                    if self.source.duration_seconds else None
                ),
                "covered_from_seconds": round(span[0], 1) if span else None,
                "covered_to_seconds": round(span[1], 1) if span else None,
                "stopped_kind": self._stop_kind,
                "complete": span is not None,
                "incomplete_reason": (
                    None if span else "没有采集到任何可分析的片段"
                ),
                "whole_video": True,
            }
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
        if source.mode == "file":
            # Plan C. Nothing is cut, nothing is read: the object's address goes
            # to Vertex and Vertex reads it. The requested time span is not
            # honoured here and that is deliberate -- see `WholeFileProducer`
            # for the measurement that made windowing this path wrong rather
            # than merely wasteful. A file that needs cutting gets cut before it
            # is put in the bucket.
            return gcs_video.WholeFileProducer(
                source, duration_seconds=source.duration_seconds
            )

        if source.mode == "stream":
            # Purely for the operator: nothing here is recorded, but a
            # dashboard showing a postage-stamp player inside a page of
            # sidebars and comments is not worth watching. Costs one call.
            if config.fullscreen_player:
                await navigator.enter_fullscreen(page)
                # Remembered on this path too, and not only on Plan B's.
                # bilibili's login nag drops the player out of web fullscreen
                # and dismissing it does not put it back (see _hold_geometry),
                # so without a baseline the operator spends the rest of the run
                # watching a postage stamp in the corner of a comments page.
                # Nothing is being recorded here, so this is only ever about
                # the picture -- which is the entire reason we went fullscreen.
                self._player_rect = await navigator.video_rect(page)

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
                "Player moved from %sx%s@(%s,%s) and could not be restored; "
                "whatever is being captured or shown now frames the wrong region.",
                round(was["width"]), round(was["height"]), round(was["x"]), round(was["y"]),
            )

    async def _watch_page(self, page, navigator, *, page_is_source: bool = True) -> None:
        """Housekeeping for the page, on both plans.

        Three jobs nothing else does:

        * Keep the player visible. Login nags and session-expiry prompts get
          re-raised while the audit runs, and the recorder would happily
          capture a dialog box sitting on top of the footage.
        * Keep it playing. A nag that pauses the video freezes everything
          downstream of the page.
        * Notice the end. A finished <video> renders its last frame forever, so
          without this a 15-minute recording keeps producing identical windows
          until the wall-clock budget expires -- billing for every one of them.

        `page_is_source` is what separates the two plans, and only the third
        job depends on it. Under Plan B the page *is* the footage, so the page
        running out is the run running out. Under Plan A ffmpeg pulls the media
        itself and the page is only the picture on the dashboard: tend it, but
        never let it end the audit. Getting that backwards would kill a healthy
        Plan A run the moment bilibili paused the player -- an audit stopped at
        01:19 of a requested 05:00, because of a dialog box in a browser
        nobody was reading the pixels of.

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
        # Three strikes only makes sense when the strikes lead to a verdict.
        # Plan B gives up and calls the footage finished; Plan A has nothing to
        # conclude -- ffmpeg is feeding the audit either way -- so it keeps
        # trying for as long as the run lasts. bilibili re-raises its login nag
        # every couple of minutes and each one pauses the player again, so a
        # Plan A run that stopped nudging after the third would show a still
        # for the remaining fifty minutes.
        max_recoveries: float = 3 if page_is_source else math.inf
        last_time: Optional[float] = None

        # Not `while not self._stop.is_set()`. Stopping means capture is over,
        # not that the page stopped mattering: the dashboard is fed from this
        # page until the browser closes, which is after the last window is
        # analysed. The caller cancels this task at that point, and that is the
        # only thing that should end it. What stopping *does* change is that
        # nothing here may end the run any more -- see `may_stop` below.
        while True:
            await asyncio.sleep(config.page_watch_seconds)
            try:
                await navigator.keep_clear(page)
                await self._hold_geometry(page, navigator)
                state = await navigator.read_playback_state(page)
            except Exception as exc:
                logger.debug("Page watchdog poll failed: %s", exc)
                continue

            if not state:
                continue

            # Only Plan B gets to end the run from what the page says. Plan A
            # keeps tending the page below either way.
            #
            # And once the run is already stopping, neither of them does. This
            # is the guard that makes outliving capture safe: the stop path
            # sets `_footage_ends_at`, and the workers drop every queued clip
            # that starts after it. During the drain those clips are already
            # captured and perfectly good, so a page that happens to reach its
            # end while the queue empties would silently delete the tail of the
            # audit -- windows the customer asked for, never analysed, reported
            # as "footage ended". Past capture the watchdog only tends pixels.
            may_stop = (
                page_is_source
                and config.stop_on_video_end
                and not self._stop.is_set()
            )

            current = state.get("current_time")
            duration = state.get("duration")
            over = bool(
                state.get("ended")
                or (duration and current is not None and current >= duration - 1.0)
            )
            if over:
                if may_stop:
                    at = round(current or 0.0, 1)
                    self._footage_ends_at = at
                    self._emit("video_ended", {"at_seconds": at})
                    self.stop(
                        f"视频已播放完毕（{Clip.format_offset(at)}）", kind="video_ended"
                    )
                    return
                # Plan A: the page finishing says nothing about the audit, and
                # nudging an ended <video> restarts it from zero -- which on the
                # dashboard looks like the audit jumped back to the beginning.
                # Say it once and leave the last frame up.
                self._note_picture_frozen(
                    "网页里的视频已播完，画面停在最后一帧（稽核不受影响，仍在继续）"
                )
                continue

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
                held_long = stalled >= (tail_stall_limit if near_end else stall_limit)
                held = round(stalled * poll, 1)
                tail = "" if near_end else f"，已尝试 {recoveries} 次恢复播放"
                if held_long and may_stop and (near_end or recoveries >= max_recoveries):
                    self._footage_ends_at = round(current, 1)
                    self._emit("playback_stalled", {
                        "at_seconds": round(current, 1),
                        "for_seconds": held,
                        "recovery_attempts": recoveries,
                    })
                    self.stop(
                        f"播放在 {Clip.format_offset(current)} 卡住 {held:.0f} 秒不再前进{tail}",
                        kind="playback_stalled",
                    )
                    return
                if held_long and not may_stop:
                    # Plan A: the nudging above carries on regardless -- this
                    # only says out loud what the operator is already looking
                    # at. Deliberately not gated on having run out of retries,
                    # because Plan A never runs out: without this, a picture
                    # that never comes back would never be mentioned either.
                    self._note_picture_frozen(
                        f"网页画面卡在 {Clip.format_offset(current)} 不动了{tail}。"
                        f"稽核不受影响，仍在继续。"
                    )
            else:
                if stalled:
                    logger.info("Playback resumed at %.1fs.", current or 0.0)
                    self._picture_frozen = False
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
                # `whole_video` opts out: that clip *is* the recording, so a
                # requested end it starts before or after says nothing about it.
                if (
                    end_offset is not None
                    and not clip.whole_video
                    and clip.start_offset >= end_offset
                ):
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
                # Nothing to delete when the bytes were never ours (Plan C).
                if clip is not None and clip.path is not None and not config.keep_clips:
                    with contextlib.suppress(OSError):
                        clip.path.unlink(missing_ok=True)
                queue.task_done()
