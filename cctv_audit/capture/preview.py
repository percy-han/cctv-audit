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

"""The live picture on the dashboard.

Deliberately separate from capture. It used to live inside `ScreenRecorder`,
which was fine only for as long as every run was a screen recording -- the day
Plan A started working, the dashboard went black, because under Plan A no
recorder is ever constructed. What the operator watches and what the analyzer
eats are two different concerns and now two different objects.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from typing import Awaitable, Callable, List, Optional

from ..cpuprobe import CpuProbe

logger = logging.getLogger("cctv_audit.preview")

# How often to re-ask the dashboard whether anyone is watching. Long enough
# that the question costs nothing next to the frames it saves, short enough
# that someone opening the page mid-audit sees a picture rather than wondering
# whether the thing is broken. Five seconds was too long for that second half:
# an operator who clicks the link and stares at an empty panel assumes it is
# broken well before five seconds are up.
_VIEWER_POLL_SECONDS = 2.0

# Matches the dashboard client's own report interval so the two lines can be
# read side by side for the same fifteen seconds.
_PUMP_REPORT_SECONDS = 15.0


class LivePreview:
    """A cheap screencast of the page, forwarded to the dashboard.

    Runs its own CDP session rather than sharing the recorder's. Chromium
    happily drives two screencasts on one page with independent parameters
    (measured: 1080p q85 and 360p q50 both delivered at 33 fps), and sharing
    one would force a choice between evidence quality and operator experience:
    a 1080p q85 frame is ~280 KB of base64, so forwarding every one of them is
    84 Mbps -- hopeless down an SSH tunnel. At 960x540 q60 a frame is ~58 KB,
    so 15 fps costs *less* than the old 4 fps did and looks four times smoother.

    Pass `has_viewers` and frames are only *forwarded* while somebody is
    actually looking. That matters once the dashboard is a separate service:
    the frames become metered egress, each viewer is their own stream, and most
    audits are watched by nobody at all. Without it the run pays to ship a
    picture that is thrown away at the other end.

    Note what that gating does *not* do: it never stops the screencast. Two
    screencasts on one page are only well-behaved if the low-resolution one is
    started first and left alone -- measured on real Chromium:

        recorder first, preview second  -> recorder receives nothing
        preview first, recorder second  -> both fine, at their own sizes
        preview calls stopScreencast    -> the recorder stops too

    An earlier version of this class started the cast on the first viewer and
    stopped it on the last, which put the start order in the hands of whoever
    opened the dashboard and hit the first row of that table. Plan B then
    starved: ffmpeg got 83 frames and died, and the audit reported zero windows
    after twenty-eight minutes. The saving was always in the network hop, not in
    Chromium's JPEG encoder -- a 640x360 q50 frame is ~2.5 KB and costs nothing
    to produce -- so the gate belongs on `on_frame`, and the cast runs for the
    whole session.
    """

    def __init__(
        self,
        page,
        on_frame: Callable[[str], None],
        fps: int = 8,
        width: int = 640,
        height: int = 360,
        quality: int = 50,
        has_viewers: Optional[Callable[[], Awaitable[bool]]] = None,
        on_repaint: Optional[Callable[[bool, float], None]] = None,
    ):
        self.page = page
        self.on_frame = on_frame
        # Called once per report window while frames are being forwarded, with
        # (is_the_page_still_repainting, window_seconds). This is the only
        # place that can tell a quiet room from a dead browser -- the pump's
        # own counters look identical either way -- and until it had somewhere
        # to go the answer was written to the container log, where the person
        # actually looking at the frozen picture will never see it.
        self.on_repaint = on_repaint
        self.fps = max(1, int(fps))
        self.width = max(160, int(width))
        self.height = max(90, int(height))
        self.quality = min(95, max(10, int(quality)))
        self.has_viewers = has_viewers

        self._cdp = None
        self._latest_b64: Optional[str] = None
        self._pump: Optional[asyncio.Task] = None
        self._watcher: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()
        # `_casting` is "is Chromium producing frames" -- true for the whole
        # session. `_forwarding` is "is anyone at the other end" -- the thing
        # the viewer poll toggles. Conflating the two is what broke Plan B.
        self._casting = False
        self._forwarding = False
        # Diagnostics for `_pump_report`. Two independent questions the
        # dashboard-side numbers cannot separate: is the loop getting to run,
        # and is Chromium giving us anything to send.
        self._cast_frames = 0
        self._ticks = 0
        self._tick_ms: List[float] = []
        self._report_at = 0.0
        self._cpu: Optional[CpuProbe] = None

    async def start(self) -> None:
        self._cdp = await self.page.context.new_cdp_session(self.page)

        async def on_frame(event):
            data = event.get("data")
            if data:
                self._latest_b64 = data
                self._cast_frames += 1
            try:
                await self._cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})
            except Exception as exc:
                logger.debug("preview ack failed (page likely closing): %s", exc)

        self._cdp.on("Page.screencastFrame", on_frame)
        # Unconditional, and before any recorder exists. See the class note:
        # the low-resolution cast has to be the one that starts first, so it
        # cannot wait for a viewer to show up.
        await self._start_cast()
        # With no way to ask, assume somebody is there: that is the `adk web`
        # case, where the dashboard is the next window over and a black screen
        # would be a regression.
        self._forwarding = self.has_viewers is None
        self._pump = asyncio.create_task(self._run())
        # Its own task, deliberately. See `_watch_viewers`.
        if self.has_viewers is not None:
            self._watcher = asyncio.create_task(self._watch_viewers())
        logger.info("Dashboard preview: %dx%d q%d at %d fps%s",
                    self.width, self.height, self.quality, self.fps,
                    "" if self.has_viewers is None else "（有人看才推帧）")

    async def _start_cast(self) -> None:
        if self._casting or self._cdp is None:
            return
        await self._cdp.send("Page.startScreencast", {
            "format": "jpeg",
            "quality": self.quality,
            "maxWidth": self.width,
            "maxHeight": self.height,
            "everyNthFrame": 1,
        })
        self._casting = True

    async def _stop_cast(self) -> None:
        """Only ever called from `aclose`.

        Stopping this cast stops the recorder's as well (measured), so there is
        no safe moment to call it while the session is still running.
        """
        if not self._casting or self._cdp is None:
            return
        self._casting = False
        self._latest_b64 = None
        try:
            await self._cdp.send("Page.stopScreencast")
        except Exception as exc:
            logger.debug("stopScreencast failed (page likely gone): %s", exc)

    async def _watch_viewers(self) -> None:
        """Asks the dashboard whether anyone is there -- off the pump's loop.

        This used to be an `await` inside `_run`, which made the frame rate a
        function of how long the dashboard took to answer. Measured in the
        cloud (job c70488): the answer took a median of ~1s and often hit the
        client's 1.5s ceiling, so each 2s cycle spent most of itself blocked
        and the pump delivered 0.6-3.2 fps against a configured 12. The
        dashboard was never the problem -- 70 KB posts to it measure 140ms from
        an idle machine -- and neither was the in-flight cap, which dropped
        nothing. The loop was simply waiting.

        Worse, a check that ran past 1.5s raised, and an unreachable dashboard
        is read as "nobody is watching", so the pump *paused* for the next
        cycle -- with a viewer connected the whole time. The container logged
        `paused`/`resumed` about twice a second.

        A slow answer must only make the answer stale, never make the video
        stutter. Hence a task of its own: it can take as long as it likes.

        The second half of that bug is here rather than in the caller: a "no"
        only counts on the second time of asking. The client cannot tell "the
        dashboard says nobody" from "the dashboard did not answer in 1.5s", and
        deliberately so -- both mean send nothing, and guessing the optimistic
        way would stream a live CCTV feed at a service that is gone. But one
        timed-out question should not blank an operator's screen. Two in a row,
        four seconds apart, still stops a run whose viewer really has left, and
        frames are cheap next to that.
        """
        missed = 0
        while not self._stopping.is_set():
            try:
                watched = await self.has_viewers()
            except Exception as exc:
                # A dashboard we cannot reach is a dashboard nobody is seeing
                # this on. Same answer, same behaviour: send nothing.
                logger.debug("Viewer check failed: %s", exc)
                watched = False
            if watched:
                missed = 0
            else:
                missed += 1
                if self._forwarding and missed < 2:
                    watched = True
            if watched != self._forwarding:
                self._forwarding = watched
                logger.info(
                    "Dashboard preview %s.",
                    "resumed -- someone is watching" if watched
                    else "paused -- nobody is watching",
                )
            # After the check, not before: a check that took two seconds has
            # already waited long enough, and sleeping the full interval on top
            # would halve the polling rate exactly when it is least accurate.
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=_VIEWER_POLL_SECONDS)
            except asyncio.TimeoutError:
                pass

    async def _run(self) -> None:
        """Resamples to `fps` and drops rather than queues.

        Chromium emits on repaint, at whatever rate the page happens to run.
        Forwarding every frame would let a busy page saturate the tunnel, and
        the operator view must never be able to stall anything upstream --
        hence a fixed tick that takes the latest frame and lets the rest go.

        Nothing in this loop awaits the network. That is the whole point of it
        being separate from `_watch_viewers`.
        """
        interval = 1.0 / self.fps
        last = time.monotonic()
        while not self._stopping.is_set():
            await asyncio.sleep(interval)
            now = time.monotonic()
            self._ticks += 1
            self._tick_ms.append((now - last) * 1000.0)
            last = now
            self._pump_report(now)
            if not self._forwarding:
                continue
            b64 = self._latest_b64
            if b64:
                try:
                    self.on_frame(b64)
                except Exception as exc:
                    logger.debug("Preview frame dropped: %s", exc)

    def _pump_report(self, now: float) -> None:
        """Says whether the loop is getting to run, once every 15 seconds.

        The dashboard-side counters in `monitor.py` cannot answer this. They
        showed 0.7-2.4 fps offered against a configured 12 with nothing dropped
        by the in-flight cap -- which says the pump is not offering frames, but
        not why. Two candidates and one number each:

          * `tick` is how long `asyncio.sleep(1/fps)` really took. Configured
            83ms coming back as several hundred means the event loop is
            starved, and the fix is upstream of this file.
          * `screencast in` is how many frames Chromium delivered. A pump
            ticking at 12 Hz with nothing to send would be the other story
            entirely.

        Moving the viewer check off this loop was supposed to fix the rate and
        did not, so the next change should follow a measurement rather than
        another hypothesis.

        Measured (v14, cloud): tick median 300-600ms, worst 15274ms, with
        `screencast in` at 0.1-0.5 fps. Both numbers are the same fact. The
        cast only advances once we return `Page.screencastFrameAck`, and that
        ack is dispatched by this same loop -- a starved loop therefore throttles
        Chromium as well as the pump, and no amount of work in this file can
        raise either. Hence `CpuProbe` on the same fifteen-second window: a
        lagging loop next to a saturated container is a resourcing problem, and
        a lagging loop next to an idle one is something blocking in the loop.
        """
        if self._report_at == 0.0:
            self._report_at = now
            self._cpu = CpuProbe()
            return
        if now - self._report_at < _PUMP_REPORT_SECONDS or not self._tick_ms:
            return
        window = now - self._report_at
        ordered = sorted(self._tick_ms)
        logger.info(
            "preview pump: %d ticks in %.0fs (%.1f/s, configured %d), "
            "tick median %.0fms p90 %.0fms worst %.0fms, screencast in %.1f fps",
            self._ticks, window, self._ticks / window, self.fps,
            statistics.median(ordered),
            ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
            ordered[-1], self._cast_frames / window,
        )
        if self._forwarding and self.on_repaint is not None:
            try:
                self.on_repaint(bool(self._cast_frames), window)
            except Exception as exc:
                # Diagnostics must never be able to stop the picture.
                logger.debug("Repaint hook failed: %s", exc)
        if self._forwarding and not self._cast_frames:
            # Chromium only emits on repaint, so a page that has stopped moving
            # produces nothing at all -- while this loop keeps forwarding the
            # cached frame at the full configured rate. Measured on bilibili
            # (job 692b53): `screencast in` fell to 0.0 fps and stayed there for
            # two minutes, every forwarded frame byte-for-byte the same 84.3 KB
            # still, and no other counter moved. Someone watching the dashboard
            # cannot tell that from a healthy feed of a quiet room, so say it.
            logger.warning(
                "Chromium sent no screencast frame for %.0fs -- the dashboard is "
                "showing a still. The page has stopped repainting (paused video, "
                "modal dialog, or a hung tab).", window,
            )
        if self._cpu is not None:
            for name, produce in (("cpu", self._cpu.report),
                                  ("sched", self._cpu.scheduling_report)):
                try:
                    line = produce(window)
                except Exception as exc:
                    # Diagnostics must never be able to stop the picture.
                    logger.debug("%s probe failed: %s", name, exc)
                else:
                    if line:
                        logger.info("%s", line)
        self._report_at = now
        self._ticks = 0
        self._cast_frames = 0
        self._tick_ms = []

    async def aclose(self) -> None:
        self._stopping.set()
        for task in (self._pump, self._watcher):
            if task and not task.done():
                task.cancel()
        self._pump = None
        self._watcher = None
        if self._cpu is not None:
            self._cpu.stop()
            self._cpu = None
        if self._cdp is not None:
            await self._stop_cast()
            self._cdp = None
