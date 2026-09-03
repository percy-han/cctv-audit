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

"""Plan B: record what the page renders.

Universal fallback -- it does not care whether the player uses a proprietary
transport, WASM decoding, or a plain <video> tag. If a human can see it, this
can record it. The costs are a second encode, real CPU, and the fact that it
can only run at playback speed (which is why PLAYBACK_RATE exists).

CDP hands us frames whenever it feels like it. We ack every frame immediately
but resample to a fixed cadence before feeding ffmpeg, because image2pipe needs
a constant frame rate for timestamps to mean anything -- and because forwarding
every frame is what previously saturated the event loop.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import shutil
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

from .ffmpeg_util import require_ffmpeg
from .types import Clip
from .window_assembler import SEGMENT_PATTERN, WindowAssembler

logger = logging.getLogger("cctv_audit.screen")


class ScreenRecorder:
    def __init__(
        self,
        page,
        work_dir: Path,
        window_seconds: float,
        overlap_seconds: float,
        capture_fps: int = 5,
        width: int = 1920,
        height: int = 1080,
        playback_rate: float = 1.0,
        start_offset: float = 0.0,
        crop: Optional[tuple] = None,
    ):
        self.page = page
        self.work_dir = work_dir
        self.segment_dir = work_dir / "segments"
        self.window_seconds = float(window_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.capture_fps = max(1, int(capture_fps))
        self.width = width
        self.height = height
        self.playback_rate = max(0.1, float(playback_rate))
        self.start_offset = float(start_offset)
        self.crop = normalise_crop(crop, width, height)

        # The dashboard's picture is not this class's job -- see
        # capture/preview.py. It runs its own screencast on the same page, at
        # both plans, so that Plan A does not black the operator out.
        self._cdp = None
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._latest_jpeg: Optional[bytes] = None
        self._pump: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()
        self._stderr_tail: list[str] = []

    async def clips(self) -> AsyncIterator[Clip]:
        ffmpeg, _ = require_ffmpeg()
        _reset_dir(self.segment_dir)

        # Segment length in *file* seconds. At 4x playback a 12 video-second
        # step is only 3 seconds of recording.
        step_video = self.window_seconds - self.overlap_seconds
        seg_video = choose_segment_seconds(self.window_seconds, step_video)
        seg_file = seg_video / self.playback_rate
        gop = max(1, int(round(self.capture_fps * seg_file)))

        args = [
            ffmpeg, "-nostdin", "-v", "warning", "-y",
            "-f", "image2pipe",
            "-framerate", str(self.capture_fps),
            "-i", "-",
            "-an",
        ]
        if self.crop:
            cw, ch, cx, cy = self.crop
            args += ["-vf", f"crop={cw}:{ch}:{cx}:{cy}"]
            logger.info("Cropping the capture to the player: %dx%d at (%d,%d)", cw, ch, cx, cy)
        args += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            # Segments must start on a keyframe or the concat in the assembler
            # produces corrupt leading frames.
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-force_key_frames", f"expr:gte(t,n_forced*{seg_file:.3f})",
            "-f", "segment",
            "-segment_time", f"{seg_file:.3f}",
            "-reset_timestamps", "1",
            "-segment_format", "mp4",
            str(self.segment_dir / SEGMENT_PATTERN),
        ]

        logger.info(
            "Plan B: screen recording at %dfps, %.1fx playback (window %.0fs video = %.1fs file, "
            "%.1fs segments, first verdict ~%.0fs in)",
            self.capture_fps, self.playback_rate, self.window_seconds,
            self.window_seconds / self.playback_rate, seg_file,
            (self.window_seconds + seg_video) / self.playback_rate,
        )
        self._proc = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        await self._start_screencast()
        self._pump = asyncio.create_task(self._pump_frames())
        stderr_drain = asyncio.create_task(self._drain_stderr())

        assembler = WindowAssembler(
            segment_dir=self.segment_dir,
            out_dir=self.work_dir / "windows",
            window_seconds=self.window_seconds,
            overlap_seconds=self.overlap_seconds,
            source_mode="screen",
            start_offset=self.start_offset,
            time_scale=self.playback_rate,
            segment_seconds=seg_video,
        )
        emitted = 0
        returncode = None
        try:
            async for clip in assembler.windows(producer_alive=self._alive):
                emitted += 1
                yield clip
        finally:
            stderr_drain.cancel()
            returncode = self._proc.returncode if self._proc else None
            await self.aclose()

        # See the matching note in stream_grabber: an encoder that dies at once
        # would otherwise be indistinguishable from a shop with nothing to flag.
        if emitted == 0 and returncode not in (0, None):
            raise RuntimeError(
                f"ffmpeg exited {returncode} without producing a single window. "
                f"Last output:\n{self.stderr_tail or '(no stderr)'}"
            )

    def _alive(self) -> bool:
        return not self._stopping.is_set() and self._proc is not None and self._proc.returncode is None

    async def _start_screencast(self) -> None:
        """The evidence feed: full resolution, high quality, nothing else."""
        self._cdp = await self.page.context.new_cdp_session(self.page)

        async def on_frame(event):
            data = event.get("data")
            if data:
                self._latest_jpeg = base64.b64decode(data)
            try:
                await self._cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})
            except Exception as exc:
                logger.debug("screencast ack failed (page likely closing): %s", exc)

        self._cdp.on("Page.screencastFrame", on_frame)
        await self._cdp.send("Page.startScreencast", {
            "format": "jpeg",
            "quality": 85,
            "maxWidth": self.width,
            "maxHeight": self.height,
            # Chromium only emits on repaint; we resample downstream anyway.
            "everyNthFrame": 1,
        })

    async def _pump_frames(self) -> None:
        """Feeds ffmpeg exactly `capture_fps` frames per second."""
        interval = 1.0 / self.capture_fps
        loop = asyncio.get_running_loop()
        next_tick = loop.time()
        written = 0
        while self._alive():
            next_tick += interval
            await asyncio.sleep(max(0.0, next_tick - loop.time()))
            frame = self._latest_jpeg
            if frame is None:
                # Nothing painted yet; skipping would desync timestamps, so
                # wait for the first real frame before the clock starts.
                next_tick = loop.time()
                continue
            try:
                self._proc.stdin.write(frame)
                await self._proc.stdin.drain()
                written += 1
            except (BrokenPipeError, ConnectionResetError):
                logger.warning("ffmpeg stdin closed after %d frames; stopping recorder", written)
                self._stopping.set()
                return
            except Exception as exc:
                logger.warning("Frame write failed: %s", exc)
                self._stopping.set()
                return

    async def _drain_stderr(self) -> None:
        """Keeps the pipe from filling (which would block ffmpeg) and retains
        the tail so a failure can be explained."""
        assert self._proc and self._proc.stderr
        try:
            async for raw in self._proc.stderr:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    self._stderr_tail.append(line)
                    del self._stderr_tail[:-20]
                    logger.debug("ffmpeg: %s", line)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    async def aclose(self) -> None:
        self._stopping.set()
        if self._pump and not self._pump.done():
            self._pump.cancel()
        self._pump = None

        if self._cdp is not None:
            try:
                await self._cdp.send("Page.stopScreencast")
            except Exception as exc:
                logger.debug("stopScreencast failed (page likely gone): %s", exc)
            self._cdp = None

        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        try:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()
            # Let ffmpeg flush and finalise the last MP4 header.
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        except Exception as exc:
            logger.debug("Recorder shutdown: %s", exc)


# Below this many pixels the "player" we measured is almost certainly a
# placeholder or a collapsed element, and cropping to it would throw the
# footage away entirely.
_MIN_CROP_EDGE = 160


def choose_segment_seconds(
    window_seconds: float,
    step_seconds: float,
    max_parts: int = 24,
    min_seconds: float = 2.0,
) -> float:
    """How finely to cut the recording, in video seconds.

    A window cannot be assembled until every segment it touches has been closed
    by ffmpeg, so the segment size is a floor on how far behind the playhead the
    audit runs. Cutting at the step -- the obvious choice, and what this used to
    do -- makes that floor `window + step`: 24 seconds for a 15-second window,
    and a shade under 20 minutes for a 10-minute one. Cutting finer brings it to
    `window + segment`, which approaches the real floor of one window.

    It is not free: each segment must open on a forced keyframe, so smaller
    pieces mean more keyframes and larger files, and every window is a concat of
    more parts. Hence the two bounds -- pieces no shorter than `min_seconds`, no
    more than `max_parts` of them per window. The result always divides the step
    exactly, which the assembler requires to keep window starts aligned.
    """
    if step_seconds <= 0:
        return max(min_seconds, window_seconds)
    # Segment = step / k keeps the division exact for any integer k. Take the
    # largest k -- the finest cut -- that stays inside both bounds.
    for k in range(math.ceil(step_seconds / min_seconds), 0, -1):
        seg = step_seconds / k
        if seg >= min_seconds and math.ceil(window_seconds / seg) <= max_parts:
            return seg
    return step_seconds


def content_box(rect: Optional[dict]) -> Optional[tuple]:
    """The footage inside a player box, minus its letterbox/pillarbox bars.

    `object-fit: contain` means a 9:16 phone video shown in a 16:9 player is
    two thirds black. Fullscreen does not help -- it makes the bars bigger.
    Given the intrinsic size we can cut them off and spend the whole frame on
    the footage. Returns a (w, h, x, y) box, or None when the bars are
    negligible or the intrinsic size is not known yet.
    """
    if not rect:
        return None
    try:
        box_w, box_h = float(rect["width"]), float(rect["height"])
        src_w, src_h = float(rect.get("intrinsic_width") or 0), float(rect.get("intrinsic_height") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    if min(box_w, box_h, src_w, src_h) <= 0:
        return None

    scale = min(box_w / src_w, box_h / src_h)
    width, height = src_w * scale, src_h * scale
    # A pixel or two of rounding slack is not worth a crop filter.
    if box_w - width < 4 and box_h - height < 4:
        return None
    return (width, height,
            float(rect.get("x", 0.0)) + (box_w - width) / 2,
            float(rect.get("y", 0.0)) + (box_h - height) / 2)


def normalise_crop(crop, frame_width: int, frame_height: int) -> Optional[tuple]:
    """Clamps a (w, h, x, y) box to the frame and rounds it for yuv420p.

    Odd dimensions make libx264 with yuv420p fail outright, and a box that
    hangs off the edge makes the crop filter error at runtime -- both of which
    would surface as "capture produced nothing" long after the misconfigured
    rect was read. Returns None when the box is unusable or is the whole frame,
    so the caller simply records uncropped.
    """
    if not crop:
        return None
    try:
        width, height, x, y = (int(round(float(v))) for v in crop)
    except (TypeError, ValueError):
        logger.warning("Ignoring malformed crop %r", crop)
        return None

    x = max(0, min(x, frame_width - 2))
    y = max(0, min(y, frame_height - 2))
    width = min(width, frame_width - x)
    height = min(height, frame_height - y)
    width -= width % 2
    height -= height % 2

    if width < _MIN_CROP_EDGE or height < _MIN_CROP_EDGE:
        logger.warning(
            "Player box %dx%d is too small to crop to; recording the full frame.", width, height
        )
        return None
    if width >= frame_width and height >= frame_height:
        return None
    return (width, height, x, y)


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
