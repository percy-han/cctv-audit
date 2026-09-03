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

"""Turns a stream of fixed-size ffmpeg segments into overlapping analysis windows.

Both capture modes (direct stream grab and screen recording) run ffmpeg with
the `segment` muxer, so they share this assembler.

Why segments plus assembly, rather than asking ffmpeg for the windows directly:
the segment muxer cannot produce overlapping output. Overlap matters here
because an action that straddles a boundary -- "picked up a phone, then touched
a cup rim without washing hands" -- is invisible to both neighbouring windows
if they are cut flush. So ffmpeg emits fixed-size pieces and we stitch each
window from the pieces it spans with a stream copy, which costs no re-encoding.

The piece size sets how far behind the playhead the audit runs, and it is not
the same thing as the step. An mp4 cannot be read until it is closed, so a
window is only assemblable once every piece it touches has been closed --
including the one holding the last three seconds of it. Cut at the step, a 15s
window (step 12) waits 24 seconds to cover 15, and a 10-minute window (step
597) waits nearly 20 minutes to cover 10. Cut finer, the wait falls to roughly
one window plus one piece, which is the floor: you cannot judge fifteen seconds
of footage before fifteen seconds have played. Callers that control their
encoder pass `segment_seconds`; the default keeps the old step-sized pieces.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from .ffmpeg_util import require_ffmpeg
from .types import Clip

logger = logging.getLogger("cctv_audit.assembler")

SEGMENT_PATTERN = "seg_%05d.mp4"
_SEGMENT_RE = re.compile(r"^seg_(\d{5})\.mp4$")


class WindowAssembler:
    """Watches a segment directory and yields overlapping `Clip` windows."""

    def __init__(
        self,
        segment_dir: Path,
        out_dir: Path,
        window_seconds: float,
        overlap_seconds: float,
        source_mode: str,
        start_offset: float = 0.0,
        time_scale: float = 1.0,
        poll_interval: float = 0.4,
        segment_seconds: Optional[float] = None,
    ):
        if overlap_seconds >= window_seconds:
            raise ValueError("overlap must be smaller than the window, else windows never advance")
        self.segment_dir = segment_dir
        self.out_dir = out_dir
        self.window_seconds = float(window_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.step_seconds = self.window_seconds - self.overlap_seconds
        self.source_mode = source_mode
        self.start_offset = float(start_offset)
        self.time_scale = float(time_scale)
        self.poll_interval = poll_interval
        # Segment size, in video seconds like everything else here. It has to
        # divide the step, or window N would start mid-segment and every
        # timestamp after it would be wrong.
        self.segment_seconds = float(segment_seconds or self.step_seconds)
        steps = self.step_seconds / self.segment_seconds
        if self.segment_seconds <= 0 or abs(steps - round(steps)) > 1e-6:
            raise ValueError(
                f"segment_seconds ({self.segment_seconds}) must divide the step "
                f"({self.step_seconds}) exactly"
            )
        # How many segments a window spans, and how many it advances by.
        self.segments_per_window = max(1, math.ceil(self.window_seconds / self.segment_seconds))
        self.segments_per_step = max(1, int(round(steps)))
        self.out_dir.mkdir(parents=True, exist_ok=True)

    async def windows(self, producer_alive) -> AsyncIterator[Clip]:
        """Yields clips as they become available.

        `producer_alive` is a zero-arg callable returning False once the ffmpeg
        producer has exited; at that point the final partial window is flushed.
        """
        emitted = 0
        while True:
            ready = self._completed_segments()
            first = emitted * self.segments_per_step
            needed = first + self.segments_per_window

            if len(ready) >= needed:
                clip = await self._assemble(emitted, ready[first:needed])
                if clip is not None:
                    yield clip
                emitted += 1
                continue

            if not producer_alive():
                # Drain: the producer is gone, so the newest file is complete
                # too. Allow the tail window to be shorter than a full window
                # rather than silently discarding the last seconds of footage.
                remaining = self._completed_segments(include_last=True)
                while emitted * self.segments_per_step < len(remaining):
                    first = emitted * self.segments_per_step
                    parts = remaining[first:first + self.segments_per_window]
                    clip = await self._assemble(
                        emitted, parts, partial=len(parts) < self.segments_per_window
                    )
                    if clip is not None:
                        yield clip
                    emitted += 1
                return

            await asyncio.sleep(self.poll_interval)

    def _completed_segments(self, include_last: bool = False) -> list[Path]:
        """Segments ffmpeg has finished writing.

        A segment is complete once its successor exists. While the producer is
        running the newest file is still being appended to, so it is withheld.
        """
        try:
            found = sorted(
                (p for p in self.segment_dir.iterdir() if _SEGMENT_RE.match(p.name)),
                key=lambda p: p.name,
            )
        except FileNotFoundError:
            return []
        if not include_last:
            found = found[:-1]
        return [p for p in found if p.stat().st_size > 0]

    async def _assemble(self, index: int, parts: list[Path], partial: bool = False) -> Optional[Clip]:
        if not parts:
            return None

        ffmpeg, _ = require_ffmpeg()
        out_path = self.out_dir / f"window_{index:05d}.mp4"
        wall_start = time.time()
        # `-t` is measured in file seconds, while window_seconds is video
        # seconds. They differ whenever the page was recorded at speed.
        file_duration = self.window_seconds / self.time_scale

        args = [ffmpeg, "-nostdin", "-v", "error", "-y"]
        list_path: Optional[Path] = None
        if len(parts) == 1:
            args += ["-i", str(parts[0])]
        else:
            list_path = self.out_dir / f"concat_{index:05d}.txt"
            list_path.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
            args += ["-f", "concat", "-safe", "0", "-i", str(list_path)]
        if not partial:
            args += ["-t", f"{file_duration:.3f}"]
        args += ["-c", "copy", "-movflags", "+faststart", str(out_path)]

        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if list_path is not None:
            list_path.unlink(missing_ok=True)

        if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            logger.warning(
                "Window %d could not be assembled from %d segment(s): %s",
                index, len(parts), stderr.decode("utf-8", "replace")[:250],
            )
            return None

        start = self.start_offset + index * self.step_seconds
        covered = (self.window_seconds if not partial
                   else min(self.window_seconds, len(parts) * self.segment_seconds))
        return Clip(
            index=index,
            path=out_path,
            start_offset=start,
            end_offset=start + covered,
            wall_clock_start=wall_start,
            source_mode=self.source_mode,
            time_scale=self.time_scale,
        )
