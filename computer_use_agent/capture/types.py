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

"""Shared value types for the capture layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CaptureSource:
    """What the probe decided to capture, and how."""

    mode: str  # "stream" | "screen"
    url: Optional[str] = None  # media URL, when mode == "stream"
    headers: Optional[dict] = None  # cookies / referer needed to fetch it
    reason: str = ""  # why this mode was chosen, surfaced in logs and the dashboard


@dataclass(frozen=True)
class Clip:
    """One analysis window, materialised as an MP4 on disk.

    Offsets are in *video* seconds (i.e. already corrected for playback rate),
    so they line up with what an operator sees on the platform's timeline.

    `time_scale` is video-seconds per clip-second. It is 1.0 for a direct
    stream grab, and equals the playback rate for a screen recording made at
    speed: a 15s window recorded at 4x is a 3.75s file, so a model-reported
    timestamp of 2.0s inside the clip means 8.0s into the window.
    """

    index: int
    path: Path
    start_offset: float
    end_offset: float
    wall_clock_start: float  # time.time() when capture of this window began
    source_mode: str
    time_scale: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_offset - self.start_offset)

    def clip_ts_to_video_offset(self, clip_seconds: float) -> float:
        """Maps a timestamp *inside the clip file* to an absolute video offset."""
        bounded = min(max(clip_seconds, 0.0), self.duration / max(self.time_scale, 1e-6))
        return self.start_offset + bounded * self.time_scale

    @staticmethod
    def format_offset(seconds: float) -> str:
        """Renders a video offset as HH:MM:SS (or MM:SS under an hour)."""
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @property
    def time_range(self) -> str:
        return f"{self.format_offset(self.start_offset)} - {self.format_offset(self.end_offset)}"
