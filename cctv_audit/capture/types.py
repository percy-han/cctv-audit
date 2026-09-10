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

    # "stream" -- the media the page is already fetching (Plan A)
    # "screen"  -- record the page playing it (Plan B)
    # "file"    -- a video object handed to us directly, no page involved
    #              (Plan C, `gs://...`). Kept distinct from "stream" rather
    #              than folded into it because everything that reads this field
    #              is deciding whether there is a page to mind, and on this path
    #              there is not.
    mode: str
    url: Optional[str] = None  # media URL, when mode == "stream"
    headers: Optional[dict] = None  # cookies / referer needed to fetch it
    reason: str = ""  # why this mode was chosen, surfaced in logs and the dashboard
    # The object's own address, `gs://bucket/name`, when mode == "file". Kept
    # alongside `url` rather than derived from it because the two go to
    # different places: `url` is the HTTPS endpoint ffmpeg reads, and this is
    # what Vertex is given, which only accepts the `gs://` form.
    object_uri: Optional[str] = None
    # How long the media actually is, when ffprobe could tell us. Preflight
    # answers "is the time span you asked for even in this recording", and it
    # needs a number for that -- the same figure was previously formatted into
    # `reason` and thrown away. None means live, or unknown.
    duration_seconds: Optional[float] = None


@dataclass(frozen=True)
class Clip:
    """One unit of analysis: either an MP4 we cut, or an object we point at.

    Offsets are in *video* seconds (i.e. already corrected for playback rate),
    so they line up with what an operator sees on the platform's timeline.

    `time_scale` is video-seconds per clip-second. It is 1.0 for a direct
    stream grab, and equals the playback rate for a screen recording made at
    speed: a 15s window recorded at 4x is a 3.75s file, so a model-reported
    timestamp of 2.0s inside the clip means 8.0s into the window.

    Plan C fills `uri` instead of `path`: the bytes stay in the bucket and
    Vertex fetches them itself. That is the whole reason this type grew a
    second way to say where the footage is -- with a `gs://` object there is
    nothing to cut, nothing to upload and nothing to delete afterwards, and
    every step in between was pure cost.
    """

    index: int
    path: Optional[Path]
    start_offset: float
    end_offset: float
    wall_clock_start: float  # time.time() when capture of this window began
    source_mode: str
    time_scale: float = 1.0
    # `gs://bucket/name`, when the footage is read by Vertex rather than by us.
    uri: Optional[str] = None
    # Whether this covers the entire recording rather than a window of it.
    # Not implied by `uri`: the two are separate decisions, and the prompt
    # needs this one (a whole video can hold the same violation five times,
    # and "one finding per rule" would report one of them).
    whole_video: bool = False

    def __post_init__(self) -> None:
        if (self.path is None) == (self.uri is None):
            raise ValueError(
                "a Clip needs exactly one of `path` (bytes we hold) or `uri` "
                "(bytes Vertex fetches)"
            )

    @property
    def is_remote(self) -> bool:
        """True when nobody local ever holds these bytes."""
        return self.uri is not None

    @property
    def duration(self) -> float:
        return max(0.0, self.end_offset - self.start_offset)

    def clip_ts_to_video_offset(self, clip_seconds: float) -> float:
        """Maps a timestamp *inside the clip file* to an absolute video offset."""
        if self.whole_video and not self.duration:
            # ffprobe could not read a length, so `end_offset` is 0 and the
            # clamp below would pull every finding in the recording back to
            # 00:00 -- silently, and with an evidence frame from the wrong
            # moment to match. An unbounded timestamp from the model can be
            # wrong; a clamped one is guaranteed to be.
            return max(clip_seconds, 0.0)
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
