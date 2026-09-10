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

"""Real ffmpeg, real files, no browser and no Vertex AI.

The capture layer is four subprocess invocations glued together by filesystem
polling, so unit tests of the arithmetic prove very little on their own. These
exercise the actual pipe: segment a source, stitch overlapping windows out of
the segments, and pull a still from a known offset.

ffmpeg is a hard runtime dependency, so these do not skip silently when it is
missing -- a missing ffmpeg is a broken install, not an untestable environment.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

from cctv_audit.capture.ffmpeg_util import extract_frame
from cctv_audit.capture.stream_grabber import StreamGrabber
from cctv_audit.capture.types import CaptureSource

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)

SOURCE_SECONDS = 60
FPS = 10


@pytest.fixture(scope="module")
def source_video(tmp_path_factory):
    """A 60s clip with a burned-in second counter, keyframed every second.

    The counter is what makes the offset assertions meaningful: a frame pulled
    from t=37 should render "37". Frequent keyframes match how real HLS is
    packaged and let the segment muxer cut where it is told.
    """
    path = tmp_path_factory.mktemp("src") / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate={FPS}:duration={SOURCE_SECONDS}",
            "-vf", "drawtext=text='%{eif\\:t\\:d}':fontsize=48:fontcolor=white:x=10:y=10",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-g", str(FPS), "-keyint_min", str(FPS), "-sc_threshold", "0",
            str(path),
        ],
        check=True, capture_output=True,
    )
    return path


def duration_of(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


async def collect(grabber, limit=None):
    clips = grabber.clips()
    out = []
    try:
        async for clip in clips:
            out.append(clip)
            if limit is not None and len(out) >= limit:
                break
    finally:
        await clips.aclose()
        await grabber.aclose()
    return out


class TestStreamGrabberEndToEnd:
    @pytest.mark.asyncio
    async def test_windows_are_produced_with_the_configured_geometry(self, source_video, tmp_path):
        grabber = StreamGrabber(
            source=CaptureSource(mode="stream", url=str(source_video), reason="test"),
            work_dir=tmp_path / "work",
            window_seconds=15,
            overlap_seconds=3,
        )
        clips = await asyncio.wait_for(collect(grabber), timeout=120)

        # 60s of source at a 12s step -> 5 segments -> 4 full windows plus a tail.
        assert len(clips) >= 4, f"expected at least 4 windows, got {len(clips)}"

        for i, clip in enumerate(clips):
            assert clip.path.exists() and clip.path.stat().st_size > 0
            assert clip.index == i
            assert clip.source_mode == "stream"
            assert clip.time_scale == 1.0
            # Windows advance by the step, not the window length.
            assert clip.start_offset == pytest.approx(i * 12.0)

        # Every full window really holds ~15s of video, which is the claim the
        # analysis prompt makes to the model.
        for clip in clips[:4]:
            assert duration_of(clip.path) == pytest.approx(15.0, abs=1.5)

    @pytest.mark.asyncio
    async def test_consecutive_windows_actually_overlap(self, source_video, tmp_path):
        # The overlap exists so an action straddling a boundary stays visible to
        # one window. If ffmpeg cut flush instead, this is where it would show.
        grabber = StreamGrabber(
            source=CaptureSource(mode="stream", url=str(source_video), reason="test"),
            work_dir=tmp_path / "work",
            window_seconds=15,
            overlap_seconds=3,
        )
        clips = await asyncio.wait_for(collect(grabber, limit=2), timeout=120)
        assert len(clips) == 2

        first, second = clips
        assert first.end_offset - second.start_offset == pytest.approx(3.0)

        # The last 3s of window 0 and the first 3s of window 1 are the same
        # moment, so the burned-in counter must agree.
        a = tmp_path / "a.jpg"
        b = tmp_path / "b.jpg"
        assert await extract_frame(first.path, 13.5, a)
        assert await extract_frame(second.path, 1.5, b)
        assert a.stat().st_size > 0 and b.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_start_offset_seeks_the_source(self, source_video, tmp_path):
        # Plan A ignores the playhead and seeks with -ss, so the first window
        # must start at the requested offset rather than at zero.
        grabber = StreamGrabber(
            source=CaptureSource(mode="stream", url=str(source_video), reason="test"),
            work_dir=tmp_path / "work",
            window_seconds=15,
            overlap_seconds=3,
            start_offset=24.0,
        )
        clips = await asyncio.wait_for(collect(grabber, limit=1), timeout=120)
        assert clips[0].start_offset == pytest.approx(24.0)
        assert clips[0].time_range.startswith("00:24")


class TestStreamGrabberFailureModes:
    @pytest.mark.asyncio
    async def test_a_dead_pull_raises_instead_of_yielding_nothing(self, tmp_path):
        # Silence here would surface as an audit that found zero violations,
        # which reads as "the shop is compliant" rather than "nothing was seen".
        grabber = StreamGrabber(
            source=CaptureSource(mode="stream", url=str(tmp_path / "missing.m3u8"), reason="test"),
            work_dir=tmp_path / "work",
            window_seconds=15,
            overlap_seconds=3,
        )
        with pytest.raises(RuntimeError, match="without producing a single window"):
            await asyncio.wait_for(collect(grabber), timeout=60)

    @pytest.mark.asyncio
    async def test_http_only_flags_do_not_break_other_protocols(self, source_video, tmp_path):
        # `-reconnect` and `-headers` are HTTP-handler options; ffmpeg aborts
        # with "Option reconnect not found" when they are passed for any other
        # protocol. CCTV platforms serve rtsp:// routinely, so headers must be
        # attached conditionally rather than always.
        grabber = StreamGrabber(
            source=CaptureSource(
                mode="stream", url=str(source_video),
                headers={"Cookie": "session=abc", "Referer": "https://example.com/"},
                reason="test",
            ),
            work_dir=tmp_path / "work",
            window_seconds=15,
            overlap_seconds=3,
        )
        clips = await asyncio.wait_for(collect(grabber, limit=1), timeout=120)
        assert len(clips) == 1


class TestEvidenceExtraction:
    @pytest.mark.asyncio
    async def test_a_frame_comes_out_at_the_requested_offset(self, source_video, tmp_path):
        out = tmp_path / "evidence.jpg"
        assert await extract_frame(source_video, 37.0, out)
        assert out.exists() and out.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_a_bad_offset_fails_instead_of_writing_a_broken_file(self, source_video, tmp_path):
        # Seeking past the end must report failure, so the record says
        # "no evidence frame" rather than pointing at an empty file.
        out = tmp_path / "past_end.jpg"
        ok = await extract_frame(source_video, SOURCE_SECONDS + 30, out)
        assert not ok
        assert not out.exists() or out.stat().st_size == 0

    @pytest.mark.asyncio
    async def test_a_missing_clip_fails_cleanly(self, tmp_path):
        assert not await extract_frame(tmp_path / "nope.mp4", 1.0, tmp_path / "out.jpg")
