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

"""Plan A: download the media the player is already fetching.

This does not "play" anything -- ffmpeg pulls the HLS/FLV/DASH segments as
fast as the server and link allow. For *recorded* footage that is typically far
faster than real time, which is what makes multi-hour audits practical. For a
*live* stream there is nothing to race: the data is produced in real time, so
Plan A runs at real time too.

No platform API is required. We only reuse the transport the player itself uses.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import AsyncIterator, Optional

from .ffmpeg_util import require_ffmpeg
from .types import CaptureSource, Clip
from .window_assembler import SEGMENT_PATTERN, WindowAssembler

logger = logging.getLogger("cctv_audit.stream")


class StreamGrabber:
    def __init__(
        self,
        source: CaptureSource,
        work_dir: Path,
        window_seconds: float,
        overlap_seconds: float,
        start_offset: float = 0.0,
    ):
        if source.mode != "stream" or not source.url:
            raise ValueError("StreamGrabber requires a CaptureSource with mode='stream'")
        self.source = source
        self.work_dir = work_dir
        self.segment_dir = work_dir / "segments"
        self.window_seconds = float(window_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.start_offset = float(start_offset)
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_tail: list[str] = []

    async def clips(self) -> AsyncIterator[Clip]:
        ffmpeg, _ = require_ffmpeg()
        _reset_dir(self.segment_dir)

        step = self.window_seconds - self.overlap_seconds
        args = [ffmpeg, "-nostdin", "-v", "warning", "-y"]

        # `-headers` and the `-reconnect*` family belong to the HTTP protocol
        # handler. ffmpeg does not merely ignore them elsewhere -- it exits with
        # "Option reconnect not found" and produces no output at all. CCTV
        # platforms hand out rtsp:// and rtmp:// URLs routinely, so applying
        # them unconditionally would make Plan A fail silently on exactly the
        # sources this project exists to audit.
        is_http = self.source.url.lower().startswith(("http://", "https://"))
        if is_http:
            if self.source.headers:
                blob = "".join(f"{k}: {v}\r\n" for k, v in self.source.headers.items())
                args += ["-headers", blob]
            # Reconnect flags matter for multi-hour pulls over flaky links.
            args += [
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "10",
            ]
        elif self.source.headers:
            logger.debug(
                "Ignoring %d request header(s): the source is not HTTP (%s).",
                len(self.source.headers), self.source.url.split("://", 1)[0],
            )
        if self.start_offset > 0:
            args += ["-ss", f"{self.start_offset:.3f}"]
        args += [
            "-i", self.source.url,
            "-an",  # audio is irrelevant to visual SOP checks and costs tokens
            "-c:v", "copy",
            "-f", "segment",
            "-segment_time", f"{step:.3f}",
            "-reset_timestamps", "1",
            "-segment_format", "mp4",
            "-segment_format_options", "movflags=+faststart",
            str(self.segment_dir / SEGMENT_PATTERN),
        ]

        logger.info("Plan A: grabbing stream directly (%s)", self.source.reason)
        self._proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        drain = asyncio.create_task(self._drain_stderr())

        assembler = WindowAssembler(
            segment_dir=self.segment_dir,
            out_dir=self.work_dir / "windows",
            window_seconds=self.window_seconds,
            overlap_seconds=self.overlap_seconds,
            source_mode="stream",
            start_offset=self.start_offset,
            time_scale=1.0,
        )
        emitted = 0
        returncode: Optional[int] = None
        try:
            async for clip in assembler.windows(producer_alive=lambda: self._proc.returncode is None):
                emitted += 1
                yield clip
        finally:
            drain.cancel()
            returncode = self._proc.returncode if self._proc else None
            await self.aclose()

        # An ffmpeg that dies on its first breath leaves an empty segment dir,
        # which the assembler reads as "nothing to do" and the pipeline as a
        # clean finish. That turns a broken pull into an audit reporting zero
        # violations, which is the worst possible way to fail.
        if emitted == 0 and returncode not in (0, None):
            raise RuntimeError(
                f"ffmpeg exited {returncode} without producing a single window. "
                f"Last output:\n{self.stderr_tail or '(no stderr)'}"
            )

    async def _drain_stderr(self) -> None:
        """Keeps the stderr pipe from filling (which would block ffmpeg) and
        retains the tail so a failure can be explained."""
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
        except Exception as exc:
            logger.debug("stderr drain ended: %s", exc)

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    async def aclose(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
