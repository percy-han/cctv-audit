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

"""Thin async wrappers around the ffmpeg / ffprobe binaries.

ffmpeg is the bridge between "a pile of frames" and "a video the model can
reason over temporally", so a missing binary is a hard startup error rather
than something to degrade around.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cctv_audit.ffmpeg")


class FFmpegMissingError(RuntimeError):
    pass


def require_ffmpeg() -> tuple[str, str]:
    """Returns (ffmpeg_path, ffprobe_path) or raises with install instructions."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FFmpegMissingError(
            "ffmpeg/ffprobe not found on PATH. They are required to assemble and "
            "slice video clips.\n"
            "  Debian/Ubuntu: sudo apt install ffmpeg\n"
            "  macOS:         brew install ffmpeg"
        )
    return ffmpeg, ffprobe


async def probe_stream(url: str, headers: Optional[dict] = None, timeout: float = 15.0) -> Optional[dict]:
    """Returns ffprobe's format/stream info for `url`, or None if it is not readable.

    This is the gate for Plan A: a URL that ffprobe cannot open (proprietary
    container, DRM, auth failure) must fall back to screen recording.
    """
    info, _ = await probe_stream_verbose(url, headers, timeout)
    return info


async def probe_stream_verbose(
    url: str, headers: Optional[dict] = None, timeout: float = 15.0
) -> tuple[Optional[dict], str]:
    """`probe_stream`, plus what ffprobe said when it said no.

    Plan A does not need the reason -- it just falls back to recording the
    page. A `gs://` object has nothing to fall back to, so "404 Not Found" and
    "403 Forbidden" and "this is a .txt" have to reach the customer, who is the
    only one who can fix any of them.
    """
    _, ffprobe = require_ffmpeg()
    args = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams"]
    args += header_args(headers)
    args += [url]

    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except OSError as exc:
        logger.warning("ffprobe could not start: %s", exc)
        return None, f"ffprobe 起不来：{exc}"

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.info("ffprobe timed out after %.0fs on %s", timeout, _redact(url))
        return None, f"{timeout:.0f} 秒内没读出这个文件的信息"

    if proc.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip()
        logger.info("ffprobe rejected %s: %s", _redact(url), detail[:200])
        # The last line is the one that names the cause; the ones before it are
        # ffmpeg's banner and per-protocol chatter.
        last = detail.splitlines()[-1] if detail else ""
        return None, _redact(last)[:300] or "打不开这个文件"

    try:
        info = json.loads(stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None, "ffprobe 的输出读不成 JSON"

    has_video = any(s.get("codec_type") == "video" for s in info.get("streams", []))
    if not has_video:
        logger.info("ffprobe found no video stream in %s", _redact(url))
        return None, "这个文件里没有视频轨"
    return info, ""


async def extract_frame(clip_path: Path, offset_seconds: float, out_path: Path) -> bool:
    """Pulls a single frame out of a clip as evidence for a finding."""
    ffmpeg, _ = require_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg, "-nostdin", "-v", "error", "-y",
        # -ss before -i is a fast keyframe seek; clips are short so accuracy is fine.
        "-ss", f"{max(0.0, offset_seconds):.3f}",
        "-i", str(clip_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not out_path.exists():
        logger.warning(
            "Evidence frame extraction failed for %s @%.2fs: %s",
            clip_path.name, offset_seconds, stderr.decode("utf-8", "replace")[:200],
        )
        return False
    return True


def header_args(headers: Optional[dict]) -> list[str]:
    if not headers:
        return []
    blob = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    return ["-headers", blob]


def _redact(url: str) -> str:
    """Media URLs routinely carry auth tokens in the query string."""
    return url.split("?", 1)[0] + ("?<redacted>" if "?" in url else "")
