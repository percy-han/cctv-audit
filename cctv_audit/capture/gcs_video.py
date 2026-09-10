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

"""Plan C: a video file handed to us directly, as `gs://bucket/path/clip.mp4`.

Plans A and B both start from a web page. The customer's own SOP recordings do
not come that way -- they are files, some of them hundreds of megabytes, and
there is no player to drive, no login to survive and no picture to record.

**Nothing is captured on this path.** The object's address goes to Vertex and
Vertex reads it; the bytes never enter this container. That is not an
optimisation, it is the removal of a step that existed only because the other
two plans need it: footage that already sits in a bucket in the same project
does not have to be pulled through here, cut into MP4s, and pushed back out
again. It also sidesteps the container filesystem being memory-backed on Cloud
Run, where a 2 GB file would have been 2 GB of the container's 8 GiB RAM.

The unit of analysis is therefore **the whole recording, in one request**,
rather than a window of it -- see `WholeFileProducer` for why windowing a
`gs://` object is actively wrong under `MEDIA_PROCESSING=agentic`, not merely
wasteful. Anything that genuinely needs cutting up can be cut before it is
uploaded to the bucket.

ffmpeg is still here, for the two things Vertex does not do: reading the
duration in preflight, and cutting the stills that back each violation. Both
are byte-range reads of a few hundred KB against the storage REST endpoint,
with a freshly minted bearer token each time -- which is what keeps evidence
frames working on an audit that outlives the token that started it.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Optional
from urllib.parse import quote

from ..gcp import access_token
from .ffmpeg_util import header_args, probe_stream_verbose, require_ffmpeg
from .types import CaptureSource, Clip

logger = logging.getLogger("cctv_audit.gcs")

GS_SCHEME = "gs://"

# Bucket naming rules, tightened: lowercase letters, digits, dashes, dots and
# underscores, 3-222 characters. The point is not to reimplement Google's
# validator -- it is that this string is about to be interpolated into a URL,
# so anything that could carry a `/`, a `?` or a `@` out of the bucket position
# and into the path has to be refused here rather than fetched.
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")

# What we are willing to hand ffmpeg. Not a security control -- the bucket's
# IAM is that -- but a customer who pastes the URI of a spreadsheet should be
# told so in preflight, not after a browserless run produces nothing.
_VIDEO_SUFFIXES = (
    ".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".flv", ".webm", ".mpg", ".mpeg", ".m3u8",
)

# What Vertex is told the object is. Only the containers it accepts are listed;
# an unknown suffix falls back to video/mp4 rather than being refused here,
# because ffprobe has already opened the thing by the time this is asked.
_MIME_BY_SUFFIX = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".flv": "video/x-flv",
    ".mpg": "video/mpeg",
    ".mpeg": "video/mpeg",
}


class BadGcsUri(ValueError):
    """The string is not a usable `gs://` object URI. Message is for the customer."""


def is_gcs_uri(target: str) -> bool:
    """True for anything that means to be a `gs://` URI, valid or not.

    Deliberately loose. Routing happens on this, and a malformed `gs://` should
    reach `parse_gs_uri` and get a specific complaint, not fall through to the
    browser and get "打不开这个视频".
    """
    return (target or "").strip().lower().startswith(GS_SCHEME)


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """`gs://bucket/a/b.mp4` -> `("bucket", "a/b.mp4")`. Raises `BadGcsUri`."""
    raw = (uri or "").strip()
    if not is_gcs_uri(raw):
        raise BadGcsUri(f"{raw!r} 不是一个 gs:// 地址")
    rest = raw[len(GS_SCHEME):]
    bucket, _, name = rest.partition("/")
    bucket = bucket.lower()
    if not _BUCKET_RE.match(bucket):
        raise BadGcsUri(f"gs:// 后面的存储桶名不合法：{bucket!r}")
    if not name:
        raise BadGcsUri(
            f"gs://{bucket} 只给到了存储桶，没给文件。"
            f"要写成 gs://{bucket}/文件夹/文件名.mp4 这样"
        )
    if name.endswith("/"):
        raise BadGcsUri(f"{raw} 指的是一个目录，不是一个视频文件")
    return bucket, name


def media_url(bucket: str, name: str) -> str:
    """The HTTPS address ffmpeg reads the object from.

    The JSON API's `?alt=media` rather than the XML endpoint, because the object
    name has to be percent-encoded *including* its slashes, and that is the one
    of the two whose contract says so. Object names legitimately contain `#`,
    `?` and spaces; leaving any of them raw truncates the path silently and
    fetches a different object, or none.
    """
    return (
        f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}"
        f"/o/{quote(name, safe='')}?alt=media"
    )


def auth_headers() -> dict:
    """The `Authorization` header ffmpeg needs. Minted fresh; see the module note."""
    return {"Authorization": f"Bearer {access_token()}"}


def looks_like_video(name: str) -> bool:
    return name.lower().endswith(_VIDEO_SUFFIXES)


async def open_source(uri: str, *, timeout: float = 30.0) -> CaptureSource:
    """Probes the object and returns the `CaptureSource` the pipeline captures from.

    Raises with a sentence the customer can act on. Every failure here is one
    only they can fix -- wrong path, wrong bucket, no permission, not a video --
    so swallowing it into a generic "打不开" would be the worst of both worlds.

    A longer default timeout than the page probe's 15s: this reads over the
    public internet from whatever region the customer's bucket is in, and the
    moov atom of a file written without `+faststart` is at the far end of it.
    """
    bucket, name = parse_gs_uri(uri)
    url = media_url(bucket, name)
    if not looks_like_video(name):
        logger.info("gs://%s/%s has no video-looking suffix; probing anyway.", bucket, name)

    info, problem = await probe_stream_verbose(url, auth_headers(), timeout=timeout)
    if info is None:
        raise RuntimeError(_explain(bucket, name, problem))

    duration = _duration_of(info)
    logger.info(
        "Plan C: reading gs://%s/%s directly (%s).",
        bucket, name,
        f"{duration:.1f}s" if duration else "时长未知",
    )
    return CaptureSource(
        mode="file",
        url=url,
        headers=auth_headers(),
        reason=f"直接读 GCS 上的视频文件 {name}",
        duration_seconds=duration,
        object_uri=f"{GS_SCHEME}{bucket}/{name}",
    )


def mime_for(name: str) -> str:
    """The `mime_type` Vertex is told the object is.

    Vertex will not sniff it, and a wrong one is rejected rather than guessed
    around. `video/mp4` is the fallback because it is what an unrecognised
    suffix is most likely to actually be -- and because a wrong guess here
    fails loudly at the first request, not silently halfway through a run.
    """
    return _MIME_BY_SUFFIX.get("." + name.rsplit(".", 1)[-1].lower(), "video/mp4")


async def grab_frame(
    source: CaptureSource, offset_seconds: float = 0.0, *, timeout: float = 30.0
) -> Optional[bytes]:
    """One JPEG from `offset_seconds` into the object, for preflight's cover.

    Plan A and Plan B both get this for free -- there is a browser open, so a
    screenshot is one call. Plan C has no browser, and dropping the cover here
    would quietly remove the one part of the preflight reply that proves we
    opened the customer's footage and not somebody else's file.

    Best effort, like the screenshot it replaces: a preflight that can answer
    every other question should not fail over a thumbnail.
    """
    return await _decode_frame(source.url, source.headers, offset_seconds, timeout)


async def frame_at(
    uri: str, offset_seconds: float, out_path, *, timeout: float = 30.0
) -> bool:
    """Cuts one still out of a `gs://` object and writes it to `out_path`.

    This is how evidence frames survive Plan C. On the other two plans the
    clip is already a file on disk and ffmpeg cuts from that; here there is no
    local copy by design, so the frame comes out of the object over a
    byte-range read -- a few hundred KB for a still, whatever the file's size.

    Mints its own headers rather than reusing the source's: on a long audit
    the token that opened the run may well have expired by the time the last
    violation needs a picture, and an evidence frame is the last thing that
    should be lost to that.
    """
    bucket, name = parse_gs_uri(uri)
    data = await _decode_frame(media_url(bucket, name), auth_headers(), offset_seconds, timeout)
    if not data:
        return False
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
    except OSError as exc:
        logger.warning("Could not write evidence frame %s: %s", out_path, exc)
        return False
    return True


async def _decode_frame(
    url: str, headers: Optional[dict], offset_seconds: float, timeout: float
) -> Optional[bytes]:
    ffmpeg, _ = require_ffmpeg()
    args = [ffmpeg, "-nostdin", "-v", "error"]
    # Before `-i`: seeks by keyframe instead of decoding up to the offset,
    # which on an hour-long file is the difference between instant and a
    # timeout. A still does not need to be the exact frame.
    if offset_seconds > 0:
        args += ["-ss", f"{offset_seconds:.3f}"]
    args += header_args(headers)
    args += [
        "-i", url,
        "-frames:v", "1",
        "-f", "image2",
        "-vcodec", "mjpeg",
        "-",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (OSError, asyncio.TimeoutError) as exc:
        logger.info("Could not decode a frame at %.1fs: %s", offset_seconds, exc)
        return None
    if proc.returncode != 0 or not stdout:
        logger.info(
            "Could not decode a frame at %.1fs: %s",
            offset_seconds, stderr.decode("utf-8", "replace").strip()[:200],
        )
        return None
    return stdout


class WholeFileProducer:
    """Hands the pipeline the entire object as a single clip, then stops.

    Plan C used to slice the object into windows with ffmpeg, the same way
    Plan A slices a CDN stream. Measured on 2026-09-09 that was the wrong
    shape twice over:

      * Vertex reads `gs://` URIs directly, so slicing meant pulling every
        byte through this container to produce files that were then uploaded
        back out again -- for footage that already lives in a bucket in the
        same project.
      * `MEDIA_PROCESSING=agentic` **ignores** `VideoMetadata` start/end
        offsets. Probed twice on the same object: asked for 240s-300s, static
        read exactly 00:09:00-00:09:59, agentic read the whole 00:00-05:00.
        So under agentic every "window" would silently re-read the entire
        file: N times the cost, and timestamps that do not mean what the
        report says they mean.

    Whole-file is also what agentic is for -- one pass in which the model
    picks where to look. A 5-minute probe cost 12,087 tokens whole against
    ~169,000 for the same span cut into agentic windows.

    Windowing has not gone away; it is where it belongs. Anything that needs
    cutting can be cut before it is uploaded to the bucket.
    """

    def __init__(self, source: CaptureSource, *, duration_seconds: Optional[float] = None):
        if source.mode != "file" or not source.object_uri:
            raise ValueError("WholeFileProducer requires a CaptureSource with mode='file'")
        self.source = source
        self.duration_seconds = duration_seconds or source.duration_seconds

    async def clips(self):
        # end_offset 0 when the duration is unknown: `time_range` then reads
        # "00:00 - 00:00", which is wrong-looking rather than wrong-and-plausible.
        # ffprobe having failed to read a duration is already logged in preflight.
        yield Clip(
            index=0,
            path=None,
            uri=self.source.object_uri,
            whole_video=True,
            start_offset=0.0,
            end_offset=float(self.duration_seconds or 0.0),
            wall_clock_start=time.time(),
            source_mode="file",
        )

    async def aclose(self) -> None:
        """Nothing was opened. Present because the pipeline closes its producer."""
        return None


def _explain(bucket: str, name: str, problem: str) -> str:
    """Turns ffprobe's complaint into the thing the customer should go check.

    ffmpeg reports an HTTP status and nothing else, and "Server returned 403
    Forbidden" is not actionable until somebody tells you *whose* permission is
    missing -- which is ours, on their bucket, and is the single most likely
    thing to be wrong the first time this is used.
    """
    where = f"gs://{bucket}/{name}"
    if "404" in problem or "Not Found" in problem:
        return f"{where} 这个文件不存在。检查一下路径和大小写（对象名是区分大小写的）。"
    if "403" in problem or "401" in problem or "Forbidden" in problem:
        return (
            f"没有权限读 {where}。需要给稽核服务账号在这个桶上加 "
            f"roles/storage.objectViewer。"
        )
    return f"读不了 {where}：{problem}"


def _duration_of(info: dict) -> Optional[float]:
    """Container duration in seconds, or None. Mirrors `probe._duration_of`."""
    try:
        value = float((info.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        return None
    # An unfinalised or fragmented recording reports 0 or Infinity. Neither is
    # a length, and claiming one would make every span check pass.
    return value if value > 0 and value != float("inf") else None
