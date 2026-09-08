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

"""Decide whether we can grab the media stream directly (Plan A) or must
screen-record the rendered page (Plan B).

This watches what the *player itself* downloads. It does not need any platform
API -- which matters, because the target CCTV platforms are not expected to
expose one. Hikvision/Dahua web clients often use a proprietary transport with
WASM decoding into a canvas; ffprobe will reject those and we fall back to
Plan B.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional
from urllib.parse import urlsplit

from .ffmpeg_util import probe_stream
from .types import CaptureSource

logger = logging.getLogger("cctv_audit.probe")

# Ordered by preference: a playlist covers the whole recording in one URL.
# A segment URL usually covers a few seconds -- but not always, see
# _MIN_WHOLE_TRACK_SECONDS below.
_PLAYLIST_RE = re.compile(r"\.m3u8(\?|$)", re.I)
_STREAM_RE = re.compile(r"\.(flv|mpd)(\?|$)", re.I)
_SEGMENT_RE = re.compile(r"\.(ts|m4s)(\?|$)", re.I)

# How long a "segment" has to be before we treat it as the whole track rather
# than one slice of it. Real HLS segments are 2-10s; a DASH `.m4s` served by
# byte range is the entire recording. Both look identical in the URL.
_MIN_WHOLE_TRACK_SECONDS = 60.0

# ffprobe costs a subprocess and up to 15s each, and a busy player racks up
# dozens of segment URLs. Only the distinct tracks are worth checking.
_MAX_SEGMENT_PROBES = 6

_MEDIA_CONTENT_TYPES = (
    "application/vnd.apple.mpegurl",
    "application/x-mpegurl",
    "video/mp2t",
    "video/x-flv",
    "application/dash+xml",
)


class StreamProbe:
    """Collects candidate media URLs seen on a page, then validates them."""

    def __init__(self, page, capture_fps: int = 5):
        self._page = page
        self._playlists: list[str] = []
        self._streams: list[str] = []
        self._segments: list[str] = []
        self._attached = False

    def attach(self) -> None:
        """Starts listening. Call this *before* navigating to the player page."""
        if self._attached:
            return
        self._page.on("response", self._on_response)
        self._page.on("request", self._on_request)
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        try:
            self._page.remove_listener("response", self._on_response)
            self._page.remove_listener("request", self._on_request)
        except Exception as exc:  # listener may already be gone with the page
            logger.debug("Probe detach was a no-op: %s", exc)
        self._attached = False

    def _on_request(self, request) -> None:
        self._classify(request.url, "")

    def _on_response(self, response) -> None:
        try:
            content_type = (response.headers or {}).get("content-type", "")
        except Exception:
            content_type = ""
        self._classify(response.url, content_type)

    def _classify(self, url: str, content_type: str) -> None:
        if not url or url.startswith("data:") or url.startswith("blob:"):
            return
        ct = content_type.lower()
        if _PLAYLIST_RE.search(url) or "mpegurl" in ct:
            self._remember(self._playlists, url)
        elif _STREAM_RE.search(url) or any(t in ct for t in ("x-flv", "dash+xml")):
            self._remember(self._streams, url)
        elif _SEGMENT_RE.search(url) or "mp2t" in ct:
            self._remember(self._segments, url)

    @staticmethod
    def _remember(bucket: list[str], url: str) -> None:
        if url not in bucket:
            bucket.append(url)
            del bucket[:-20]  # only the most recent handful are useful

    async def decide(
        self,
        mode: str,
        wait_seconds: float = 12.0,
        poll_interval: float = 0.5,
    ) -> CaptureSource:
        """Waits for a grabbable stream to appear, then picks a capture mode.

        `mode` is the operator's CAPTURE_MODE: auto | stream | screen.
        """
        if mode == "screen":
            return CaptureSource(mode="screen", reason="CAPTURE_MODE=screen (forced)")

        deadline = asyncio.get_running_loop().time() + wait_seconds
        while asyncio.get_running_loop().time() < deadline:
            if self._playlists or self._streams:
                break
            await asyncio.sleep(poll_interval)

        base_headers = await self._build_headers()
        for url in self._playlists + self._streams:
            headers = self._headers_for(url, base_headers)
            info = await probe_stream(url, headers)
            if info:
                duration = _duration_of(info)
                return CaptureSource(
                    mode="stream",
                    url=url,
                    headers=headers,
                    duration_seconds=duration,
                    reason=(
                        f"ffprobe opened the player's own media URL"
                        + (f" (duration {duration:.0f}s)" if duration else " (live / unknown duration)")
                    ),
                )

        # No playlist. Before giving up on Plan A, check whether what looks
        # like a segment is really the whole track: bilibili, and DASH sites
        # generally, never fetch a manifest URL at all -- the manifest is
        # embedded in the page's JSON and the player byte-ranges a single
        # `.m4s` that ffprobe can open end to end. Assuming ".m4s means a few
        # seconds" cost us Plan A on every such site.
        whole = await self._best_whole_track(base_headers)
        if whole is not None:
            url, info = whole
            duration = _duration_of(info) or 0.0
            return CaptureSource(
                mode="stream",
                url=url,
                headers=self._headers_for(url, base_headers),
                duration_seconds=duration or None,
                reason=(f"ffprobe opened a complete media track ({duration:.0f}s, "
                        f"{_shape_of(info)}) despite there being no playlist"),
            )

        if mode == "stream":
            # The operator explicitly asked for Plan A; refusing loudly beats
            # silently recording the screen and pretending it was a stream grab.
            raise RuntimeError(
                "CAPTURE_MODE=stream but no ffprobe-readable media URL was found. "
                f"Saw {len(self._playlists)} playlist(s), {len(self._streams)} stream(s), "
                f"{len(self._segments)} segment(s). Use CAPTURE_MODE=auto to fall back "
                "to screen recording."
            )

        if self._segments and not self._playlists:
            reason = (
                "no playlist, and the media segments seen are genuine short "
                "slices rather than a whole track; falling back to screen recording"
            )
        elif self._playlists or self._streams:
            reason = "candidate media URLs found but ffprobe could not open them (proprietary or DRM)"
        else:
            reason = "no standard media transport observed (likely canvas + WASM decoding)"
        return CaptureSource(mode="screen", reason=reason)

    async def _best_whole_track(self, base_headers: dict) -> Optional[tuple[str, dict]]:
        """The highest-resolution segment URL that is actually a whole track.

        Returns `(url, ffprobe_info)`, or None if every candidate is a real
        short slice, unopenable, or audio-only (`probe_stream` drops those).
        Resolution is the tie-breaker rather than arrival order because a DASH
        player fetches several quality ladders and whichever lands first is
        arbitrary -- auditing 640x360 when 852x480 was available for the same
        price is a worse audit.
        """
        best: Optional[tuple[int, str, dict]] = None
        seen_paths: set[str] = set()
        for url in self._segments:
            # Mirrors serve the identical track from different hosts; the path
            # is what distinguishes one quality ladder from another.
            path = urlsplit(url).path
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if len(seen_paths) > _MAX_SEGMENT_PROBES:
                logger.info("Stopping after %d segment probes; %d candidates unchecked.",
                            _MAX_SEGMENT_PROBES, len(self._segments) - _MAX_SEGMENT_PROBES)
                break

            info = await probe_stream(url, self._headers_for(url, base_headers))
            if not info:
                continue
            duration = _duration_of(info) or 0.0
            if duration < _MIN_WHOLE_TRACK_SECONDS:
                logger.info("Segment is only %.1fs long, so it is a slice, not a track.", duration)
                continue
            video = _video_stream(info)
            area = int(video.get("width") or 0) * int(video.get("height") or 0)
            if best is None or area > best[0]:
                best = (area, url, info)

        return (best[1], best[2]) if best else None

    @staticmethod
    def _headers_for(url: str, base: dict) -> dict:
        """`base` plus a Google ID token, if this URL's origin needs one.

        Per URL and not once per page, because the page and its media are not
        always on the same host. Adding the token to the shared header block
        would hand this deployment's identity to whichever CDN the player
        happens to pull from -- and the audit works either way, so nothing
        would ever surface the mistake.
        """
        from ..gcp import id_token_for

        token = id_token_for(url)
        return {**base, "Authorization": f"Bearer {token}"} if token else base

    async def _build_headers(self) -> dict:
        """Media URLs are usually auth'd by cookie and gated on Referer."""
        headers: dict[str, str] = {}
        try:
            page_url = self._page.url or ""
            cookies = await self._page.context.cookies()
            host = urlsplit(page_url).hostname or ""
            relevant = [
                c for c in cookies
                if not host or host.endswith(str(c.get("domain", "")).lstrip("."))
            ]
            if relevant:
                headers["Cookie"] = "; ".join(f"{c['name']}={c['value']}" for c in relevant)
            if page_url:
                headers["Referer"] = page_url
                headers["Origin"] = f"{urlsplit(page_url).scheme}://{urlsplit(page_url).netloc}"
            user_agent = await self._page.evaluate("() => navigator.userAgent")
            if user_agent:
                headers["User-Agent"] = user_agent
        except Exception as exc:
            logger.warning("Could not build stream auth headers: %s", exc)
        return headers


def _video_stream(info: dict) -> dict:
    """`probe_stream` already rejected anything without one, so this always hits."""
    for stream in info.get("streams") or []:
        if stream.get("codec_type") == "video":
            return stream
    return {}


def _shape_of(info: dict) -> str:
    video = _video_stream(info)
    return f"{video.get('width', '?')}x{video.get('height', '?')} {video.get('codec_name', '?')}"


def _duration_of(info: dict) -> Optional[float]:
    raw = (info.get("format") or {}).get("duration")
    try:
        return float(raw) if raw else None
    except (TypeError, ValueError):
        return None
