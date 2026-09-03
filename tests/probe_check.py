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

"""Does this page support Plan A (stream grab), or only Plan B (screen record)?

Opens the page exactly as the pipeline does, watches what the player downloads,
then runs ffprobe against every candidate URL and prints the verdict. Costs
nothing at Vertex AI -- no video is analysed -- so it is the cheap way to try a
new platform before committing a real run to it.

    .venv/bin/python tests/probe_check.py <url>

`--all` additionally ffprobes every segment URL and prints each one's duration
and resolution, which is how you see *why* the verdict came out the way it did:
a 5s duration is a real HLS slice, a 500s one is the whole recording.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys


async def _probe(url: str, wait_seconds: float, all_candidates: bool) -> int:
    from playwright.async_api import async_playwright

    from computer_use_agent.capture.ffmpeg_util import probe_stream
    from computer_use_agent.capture.probe import StreamProbe
    from computer_use_agent.config import config
    from computer_use_agent.navigator import for_target, platform_of
    from computer_use_agent.navigator.base import load_state

    platform = platform_of(url)
    playwright = await async_playwright().start()
    browser = context = None
    try:
        browser = await playwright.chromium.launch(headless=config.headless)
        context = await browser.new_context(
            viewport={"width": config.screen_width, "height": config.screen_height},
            storage_state=load_state(platform),
        )
        page = await context.new_page()

        probe = StreamProbe(page)
        probe.attach()

        navigator = for_target(url)
        print(f"opening {url}  (platform adapter: {platform})")
        await navigator.login(page)
        await navigator.open_target(page, url)
        await navigator.ensure_playing(page)

        print(f"playing; watching the network for {wait_seconds:.0f}s ...\n")
        await asyncio.sleep(wait_seconds)

        headers = await probe._build_headers()
        buckets = [
            ("playlist (.m3u8)", probe._playlists, True),
            ("stream   (.flv/.mpd)", probe._streams, True),
            ("segment  (.ts/.m4s)", probe._segments, all_candidates),
        ]

        for label, urls, do_probe in buckets:
            print(f"{label}: {len(urls)} seen")
            for candidate in urls[:5]:
                shown = candidate.split("?")[0]
                if not do_probe:
                    print(f"   - {shown}   (not probed; pass --all)")
                    continue
                info = await probe_stream(candidate, headers)
                if not info:
                    print(f"   ✗ {shown}   ffprobe could not open it")
                    continue
                fmt = info.get("format") or {}
                duration = fmt.get("duration", "?")
                video = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
                shape = (f"{video[0].get('width')}x{video[0].get('height')} "
                         f"{video[0].get('codec_name')}") if video else "no video stream"
                print(f"   ✓ {shown}")
                print(f"       ffprobe OK: {fmt.get('format_name', '?')}, "
                      f"duration {duration}s, {shape}")
            print()

        # The real thing, so the answer matches what a run would actually do.
        source = await probe.decide("auto", wait_seconds=1.0)
        print("=" * 70)
        if source.mode == "stream":
            print(f"VERDICT: Plan A -- {source.reason}")
            print(f"         {(source.url or '').split('?')[0]}")
        else:
            print(f"VERDICT: Plan B -- {source.reason}")
        print("=" * 70)
        return 0
    finally:
        for closer in (
            lambda: context.close() if context else None,
            lambda: browser.close() if browser else None,
            playwright.stop,
        ):
            try:
                result = closer()
                if result is not None:
                    await result
            except Exception as exc:
                logging.getLogger("probe_check").debug("Shutdown step failed: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--wait", type=float, default=12.0,
                        help="seconds to watch the network before deciding")
    parser.add_argument("--all", action="store_true",
                        help="also ffprobe the segment bucket the pipeline ignores")
    args = parser.parse_args()

    os.environ.setdefault("CAPTURE_MODE", "auto")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-24s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return asyncio.run(_probe(args.url, args.wait, args.all))


if __name__ == "__main__":
    sys.exit(main())
