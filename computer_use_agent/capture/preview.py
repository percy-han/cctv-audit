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

"""The live picture on the dashboard.

Deliberately separate from capture. It used to live inside `ScreenRecorder`,
which was fine only for as long as every run was a screen recording -- the day
Plan A started working, the dashboard went black, because under Plan A no
recorder is ever constructed. What the operator watches and what the analyzer
eats are two different concerns and now two different objects.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

logger = logging.getLogger("cctv_audit.preview")


class LivePreview:
    """A cheap screencast of the page, forwarded to the dashboard.

    Runs its own CDP session rather than sharing the recorder's. Chromium
    happily drives two screencasts on one page with independent parameters
    (measured: 1080p q85 and 360p q50 both delivered at 33 fps), and sharing
    one would force a choice between evidence quality and operator experience:
    a 1080p q85 frame is ~280 KB of base64, so forwarding every one of them is
    84 Mbps -- hopeless down an SSH tunnel. At 960x540 q60 a frame is ~58 KB,
    so 15 fps costs *less* than the old 4 fps did and looks four times smoother.
    """

    def __init__(
        self,
        page,
        on_frame: Callable[[str], None],
        fps: int = 15,
        width: int = 960,
        height: int = 540,
        quality: int = 60,
    ):
        self.page = page
        self.on_frame = on_frame
        self.fps = max(1, int(fps))
        self.width = max(160, int(width))
        self.height = max(90, int(height))
        self.quality = min(95, max(10, int(quality)))

        self._cdp = None
        self._latest_b64: Optional[str] = None
        self._pump: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        self._cdp = await self.page.context.new_cdp_session(self.page)

        async def on_frame(event):
            data = event.get("data")
            if data:
                self._latest_b64 = data
            try:
                await self._cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})
            except Exception as exc:
                logger.debug("preview ack failed (page likely closing): %s", exc)

        self._cdp.on("Page.screencastFrame", on_frame)
        await self._cdp.send("Page.startScreencast", {
            "format": "jpeg",
            "quality": self.quality,
            "maxWidth": self.width,
            "maxHeight": self.height,
            "everyNthFrame": 1,
        })
        self._pump = asyncio.create_task(self._run())
        logger.info("Dashboard preview: %dx%d q%d at %d fps",
                    self.width, self.height, self.quality, self.fps)

    async def _run(self) -> None:
        """Resamples to `fps` and drops rather than queues.

        Chromium emits on repaint, at whatever rate the page happens to run.
        Forwarding every frame would let a busy page saturate the tunnel, and
        the operator view must never be able to stall anything upstream --
        hence a fixed tick that takes the latest frame and lets the rest go.
        """
        interval = 1.0 / self.fps
        while not self._stopping.is_set():
            await asyncio.sleep(interval)
            b64 = self._latest_b64
            if b64:
                try:
                    self.on_frame(b64)
                except Exception as exc:
                    logger.debug("Preview frame dropped: %s", exc)

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
