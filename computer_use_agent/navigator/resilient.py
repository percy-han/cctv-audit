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

"""Deterministic first, agent second.

Wraps a platform adapter so that any step which fails is retried by the
Computer Use fallback with a plain-language goal. The fast, cheap, reproducible
path runs every time; the expensive one only pays for itself when a selector
has actually broken.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import (
    HumanGate,
    NavigationError,
    Navigator,
    TargetUnavailable,
    detect_challenge,
    diagnose_block,
)
from .generic_agent import ComputerUseFallback

logger = logging.getLogger("cctv_audit.navigator.resilient")


class ResilientNavigator:
    """A Navigator with a Computer Use safety net."""

    def __init__(
        self,
        inner: Navigator,
        fallback: Optional[ComputerUseFallback] = None,
        gate: Optional[HumanGate] = None,
    ):
        self.inner = inner
        self.fallback = fallback
        self.gate = gate
        self.name = getattr(inner, "name", "unknown")
        self.used_fallback = False

    async def login(self, page) -> None:
        await self.inner.login(page)

    async def open_target(self, page, target: str) -> None:
        try:
            await self.inner.open_target(page, target)
        except NavigationError as exc:
            await self._rescue(
                page, exc,
                goal=f"Open the video page at {target} and make the video player visible.",
                # An objective check, because there is one: either the page has
                # a player we can read or it does not. Without it, success was
                # decided by whether the model's prose began with the word
                # "BLOCKED" -- so "I could not open it" counted as opened.
                success_check=lambda: self._player_present(page),
            )

    async def ensure_playing(self, page) -> None:
        try:
            await self.inner.ensure_playing(page)
        except NavigationError as exc:
            await self._rescue(
                page, exc,
                goal=(
                    "Start video playback on this page. Dismiss any modal, advertisement, "
                    "or autoplay prompt that blocks it. Success means the playhead is moving."
                ),
                success_check=lambda: self._playhead_advancing(page),
            )

    async def seek_to(self, page, seconds: float) -> None:
        try:
            await self.inner.seek_to(page, seconds)
        except NavigationError as exc:
            await self._rescue(
                page, exc,
                goal=(
                    f"Move the video playhead to {int(seconds // 60)}m{int(seconds % 60)}s "
                    "by dragging the progress bar or using the player's time input."
                ),
                # Dragging a progress bar lands where it lands. Asking the
                # player where the playhead actually is costs one round trip
                # and is the difference between auditing the requested minute
                # and auditing whichever minute the drag happened to hit.
                success_check=lambda: self._playhead_near(page, seconds),
            )

    async def read_player_time(self, page):
        # No fallback: an agent guessing at the progress bar is exactly the
        # unreliable timestamp source this design set out to remove. None is a
        # legitimate answer, and callers handle it.
        return await self.inner.read_player_time(page)

    # The three below are observations and housekeeping, not navigation steps.
    # None of them gets a Computer Use rescue: a failure means "unknown", which
    # every caller already handles, and paying for an agent turn on a five
    # second timer would dwarf the cost of the audit itself.
    async def read_playback_state(self, page):
        reader = getattr(self.inner, "read_playback_state", None)
        return await reader(page) if reader else None

    async def video_rect(self, page):
        reader = getattr(self.inner, "video_rect", None)
        return await reader(page) if reader else None

    async def keep_clear(self, page) -> None:
        cleaner = getattr(self.inner, "keep_clear", None)
        if cleaner:
            await cleaner(page)

    async def enter_fullscreen(self, page) -> bool:
        maximiser = getattr(self.inner, "enter_fullscreen", None)
        return bool(await maximiser(page)) if maximiser else False

    async def set_playback_rate(self, page, rate: float) -> None:
        try:
            await self.inner.set_playback_rate(page, rate)
        except NavigationError as exc:
            # Do not rescue: if the rate is wrong, every clip's timestamps are
            # wrong, and a silently-mismatched rate is worse than 1x.
            raise NavigationError(
                f"无法把播放速度设为 {rate}x（{exc}）。请设置 PLAYBACK_RATE=1，"
                "或换一个支持倍速的播放器。"
            ) from exc

    async def _player_present(self, page) -> bool:
        """A readable playhead is the cheapest proof that a player exists."""
        return await self.inner.read_player_time(page) is not None

    async def _playhead_near(self, page, target: float, tolerance: float = 5.0) -> bool:
        """Did the seek actually land where it was asked to?

        The tolerance is generous because a player snaps to the nearest
        keyframe, but it is far tighter than "the model said it worked".
        """
        where = await self.inner.read_player_time(page)
        return where is not None and abs(where - target) <= tolerance

    async def _playhead_advancing(self, page) -> bool:
        import asyncio
        first = await self.inner.read_player_time(page)
        if first is None:
            return False
        await asyncio.sleep(1.0)
        second = await self.inner.read_player_time(page)
        return second is not None and second > first + 0.05

    async def _route(self, page, exc: Exception, blocker: dict) -> bool:
        """Sends a diagnosed block where it belongs. True if it was handled.

        Each verdict has one correct destination, and they are not
        interchangeable: paging a human for an expired recording wastes
        somebody's evening, and letting Computer Use grind against a CAPTCHA
        burns its whole turn budget to arrive at the same place.
        """
        kind = blocker["blocker"]
        what = blocker["description"] or kind
        if kind == "unavailable":
            # No amount of clicking brings back a recording past its retention.
            raise TargetUnavailable(f"{what}（页面识别）") from exc
        if kind in ("challenge", "login") and self.gate is not None:
            logger.info("Page reads as a %s: %s", kind, what[:200])
            await self.gate.wait_for_human(page, f"{self.name}: {what}")
            return True
        logger.info("Page diagnosis: %s -- %s", kind, what[:200])
        return False

    async def _rescue(self, page, exc: Exception, goal: str, success_check=None) -> None:
        # An agent cannot navigate its way out of a link that does not exist.
        # Letting it try costs a Computer Use turn and buys a worse-worded
        # version of the message we already have.
        if isinstance(exc, TargetUnavailable):
            raise exc

        challenge = await detect_challenge(page)
        if challenge:
            # Sites raise a login nag on top of the player mid-playback, which
            # is both why the step failed and what the detector is looking at.
            # Clearing it costs one call and resolves the common case without
            # anyone being paged; only what survives cleanup is a real block.
            try:
                await self.keep_clear(page)
                challenge = await detect_challenge(page)
            except Exception as exc:
                logger.debug("Could not clear overlays before gating: %s", exc)
        if challenge and self.gate is not None:
            await self.gate.wait_for_human(page, f"{self.name}: {challenge}")
            return

        # The keyword lists only know bilibili's exact wording. On a platform
        # they have never seen -- which is every platform the customer actually
        # runs -- they return None and the run dies with a generic message
        # while the page says in plain text what is wrong. Read it.
        if challenge is None:
            blocker = await diagnose_block(page)
            if blocker and blocker["blocker"] != "none":
                if await self._route(page, exc, blocker):
                    return

        if self.fallback is None:
            raise exc

        logger.warning("Deterministic step failed (%s); handing over to Computer Use.", exc)
        self.used_fallback = True
        succeeded, detail = await self.fallback.run(page, goal, success_check=success_check)
        if not succeeded:
            raise NavigationError(
                f"{exc}\n兜底的 Computer Use 也没能恢复：{detail[:300]}"
            )
        logger.info("Computer Use fallback recovered: %s", detail[:200])
