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

"""bilibili adapter -- the stand-in target until the customer's CCTV platform
is available.

It exists to exercise the full pipeline end to end against a real HLS/DASH
player. Everything platform-specific is confined to this file; the pipeline,
capture, and analyzer layers know nothing about bilibili.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .base import (
    HumanGate,
    NavigationError,
    TargetUnavailable,
    detect_challenge,
    save_state,
)

logger = logging.getLogger("cctv_audit.navigator.bilibili")

# bilibili answers a dead BV id with a 404, but it also serves a styled
# "video does not exist" page with HTTP 200 for withdrawn and region-locked
# videos. Both have to be recognised, or the run burns a Computer Use turn to
# be told what the page already says in plain text.
# "Go away" rather than "not here". A retry, a saved session, or a human can
# resolve these; a 404 cannot.
_BLOCKED_STATUSES = frozenset({401, 403, 412, 429})

_DEAD_PAGE_MARKERS = (
    "视频不存在", "该视频不存在", "啊叻？视频不见了", "视频已失效",
    "稿件不可见", "抱歉，你所访问的页面不存在", "该页面不存在",
)

# bilibili renders the feature video into a plain <video>, but the page also
# hosts preview players in the sidebar. Prefer the one inside the main player.
_VIDEO_SELECTOR = (
    "() => document.querySelector('#bilibili-player video')"
    " || document.querySelector('.bpx-player-video-wrap video')"
    " || document.querySelector('video')"
)


class BilibiliNavigator:
    name = "bilibili"

    def __init__(self, gate: Optional[HumanGate] = None):
        self.gate = gate

    async def login(self, page) -> None:
        """No-op by design.

        Most bilibili videos play without an account, and a saved storage_state
        covers the rest. We do not automate the slider -- see the login strategy
        in the plan. If a challenge is on screen, hand over to a human.
        """
        reason = await detect_challenge(page)
        if reason and self.gate is not None:
            await self.gate.wait_for_human(page, f"bilibili login: {reason}")
        elif reason:
            raise NavigationError(f"出现登录验证（{reason}），但没有配置人工介入通道。")

    async def open_target(self, page, target: str) -> None:
        if not target.startswith("http"):
            target = f"https://www.bilibili.com/video/{target}"
        logger.info("Opening %s", target)
        response = await page.goto(target, timeout=60000, wait_until="domcontentloaded")
        await self._reject_dead_page(page, target, response)

        try:
            await page.wait_for_selector("video", timeout=30000, state="attached")
        except Exception as exc:
            reason = await detect_challenge(page)
            if reason and self.gate is not None:
                await self.gate.wait_for_human(page, f"bilibili playback: {reason}")
                await page.wait_for_selector("video", timeout=30000, state="attached")
            else:
                # Re-check after the wait: a soft-404 shell can finish
                # rendering its error text well after DOMContentLoaded.
                await self._reject_dead_page(page, target, response)
                raise NavigationError(
                    f"页面已打开但 30 秒内没有出现视频播放器：{target}"
                ) from exc

        await self._dismiss_overlays(page)
        await self._disable_danmaku(page)

    async def _reject_dead_page(self, page, target: str, response) -> None:
        """Fails fast and in plain language when the link goes nowhere."""
        status = getattr(response, "status", None) if response else None
        if status and status >= 400:
            if status in _BLOCKED_STATUSES:
                # Not the same as "gone": bilibili answers a crawler-looking
                # request with 412 even for a perfectly good video. Calling
                # that a dead link would send someone hunting for a URL bug
                # that is not there, so this stays rescuable.
                raise NavigationError(
                    f"bilibili 拒绝了这次请求（HTTP {status}），通常是反爬或频率限制，"
                    f"不代表视频不存在：{target}"
                )
            raise TargetUnavailable(f"这个链接打不开（HTTP {status}）：{target}")

        try:
            title = (await page.title()) or ""
            body = await page.evaluate("() => document.body ? document.body.innerText.slice(0, 2000) : ''")
        except Exception:
            return
        haystack = f"{title}\n{body}"
        marker = next((m for m in _DEAD_PAGE_MARKERS if m in haystack), None)
        if marker:
            raise TargetUnavailable(f"页面提示「{marker}」，视频不可观看：{target}")

    async def ensure_playing(self, page) -> None:
        """Starts playback and confirms the playhead actually advances.

        Confirming matters: a paused player still renders a frame, and every
        window would then be the same still image with no error anywhere.
        """
        # Muting is what makes autoplay permissible under Chromium's policy.
        # Audio is irrelevant to visual SOP checks anyway.
        await page.evaluate(
            f"""async () => {{
                const v = ({_VIDEO_SELECTOR})();
                if (!v) return;
                v.muted = true;
                try {{ await v.play(); }} catch (e) {{ /* fall through to a click */ }}
            }}"""
        )
        if await self._is_advancing(page):
            return

        # Autoplay was refused; a real user gesture usually clears it.
        for attempt in ("click", "space"):
            try:
                if attempt == "click":
                    await page.locator("video").first.click(timeout=5000, force=True)
                else:
                    await page.keyboard.press("Space")
            except Exception as exc:
                logger.debug("Play gesture '%s' failed: %s", attempt, exc)
            if await self._is_advancing(page):
                return

        raise NavigationError(
            "播放始终没有推进。播放器可能停在广告、登录墙或地区限制上。"
        )

    async def seek_to(self, page, seconds: float) -> None:
        ok = await page.evaluate(
            f"""(t) => {{
                const v = ({_VIDEO_SELECTOR})();
                if (!v) return false;
                v.currentTime = t;
                return true;
            }}""",
            float(seconds),
        )
        if not ok:
            raise NavigationError("无法跳转：页面上找不到视频播放器。")
        # Give the player time to buffer, or the first window records a spinner.
        await asyncio.sleep(1.5)

    async def read_player_time(self, page) -> Optional[float]:
        try:
            value = await page.evaluate(
                f"() => {{ const v = ({_VIDEO_SELECTOR})(); return v ? v.currentTime : null; }}"
            )
            return float(value) if value is not None else None
        except Exception as exc:
            logger.debug("Could not read the playhead: %s", exc)
            return None

    async def read_playback_state(self, page) -> Optional[dict]:
        """currentTime / duration / ended / paused, read in one round trip.

        They have to arrive together: polling them separately can straddle a
        state change and make a finished recording look merely stalled, which
        is the difference between stopping cleanly and encoding a frozen frame
        until the wall-clock budget runs out.
        """
        try:
            return await page.evaluate(
                f"""() => {{
                    const v = ({_VIDEO_SELECTOR})();
                    if (!v) return null;
                    return {{
                        current_time: v.currentTime,
                        duration: isFinite(v.duration) && v.duration > 0 ? v.duration : null,
                        ended: !!v.ended,
                        paused: !!v.paused,
                    }};
                }}"""
            )
        except Exception as exc:
            logger.debug("Could not read playback state: %s", exc)
            return None

    async def video_rect(self, page) -> Optional[dict]:
        """The player's box in viewport pixels, for cropping the recording.

        A full-page 1920x1080 capture of a video site is mostly header,
        sidebar and comments. Those pixels are re-encoded, uploaded, and then
        downsampled by the model along with the footage -- which is how a
        gloved hand and a bare one end up looking the same. Cropping to the
        player spends the whole frame budget on the only region the SOP rules
        are about.
        """
        try:
            return await page.evaluate(
                f"""() => {{
                    const v = ({_VIDEO_SELECTOR})();
                    if (!v) return null;
                    const r = v.getBoundingClientRect();
                    return {{x: r.x, y: r.y, width: r.width, height: r.height,
                             // Intrinsic size: the element box includes the
                             // letterbox/pillarbox bars, the footage does not.
                             intrinsic_width: v.videoWidth, intrinsic_height: v.videoHeight,
                             view_width: window.innerWidth, view_height: window.innerHeight}};
                }}"""
            )
        except Exception as exc:
            logger.debug("Could not measure the player: %s", exc)
            return None

    # bilibili's own "网页全屏" control. Web fullscreen is CSS-driven, so it
    # works headless; the real Fullscreen API needs a user activation Chromium
    # will not always grant.
    _FULLSCREEN_BUTTONS = (
        ".bpx-player-ctrl-web",
        ".bpx-player-ctrl-btn.bpx-player-ctrl-web",
        ".squirtle-video-pagefullscreen",
    )

    async def enter_fullscreen(self, page) -> bool:
        """Makes the player fill the viewport. Returns whether it worked.

        Worth doing before recording: a windowed player occupies well under
        half of a 1920x1080 page, and every pixel outside it is re-encoded,
        uploaded and downsampled along with the footage. Filling the viewport
        spends the whole frame budget on the only region the SOP is about --
        the same goal as cropping, but the *player* renders at full size too,
        so the detail is genuinely there rather than merely un-cropped.
        """
        before = await self.video_rect(page)
        for selector in self._FULLSCREEN_BUTTONS:
            try:
                button = page.locator(selector).first
                if not await button.count():
                    continue
                await button.click(timeout=3000, force=True)
                await asyncio.sleep(1.0)
                if await self._fills_viewport(page):
                    logger.info("Player is in web fullscreen (%s).", selector)
                    return True
            except Exception as exc:
                logger.debug("Fullscreen via %s failed: %s", selector, exc)

        # Keyboard shortcut, then the standard API as a last resort. The click
        # attempts above count as user activation for requestFullscreen.
        for attempt in ("keyboard", "api"):
            try:
                if attempt == "keyboard":
                    await page.keyboard.press("w")  # bilibili: toggle web fullscreen
                else:
                    await page.evaluate(
                        f"""async () => {{
                            const v = ({_VIDEO_SELECTOR})();
                            const host = v && (v.closest('#bilibili-player') || v.parentElement);
                            if (host && host.requestFullscreen) await host.requestFullscreen();
                        }}"""
                    )
                await asyncio.sleep(1.0)
                if await self._fills_viewport(page):
                    logger.info("Player is fullscreen (via %s).", attempt)
                    return True
            except Exception as exc:
                logger.debug("Fullscreen via %s failed: %s", attempt, exc)

        after = await self.video_rect(page)
        logger.info(
            "Could not enter fullscreen; the player stays at %sx%s (was %sx%s). "
            "Capture will be cropped to it instead.",
            (after or {}).get("width"), (after or {}).get("height"),
            (before or {}).get("width"), (before or {}).get("height"),
        )
        return False

    async def _fills_viewport(self, page, threshold: float = 0.9) -> bool:
        rect = await self.video_rect(page)
        if not rect or not rect.get("view_width") or not rect.get("view_height"):
            return False
        covered = (rect["width"] * rect["height"]) / (rect["view_width"] * rect["view_height"])
        return covered >= threshold

    async def keep_clear(self, page) -> None:
        """Re-runs the overlay and danmaku cleanup mid-playback.

        bilibili re-raises its login modal every couple of minutes, and the
        recorder has no idea: it keeps capturing a dialog box sitting on top of
        the footage. Dismissing once at open_target is not enough, so the
        pipeline calls this on a timer. Real CCTV platforms do the same thing
        with session-expiry prompts.

        Closing the modal is only half of it, because bilibili *pauses* the
        video when it raises one. Measured (job 692b53): the nag appeared about
        70 seconds in, the player stopped, and Chromium -- which only emits a
        screencast frame on repaint -- delivered 0 fps for the remaining two
        minutes while the dashboard happily forwarded the same still. So
        whatever we just closed, put the player back in motion before leaving.
        Conditional on having actually closed something: calling `play()` on
        every poll would override a human working the gate.
        """
        closed = await self._dismiss_overlays(page)
        await self._disable_danmaku(page)
        if not closed:
            return
        try:
            await self.ensure_playing(page)
        except Exception as exc:
            # The caller polls; there will be another chance in a few seconds.
            # Raising here would skip the rest of its housekeeping instead.
            logger.warning("Dismissed %d overlay(s) but could not resume "
                           "playback: %s", closed, exc)

    async def read_duration(self, page) -> Optional[float]:
        try:
            value = await page.evaluate(
                f"() => {{ const v = ({_VIDEO_SELECTOR})();"
                f" return v && isFinite(v.duration) ? v.duration : null; }}"
            )
            return float(value) if value is not None else None
        except Exception:
            return None

    async def set_playback_rate(self, page, rate: float) -> None:
        applied = await page.evaluate(
            f"""(r) => {{
                const v = ({_VIDEO_SELECTOR})();
                if (!v) return null;
                v.playbackRate = r;
                return v.playbackRate;
            }}""",
            float(rate),
        )
        if applied is None:
            raise NavigationError("无法设置倍速：页面上找不到视频播放器。")
        if abs(applied - rate) > 0.01:
            # Players clamp this. The caller must know, because the recorder's
            # time_scale is derived from the rate and would misdate every clip.
            raise NavigationError(
                f"Player clamped the playback rate to {applied}x (asked for {rate}x)."
            )
        logger.info("Playback rate set to %.2fx", applied)

    # -- internals ---------------------------------------------------------

    async def _is_advancing(self, page, samples: int = 3, interval: float = 0.7) -> bool:
        first = await self.read_player_time(page)
        if first is None:
            return False
        for _ in range(samples):
            await asyncio.sleep(interval)
            current = await self.read_player_time(page)
            if current is not None and current > first + 0.05:
                return True
        return False

    # Scoped deliberately. This runs on a timer during capture, so a selector
    # broad enough to match a player control (the old `[class*='close-btn']`
    # did) would click something in the footage every few seconds.
    _OVERLAY_CLOSERS = (
        ".bili-mini-mask .bili-mini-close-icon",
        ".bili-mini-close-icon",
        ".login-scan-box .close",
        ".lt-row .btn-close",
        ".bili-header-channel-menu__close",
        ".bpx-player-toast-close",
    )
    # Some nags have no close button at all; hiding them is the only way out.
    _OVERLAY_HIDE = (
        ".bili-mini-mask",
        ".lt-login-alert",
        ".login-panel-popover",
        ".bili-header__banner",
    )

    async def _dismiss_overlays(self, page) -> int:
        """Closes the login nag and cookie banners that cover the player.

        Returns how many things it actually got rid of, and says so at INFO
        rather than DEBUG. That is not tidying: DEBUG does not reach Cloud
        Logging, so on the run where the picture froze (job 692b53) there was
        no way to tell "we never saw the modal" from "we closed it and the
        player stayed paused anyway" -- two different bugs with two different
        fixes. The count is the difference.
        """
        closed = 0
        for selector in self._OVERLAY_CLOSERS:
            try:
                locator = page.locator(selector).first
                if await locator.count() and await locator.is_visible(timeout=500):
                    await locator.click(timeout=2000)
                    closed += 1
                    logger.info("Dismissed overlay %s", selector)
            except Exception:
                continue
        try:
            hidden = await page.evaluate(
                """(selectors) => {
                    let n = 0;
                    for (const sel of selectors) {
                        for (const el of document.querySelectorAll(sel)) {
                            if (el.style.display !== 'none') { el.style.display = 'none'; n++; }
                        }
                    }
                    return n;
                }""",
                list(self._OVERLAY_HIDE),
            )
            if hidden:
                closed += hidden
                logger.info("Hid %d stubborn overlay element(s)", hidden)
        except Exception as exc:
            logger.debug("Could not hide overlays: %s", exc)
        return closed

    async def _disable_danmaku(self, page) -> None:
        """Turns off bullet comments.

        Scrolling text across the frame is not part of the scene, and the model
        will happily describe it as if it were. Worse, it occludes exactly the
        hands-and-counter region the SOP rules care about.
        """
        try:
            await page.evaluate(
                """() => {
                    document.querySelectorAll(
                        '.bpx-player-row-dm-wrap, .bili-danmaku-x, .bpx-player-dm-wrap, #danmukuBox'
                    ).forEach(el => { el.style.display = 'none'; });
                    const sw = document.querySelector('.bpx-player-dm-switch input');
                    if (sw && sw.checked) sw.click();
                }"""
            )
        except Exception as exc:
            logger.debug("Could not disable danmaku: %s", exc)

    async def persist_session(self, context) -> None:
        await save_state(context, self.name)
