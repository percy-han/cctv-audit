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

"""Playwright-based implementation of BaseComputer for ADK Computer Use.

Follows the Gemini Enterprise Agent Platform Computer Use specifications:
https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/computer-use
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Literal, Optional, Union

from google.adk.features import FeatureName, experimental
from google.adk.tools.computer_use.base_computer import (
    BaseComputer,
    ComputerEnvironment,
    ComputerState,
)
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

logger = logging.getLogger("google_adk.computer_use.playwright")


@experimental(FeatureName.COMPUTER_USE)
class PlaywrightComputer(BaseComputer):
  """A local browser computer environment powered by Playwright.

  This environment executes browser automation actions requested by Gemini
  Computer Use (click_at, type_text_at, scroll_document, navigate, etc.)
  and returns the updated GUI screenshot and current URL as ComputerState.
  """

  def __init__(
      self,
      *,
      width: int = 1280,
      height: int = 800,
      headless: Optional[bool] = None,
      search_url: str = "https://www.google.com",
  ):
    """Initializes the Playwright computer environment.

    Args:
      width: Viewport width in pixels (default 1280).
      height: Viewport height in pixels (default 800).
      headless: Whether to run browser headlessly. Defaults to env
        var BROWSER_HEADLESS (or False if not set, enabling visual inspection).
      search_url: Default search URL when search() is called.
    """
    self.width = width
    self.height = height
    if headless is not None:
      self.headless = headless
    else:
      self.headless = os.getenv("BROWSER_HEADLESS", "true").lower() in (
          "true",
          "1",
          "yes",
      )

    self.search_url = search_url

    self._playwright: Optional[Playwright] = None
    self._browser: Optional[Browser] = None
    self._context: Optional[BrowserContext] = None
    self._page: Optional[Page] = None
    self._lock = asyncio.Lock()

  async def screen_size(self) -> tuple[int, int]:
    """Returns the screen size of the environment (width, height)."""
    return (self.width, self.height)

  async def environment(self) -> ComputerEnvironment:
    """Returns the environment type (ENVIRONMENT_BROWSER)."""
    return ComputerEnvironment.ENVIRONMENT_BROWSER

  async def _ensure_browser(self) -> Page:
    """Ensures that the browser, context, and page are initialized and active."""
    async with self._lock:
      if self._page is not None and not self._page.is_closed():
        return self._page

      logger.info(
          "Launching Playwright Chromium (headless=%s, %dx%d)...",
          self.headless,
          self.width,
          self.height,
      )
      self._playwright = await async_playwright().start()
      self._browser = await self._playwright.chromium.launch(
          headless=self.headless,
          args=[
              "--no-sandbox",
              "--disable-setuid-sandbox",
              "--disable-dev-shm-usage",
              "--disable-blink-features=AutomationControlled",
              "--mute-audio",
              "--js-flags=--max-old-space-size=512",
              "--disk-cache-size=104857600",
              "--disable-background-networking",
          ],
      )
      self._context = await self._browser.new_context(
          viewport={"width": self.width, "height": self.height},
          user_agent=(
              "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0.0.0 Safari/537.36"
          ),
      )
      self._page = await self._context.new_page()
      return self._page

  async def current_state(self) -> ComputerState:
    """Captures and returns the current state (screenshot and URL)."""
    page = await self._ensure_browser()
    await asyncio.sleep(0.3)
    screenshot_bytes = await page.screenshot(type="png", full_page=False)
    current_url = page.url or "about:blank"
    return ComputerState(screenshot=screenshot_bytes, url=current_url)

  async def open_web_browser(self) -> ComputerState:
    """Opens the web browser."""
    page = await self._ensure_browser()
    if page.url == "about:blank":
      try:
        await page.goto(self.search_url, wait_until="domcontentloaded", timeout=15000)
      except Exception as e:
        logger.warning("Failed to load initial search URL %s: %s", self.search_url, e)
    return await self.current_state()

  async def navigate(self, url: str) -> ComputerState:
    """Navigates directly to a specified URL."""
    page = await self._ensure_browser()
    target_url = url.strip()
    if not (target_url.startswith("http://") or target_url.startswith("https://")):
      target_url = "https://" + target_url

    logger.info("Navigating to: %s", target_url)
    try:
      await page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
      logger.warning("Navigation error/timeout for %s: %s", target_url, e)

    await asyncio.sleep(1.0)
    return await self.current_state()

  async def search(self) -> ComputerState:
    """Jumps directly to search engine."""
    return await self.navigate(self.search_url)

  async def click_at(self, x: int, y: int) -> ComputerState:
    """Clicks at coordinate (x, y).

    Coordinates are pre-scaled from 0-1000 grid to actual pixel coordinates
    by ADK's ComputerUseTool.
    """
    page = await self._ensure_browser()
    logger.info("Clicking at: (%d, %d)", x, y)
    await page.mouse.click(x, y)
    await asyncio.sleep(0.6)
    return await self.current_state()

  async def hover_at(self, x: int, y: int) -> ComputerState:
    """Hovers mouse at coordinate (x, y)."""
    page = await self._ensure_browser()
    logger.info("Hovering at: (%d, %d)", x, y)
    await page.mouse.move(x, y)
    await asyncio.sleep(0.4)
    return await self.current_state()

  async def type_text_at(
      self,
      x: int,
      y: int,
      text: str,
      press_enter: bool = True,
      clear_before_typing: bool = True,
  ) -> ComputerState:
    """Types text at coordinate (x, y)."""
    page = await self._ensure_browser()
    logger.info(
        "Typing text at (%d, %d): %r (press_enter=%s, clear=%s)",
        x,
        y,
        text,
        press_enter,
        clear_before_typing,
    )
    await page.mouse.click(x, y)
    await asyncio.sleep(0.2)

    if clear_before_typing:
      await page.keyboard.press("ControlOrMeta+A")
      await page.keyboard.press("Backspace")
      await asyncio.sleep(0.1)

    await page.keyboard.type(text)

    if press_enter:
      await page.keyboard.press("Enter")
      await asyncio.sleep(1.0)
    else:
      await asyncio.sleep(0.3)

    return await self.current_state()

  async def scroll_document(
      self, direction: Literal["up", "down", "left", "right"]
  ) -> ComputerState:
    """Scrolls the entire document."""
    page = await self._ensure_browser()
    scroll_amount = int(self.height * 0.7)
    delta_x, delta_y = 0, 0
    if direction == "down":
      delta_y = scroll_amount
    elif direction == "up":
      delta_y = -scroll_amount
    elif direction == "right":
      delta_x = scroll_amount
    elif direction == "left":
      delta_x = -scroll_amount

    logger.info("Scrolling document %s (delta_x=%d, delta_y=%d)", direction, delta_x, delta_y)
    await page.mouse.wheel(delta_x, delta_y)
    await asyncio.sleep(0.6)
    return await self.current_state()

  async def scroll_at(
      self,
      x: int,
      y: int,
      direction: Literal["up", "down", "left", "right"],
      magnitude: int,
  ) -> ComputerState:
    """Scrolls at a specific element/coordinate."""
    page = await self._ensure_browser()
    await page.mouse.move(x, y)
    delta_x, delta_y = 0, 0
    if direction == "down":
      delta_y = magnitude
    elif direction == "up":
      delta_y = -magnitude
    elif direction == "right":
      delta_x = magnitude
    elif direction == "left":
      delta_x = -magnitude

    logger.info(
        "Scrolling at (%d, %d) %s by %d pixels", x, y, direction, magnitude
    )
    await page.mouse.wheel(delta_x, delta_y)
    await asyncio.sleep(0.6)
    return await self.current_state()

  async def wait(self, seconds: int) -> ComputerState:
    """Waits for n seconds to allow async operations/animations to finish."""
    wait_time = max(1, min(seconds, 30))
    logger.info("Waiting for %d seconds...", wait_time)
    await asyncio.sleep(wait_time)
    return await self.current_state()

  async def go_back(self) -> ComputerState:
    """Navigates backward in history."""
    page = await self._ensure_browser()
    logger.info("Navigating back...")
    try:
      await page.go_back(timeout=10000)
    except Exception as e:
      logger.warning("Failed to go back: %s", e)
    await asyncio.sleep(0.5)
    return await self.current_state()

  async def go_forward(self) -> ComputerState:
    """Navigates forward in history."""
    page = await self._ensure_browser()
    logger.info("Navigating forward...")
    try:
      await page.go_forward(timeout=10000)
    except Exception as e:
      logger.warning("Failed to go forward: %s", e)
    await asyncio.sleep(0.5)
    return await self.current_state()

  async def key_combination(self, keys: Union[list[str], str]) -> ComputerState:
    """Presses key combination (e.g. 'Control+A', 'Enter', ['Control', 'C'])."""
    page = await self._ensure_browser()
    if isinstance(keys, list):
      combo = "+".join(keys)
    else:
      combo = str(keys)
    logger.info("Pressing key combination: %s", combo)
    await page.keyboard.press(combo)
    await asyncio.sleep(0.5)
    return await self.current_state()

  async def drag_and_drop(
      self, x: int, y: int, destination_x: int, destination_y: int
  ) -> ComputerState:
    """Drags an element from (x, y) to (destination_x, destination_y)."""
    page = await self._ensure_browser()
    logger.info("Drag and drop from (%d, %d) to (%d, %d)", x, y, destination_x, destination_y)
    await page.mouse.move(x, y)
    await page.mouse.down()
    await page.mouse.move(destination_x, destination_y, steps=10)
    await page.mouse.up()
    await asyncio.sleep(0.5)
    return await self.current_state()

  async def get_page_text(self) -> str:
    """Helper method: extracts visible text content from the current page."""
    page = await self._ensure_browser()
    try:
      text = await page.evaluate("() => document.body.innerText")
      return text[:5000] if text else "Page has no text."
    except Exception as e:
      return f"Error extracting page text: {e}"

  async def close(self) -> None:
    """Closes all browser resources."""
    async with self._lock:
      logger.info("Closing Playwright browser...")
      if self._page and not self._page.is_closed():
        await self._page.close()
      if self._context:
        await self._context.close()
      if self._browser:
        await self._browser.close()
      if self._playwright:
        await self._playwright.stop()
      self._page = None
      self._context = None
      self._browser = None
      self._playwright = None
