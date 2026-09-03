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

"""The single implementation of "Computer Use action -> Playwright call".

This used to exist twice (agent_loop.py and computer.py) with subtly different
behaviour. Everything that dispatches model-emitted browser actions goes
through `execute_action` / `execute_function_calls` here.

Notably absent: `record_video_segment`. Audit findings no longer come from the
navigation model -- they come from the analyzer, which sees actual video rather
than a single screenshot.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger("cctv_audit.actions")

# Computer Use emits coordinates normalised to a 0-1000 box, independent of the
# real viewport size.
COORD_SCALE = 1000


def normalize_x(x: int, screen_width: int) -> int:
    return int(int(x) / COORD_SCALE * screen_width)


def normalize_y(y: int, screen_height: int) -> int:
    return int(int(y) / COORD_SCALE * screen_height)


async def execute_action(
    name: str,
    args: dict,
    page,
    screen_width: int,
    screen_height: int,
) -> str:
    """Runs one model-requested browser action. Returns a short status string."""
    args = args or {}

    def px(key: str, default: Optional[int] = None) -> int:
        value = args.get(key, default)
        axis_w = key.endswith("x")
        scale = screen_width if axis_w else screen_height
        return int(int(value) / COORD_SCALE * scale)

    if name == "open_web_browser":
        return "success"

    if name in ("navigate", "goto"):
        await page.goto(args["url"], timeout=60000, wait_until="domcontentloaded")
        return "success"

    if name in ("click_at", "click"):
        await page.mouse.click(px("x"), px("y"))
        return "success"

    if name in ("move_cursor", "move", "hover_at", "hover"):
        await page.mouse.move(px("x"), px("y"))
        return "success"

    if name == "mouse_down":
        await page.mouse.down(button=args.get("button", "left"))
        return "success"

    if name == "mouse_up":
        await page.mouse.up(button=args.get("button", "left"))
        return "success"

    if name in ("context_click", "right_click"):
        await page.mouse.click(px("x"), px("y"), button="right")
        return "success"

    if name in ("double_click_at", "double_click"):
        await page.mouse.dblclick(px("x"), px("y"))
        return "success"

    if name in ("triple_click_at", "triple_click"):
        await page.mouse.click(px("x"), px("y"), click_count=3)
        return "success"

    if name in ("middle_click_at", "middle_click"):
        await page.mouse.click(px("x"), px("y"), button="middle")
        return "success"

    if name in ("type_text_at", "type_text", "type"):
        if "x" in args and "y" in args:
            await page.mouse.click(px("x"), px("y"))
            await asyncio.sleep(0.1)
        if args.get("clear_before_typing", True):
            await page.keyboard.press("ControlOrMeta+A")
            await page.keyboard.press("Backspace")
        await page.keyboard.type(args.get("text", ""))
        if args.get("press_enter", False):
            await page.keyboard.press("Enter")
        return "success"

    if name in ("drag_and_drop", "drag"):
        # Traced as a series of small steps: instantaneous jumps are ignored by
        # some drag implementations, which listen for intermediate mousemove.
        start = (px("start_x"), px("start_y"))
        end = (px("end_x"), px("end_y"))
        await page.mouse.move(*start)
        await page.mouse.down()
        steps = 20
        for i in range(1, steps + 1):
            await page.mouse.move(
                start[0] + (end[0] - start[0]) * i / steps,
                start[1] + (end[1] - start[1]) * i / steps,
            )
            await asyncio.sleep(0.01)
        await page.mouse.up()
        return "success"

    if name in ("scroll_at", "scroll"):
        delta_x = args.get("delta_x", 0)
        delta_y = args.get("delta_y", 0)
        direction = args.get("direction", "")
        if delta_y == 0 and direction in ("down", "up"):
            delta_y = 500 if direction == "down" else -500
        if delta_x == 0 and direction in ("left", "right"):
            delta_x = 500 if direction == "right" else -500
        await page.mouse.move(px("x", 500), px("y", 500))
        await page.mouse.wheel(delta_x, delta_y)
        return "success"

    if name == "scroll_document":
        direction = args.get("direction", "down")
        expr = {
            "down": "window.scrollBy(0, window.innerHeight * 0.8)",
            "up": "window.scrollBy(0, -window.innerHeight * 0.8)",
            "left": "window.scrollBy(-window.innerWidth * 0.8, 0)",
            "right": "window.scrollBy(window.innerWidth * 0.8, 0)",
        }.get(direction)
        if not expr:
            return f"error: unknown scroll direction '{direction}'"
        await page.evaluate(expr)
        return "success"

    if name in ("wait", "sleep"):
        await asyncio.sleep(min(float(args.get("seconds", 2)), 60))
        return "success"

    if name in ("key_combination", "press_key", "key"):
        await page.keyboard.press(args.get("keys") or args.get("key", "Space"))
        return "success"

    if name == "key_down":
        await page.keyboard.down(args["key"])
        return "success"

    if name == "key_up":
        await page.keyboard.up(args["key"])
        return "success"

    if name in ("go_back", "back"):
        await page.go_back()
        return "success"

    if name in ("go_forward", "forward"):
        await page.go_forward()
        return "success"

    logger.warning("Unrecognised Computer Use action: %s", name)
    return "unknown_function"


async def execute_function_calls(
    response,
    page,
    screen_width: int,
    screen_height: int,
    on_action: Optional[Callable[[str, dict, Any, Any, str], None]] = None,
) -> Tuple[str, List[Tuple[str, Any]], str]:
    """Executes every function call in a Computer Use response.

    Returns `(status, [(name, result), ...], reasoning_text)` where status is
    "SUCCESS" or "NO_ACTION". `on_action` is the hook the live dashboard uses;
    passing it keeps this module free of any monitor import.
    """
    candidate = response.candidates[0]
    function_calls = []
    thoughts = []
    for part in (candidate.content.parts or []):
        if getattr(part, "function_call", None):
            function_calls.append(part.function_call)
        elif getattr(part, "text", None):
            thoughts.append(part.text)

    reasoning = " ".join(thoughts).strip()
    if reasoning:
        logger.info("Model reasoning: %s", reasoning[:500])

    if not function_calls:
        return "NO_ACTION", [], reasoning

    results: List[Tuple[str, Any]] = []
    for call in function_calls:
        name = call.name
        args = dict(call.args or {})
        logger.info("Action %s %s", name, args)
        try:
            result = await execute_action(name, args, page, screen_width, screen_height)
        except Exception as exc:
            logger.warning("Action %s failed: %s", name, exc)
            result = f"error: {exc}"

        if on_action is not None:
            try:
                on_action(name, args, args.get("x"), args.get("y"), page.url)
            except Exception as exc:
                logger.debug("Action hook raised (ignored): %s", exc)

        results.append((name, result))

    return "SUCCESS", results, reasoning
