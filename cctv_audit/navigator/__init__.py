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

"""Platform adapters: the deterministic "get to the video" layer.

Adding Hikvision or Dahua means adding one module here and one line in
`for_target`. Nothing else in the pipeline changes.
"""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlsplit

from .base import (
    HumanGate,
    HumanInterventionRequired,
    NavigationError,
    Navigator,
    detect_challenge,
    load_state,
    save_state,
    state_path,
)
from .bilibili import BilibiliNavigator
from .generic_agent import ComputerUseFallback
from .resilient import ResilientNavigator

__all__ = [
    "Navigator", "NavigationError", "HumanInterventionRequired", "HumanGate",
    "BilibiliNavigator", "ComputerUseFallback", "ResilientNavigator",
    "detect_challenge", "load_state", "save_state", "state_path",
    "for_target", "platform_of",
]

logger = logging.getLogger("cctv_audit.navigator")

_BY_HOST = {
    "bilibili.com": BilibiliNavigator,
    "b23.tv": BilibiliNavigator,
}


def platform_of(target: str) -> str:
    """Platform key for a target URL, used to name the saved session file."""
    host = (urlsplit(target).hostname or "").lower()
    for suffix, adapter in _BY_HOST.items():
        if host == suffix or host.endswith("." + suffix):
            return adapter.name
    return "generic"


def for_target(
    target: str,
    gate: Optional[HumanGate] = None,
    fallback: Optional[ComputerUseFallback] = None,
) -> ResilientNavigator:
    """Picks the adapter for `target` and wraps it in the Computer Use safety net."""
    host = (urlsplit(target).hostname or "").lower()
    for suffix, adapter_cls in _BY_HOST.items():
        if host == suffix or host.endswith("." + suffix):
            logger.info("Using the %s adapter for %s", adapter_cls.name, host)
            return ResilientNavigator(adapter_cls(gate=gate), fallback=fallback, gate=gate)

    # No adapter yet -- which is the expected state for the customer's CCTV
    # platform until we can see it. The generic adapter handles any page whose
    # player is a standard <video>; anything stranger falls through to Computer
    # Use on every step.
    logger.warning("No adapter for '%s'; using the generic <video> adapter.", host or target)
    return ResilientNavigator(GenericVideoNavigator(gate=gate), fallback=fallback, gate=gate)


class GenericVideoNavigator(BilibiliNavigator):
    """Adapter of last resort for an unknown platform.

    Inherits the plain-<video> handling and drops the bilibili-specific overlay
    and danmaku cleanup, which would be meaningless elsewhere.
    """

    name = "generic"

    async def _dismiss_overlays(self, page) -> None:
        return

    async def _disable_danmaku(self, page) -> None:
        return
