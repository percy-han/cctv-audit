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

"""Central configuration for the CCTV audit agent.

Importing this module is the single supported way to read settings: it loads
`.env` (via python-dotenv), pins the SSL bundle, and exposes one frozen
`Config` instance. Nothing else in the package should call `os.getenv`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import certifi
from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

# Some Python builds (notably macOS 3.13) do not pick up the system trust store,
# which breaks aiohttp and the Google API clients.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

load_dotenv(PACKAGE_DIR / ".env")


def _env_str(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _env_int(key: str, default: int) -> int:
    raw = _env_str(key)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = _env_str(key)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env_str(key).lower()
    if not raw:
        return default
    return raw in ("true", "1", "yes", "on")


def _project_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_DIR / path)


def _env_opt_int(key: str) -> Optional[int]:
    raw = _env_str(key)
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


@dataclass(frozen=True)
class Config:
    # ---- Vertex AI ----------------------------------------------------
    # NOTE: location is pinned to "global" by product decision. Computer Use is
    # a preview feature and the codebase has previously hit
    # "computer use is not supported for this model in this region" on regional
    # endpoints. "global" means Google routes by capacity with no geographic
    # guarantee -- see the Region section of the plan before promising data
    # residency to a customer.
    gcp_project: str = field(default_factory=lambda: _env_str("GOOGLE_CLOUD_PROJECT") or _env_str("GCP_PROJECT"))
    gcp_location: str = "global"
    nav_model: str = field(default_factory=lambda: _env_str("ADK_MODEL", "gemini-3.5-flash"))
    analysis_model: str = field(default_factory=lambda: _env_str("ANALYSIS_MODEL", "gemini-3.5-flash"))

    # ---- Browser ------------------------------------------------------
    headless: bool = field(default_factory=lambda: _env_bool("BROWSER_HEADLESS", True))
    screen_width: int = field(default_factory=lambda: _env_int("SCREEN_WIDTH", 1920))
    screen_height: int = field(default_factory=lambda: _env_int("SCREEN_HEIGHT", 1080))
    allow_private_network: bool = field(default_factory=lambda: _env_bool("ALLOW_PRIVATE_NETWORK_ACCESS", True))
    playback_rate: float = field(default_factory=lambda: _env_float("PLAYBACK_RATE", 1.0))

    # ---- Human-in-the-loop gate ----------------------------------------
    # Where this runs decides whether waiting for a person is a strategy or a
    # way to burn fifteen minutes and then fail anyway. On a workstation with
    # the dashboard open, waiting is right. On Agent Engine there is no browser
    # to hand over and nobody watching, so the honest move is to fail at once
    # and say what credential was missing.
    #   auto -- wait only if the dashboard answers a probe (the default)
    #   wait -- always wait, even with no dashboard (attended debugging)
    #   off  -- never wait; fail the run the moment a challenge appears
    human_gate_mode: str = field(default_factory=lambda: _env_str("HUMAN_GATE_MODE", "auto").lower())
    human_gate_timeout_seconds: float = field(
        default_factory=lambda: _env_float("HUMAN_GATE_TIMEOUT_SECONDS", 900.0)
    )

    # ---- Capture ------------------------------------------------------
    # auto -> probe for a grabbable stream, fall back to screen recording.
    capture_mode: str = field(default_factory=lambda: _env_str("CAPTURE_MODE", "auto").lower())
    window_seconds: int = field(default_factory=lambda: _env_int("WINDOW_SECONDS", 15))
    window_overlap_seconds: int = field(default_factory=lambda: _env_int("WINDOW_OVERLAP_SECONDS", 3))
    capture_fps: int = field(default_factory=lambda: _env_int("CAPTURE_FPS", 5))
    stream_probe_seconds: float = field(default_factory=lambda: _env_float("STREAM_PROBE_SECONDS", 12.0))
    # A full-page capture spends most of its pixels on sidebars and headers.
    # Cropping to the player is what keeps small details (a glove, a mask edge)
    # above the resolution the model can actually resolve.
    crop_to_player: bool = field(default_factory=lambda: _env_bool("CROP_TO_PLAYER", True))
    # Better than cropping where the player supports it: the video renders at
    # full viewport size, so the detail is really there rather than just
    # un-cropped. Falls back to CROP_TO_PLAYER when the player refuses.
    fullscreen_player: bool = field(default_factory=lambda: _env_bool("FULLSCREEN_PLAYER", True))
    # Recorded footage ends; the recorder does not notice on its own and would
    # keep encoding the frozen last frame until a budget stops it.
    stop_on_video_end: bool = field(default_factory=lambda: _env_bool("STOP_ON_VIDEO_END", True))
    page_watch_seconds: float = field(default_factory=lambda: _env_float("PAGE_WATCH_SECONDS", 5.0))

    # ---- Analysis -----------------------------------------------------
    analysis_concurrency: int = field(default_factory=lambda: _env_int("ANALYSIS_CONCURRENCY", 3))
    analysis_fps: float = field(default_factory=lambda: _env_float("ANALYSIS_FPS", 1.0))
    media_resolution: str = field(default_factory=lambda: _env_str("MEDIA_RESOLUTION", "low").lower())
    clip_queue_size: int = field(default_factory=lambda: _env_int("CLIP_QUEUE_SIZE", 8))

    # ---- Budget guards -------------------------------------------------
    max_wall_clock_seconds: int = field(default_factory=lambda: _env_int("MAX_WALL_CLOCK_SECONDS", 3600))
    max_cost_tokens: Optional[int] = field(default_factory=lambda: _env_opt_int("MAX_COST_TOKENS"))
    max_windows: Optional[int] = field(default_factory=lambda: _env_opt_int("MAX_WINDOWS"))

    # ---- Paths ---------------------------------------------------------
    auth_state_dir: Path = field(default_factory=lambda: Path(_env_str("AUTH_STATE_DIR", str(PROJECT_DIR / "auth"))))
    data_dir: Path = field(default_factory=lambda: Path(_env_str("DATA_DIR", str(PACKAGE_DIR / "checkpoints"))))
    work_dir: Path = field(default_factory=lambda: Path(_env_str("WORK_DIR", str(PROJECT_DIR / ".work"))))
    keep_clips: bool = field(default_factory=lambda: _env_bool("KEEP_CLIPS", False))
    # The audit standard itself. Point this at the customer's own file rather
    # than editing the one in the repo.
    # A relative path is resolved against the repo root, not the working
    # directory: the agent is launched from ADK, cron and the shell alike, and
    # a standard that silently fails to load is worse than one that is missing.
    sop_rules_path: Path = field(default_factory=lambda: _project_path(
        _env_str("SOP_RULES_PATH", str(PACKAGE_DIR / "analyzer" / "sop_rules.yaml"))
    ))

    # ---- Monitor -------------------------------------------------------
    monitor_host: str = field(default_factory=lambda: _env_str("MONITOR_HOST", "127.0.0.1"))
    monitor_port: int = field(default_factory=lambda: _env_int("MONITOR_PORT", 8080))
    monitor_token: str = field(default_factory=lambda: _env_str("MONITOR_TOKEN"))
    # The dashboard is a live view of the run in progress. Replaying the
    # append-only trail on startup showed old runs' findings -- possibly graded
    # against a different SOP file -- as if they belonged to this one.
    monitor_replay_history: bool = field(
        default_factory=lambda: _env_bool("MONITOR_REPLAY_HISTORY", False)
    )
    # The dashboard gets its own screencast, so these cost the evidence feed
    # nothing. 960x540 q60 is ~58 KB a frame, which makes 15 fps cheaper than
    # the old 4 fps of full-size frames (~280 KB) and far smoother.
    preview_fps: int = field(default_factory=lambda: _env_int("PREVIEW_FPS", 15))
    preview_width: int = field(default_factory=lambda: _env_int("PREVIEW_WIDTH", 960))
    preview_height: int = field(default_factory=lambda: _env_int("PREVIEW_HEIGHT", 540))
    preview_quality: int = field(default_factory=lambda: _env_int("PREVIEW_QUALITY", 60))

    # ---- Computer Use fallback ------------------------------------------
    fallback_max_turns: int = field(default_factory=lambda: _env_int("FALLBACK_MAX_TURNS", 25))
    enable_injection_detection: bool = field(default_factory=lambda: _env_bool("ENABLE_INJECTION_DETECTION", True))

    @property
    def evidence_dir(self) -> Path:
        return self.data_dir / "evidence"

    @property
    def clips_dir(self) -> Path:
        return self.work_dir / "clips"

    @property
    def records_path(self) -> Path:
        return self.data_dir / "audit_records.jsonl"

    @property
    def effective_window_seconds(self) -> int:
        """Wall-clock seconds a screen-recorded window takes at the configured rate."""
        return max(1, int(self.window_seconds / max(self.playback_rate, 0.1)))

    def ensure_dirs(self) -> None:
        for path in (self.auth_state_dir, self.data_dir, self.evidence_dir, self.clips_dir):
            path.mkdir(parents=True, exist_ok=True)
        # storage_state files hold live session cookies.
        try:
            self.auth_state_dir.chmod(0o700)
        except OSError:
            pass

    def validate(self) -> list[str]:
        """Returns a list of human-readable problems; empty means good to go."""
        problems: list[str] = []
        if not self.gcp_project:
            problems.append("GOOGLE_CLOUD_PROJECT is not set (no default is assumed).")
        if self.window_seconds <= 0:
            problems.append("WINDOW_SECONDS must be > 0.")
        if self.window_overlap_seconds >= self.window_seconds:
            problems.append(
                f"WINDOW_OVERLAP_SECONDS ({self.window_overlap_seconds}) must be smaller than "
                f"WINDOW_SECONDS ({self.window_seconds}); otherwise windows never advance."
            )
        if self.media_resolution not in ("low", "medium", "high"):
            problems.append(f"MEDIA_RESOLUTION must be low|medium|high, got '{self.media_resolution}'.")
        if self.human_gate_mode not in ("auto", "wait", "off"):
            problems.append(f"HUMAN_GATE_MODE must be auto|wait|off, got '{self.human_gate_mode}'.")
        if self.capture_mode not in ("auto", "stream", "screen"):
            problems.append(f"CAPTURE_MODE must be auto|stream|screen, got '{self.capture_mode}'.")
        if self.analysis_concurrency < 1:
            problems.append("ANALYSIS_CONCURRENCY must be >= 1.")
        if not self.sop_rules_path.exists():
            problems.append(f"SOP_RULES_PATH does not exist: {self.sop_rules_path}")
        if self.capture_fps < self.analysis_fps:
            problems.append(
                f"CAPTURE_FPS ({self.capture_fps}) is below ANALYSIS_FPS ({self.analysis_fps}); "
                "the model would be starved of frames."
            )
        return problems


config = Config()
