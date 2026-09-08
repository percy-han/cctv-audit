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

"""Makes the container's own logs visible. It is not decoration.

Until this module existed, `server.py` created loggers and never gave the root
logger a handler. Python's handler-of-last-resort then emitted **WARNING and
above only**, so every `logger.info` in the pipeline -- which window is being
captured, which plan was chosen, how long a clip took -- went nowhere. Cloud
Logging showed uvicorn's access lines and nothing else, on failed and
successful runs alike.

The cost of that was a job that sat at `running` for 62 minutes with no way to
tell a hang from slow progress, because the only evidence available was the
absence of evidence. **Absence of audit logs did not mean the audit was dead**
-- it meant nothing at all. That ambiguity is what this removes.

Two details are specific to running on Google's serverless platforms:

  * **stdout, not stderr.** Cloud Run/Agent Runtime tags anything written to
    stderr as ERROR. `basicConfig`'s default handler is stderr, so an INFO line
    would arrive coloured red and a real error would be indistinguishable from
    a heartbeat.
  * **JSON, not text.** A line of JSON carrying a `severity` key is parsed into
    a structured entry, which is what makes `severity>=WARNING` a usable filter
    and what puts the traceback in one entry instead of forty. Plain text is
    kept for local runs, where a human is reading it directly.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

_CONFIGURED = False

# Loggers that say a great deal about their own health and nothing about ours.
# Left at WARNING so that turning our level up to DEBUG stays readable.
_NOISY = (
    "google.auth",
    "google.api_core",
    "google_genai",
    "httpx",
    "httpcore",
    "urllib3",
    "websockets",
    "asyncio",
)

# Python's level names are not Cloud Logging's. Unmapped names arrive as
# DEFAULT, which sorts below INFO and hides in the console's default view.
_SEVERITY = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


class CloudJsonFormatter(logging.Formatter):
    """One log record, one line of JSON that Cloud Logging understands."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": _SEVERITY.get(record.levelname, "INFO"),
            "message": record.getMessage(),
            "logger": record.name,
            "instance": os.environ.get("K_REVISION") or os.environ.get("HOSTNAME") or "local",
        }
        if record.exc_info:
            # Appended to `message` rather than put in a field of its own: the
            # log viewer shows `message` and would otherwise hide the traceback
            # behind an expander, which is where tracebacks go to be missed.
            payload["message"] += "\n" + self.formatException(record.exc_info)
        # `default=str` because a %-format argument can be anything, and a
        # logging call must never be the thing that raises.
        return json.dumps(payload, ensure_ascii=False, default=str)


def _wants_json() -> bool:
    fmt = os.environ.get("LOG_FORMAT", "").strip().lower()
    if fmt in ("json", "text", "plain"):
        return fmt == "json"
    # K_REVISION is set by Cloud Run and by Agent Runtime, which runs on it.
    return bool(os.environ.get("K_REVISION") or os.environ.get("K_SERVICE"))


def setup_logging(level: Optional[str] = None, force: bool = False) -> None:
    """Installs one handler on the root logger. Safe to call more than once.

    Called at import time by both entrypoints. Importing an entrypoint is
    already a decision to run a server, and uvicorn configures its own loggers
    before it imports the app, so by the time this runs there is a gap to fill
    and nothing to fight with.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    wanted = (level or os.environ.get("LOG_LEVEL") or "INFO").strip().upper()
    numeric = getattr(logging, wanted, logging.INFO)
    if not isinstance(numeric, int):
        numeric = logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        CloudJsonFormatter()
        if _wants_json()
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    # Replacing rather than adding: a second call (tests, `--force`) must not
    # double every line, and a handler installed by an imported library is not
    # the one we want deciding where audit logs go.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(numeric)

    for name in _NOISY:
        logging.getLogger(name).setLevel(max(numeric, logging.WARNING))

    _CONFIGURED = True
