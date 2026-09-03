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

"""Vertex AI client plumbing, shared by the navigator fallback and the analyzer.

Both layers call Gemini, but for different things (browser actions vs. video
understanding). They share credentials, the client, and the retry policy.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from typing import Optional

import google.auth
import google.auth.transport.requests
from google import genai
from google.auth.credentials import Credentials as BaseCredentials

from .config import config

logger = logging.getLogger("cctv_audit.gcp")

_client: Optional[genai.Client] = None


class GCloudCredentials(BaseCredentials):
    """Credentials backed by the gcloud CLI, falling back to ADC."""

    def __init__(self):
        super().__init__()
        self.token = None
        self.refresh(None)

    def refresh(self, request=None):
        try:
            self.token = subprocess.check_output(
                ["gcloud", "auth", "print-access-token"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception as exc:
            logger.info("gcloud print-access-token failed (%s); falling back to ADC.", exc)
            creds, _ = google.auth.default()
            creds.refresh(request or google.auth.transport.requests.Request())
            self.token = creds.token


def get_genai_client() -> genai.Client:
    """Returns the shared Vertex AI client (created on first use)."""
    global _client
    if _client is None:
        if not config.gcp_project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set; cannot reach Vertex AI.")
        _client = genai.Client(
            vertexai=True,
            project=config.gcp_project,
            location=config.gcp_location,
            credentials=GCloudCredentials(),
        )
        logger.info("Vertex AI client ready (project=%s, location=%s).",
                    config.gcp_project, config.gcp_location)
    return _client


# Retry these and nothing else. Read off the exception's status code, not out
# of its message: the previous version looked for the substrings "500", "429"
# and "503" anywhere in the text, so an error mentioning a 15000-token limit or
# a model named `...-0429` was retried three times for no reason, while the
# same digits appearing in a genuinely permanent error hid it behind two
# pointless backoffs.
_TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

# Named status symbols. These are gRPC/Vertex enum names, not prose -- they are
# safe to match exactly, and they cover the transports that report no HTTP code.
_TRANSIENT_STATUSES = frozenset({
    "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED", "ABORTED",
})

# One known-transient condition that arrives as prose with no useful code: the
# request was routed to a region where the model is not deployed. Retrying hits
# a different region and usually succeeds.
_TRANSIENT_PHRASE = "computer use is not supported for this model in this region"


def is_transient(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, bool):  # bool is an int; nobody means HTTP 1.
        code = None
    if isinstance(code, int) and code in _TRANSIENT_STATUS:
        return True

    # google-genai puts the enum name in `.status`; grpc in `.code().name`.
    status = getattr(exc, "status", None)
    if isinstance(status, str) and status.strip().upper() in _TRANSIENT_STATUSES:
        return True

    text = str(exc)
    if _TRANSIENT_PHRASE in text.lower():
        return True
    # Last resort for SDK versions that carry the code only in the message.
    # Anchored to the shapes an error actually uses -- "503 UNAVAILABLE",
    # "code: 429", "[500]" -- so a stray number in a sentence does not match.
    if re.search(r"(?:^|[\[\s:(])(?:408|409|429|500|502|503|504)(?:[\]\s,.:)]|$)", text):
        return True
    return any(re.search(rf"\b{name}\b", text) for name in _TRANSIENT_STATUSES)


async def generate_content_with_retry(
    model: str,
    contents,
    generate_config,
    max_retries: int = 3,
    client: Optional[genai.Client] = None,
):
    """generate_content with exponential backoff on transient routing/capacity errors."""
    client = client or get_genai_client()
    for attempt in range(max_retries):
        try:
            return await client.aio.models.generate_content(
                model=model, contents=contents, config=generate_config
            )
        except Exception as exc:
            if is_transient(exc) and attempt < max_retries - 1:
                delay = 2 ** attempt
                logger.warning(
                    "Transient Vertex AI error on attempt %d/%d: %s -- retrying in %ds",
                    attempt + 1, max_retries, str(exc)[:150], delay,
                )
                await asyncio.sleep(delay)
                continue
            raise
