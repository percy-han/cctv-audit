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
import time
from typing import Optional
from urllib.parse import urlsplit

import google.auth
import google.auth.transport.requests
from google import genai

from .config import config

logger = logging.getLogger("cctv_audit.gcp")

_client: Optional[genai.Client] = None


# Vertex, Storage and Firestore all sit under this one scope. Asking for it by
# name matters on a workstation, where ADC may have been minted for something
# narrower; on a metadata-server identity it is what you get anyway.
_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


# There used to be a `GCloudCredentials` class here that shelled out to
# `gcloud auth print-access-token` and fell back to ADC. It is gone, and the
# reason is worth keeping:
#
# It subclassed `Credentials` but never set `self.expiry`. In google-auth,
# `expired` is False whenever `expiry` is None and `valid` is `token is not
# None and not expired` -- so the object reported itself valid forever and
# `before_request` never called `refresh()`. The access token minted when the
# container booted was reused until the process died. Google access tokens
# last an hour.
#
# What that looked like from outside, on 2026-09-04: an audit ran 22 windows
# clean and then 401'd on windows 23 through 32, and the job still finished and
# filed a report -- for a video it had stopped watching. A GE turn 50 minutes
# after the container started came back "I can't read that, say it again",
# because the intent call had 401'd and the honest-sounding fallback message
# hid an auth failure behind a comprehension failure.
#
# Nothing here needs hand-rolled credentials. ADC already refreshes itself.
def credentials_without_quota_project():
    """ADC with the billing/quota project stripped off.

    Agent Runtime hands the container credentials that already carry a quota
    project -- the project *number*. Any client built on them sends
    `x-goog-user-project`, and Cloud Storage reads that as "bill this project",
    which needs `serviceusage.services.use`. An agent service account granted
    exactly the roles it uses does not have that, so every GCS read came back:

        403 GET .../sop%2Fchagee-store-v1.yaml?alt=media:
        cctv-audit-agent@... does not have serviceusage.services.use access to
        the Google Cloud project.

    Note whose name is in that message: the storage role was never the problem,
    and the bucket policy looked correct the whole time. Firestore is not
    affected -- same credentials, same run, it read fine -- which is why only
    the SOP fetch broke.

    Granting `roles/serviceusage.serviceUsageConsumer` would also fix it. We do
    not, because the bucket is in the same project we would be billing: there
    is nothing to charge elsewhere, so the header has no purpose here, and
    dropping it keeps one more role off the customer's deployment checklist.

    Returns None if ADC cannot be resolved at all, which lets the caller fall
    back to the library default and fail with its own clearer message.
    """
    try:
        creds, _ = google.auth.default(scopes=list(_SCOPES))
    except Exception as exc:  # no ADC -- local runs without gcloud, tests
        logger.info("no ADC available (%s); leaving credentials to the client.", exc)
        return None
    strip = getattr(creds, "with_quota_project", None)
    return strip(None) if strip else creds


# ID tokens are good for an hour. Refreshing a few minutes early costs one
# extra mint per audit and removes the case where a token issued at the top of
# a long capture expires halfway through it.
_ID_TOKEN_TTL_SECONDS = 3000.0
_id_tokens: dict[str, tuple[float, str]] = {}


def origin_of(url: str) -> str:
    """`scheme://host[:port]`, lowercased, or "" if there is no host."""
    parts = urlsplit(url or "")
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}" if parts.netloc else ""


def is_oidc_origin(url: str) -> bool:
    """True if `url` lives on an origin the operator marked as IAM-protected."""
    return origin_of(url) in set(config.oidc_origins)


def id_token_for(url: str, *, now: Optional[float] = None) -> Optional[str]:
    """A Google ID token audienced at `url`'s origin, or None if not needed.

    Returns None -- rather than raising -- for an origin that is not on the
    list, because that is the overwhelmingly common case: every ordinary CCTV
    console and every public video site goes down this path, and none of them
    should see a token.

    Minting *does* raise if the origin is on the list and no identity can be
    obtained. That is deliberate: the operator has said this origin needs a
    token, so continuing without one produces a 403 whose cause is three
    layers away from the message.
    """
    origin = origin_of(url)
    if origin not in set(config.oidc_origins):
        return None

    now = time.monotonic() if now is None else now
    cached = _id_tokens.get(origin)
    if cached and cached[0] > now:
        return cached[1]

    from google.oauth2 import id_token as google_id_token

    # `fetch_id_token` uses ADC: on Agent Runtime and Cloud Run that is the
    # service account's metadata-server identity, which is what the receiving
    # service checks against its `run.invoker` binding.
    token = google_id_token.fetch_id_token(
        google.auth.transport.requests.Request(), origin
    )
    _id_tokens[origin] = (now + _ID_TOKEN_TTL_SECONDS, token)
    logger.info("Minted a Google ID token for %s.", origin)
    return token


def reset_id_token_cache() -> None:
    """Drops every cached ID token. For tests and for a fresh audit run."""
    _id_tokens.clear()


def access_token() -> str:
    """A fresh OAuth access token for this deployment's own identity.

    For handing to something that is not a Google client library -- ffmpeg,
    which reads `gs://` objects over the storage.googleapis.com REST endpoint
    with an `Authorization: Bearer` header because it has no idea what a
    service account is.

    Not cached. `refresh()` is a no-op on a credential that is still valid, and
    the alternative -- caching a string we cannot invalidate -- is the exact
    shape of the bug documented above `credentials_without_quota_project`.

    Raises if no identity can be obtained. There is no useful fallback: a
    `gs://` object in the customer's bucket is not readable anonymously, and
    the 401 that would follow is three layers away from the cause.
    """
    creds = credentials_without_quota_project()
    if creds is None:
        raise RuntimeError(
            "读不到这台机器的 Google 身份（ADC），没法访问 GCS 上的视频。"
            "本机跑的话先执行 `gcloud auth application-default login`。"
        )
    creds.refresh(google.auth.transport.requests.Request())
    token = getattr(creds, "token", None)
    if not token:
        raise RuntimeError("刷新 Google 凭据之后仍然没拿到 access token。")
    return token


def get_genai_client() -> genai.Client:
    """Returns the shared Vertex AI client (created on first use)."""
    global _client
    if _client is None:
        if not config.gcp_project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set; cannot reach Vertex AI.")
        # `credentials=None` would let google-genai resolve ADC itself, which
        # is almost right -- but it would keep the quota project, and on Agent
        # Runtime that is the project *number* in an `x-goog-user-project`
        # header the agent service account is not allowed to bill to.
        _client = genai.Client(
            vertexai=True,
            project=config.gcp_project,
            location=config.gcp_location,
            credentials=credentials_without_quota_project(),
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
