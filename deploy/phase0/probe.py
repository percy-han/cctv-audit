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

"""A container that does nothing useful, on purpose.

Phase 0 of the Agent Runtime migration. Before any of the real pipeline is
touched, four questions have to be answered, and none of them can be answered
from documentation -- I looked:

  1. How long may a single query hang before something cuts it? Neither the
     runtime contract nor the scaling guide states a timeout. An audit runs for
     minutes to hours, so this number decides whether the design is synchronous,
     polling, or fire-and-notify.
  2. Can a custom Agent Runtime deployment be registered as a tool in Gemini
     Enterprise at all?
  3. Will Gemini Enterprise carry a two-turn exchange -- ask, wait for the
     human, then continue? The whole confirm-before-you-audit requirement rests
     on this.
  4. Can the container reach the public internet? Without that there is nothing
     to record.

Deliberately dependency-light: FastAPI, uvicorn and aiohttp, all of which the
project already pins. No Chromium, no ffmpeg, no Vertex client. If this image
misbehaves the cause is the platform, not our code -- which is the only reason
to build a throwaway container instead of just deploying the real one.

Run locally:   python -m deploy.phase0.probe      (or: uvicorn deploy.phase0.probe:app)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("phase0.probe")

# Identifies this process. Two calls that report the same value ran on the same
# container; two that differ did not. That distinction is the reason preflight
# is planned to externalise its state instead of holding a browser open between
# calls -- this is where that assumption gets confirmed or overturned.
INSTANCE_ID = f"{uuid.uuid4().hex[:8]}-pid{os.getpid()}"
BOOT_TIME = time.time()

# The port is not ours to choose. From the runtime contract: the container must
# listen on 0.0.0.0 and port 8080.
PORT = int(os.environ.get("PORT", "8080"))

app = FastAPI(title="Agent Runtime Phase 0 probe")


# ------------------------------------------------------------------
# Event log
# ------------------------------------------------------------------
# The point of the whole timeout probe. When a caller gives up on a hanging
# request it learns exactly one thing: that *it* stopped waiting. It cannot tell
# whether the container was killed, whether the request was severed while the
# work continued, or whether an intermediate proxy simply hung up.
#
# So the server records what it finished, in memory, and `probe_log` reads it
# back on a later call. A `hang` that vanished from the client's point of view
# but shows `completed` here means the work survived and only the connection
# died -- which would make fire-and-notify viable. No entry, or a fresh
# INSTANCE_ID, means the container itself went away.
_EVENTS: List[Dict[str, Any]] = []
_MAX_EVENTS = 500


def record(kind: str, **fields: Any) -> Dict[str, Any]:
    """Appends one line to the in-memory log and returns it."""
    event = {
        "kind": kind,
        "instance": INSTANCE_ID,
        "at": round(time.time() - BOOT_TIME, 3),
        **fields,
    }
    _EVENTS.append(event)
    del _EVENTS[:-_MAX_EVENTS]
    logger.info("%s %s", kind, fields)
    return event


# ------------------------------------------------------------------
# The probes
# ------------------------------------------------------------------
async def m_hello(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Smallest possible round trip. Confirms the deployment is reachable."""
    record("hello")
    return {
        "ok": True,
        "message": payload.get("message", "hello from Agent Runtime"),
        "instance": INSTANCE_ID,
        "uptime_seconds": round(time.time() - BOOT_TIME, 1),
        "echo": payload,
    }


async def m_hang(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sleeps for `seconds`, then reports how long it actually slept.

    Probe 1. Call it with 60, 300, 900, 1800 and note where the caller stops
    getting an answer. Then call `probe_log` and see whether the sleep finished
    anyway -- the difference between "the platform cut the connection" and "the
    platform killed the container" is the difference between two designs.
    """
    seconds = float(payload.get("seconds", 60))
    ticket = uuid.uuid4().hex[:8]
    record("hang_start", ticket=ticket, requested=seconds)

    started = time.monotonic()
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        # Reaching here is itself a finding: the platform propagated a cancel
        # rather than letting the coroutine run on unattached.
        record(
            "hang_cancelled",
            ticket=ticket,
            requested=seconds,
            elapsed=round(time.monotonic() - started, 2),
        )
        raise

    elapsed = round(time.monotonic() - started, 2)
    record("hang_completed", ticket=ticket, requested=seconds, elapsed=elapsed)
    return {
        "ok": True,
        "ticket": ticket,
        "requested_seconds": seconds,
        "elapsed_seconds": elapsed,
        "instance": INSTANCE_ID,
    }


# A plain HTTP client is not what the real agent uses, and some sites can tell.
# Measured from this VM: bilibili answers a bare aiohttp request with 412 and a
# 3.4 KB block page, and the same request carrying these headers with 200 and
# 115 KB. Either way packets got out and back, so the first reading was already
# a *reachable* result -- which is the distinction this probe has to get right,
# because filing 412 as "no egress" would send us configuring Private Service
# Connect for a problem that does not exist.
_BROWSERISH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# Two targets by default, because they answer two different questions.
_DEFAULT_TARGETS = [
    # Is there any route to the internet at all? Returns 204 and no body, and
    # nothing about it is anti-bot sensitive.
    "https://www.google.com/generate_204",
    # Can we reach the site the first version will actually be watching?
    "https://www.bilibili.com",
]


async def _fetch_one(
    session: aiohttp.ClientSession, url: str
) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        async with session.get(url, headers=_BROWSERISH_HEADERS) as response:
            body = await response.read()
            return {
                # Any HTTP status means packets made the round trip. Whether the
                # site liked us is a separate question, answered by `status`.
                "reachable": True,
                "url": url,
                "status": response.status,
                "bytes": len(body),
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
    except Exception as exc:  # noqa: BLE001 -- the failure itself is the result
        return {
            "reachable": False,
            "url": url,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }


async def m_egress(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fetches one or more URLs from inside the container.

    Probe 4. Reports `reachable` (did packets get out and back) separately from
    `status` (did the site serve us). Only the first can sink the plan: no
    egress means a Private Service Connect interface or Agent Gateway has to be
    configured before anything else works. A 403/412 from an anti-bot page is
    expected here and is not a problem -- the real agent drives a browser.
    """
    if payload.get("url"):
        targets = [payload["url"]]
    else:
        targets = list(payload.get("urls") or _DEFAULT_TARGETS)
    timeout = float(payload.get("timeout_seconds", 20))

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:
        results = await asyncio.gather(*(_fetch_one(session, u) for u in targets))

    any_reachable = any(r["reachable"] for r in results)
    record("egress", any_reachable=any_reachable, results=results)
    return {
        "ok": any_reachable,
        "verdict": (
            "container has internet egress"
            if any_reachable
            else "NO egress -- configure PSC interface or Agent Gateway"
        ),
        "results": results,
        "instance": INSTANCE_ID,
    }


# Two-turn state. In-memory on purpose: if the second turn lands on a different
# container the session will be missing, and that missing session is the answer
# to "does preflight have to externalise its state?"
_SESSIONS: Dict[str, Dict[str, Any]] = {}


async def m_confirm_flow(payload: Dict[str, Any]) -> Dict[str, Any]:
    """A miniature of the real confirm-before-you-audit handshake.

    Probe 3. Turn one returns a question and a session id and stops. Turn two
    carries the human's answer. What is being tested is not this code -- it is
    whether Gemini Enterprise will relay a question to the user, wait, and then
    call back with the reply instead of inventing one.
    """
    session_id = payload.get("session_id")
    reply = payload.get("reply")

    if not session_id:
        session_id = uuid.uuid4().hex[:8]
        _SESSIONS[session_id] = {"asked_at": time.time()}
        record("confirm_asked", session_id=session_id)
        return {
            "ok": True,
            "stage": "awaiting_confirmation",
            "session_id": session_id,
            "question": (
                "Found the video: Plan B (screen recording), 14:00-15:00 is "
                "available. Reply 'confirm' to start the audit, or 'cancel'."
            ),
            "instance": INSTANCE_ID,
        }

    known = _SESSIONS.pop(session_id, None)
    if known is None:
        # Not necessarily an error -- most likely the second turn landed on a
        # different instance. Either way it is a finding, so say which.
        record("confirm_unknown_session", session_id=session_id)
        return {
            "ok": False,
            "stage": "unknown_session",
            "session_id": session_id,
            "detail": (
                "No such session on this instance. Either the id was wrong or "
                "turn two landed on a different container."
            ),
            "instance": INSTANCE_ID,
        }

    confirmed = str(reply).strip().lower() in {"confirm", "yes", "y", "确认", "是"}
    record("confirm_answered", session_id=session_id, confirmed=confirmed)
    return {
        "ok": True,
        "stage": "confirmed" if confirmed else "cancelled",
        "session_id": session_id,
        "waited_seconds": round(time.time() - known["asked_at"], 1),
        "reply": reply,
        "instance": INSTANCE_ID,
    }


async def m_probe_log(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Reads back what this instance has finished. See the note on `_EVENTS`."""
    limit = int(payload.get("limit", 50))
    return {
        "ok": True,
        "instance": INSTANCE_ID,
        "uptime_seconds": round(time.time() - BOOT_TIME, 1),
        "events": _EVENTS[-limit:],
    }


UNARY_METHODS = {
    "hello": m_hello,
    "hang": m_hang,
    "egress": m_egress,
    "confirm_flow": m_confirm_flow,
    "probe_log": m_probe_log,
}


# ------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------
async def m_stream_query(payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
    """Emits a heartbeat every `interval` seconds for `seconds` total.

    Two jobs. The runtime contract says the Console playground will not open
    without a `stream_query`, so one has to exist regardless. Beyond that, a
    stream that keeps producing bytes is a genuinely different case from a
    unary request that goes quiet -- idle timeouts usually only punish silence.
    If the stream survives thirty minutes while `hang` dies at five, progress
    streaming becomes the design and the polling fallback is unnecessary.
    """
    seconds = float(payload.get("seconds", 60))
    interval = max(1.0, float(payload.get("interval_seconds", 5)))
    ticket = uuid.uuid4().hex[:8]
    record("stream_start", ticket=ticket, requested=seconds, interval=interval)

    started = time.monotonic()
    tick = 0
    try:
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= seconds:
                break
            tick += 1
            yield {
                "tick": tick,
                "elapsed_seconds": round(elapsed, 1),
                "remaining_seconds": round(seconds - elapsed, 1),
                "instance": INSTANCE_ID,
            }
            await asyncio.sleep(min(interval, seconds - elapsed))
    except asyncio.CancelledError:
        record(
            "stream_cancelled",
            ticket=ticket,
            ticks=tick,
            elapsed=round(time.monotonic() - started, 2),
        )
        raise

    elapsed = round(time.monotonic() - started, 2)
    record("stream_completed", ticket=ticket, ticks=tick, elapsed=elapsed)
    yield {"done": True, "ticks": tick, "elapsed_seconds": elapsed, "instance": INSTANCE_ID}


# ------------------------------------------------------------------
# The method Gemini Enterprise actually calls
# ------------------------------------------------------------------
# Measured, not documented. Registering the engine under
# `adkAgentDefinition.provisionedReasoningEngine` and sending it one A2A
# message produced exactly this on the wire:
#
#   POST /api/stream_reasoning_engine
#   {"class_method": "streaming_agent_run_with_events",
#    "input": {"request_json": "{\"message\": {\"role\": \"user\",
#                                \"parts\": [{\"text\": \"say hello\"}]},
#                                \"session_id\": \"...\",
#                                \"user_id\": \"you@example.com\"}"}}
#
# Three things follow, and all three shape Phase 2:
#
#   * GE speaks one method, not many. The careful per-method `description` and
#     `parameters` in the classMethods list are irrelevant on this path -- GE
#     never sees preflight/start_audit/get_status as separate tools. Routing
#     between them has to happen *inside* the container, or via one GE agent
#     per operation.
#   * `request_json` is a JSON string inside the JSON body, not a nested
#     object. Double-encoded.
#   * It is a *streaming* call, so whatever ceiling applies is a streaming
#     ceiling. That is the number the audit design hangs on.
GE_METHOD = "streaming_agent_run_with_events"


def _adk_event(text: str, invocation: str, partial: bool = False) -> Dict[str, Any]:
    """One ADK `Event`, in the shape `model_dump_json(exclude_none=True)` gives.

    Not invented. Generated from the real `google.adk.events.event.Event` and
    copied field for field, because two rounds of plausible-looking guesses
    were rejected with no diagnostic beyond "produced no events". Note it is
    snake_case: ADK's model has a camelCase alias generator, but the dump is
    taken without `by_alias`, so the wire form is snake.
    """
    event: Dict[str, Any] = {
        "content": {"parts": [{"text": text}], "role": "model"},
        "invocation_id": invocation,
        "author": "cctv-phase0-probe",
        "actions": {
            "state_delta": {},
            "artifact_delta": {},
            "requested_auth_configs": {},
            "requested_tool_confirmations": {},
        },
        "id": uuid.uuid4().hex[:8],
        "timestamp": time.time(),
    }
    if partial:
        # `exclude_none=True` means ADK omits this entirely when it is not set,
        # so only add it when it is true.
        event["partial"] = True
    return event


# The envelope. This is what was missing, and it is the whole reason GE kept
# saying "stream closed cleanly without producing any events": the container
# was emitting bare events, and GE reads `_StreamingRunResponse`, which wraps
# them:
#
#     {"events": [<event>, ...], "artifacts": [...], "session_id": "..."}
#
# Found by reading `vertexai/agent_engines/templates/adk.py` out of the
# google-cloud-aiplatform wheel -- `streaming_agent_run_with_events` yields
# `self._convert_response_events(...)`, which returns exactly that dict. No
# documentation states it. Guessing at the encoding (bare object vs quoted
# string) was the wrong axis entirely; the shape was wrong, not the escaping.
#
# The line framing is `json.dumps(chunk) + "\n"` with media type
# `application/json`, which is what ADK's own `/api/stream_reasoning_engine`
# does -- see `_encode_chunk_to_json` in google/adk/cli/fast_api.py.
def _stream_response(event: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    envelope: Dict[str, Any] = {"events": [event]}
    if session_id:
        envelope["session_id"] = session_id
    return envelope


def _user_text(request: Dict[str, Any]) -> str:
    parts = (request.get("message") or {}).get("parts") or []
    return " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()


def _command(text: str) -> tuple:
    """Reads a fixed probe vocabulary out of the turn. Deliberately literal.

    This is a debug console, not the product. It recognises a handful of exact
    keywords and, when it recognises none, echoes rather than guessing -- the
    same rule the real pipeline follows for SOP input. Nothing here is meant to
    understand a sentence; the moment it needs to, that job belongs to a model.
    """
    lowered = text.lower()
    for word in ("hang", "挂"):
        if word in lowered:
            digits = "".join(c if c.isdigit() else " " for c in lowered).split()
            if digits:
                return "hang", float(digits[0])
    for word in ("stream", "心跳", "进度"):
        if word in lowered:
            digits = "".join(c if c.isdigit() else " " for c in lowered).split()
            return "stream", float(digits[0]) if digits else 60.0
    for word in ("egress", "出网", "bilibili", "哔哩"):
        if word in lowered:
            return "egress", None
    for word in ("confirm", "确认", "是的", "yes"):
        if word in lowered:
            return "confirm", None
    for word in ("audit", "稽核", "查一下"):
        if word in lowered:
            return "ask", None
    return "echo", None


# Keyed by the GE session id, which is the only handle we get on "the same
# conversation". Whether it survives to the second turn is precisely probe 3.
_GE_SESSIONS: Dict[str, Dict[str, Any]] = {}


async def m_ge_turn(payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
    """Serves one Gemini Enterprise turn."""
    import json as _json

    raw = payload.get("request_json")
    request = _json.loads(raw) if isinstance(raw, str) else (raw or {})
    text = _user_text(request)
    session_id = str(request.get("session_id") or "")
    invocation = uuid.uuid4().hex[:12]
    verb, amount = _command(text)

    # `partial` is keyword-only on purpose. It landed as a positional once and
    # silently became the invocation id, which shipped an event with
    # `"partial": "d84079096b6e"` -- a type error the far side is entitled to
    # reject, and one that cost a deploy cycle to find.
    def out(message: str, *, partial: bool = False):
        return _stream_response(_adk_event(message, invocation, partial), session_id)

    record(
        "ge_turn",
        session_id=session_id,
        user_id=request.get("user_id"),
        text=text,
        verb=verb,
        amount=amount,
        invocation=invocation,
    )

    if verb == "hang":
        # Silence for `amount` seconds, then one answer. If GE cuts before this
        # lands, the ceiling punishes duration. Compare against "stream".
        started = time.monotonic()
        try:
            await asyncio.sleep(amount)
        except asyncio.CancelledError:
            record("ge_hang_cancelled", requested=amount,
                   elapsed=round(time.monotonic() - started, 2))
            raise
        elapsed = round(time.monotonic() - started, 2)
        record("ge_hang_completed", requested=amount, elapsed=elapsed)
        yield out(
            f"挂了 {elapsed} 秒后返回（要求 {amount} 秒），实例 {INSTANCE_ID}。",
        )
        return

    if verb == "stream":
        # Same total duration, but never silent. If this survives and `hang`
        # does not, the ceiling is an idle timeout and progress streaming is
        # the design.
        started = time.monotonic()
        tick = 0
        while time.monotonic() - started < amount:
            tick += 1
            yield out(
                f"进度 {tick}：已跑 {round(time.monotonic() - started)} 秒 / {amount} 秒。",
                partial=True,
            )
            await asyncio.sleep(min(10.0, amount - (time.monotonic() - started)))
        elapsed = round(time.monotonic() - started, 2)
        record("ge_stream_completed", requested=amount, ticks=tick, elapsed=elapsed)
        yield out(f"流式跑完 {elapsed} 秒，共 {tick} 个心跳。")
        return

    if verb == "egress":
        result = await m_egress({})
        lines = "；".join(
            f"{r['url']} -> {r.get('status', r.get('error'))}" for r in result["results"]
        )
        yield out(f"{result['verdict']}。{lines}")
        return

    if verb == "ask":
        # Turn one of the confirmation handshake: state a finding, ask, stop.
        _GE_SESSIONS[session_id] = {"asked_at": time.time(), "text": text}
        record("ge_confirm_asked", session_id=session_id)
        yield out(
            "找到视频了：Plan B 录屏，14:00-15:00 时间段够。确认开始稽核吗？"
            "回「确认」我就开始。",
        )
        return

    if verb == "confirm":
        # Turn two. A hit proves GE relayed the question, waited for a human,
        # and came back on the same session -- and that the session id is a
        # usable key. A miss says state must be externalised.
        known = _GE_SESSIONS.pop(session_id, None)
        waited = round(time.time() - known["asked_at"], 1) if known else None
        record("ge_confirm_answered", session_id=session_id, found=bool(known),
               waited_seconds=waited)
        if known is None:
            yield out(
                f"这个实例上没有会话 {session_id or '(空)'} 的记录——"
                "要么第二轮落到了别的容器，要么 GE 没把 session_id 传回来。"
                f"实例 {INSTANCE_ID}。",
            )
        else:
            yield out(
                f"收到确认，等了 {waited} 秒。上一轮问的是「{known['text']}」。"
                f"同一个实例 {INSTANCE_ID}，session_id 传回来了。",
            )
        return

    yield out(
        f"探针在线。实例 {INSTANCE_ID}，已运行 "
        f"{round(time.time() - BOOT_TIME, 1)} 秒，收到「{text}」。"
        "可以说：hang 300 / stream 300 / 出网 / 稽核 → 确认。",
    )


STREAM_METHODS = {
    "stream_query": m_stream_query,
    GE_METHOD: m_ge_turn,
}


# ------------------------------------------------------------------
# The routes Agent Runtime calls
# ------------------------------------------------------------------
# Both names and both shapes come from the runtime contract, not from taste:
# unary lands on /api/reasoning_engine, streaming on
# /api/stream_reasoning_engine, and the body is {"class_method", "input"}.
def _read_call(body: Dict[str, Any]) -> tuple:
    method = body.get("class_method") or body.get("classMethod") or ""
    payload = body.get("input") or {}
    if not isinstance(payload, dict):
        payload = {"value": payload}
    return method, payload


# Truncated, because a Gemini Enterprise turn can carry the whole conversation
# and `probe_log` has to stay readable. The head is where the method name and
# the argument shape live, which is all this is for.
_BODY_CHARS = 1200


def _record_call(route: str, body: Dict[str, Any], method: str) -> None:
    """Writes down exactly what arrived, before anything is dispatched.

    The middleware only sees the path, and that turned out not to be enough:
    when Gemini Enterprise called this container it landed on the streaming
    route and was rejected, and the log could not say *which* method it had
    asked for. The whole reason the probe exists is to answer questions like
    that from evidence, so the body gets recorded verbatim (head only).
    """
    import json as _json

    record(
        "call",
        route=route,
        class_method=method or None,
        body=_json.dumps(body, ensure_ascii=False)[:_BODY_CHARS],
    )


@app.post("/api/reasoning_engine")
async def unary(request: Request):
    body = await request.json()
    method, payload = _read_call(body)
    _record_call("unary", body, method)

    handler = UNARY_METHODS.get(method)
    if handler is None:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": f"unknown class_method {method!r}",
                "available": sorted(UNARY_METHODS),
                "instance": INSTANCE_ID,
            },
        )
    # The `output` wrapper is mandatory, and nothing says so in prose. The
    # runtime contract shows `{"output": "..."}` as an example; the API's own
    # QueryReasoningEngineResponse has exactly one field, `output`. Return a
    # bare dict and the container answers 200 while the caller gets
    # `500 Internal error` -- the platform cannot parse it and says nothing
    # useful. Streaming has no such wrapper: ndjson chunks pass through as-is,
    # which is why stream_query worked while every unary method 500'd.
    return JSONResponse(content={"output": await handler(payload)})


@app.post("/api/stream_reasoning_engine")
async def stream(request: Request):
    body = await request.json()
    method, payload = _read_call(body)
    _record_call("stream", body, method)

    handler = STREAM_METHODS.get(method)
    if handler is None:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": f"unknown streaming class_method {method!r}",
                "available": sorted(STREAM_METHODS),
                "instance": INSTANCE_ID,
            },
        )

    async def lines() -> AsyncIterator[bytes]:
        # `json.dumps(chunk) + "\n"`, one chunk per line. Copied from ADK's own
        # `_encode_chunk_to_json` (google/adk/cli/fast_api.py) rather than
        # invented, and the media type is `application/json` for the same
        # reason -- that is what the reference server sends, and this is the
        # route Gemini Enterprise reads.
        import json

        async for chunk in handler(payload):
            yield (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(lines(), media_type="application/json")


# Records every inbound request, including ones that 404.
#
# Added after finding `asyncQuery` in the Vertex discovery document: it takes an
# `inputGcsUri`, returns a long-running Operation, writes the result to
# `outputGcsUri`, and explicitly supports BYOC. That is the platform's own
# answer to long jobs -- but the runtime contract only ever documents
# /api/reasoning_engine and /api/stream_reasoning_engine, so *how* an async
# query reaches a custom container is unstated. Rather than guess, log what
# actually arrives and read it back with `probe_log`.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path != "/":  # the index is only ever us poking it
        record(
            "request",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query) or None,
        )
    return await call_next(request)


@app.get("/")
async def index():
    """Not part of the contract. Here so `curl`ing the service says something."""
    return {
        "service": "Agent Runtime Phase 0 probe",
        "instance": INSTANCE_ID,
        "uptime_seconds": round(time.time() - BOOT_TIME, 1),
        "unary_methods": sorted(UNARY_METHODS),
        "stream_methods": sorted(STREAM_METHODS),
    }


if __name__ == "__main__":
    import uvicorn

    # timeout_keep_alive is raised well past any plausible platform limit so
    # that a cut connection is attributable to the platform rather than to us.
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=3600)
