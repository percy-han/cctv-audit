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

"""Live audit dashboard: WebSocket fan-out plus the human-intervention channel.

Security posture, deliberately chosen:

* Binds to 127.0.0.1 by default. This server streams live CCTV frames that
  contain identifiable faces of staff and customers. It previously bound
  0.0.0.0 with no authentication, i.e. anyone on the network could watch.
* `/api/event` requires `MONITOR_TOKEN` when one is set, because it is a write
  endpoint that injects records into the audit view.
* Evidence frames are served only from inside the configured evidence
  directory, with the path resolved and re-checked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cctv_audit.monitor_server")

app = FastAPI(title="CCTV Audit Live Monitor")

state: Dict[str, Any] = {
    "status": "IDLE",
    "prompt": "",
    "capture_settings": "",
    "current_url": "about:blank",
    "last_action": "",
    "last_action_args": {},
    "click_target": None,
    "latest_frame_b64": None,
    "timeline": [],
    "final_result": "",
    "video_segments": [],
    "sop_violations": [],
    "intervention": None,
}

active_websockets: Set[WebSocket] = set()

# Set by the pipeline process when it runs in-process with the server, so the
# "continue" button can release the gate directly.
_human_gate = None


def attach_human_gate(gate) -> None:
    global _human_gate
    _human_gate = gate


def _to_segment(record: Dict[str, Any]) -> Dict[str, Any]:
    """Maps an AuditStore record onto the shape the sidebar renders."""
    violations = [f for f in record.get("findings", []) if f.get("status") == "VIOLATION"]
    return {
        "id": record.get("id"),
        "time_range": record.get("time_range", ""),
        "description": record.get("scene_summary", record.get("description", "")),
        "sop_status": record.get("sop_status", "COMPLIANT"),
        "violation_detail": record.get("violation_detail") or "；".join(
            f"[{f.get('rule_id')}@{f.get('timestamp', '')}] {f.get('evidence', '')}"
            for f in violations
        ),
        "severity": record.get("severity", "NONE"),
        "findings": record.get("findings", []),
        "people_count": record.get("people_count", 0),
        "visibility_ok": record.get("visibility_ok", True),
        "start_seconds": record.get("start_offset_seconds", 0.0),
    }


def _segment_order(segment: Dict[str, Any]) -> tuple:
    """Video time first, id as the tie-break.

    `start_seconds` was added late, so a record replayed from an older JSONL
    may not have it. Falling back to the id keeps those in insertion order
    rather than collapsing them all onto zero.
    """
    return (float(segment.get("start_seconds") or 0.0), segment.get("id") or 0)


def load_history() -> None:
    """Replays previously stored records so a reconnect is not a blank screen.

    Off by default. The JSONL is an append-only trail across every run ever
    made, so replaying it opened the dashboard on last week's findings -- under
    rule ids from whichever standard was configured at the time -- with nothing
    to say they were not from the run about to start. The dashboard is a live
    view of one audit; the trail on disk is the archive.
    """
    path = config.records_path
    if not path.exists():
        return
    loaded = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    segment = _to_segment(json.loads(line))
                except ValueError:
                    continue
                state["video_segments"].append(segment)
                if segment["sop_status"] == "VIOLATION":
                    state["sop_violations"].append(segment)
                loaded += 1
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return
    logger.info("Loaded %d historical windows (%d with violations) from %s",
                loaded, len(state["sop_violations"]), path.name)


if config.monitor_replay_history:
    load_history()
else:
    logger.info(
        "Dashboard starts empty; set MONITOR_REPLAY_HISTORY=true to replay %s.",
        config.records_path.name,
    )


def _check_token(request: Request) -> None:
    if not config.monitor_token:
        return
    supplied = request.headers.get("X-Monitor-Token") or request.query_params.get("token")
    if supplied != config.monitor_token:
        raise HTTPException(status_code=401, detail="invalid or missing monitor token")


async def _broadcast(payload: Dict[str, Any]) -> None:
    if not active_websockets:
        return
    message = json.dumps(payload, ensure_ascii=False)
    dead = set()
    for websocket in list(active_websockets):
        try:
            await websocket.send_text(message)
        except Exception:
            dead.add(websocket)
    active_websockets.difference_update(dead)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    from .monitor import HTML_PAGE
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/state")
async def get_state():
    return {k: v for k, v in state.items() if k != "latest_frame_b64"}


@app.get("/api/evidence/{path:path}")
async def get_evidence(path: str):
    """Serves a stored evidence frame.

    Resolved and then re-checked against the evidence directory so a crafted
    path cannot read arbitrary files off the host.
    """
    base = config.evidence_dir.resolve()
    candidate = (config.data_dir / path).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(candidate)


@app.post("/api/event")
async def post_event(request: Request):
    _check_token(request)
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    event_type = payload.get("type")

    if event_type == "state":
        state.update(payload.get("data", {}))
    elif event_type == "frame":
        state["latest_frame_b64"] = payload.get("frame")
        if payload.get("url"):
            state["current_url"] = payload["url"]
    elif event_type == "action":
        action = payload.get("action", {})
        state["last_action"] = action.get("action", "")
        state["last_action_args"] = action.get("args", {})
        state["click_target"] = payload.get("click_target")
        if action.get("url"):
            state["current_url"] = action["url"]
        if action:
            state["timeline"] = (state["timeline"] + [action])[-50:]
    elif event_type == "segment":
        segment = payload.get("segment") or {}
        if segment:
            # Dedup by id when present, else by time range. No file write here:
            # AuditStore already persisted this record before it was pushed.
            key = segment.get("id")
            existing = next(
                (s for s in state["video_segments"]
                 if (key is not None and s.get("id") == key)
                 or (key is None and s.get("time_range") == segment.get("time_range"))),
                None,
            )
            if existing is None:
                state["video_segments"].append(segment)
                if segment.get("sop_status") == "VIOLATION":
                    state["sop_violations"].append(segment)
                # Windows arrive in completion order, not video order: with
                # ANALYSIS_CONCURRENCY > 1 window 2 routinely lands before
                # window 1. Insert-sort here so the sidebar always reads as a
                # timeline, and late arrivals slot into place instead of
                # appending to the bottom.
                for key in ("video_segments", "sop_violations"):
                    state[key].sort(key=_segment_order)
            else:
                existing.update(segment)
    elif event_type == "intervention":
        state["intervention"] = payload.get("reason")
    elif event_type == "intervention_cleared":
        state["intervention"] = None

    await _broadcast(payload)
    return {"status": "ok"}


@app.post("/api/interact")
async def post_interact(request: Request):
    """The operator's side of the human-in-the-loop gate.

    `resolve` tells a waiting pipeline the challenge is cleared. This is the
    supported answer to CAPTCHAs and OTP prompts -- we do not try to solve them
    programmatically.
    """
    _check_token(request)
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    action = payload.get("action")
    if action != "resolve":
        return JSONResponse({"error": f"unsupported action '{action}'"}, status_code=400)

    state["intervention"] = None
    if _human_gate is not None:
        _human_gate.resolve()
        logger.info("Operator cleared the intervention gate.")
    else:
        # The usual case: the pipeline runs in the ADK process, so there is no
        # in-process gate to poke. Clearing `state["intervention"]` above is the
        # signal -- HumanGate polls /api/state and resumes when the banner goes.
        logger.info("Intervention cleared; the pipeline will pick it up from /api/state.")
    await _broadcast({"type": "intervention_cleared"})
    return {"status": "ok", "gate_attached": _human_gate is not None}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        await websocket.send_text(json.dumps(
            {"type": "state", "data": {k: v for k, v in state.items() if k != "latest_frame_b64"}},
            ensure_ascii=False,
        ))
        if state.get("latest_frame_b64"):
            await websocket.send_text(json.dumps({
                "type": "frame",
                "frame": state["latest_frame_b64"],
                "url": state["current_url"],
            }))
        if state.get("intervention"):
            await websocket.send_text(json.dumps(
                {"type": "intervention", "reason": state["intervention"]}, ensure_ascii=False))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WebSocket closed: %s", exc)
    finally:
        active_websockets.discard(websocket)


def main():
    parser = argparse.ArgumentParser(description="Start the CCTV audit live monitor")
    parser.add_argument("--port", type=int, default=config.monitor_port)
    parser.add_argument(
        "--host", type=str, default=config.monitor_host,
        help="Defaults to 127.0.0.1. The stream shows identifiable faces -- only "
             "expose it beyond localhost behind an authenticating proxy.",
    )
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1") and not config.monitor_token:
        logger.warning(
            "Binding to %s with no MONITOR_TOKEN set: the live CCTV view and the "
            "write endpoints are open to anyone who can reach this port.", args.host,
        )

    print("=" * 60)
    print(f"📺 CCTV audit monitor: http://{args.host}:{args.port}/")
    print("=" * 60)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
