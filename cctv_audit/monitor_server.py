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
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .config import config
from .logsetup import setup_logging

# Was a bare `basicConfig` to stderr. Same intent, but shared with the agent
# container so both halves of one image log the same way -- and on stdout, so
# Cloud Run stops filing every INFO line as an error.
setup_logging()
logger = logging.getLogger("cctv_audit.monitor_server")

app = FastAPI(title="CCTV Audit Live Monitor")


def _blank_state() -> Dict[str, Any]:
    return {
        "status": "IDLE",
        "prompt": "",
        "capture_settings": "",
        # Which models this run is on. Empty until a run says so, and the page
        # then shows the platform name alone rather than the model name of
        # whatever ran last -- an idle dashboard naming a model is exactly the
        # lie the hardcoded footer used to tell.
        "engine": "",
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
        # When the run stopped, epoch seconds, or None while it is still going.
        # Stamped here rather than sent by the container so it cannot disagree
        # with the clock the page formats it against.
        "finished_at": None,
    }


class Room:
    """One audit's picture and the people watching it.

    Rooms exist because this is a single service that several audits share.
    Frames arrive at `/api/event` from the container running the audit and
    leave over the WebSockets of whoever opened that audit's link; with one
    global state dict, two audits running at once overwrite each other's
    picture and every viewer sees whichever frame landed last.

    The empty key is the local case -- `adk web`, one audit, one dashboard,
    nobody passing a job id -- and behaves exactly as the whole server did
    before rooms existed.
    """

    __slots__ = ("job_id", "state", "sockets", "touched")

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.state = _blank_state()
        self.sockets: Set[WebSocket] = set()
        self.touched = time.time()


# Enough for far more concurrent audits than one instance can actually run,
# small enough that a stream of unknown job ids cannot grow this without
# bound -- `/api/event` is authenticated, but a bug upstream that stamped a
# fresh id on every frame would otherwise be a slow memory leak.
_MAX_ROOMS = 64

_rooms: Dict[str, Room] = {"": Room("")}

# The local room, and the one every pre-rooms caller still lands in.
state: Dict[str, Any] = _rooms[""].state
active_websockets: Set[WebSocket] = _rooms[""].sockets


def _room(job_id: Optional[str], *, create: bool = True) -> Room:
    key = (job_id or "").strip()
    room = _rooms.get(key)
    if room is None:
        if not create:
            return _rooms[""]
        _evict_rooms()
        room = _rooms[key] = Room(key)
        logger.info("Dashboard room opened for job %s (%d open)", key, len(_rooms))
    room.touched = time.time()
    return room


def _evict_rooms() -> None:
    """Drops the least recently touched empty rooms once there are too many.

    Only rooms nobody is watching, and never the local one: an audit whose
    viewer walked away still has frames coming, and evicting it mid-run would
    show the next person to open the link a dashboard that had never started.
    """
    while len(_rooms) >= _MAX_ROOMS:
        idle = [r for r in _rooms.values() if r.job_id and not r.sockets]
        if not idle:
            return
        oldest = min(idle, key=lambda r: r.touched)
        _rooms.pop(oldest.job_id, None)
        logger.info("Dashboard room %s evicted (idle, %d rooms)", oldest.job_id, len(_rooms))

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


async def _broadcast(room: Room, payload: Dict[str, Any]) -> None:
    """Sends to this room only.

    Not "this room plus everyone else, just in case" -- a live CCTV frame is
    footage of identifiable staff and customers, and the room is who is
    entitled to see it. A fan-out that treats frames as a special case worth
    delivering everywhere is how two audits end up cross-streaming.
    """
    if not room.sockets:
        return
    message = json.dumps(payload, ensure_ascii=False)
    dead = set()
    for websocket in list(room.sockets):
        try:
            await websocket.send_text(message)
        except Exception:
            dead.add(websocket)
    room.sockets.difference_update(dead)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    from .monitor import HTML_PAGE
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/state")
async def get_state(job: str = ""):
    room = _room(job, create=False)
    return {k: v for k, v in room.state.items() if k != "latest_frame_b64"}


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


@app.get("/api/viewers")
async def get_viewers(request: Request, job: str = ""):
    """How many people have *this audit's* page open.

    The audit asks before it streams anything: preview frames are the only
    high-rate traffic in the system (every viewer is a separate stream), and
    most audits run with nobody watching. Answering 0 turns that bandwidth off
    entirely.

    Scoped to the room, because the alternative is that someone watching a
    different audit keeps this one streaming frames to nobody.

    Token-checked like the rest: the count says whether an audit is being
    observed right now, which is not a stranger's business.
    """
    _check_token(request)
    return {"viewers": len(_room(job, create=False).sockets)}


@app.post("/api/event")
async def post_event(request: Request):
    _check_token(request)
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    event_type = payload.get("type")
    room = _room(payload.get("job_id"))
    room_state = room.state

    if event_type == "state":
        was = room_state.get("status")
        room_state.update(payload.get("data", {}))
        now = room_state.get("status")
        # Live frames stop the moment a run ends, and nothing replays them: open
        # the link afterwards and the video panel is blank with no explanation.
        # Job 781c96 was read as "the dashboard broke" for exactly this reason --
        # the WebSocket joined 55s after the audit had finished. Stamping the
        # moment lets the page say so instead of showing an empty box.
        if now in ("COMPLETED", "ERROR") and was not in ("COMPLETED", "ERROR"):
            room_state["finished_at"] = time.time()
        elif now == "RUNNING":
            # A room is reused when the same job id starts again locally.
            room_state["finished_at"] = None
    elif event_type == "frame":
        room_state["latest_frame_b64"] = payload.get("frame")
        if payload.get("url"):
            room_state["current_url"] = payload["url"]
    elif event_type == "action":
        action = payload.get("action", {})
        room_state["last_action"] = action.get("action", "")
        room_state["last_action_args"] = action.get("args", {})
        room_state["click_target"] = payload.get("click_target")
        if action.get("url"):
            room_state["current_url"] = action["url"]
        if action:
            room_state["timeline"] = (room_state["timeline"] + [action])[-50:]
    elif event_type == "segment":
        segment = payload.get("segment") or {}
        if segment:
            # Dedup by id when present, else by time range. No file write here:
            # AuditStore already persisted this record before it was pushed.
            key = segment.get("id")
            existing = next(
                (s for s in room_state["video_segments"]
                 if (key is not None and s.get("id") == key)
                 or (key is None and s.get("time_range") == segment.get("time_range"))),
                None,
            )
            if existing is None:
                room_state["video_segments"].append(segment)
                if segment.get("sop_status") == "VIOLATION":
                    room_state["sop_violations"].append(segment)
                # Windows arrive in completion order, not video order: with
                # ANALYSIS_CONCURRENCY > 1 window 2 routinely lands before
                # window 1. Insert-sort here so the sidebar always reads as a
                # timeline, and late arrivals slot into place instead of
                # appending to the bottom.
                for key in ("video_segments", "sop_violations"):
                    room_state[key].sort(key=_segment_order)
            else:
                existing.update(segment)
    elif event_type == "intervention":
        room_state["intervention"] = payload.get("reason")
    elif event_type == "intervention_cleared":
        room_state["intervention"] = None

    await _broadcast(room, payload)
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

    room = _room(payload.get("job_id"), create=False)
    room.state["intervention"] = None
    if _human_gate is not None:
        _human_gate.resolve()
        logger.info("Operator cleared the intervention gate.")
    else:
        # The usual case: the pipeline runs in the ADK process, so there is no
        # in-process gate to poke. Clearing `state["intervention"]` above is the
        # signal -- HumanGate polls /api/state and resumes when the banner goes.
        logger.info("Intervention cleared; the pipeline will pick it up from /api/state.")
    await _broadcast(room, {"type": "intervention_cleared"})
    return {"status": "ok", "gate_attached": _human_gate is not None}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, job: str = ""):
    await websocket.accept()
    # Created on join, not only on the first frame: the link goes out to the
    # customer the moment preflight returns, so people routinely open the page
    # before the audit they were sent has posted anything.
    room = _room(job)
    room.sockets.add(websocket)
    room_state = room.state
    try:
        await websocket.send_text(json.dumps(
            {"type": "state",
             "data": {k: v for k, v in room_state.items() if k != "latest_frame_b64"}},
            ensure_ascii=False,
        ))
        if room_state.get("latest_frame_b64"):
            await websocket.send_text(json.dumps({
                "type": "frame",
                "frame": room_state["latest_frame_b64"],
                "url": room_state["current_url"],
            }))
        if room_state.get("intervention"):
            await websocket.send_text(json.dumps(
                {"type": "intervention", "reason": room_state["intervention"]}, ensure_ascii=False))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WebSocket closed: %s", exc)
    finally:
        room.sockets.discard(websocket)
        room.touched = time.time()


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
