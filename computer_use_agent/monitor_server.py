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

"""Real-time Browser Use Live Monitor Server with WebSockets and Action Radar HUD."""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("browser_monitor_server")

app = FastAPI(title="Browser Use Live Monitor Server")

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "audit_records.jsonl")

# Global monitor state
state: Dict[str, Any] = {
    "status": "IDLE",
    "prompt": "",
    "turn": 0,
    "max_turns": 100000,
    "current_url": "about:blank",
    "current_reasoning": "",
    "last_action": "",
    "last_action_args": {},
    "click_target": None,
    "latest_frame_b64": None,
    "timeline": [],
    "final_result": "",
    "video_segments": [],
    "sop_violations": [],
}

def load_checkpoints():
    """Loads historical video segments and violations from local checkpoint file on startup."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            loaded_count = 0
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        state["video_segments"].append(record)
                        if record.get("sop_status") == "VIOLATION":
                            state["sop_violations"].append(record)
                        loaded_count += 1
            logger.info("Loaded %d historical segments (%d violations) from checkpoint %s",
                        loaded_count, len(state["sop_violations"]), CHECKPOINT_FILE)
        except Exception as e:
            logger.warning("Failed to load checkpoints from %s: %s", CHECKPOINT_FILE, e)

load_checkpoints()

active_websockets: Set[WebSocket] = set()


@app.get("/", response_class=HTMLResponse)
async def get_monitor_dashboard():
    # Import HTML from monitor module or return template
    from .monitor import HTML_PAGE
    return HTMLResponse(content=HTML_PAGE)


@app.get("/api/state")
async def get_state():
    return {k: v for k, v in state.items() if k != "latest_frame_b64"}


@app.post("/api/event")
async def post_event(request: Request):
    global state
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    event_type = payload.get("type")

    if event_type == "state":
        data = payload.get("data", {})
        state.update(data)
    elif event_type == "frame":
        state["latest_frame_b64"] = payload.get("frame")
        if "url" in payload:
            state["current_url"] = payload["url"]
    elif event_type == "action":
        action = payload.get("action", {})
        state["last_action"] = action.get("action", "")
        state["last_action_args"] = action.get("args", {})
        state["click_target"] = payload.get("click_target")
        if "url" in action and action["url"]:
            state["current_url"] = action["url"]
        if action:
            state["timeline"].append(action)
            if len(state["timeline"]) > 50:
                state["timeline"] = state["timeline"][-50:]
    elif event_type == "segment":
        seg = payload.get("segment", {})
        if seg:
            # Check duplicate by time_range or id
            exists = any(s.get("time_range") == seg.get("time_range") for s in state["video_segments"])
            if not exists:
                state["video_segments"].append(seg)
                if seg.get("sop_status") == "VIOLATION":
                    state["sop_violations"].append(seg)
                # Persist to local checkpoint file for 24h disaster recovery
                try:
                    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(seg, ensure_ascii=False) + "\n")
                except Exception as save_err:
                    logger.warning("Failed to persist segment to checkpoint: %s", save_err)

    # Broadcast to all connected WebSocket clients
    if active_websockets:
        msg_str = json.dumps(payload)
        disconnected = set()
        for ws in list(active_websockets):
            try:
                await ws.send_text(msg_str)
            except Exception:
                disconnected.add(ws)
        active_websockets.difference_update(disconnected)

    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        # Send current state & frame immediately
        init_state = {k: v for k, v in state.items() if k != "latest_frame_b64"}
        await websocket.send_text(json.dumps({"type": "state", "data": init_state}))
        if state.get("latest_frame_b64"):
            await websocket.send_text(json.dumps({
                "type": "frame",
                "frame": state["latest_frame_b64"],
                "url": state["current_url"],
            }))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.discard(websocket)
    except Exception:
        active_websockets.discard(websocket)


def main():
    parser = argparse.ArgumentParser(description="Start Browser Use Live Monitor Server")
    parser.add_argument("--port", type=int, default=int(os.getenv("MONITOR_PORT", "8080")))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"============================================================")
    print(f"📺 Browser Use Live Monitor running on http://127.0.0.1:{args.port}/")
    print(f"============================================================")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
