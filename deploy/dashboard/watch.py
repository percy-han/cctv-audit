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

"""A viewer for the dashboard, with no browser attached.

    .venv/bin/python deploy/dashboard/watch.py <job-id> [--seconds 600]

It opens the same websocket the page opens, joins the same room, and reports
what arrives: frames, frame rate, kilobytes, and every state change. Nothing
is displayed.

Why this exists. The dev VM cannot open the dashboard -- `gcloud run services
proxy` needs a component that will not install here -- so every claim about the
live view has so far rested on the page being opened on somebody's laptop.
Worse, the interesting behaviour only happens *when* somebody is looking:
frames are gated on `has_viewers`, and the last three bugs that broke a demo
were all in that path. An end-to-end run with nobody watching exercises the one
case the demo will never be in.

So this stands in for the human. To the server it is indistinguishable from the
page -- same URL, same room, same identity token -- which is the point: what it
measures is what the page would have shown.

It is also the quickest answer to "is the dashboard actually receiving
anything", which is otherwise a question you can only ask by asking a person to
look at a screen and describe it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import google.auth.transport.requests
import websockets
from google.oauth2 import id_token as id_token_lib

DEFAULT_MONITOR = os.environ.get(
    "MONITOR_URL", "https://cctv-monitor-c72222uunq-uc.a.run.app")


def identity_token(base_url: str) -> str:
    """A Google ID token for the dashboard.

    The audience is the service URL because the service is guarded by Cloud Run
    IAM. If IAP is ever put in front of it the audience becomes the IAP OAuth
    client id instead -- a different string, and the failure when it is wrong
    says `Invalid JWT audience` rather than anything about audiences being
    configurable.
    """
    return id_token_lib.fetch_id_token(
        google.auth.transport.requests.Request(), base_url)


async def watch(base_url: str, job_id: str, seconds: float) -> int:
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    query = f"?job={job_id}" if job_id else ""
    url = f"{ws_url}/ws{query}"

    headers = {"Authorization": f"Bearer {identity_token(base_url)}"}
    print(f"connecting  {url}")

    frames = 0
    frame_bytes = 0
    first_frame = 0.0
    # An average frame rate hides the thing operators actually complain about.
    # "Twelve a second" and "twenty-four in one second, then nothing for two"
    # are the same average and only one of them is watchable, so the gaps
    # between arrivals get measured too.
    gaps: list = []
    previous_frame = 0.0
    last_report = time.time()
    began = time.time()
    finished = False

    async with websockets.connect(url, additional_headers=headers,
                                  max_size=None, open_timeout=30) as socket:
        print(f"connected   room={job_id or '(default)'}\n")
        while time.time() - began < seconds:
            try:
                raw = await asyncio.wait_for(socket.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                # Silence is a legitimate reading, not an error: it is exactly
                # what a gated stream looks like when the audit is between
                # windows. Report it and keep waiting.
                pass
            else:
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "frame":
                    frames += 1
                    # base64, so this is the wire size, which is the number
                    # that costs money -- not the decoded JPEG size.
                    frame_bytes += len(message.get("frame") or "")
                    arrived = time.time()
                    if not first_frame:
                        first_frame = arrived
                        print(f"[{first_frame - began:6.1f}s] first frame")
                    else:
                        gaps.append(arrived - previous_frame)
                    previous_frame = arrived
                elif kind == "state":
                    state = message.get("data") or {}
                    status = state.get("status")
                    if status:
                        print(f"[{time.time() - began:6.1f}s] state: {status}")
                    if status in ("COMPLETED", "FAILED", "ERROR"):
                        finished = True
                elif kind == "segment":
                    seg = message.get("segment") or {}
                    print(f"[{time.time() - began:6.1f}s] window "
                          f"{seg.get('time_range', '?')} -> {seg.get('sop_status', '?')}")
                else:
                    detail = json.dumps(message, ensure_ascii=False)[:160]
                    print(f"[{time.time() - began:6.1f}s] {kind}: {detail}")

            now = time.time()
            if now - last_report >= 15.0:
                window = now - last_report
                print(f"[{now - began:6.1f}s] frames={frames} "
                      f"({frames / max(now - (first_frame or now), 1e-9):.1f}/s "
                      f"since first) {frame_bytes / 1024:.0f} KB total")
                last_report = now
            if finished:
                break

    elapsed = time.time() - began
    streaming = (time.time() - first_frame) if first_frame else 0.0
    print("\n== viewer summary")
    print(f"  connected for      {elapsed:.1f}s")
    print(f"  frames             {frames}")
    if frames:
        print(f"  average frame      {frame_bytes / frames / 1024:.1f} KB")
        print(f"  rate while live    {frames / max(streaming, 1e-9):.1f} fps")
        print(f"  total received     {frame_bytes / 1024 / 1024:.2f} MB")
    if gaps:
        ordered = sorted(gaps)
        p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
        stalls = sum(1 for gap in ordered if gap > 1.0)
        print(f"  gap between frames median {ordered[len(ordered) // 2] * 1000:.0f}ms "
              f"p90 {p90 * 1000:.0f}ms worst {ordered[-1] * 1000:.0f}ms")
        # The old symptom in one number: a frozen picture the operator waits on.
        print(f"  freezes over 1s    {stalls}")
    else:
        # Said plainly, because a viewer that saw nothing is the failure this
        # script was written to detect, and a zero is easy to read past.
        print("  NOTHING ARRIVED -- the page would have shown a blank screen")
    return 0 if frames else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", nargs="?", default="",
                        help="the room to join; omit for the unscoped room")
    parser.add_argument("--monitor", default=DEFAULT_MONITOR)
    parser.add_argument("--seconds", type=float, default=900.0)
    args = parser.parse_args()
    return asyncio.run(watch(args.monitor.rstrip("/"), args.job_id, args.seconds))


if __name__ == "__main__":
    sys.exit(main())
