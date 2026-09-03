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

"""Talks to the Gemini Enterprise app the way the chat box does.

Phase 0 needs two things the console cannot give us: a reproducible transcript,
and a stopwatch. `assistants.streamAssist` is what the GE UI calls underneath,
so driving it from here exercises the identical path -- GE decides which agent
to route to, calls our reasoningEngine as a tool, and streams the answer back.

That makes it the instrument for two of the four Phase 0 questions:

  * "can GE mount an agent that lives on Agent Runtime" -- registration alone
    proves nothing, because `agents.create` happily accepts a resource name of
    all zeros. Only a round trip that comes back with the container's own
    instance id proves the link is live.
  * "how long will a GE session wait" -- a separate number from the Agent
    Runtime request timeout, and the one that decides whether the audit can be
    synchronous. Ask the agent to hang, and read the elapsed column.

Multi-turn (question 3, mid-flow confirmation) works by passing the session
name back in: the first call leaves `--session` off, prints the session it was
given, and the second call reuses it.

    python deploy/phase0/assist.py "探针自检：说一声 hello"
    python deploy/phase0/assist.py --session <name> "确认"
    python deploy/phase0/assist.py --agent <id> "挂 900 秒不返回"

Timings are printed per chunk, because *when* the stream goes quiet is the
measurement -- a total duration cannot tell a slow answer apart from a stall
followed by a cut.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

import google.auth
import google.auth.transport.requests

PROJECT = "study-project-496907"
LOCATION = "global"
COLLECTION = "default_collection"
APP_ID = "cctv-audit"
ASSISTANT = "default_assistant"

HOST = "discoveryengine.googleapis.com"
ENGINE = (
    f"projects/{PROJECT}/locations/{LOCATION}/collections/{COLLECTION}"
    f"/engines/{APP_ID}"
)
ASSISTANT_NAME = f"{ENGINE}/assistants/{ASSISTANT}"


def token() -> str:
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def stream(body: dict, timeout: float):
    """Yield (elapsed, raw_bytes) for each chunk the server flushes.

    Deliberately no incremental JSON parsing. Google's REST streaming sends one
    big JSON array and splits it wherever it likes, so a chunk is rarely a
    whole object -- but the *arrival time* of each chunk is exactly what we are
    here to measure, and that would be lost by buffering to the end.
    """
    url = f"https://{HOST}/v1alpha/{ASSISTANT_NAME}:streamAssist"
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    request.add_header("Authorization", f"Bearer {token()}")
    request.add_header("Content-Type", "application/json")
    request.add_header("X-Goog-User-Project", PROJECT)

    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        while True:
            chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
            if not chunk:
                return
            yield time.monotonic() - start, chunk


def render(payload: list) -> str | None:
    """Pull the human-readable text and the session name out of the array."""
    session = None
    for item in payload:
        info = item.get("sessionInfo") or {}
        if info.get("session"):
            session = info["session"]
        for tool in item.get("invocationTools") or []:
            print(f"  [tool] {tool}")
        info = item.get("agentInfo") or {}
        if info.get("displayName"):
            print(f"  [agent] {info['displayName']}")
        answer = item.get("answer") or {}
        for reply in answer.get("replies") or []:
            # `content.text` is a plain string here, not the `parts` list the
            # Query side uses. Different shape on the way out than on the way
            # in.
            content = (reply.get("groundedContent") or {}).get("content") or {}
            if content.get("text"):
                print(f"  {content['text']}")
        for reason in answer.get("assistSkippedReasons") or []:
            print(f"  [skipped] {reason}")
    return session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--session", default="")
    parser.add_argument("--agent", default="")
    # Generous by default: the point of the exercise is to let *GE* be the one
    # that gives up, never this client. A short client timeout would produce a
    # number that describes urllib.
    parser.add_argument("--timeout", type=float, default=4200)
    args = parser.parse_args()

    body: dict = {"query": {"text": args.text}}
    # An empty/`-` session makes the server mint one; we print it so the next
    # turn can carry the history forward.
    body["session"] = args.session or f"{ENGINE}/sessions/-"
    if args.agent:
        body["agentsSpec"] = {"agentSpecs": [{"agentId": args.agent}]}

    print(f"POST {ASSISTANT_NAME}:streamAssist")
    print(json.dumps(body, indent=2, ensure_ascii=False))
    print()

    buffer = bytearray()
    last = 0.0
    try:
        for elapsed, chunk in stream(body, args.timeout):
            buffer += chunk
            print(f"  [{elapsed:8.1f}s  +{elapsed - last:6.1f}s] {len(chunk)} bytes")
            last = elapsed
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}\n{exc.read().decode(errors='replace')}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001 -- a cut mid-stream is a result
        print(f"\n  stream died after {last:.1f}s: {type(exc).__name__}: {exc}")

    print(f"\n  total {last:.1f}s, {len(buffer)} bytes\n")
    try:
        payload = json.loads(buffer.decode())
    except Exception:
        print(buffer.decode(errors="replace")[-4000:])
        return
    session = render(payload)
    print(f"\nsession: {session}")
    print(f"next turn:\n  python {sys.argv[0]} --session {session} '确认'")


if __name__ == "__main__":
    main()
