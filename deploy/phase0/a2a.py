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

"""Talks to one Gemini Enterprise agent over A2A, and times it.

This is the sibling of `assist.py`, and the difference between them is a Phase 0
finding in itself:

  * `assist.py` calls v1alpha `assistants:streamAssist` with an `agentsSpec`.
    That is what the console's chat box calls -- but `agentsSpec` does **not**
    pin routing. Tested twice, including with a query written to match the
    agent description; the stock assistant answered both times and our
    container was never touched.
  * this script calls v1 `.../agents/{id}/a2a/v1/message:send`, which addresses
    one agent by resource name. That one lands in our container, every time.

So this is the instrument for the two Phase 0 questions `assist.py` could not
reach:

  * **mid-flow confirmation.** `A2aV1Message.contextId` is the conversation
    handle. Turn one leaves it off and prints what came back; turn two passes
    it in. If the probe's second turn can still find what the first turn stored,
    GE can hold a "ask the user, then continue" flow -- and the audit can
    confirm before it runs.
  * **the GE-side timeout**, which is a different number from the Agent Runtime
    one and the one that actually decides whether the audit can be synchronous.
    `hang N` and read the elapsed column.

    python deploy/phase0/a2a.py "稽核 XX 店 昨天 14:00-15:00"
    python deploy/phase0/a2a.py --context <contextId> "确认"
    python deploy/phase0/a2a.py --stream "挂 900 秒"

Every response is written to results/ as raw JSON, because the interesting
fields (taskId, task state, diagnosticInfo) are ones we are still discovering
and a summary would throw away the parts we do not yet know matter.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

import google.auth
import google.auth.transport.requests

PROJECT = "study-project-496907"
HOST = "discoveryengine.googleapis.com"
APP = (
    f"projects/{PROJECT}/locations/global/collections/default_collection"
    f"/engines/cctv-audit/assistants/default_assistant"
)
DEFAULT_AGENT = "16091433261218097511"

RESULTS = pathlib.Path(__file__).parent / "results"


def token() -> str:
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def post(url: str, body: dict, timeout: float) -> tuple:
    """Returns (elapsed, status, raw_text). Never raises on an HTTP error.

    A 4xx/5xx after fourteen minutes is a *result* -- it is the number we came
    for. Turning it into a traceback would lose the elapsed time, which is the
    only part that matters.
    """
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST"
    )
    request.add_header("Authorization", f"Bearer {token()}")
    request.add_header("Content-Type", "application/json")
    request.add_header("X-Goog-User-Project", PROJECT)

    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return time.monotonic() - start, response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return time.monotonic() - start, exc.code, exc.read().decode(errors="replace")
    except Exception as exc:  # noqa: BLE001 -- a cut is the measurement
        return time.monotonic() - start, 0, f"{type(exc).__name__}: {exc}"


def summarise(payload: dict) -> None:
    """Prints the handful of fields a human needs; the file keeps the rest."""
    message = payload.get("message") or {}
    task = payload.get("task") or {}

    if task:
        status = task.get("status") or {}
        print(f"  [task] {task.get('id')}  state={status.get('state')}")
        message = status.get("message") or message

    for part in message.get("content") or []:
        if part.get("text"):
            print(f"  {part['text']}")

    metadata = message.get("metadata") or {}
    agent = (metadata.get("agentInfo") or {}).get("displayName")
    if agent:
        print(f"  [agent] {agent}")
    answer = metadata.get("answer") or {}
    if answer.get("state"):
        print(f"  [state] {answer['state']}")

    context = message.get("contextId") or task.get("contextId")
    if context:
        print(f"\ncontextId: {context}")
        print(f"next turn:\n  python {sys.argv[0]} --context {context} '确认'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--context", default="", help="contextId from a prior turn")
    parser.add_argument("--task", default="", help="taskId from a prior turn")
    parser.add_argument("--stream", action="store_true", help="use message:stream")
    parser.add_argument("--tag", default="", help="filename for the saved response")
    # Generous on purpose: GE must be the one that gives up, never urllib. A
    # short client timeout would produce a number that describes this script.
    parser.add_argument("--timeout", type=float, default=4200)
    args = parser.parse_args()

    verb = "stream" if args.stream else "send"
    url = f"https://{HOST}/v1/{APP}/agents/{args.agent}/a2a/v1/message:{verb}"

    message: dict = {
        # A2A wants a client-supplied id per message. Time-based so two turns of
        # the same conversation never collide.
        "messageId": f"probe-{int(time.time())}",
        "role": "ROLE_USER",
        "content": [{"text": args.text}],
    }
    if args.context:
        message["contextId"] = args.context
    if args.task:
        message["taskId"] = args.task
    # `request` *is* the message -- not `{"request": {"message": ...}}`, which
    # the server rejects with `Unknown name "message" at 'request'`. The A2A
    # spec nests it; this REST binding flattens it.
    body = {"request": message}

    print(f"POST {url}")
    print(json.dumps(body, indent=2, ensure_ascii=False))
    print()

    elapsed, status, raw = post(url, body, args.timeout)
    print(f"  HTTP {status} after {elapsed:.1f}s, {len(raw)} bytes\n")

    tag = args.tag or f"a2a-{int(time.time())}"
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{tag}.json"
    path.write_text(raw)

    try:
        payload = json.loads(raw)
    except Exception:
        print(raw[:2000])
        print(f"\nsaved: {path}")
        return

    # message:stream answers with a JSON array of chunks; message:send with one
    # object. Flatten so the rest does not care which was used.
    for item in payload if isinstance(payload, list) else [payload]:
        summarise(item.get("result") if "result" in item else item)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
