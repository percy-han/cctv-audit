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

"""One real audit against the deployed engine, start to finish.

    .venv/bin/python deploy/agent_runtime/run_audit.py <url> [--start 300] \\
        [--duration 120] [--sop chagee-store-v1] [--engine <resource-name>]

This drives the three `:query` methods the way GE's turn router drives them --
preflight, then start_audit, then poll get_status -- and prints the numbers
Phase 2 has to produce: capture mode, window count, failures, violations,
tokens per window, elapsed.

Why a script and not `curl`: the interesting part is the *poll*, and a run
takes minutes. Doing that by hand invites the two mistakes that make a result
useless -- giving up before the job finishes and calling it a hang, and
reading a stale status as the final one because `done` was never checked.

It talks to `:query`, not to GE. That is deliberate for a measurement: `:query`
holds a connection for 896s and GE's stream is cut at 602s, so a real audit
cannot be watched from the GE side at all. The container does the same work
either way -- GE's turn router calls these same three methods -- so this
measures the pipeline without the platform's stopwatch on top of it.
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

DEFAULT_ENGINE = (
    "projects/596821501265/locations/us-central1/reasoningEngines/"
    "6844158066963775488"
)
HOST = "us-central1-aiplatform.googleapis.com"


def _call(engine: str, method: str, payload: dict) -> dict:
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    body = json.dumps({"classMethod": method, "input": payload}).encode()
    request = urllib.request.Request(
        f"https://{HOST}/v1/{engine}:query", data=body, method="POST")
    request.add_header("Authorization", f"Bearer {creds.token}")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            raw = json.loads(response.read() or "{}")
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}\n{exc.read().decode(errors='replace')}",
              file=sys.stderr)
        raise SystemExit(1)
    # `:query` wraps the method's return value in `output`.
    return raw.get("output", raw)


def _line(label: str, value) -> None:
    print(f"  {label:<22} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None,
                        help="seconds of footage to audit, from --start")
    parser.add_argument("--sop", default="chagee-store-v1")
    parser.add_argument("--user", default="e2e@percyhan.altostrat.com")
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--give-up-after", type=float, default=2400.0)
    args = parser.parse_args()

    began = time.time()

    print(f"== preflight  {args.url}")
    payload = {"user_id": args.user, "target": args.url,
               "start": args.start, "sop_id": args.sop}
    if args.duration is not None:
        payload["duration"] = args.duration
    job = _call(args.engine, "preflight", payload)
    print(json.dumps(job, indent=2, ensure_ascii=False))

    # Every method returns the whole job document, so the state field is the
    # verdict: `ready` means preflight opened the page and found something
    # auditable, `rejected` means it opened it and there is nothing to audit.
    pre = job.get("preflight") or {}
    job_id = job.get("job_id")
    if job.get("state") != "ready" or not job_id:
        # Not a failure of this script. A refusal here is preflight doing its
        # job, and the reason it gives is the whole output.
        print("\npreflight said no; nothing was started.")
        return 1

    print(f"\n== start_audit  {job_id}")
    started_at = time.time()
    started = _call(args.engine, "start_audit",
                    {"user_id": args.user, "job_id": job_id})
    # This number is the one that decides whether GE can be the front door at
    # all: the turn has to end long before 602s.
    print(json.dumps(started, indent=2, ensure_ascii=False))
    print(f"  returned in {time.time() - started_at:.2f}s")

    print("\n== polling get_status")
    last = ""
    status: dict = {}
    while time.time() - began < args.give_up_after:
        time.sleep(args.poll_seconds)
        status = _call(args.engine, "get_status",
                       {"user_id": args.user, "job_id": job_id})
        state = status.get("state", "?")
        progress = status.get("progress") or {}
        shown = (f"{state} "
                 f"windows={progress.get('windows_analyzed', '-')} "
                 f"violations={progress.get('violations_so_far', '-')} "
                 f"at={progress.get('last_window_returned', '-')}")
        if shown != last:
            print(f"  [{time.time() - began:7.1f}s] {shown}")
            last = shown
        if state in ("done", "failed", "cancelled"):
            break
    else:
        print(f"  gave up after {args.give_up_after:.0f}s; job may still be running")
        return 2

    print("\n== result")
    _line("state", status.get("state"))
    _line("capture", f"{pre.get('capture_mode')} -- {pre.get('capture_reason')}")
    _line("elapsed", f"{time.time() - began:.1f}s")
    summary = (status.get("result") or {}).get("summary") or {}
    for key in ("capture_mode", "windows_analyzed", "windows_failed",
                "violations", "red_line_violations", "covered_from_seconds",
                "covered_to_seconds", "complete", "incomplete_reason",
                "stopped_because", "elapsed_seconds",
                "input_tokens", "output_tokens"):
        if key in summary:
            _line(key, summary[key])
    # The number the plan asks for by name. It is supposed to stay flat as a
    # run gets longer -- a rising figure means the analyzer is carrying history
    # it should not be.
    if summary.get("windows_analyzed"):
        _line("input tokens/window",
              f"{summary.get('input_tokens', 0) / summary['windows_analyzed']:.0f}")
    report = (status.get("result") or {}).get("report")
    if report:
        print("\n" + report)
    if status.get("error"):
        print("\nerror: " + status["error"])
    return 0 if status.get("state") == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
