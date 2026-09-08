#!/usr/bin/env python3
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

"""Does incoming traffic un-starve the audit? One question, one experiment.

The dashboard has been a slideshow since the move to Agent Runtime, and three
rounds of instrumentation have now cornered the reason. The preview pump asks
for a frame every 83ms and gets one every 300-500ms, with freezes up to 15
seconds -- while the container burns 0.0 of its 3 cores at load 0.0. Nothing
is competing for the CPU. The CPU is simply not being given to us.

That is the documented behaviour of a Cloud Run instance outside of request
processing, and an audit runs in a task deliberately detached from the request
that started it -- the only shape that survives the 900s streaming cancel. So
the very decision that keeps the audit alive is what starves it.

This script tests that and nothing else. It holds one `get_status` request in
flight at all times for a fixed stretch of a live audit. One at a time, because
`container_concurrency` is 1 and a second would be routed to a second instance
that is not running the audit. If the reason is throttling, `preview pump` in
the container's logs climbs while this runs and falls back when it stops. If
the numbers do not move, the reason is something else and this file was worth
the ten minutes it cost to be sure.

    python deploy/agent_runtime/keepalive_probe_test.py <job_id> --seconds 60
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

import google.auth
import google.auth.transport.requests

ENGINE = ("projects/596821501265/locations/us-central1/reasoningEngines/"
          "6844158066963775488")
HOST = "us-central1-aiplatform.googleapis.com"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--engine", default=ENGINE)
    parser.add_argument("--user", default="e2e@percyhan.altostrat.com")
    args = parser.parse_args()

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"])
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)

    body = json.dumps({
        "classMethod": "get_status",
        "input": {"job_id": args.job_id, "user_id": args.user},
    }).encode()
    url = f"https://{HOST}/v1/{args.engine}:query"

    started = time.monotonic()
    calls = 0
    failures = 0
    busy = 0.0
    while time.monotonic() - started < args.seconds:
        if not credentials.valid:
            credentials.refresh(request)
        call = urllib.request.Request(url, data=body, method="POST")
        call.add_header("Authorization", f"Bearer {credentials.token}")
        call.add_header("Content-Type", "application/json")
        at = time.monotonic()
        try:
            urllib.request.urlopen(call, timeout=30).read()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            failures += 1
            if failures <= 3:
                print(f"  call failed: {str(exc)[:120]}")
        calls += 1
        busy += time.monotonic() - at

    window = time.monotonic() - started
    # The share of the window with a request in flight is the whole point: it
    # is the fraction of the time Cloud Run had to give the instance a CPU.
    print(f"held {calls} calls over {window:.0f}s "
          f"({failures} failed), in flight {busy / window * 100:.0f}% of the time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
