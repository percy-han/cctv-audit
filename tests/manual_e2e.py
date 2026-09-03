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

"""Manual end-to-end run: real browser, real capture, real Vertex AI.

Not part of `pytest` -- it costs money and needs credentials, a network, and a
working Chromium. It exists to check the things unit tests structurally cannot:
that a page really plays, that clips really come out, and above all that the
input-token count per window stays flat as the run gets longer.

    .venv/bin/python tests/manual_e2e.py <url> [--windows 3] [--mode auto]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--window-seconds", type=int, default=15)
    parser.add_argument("--mode", default="auto", choices=["auto", "stream", "screen"])
    parser.add_argument("--start", type=float, default=0.0)
    # Without a duration the run has nothing to fall short of, so the coverage
    # reporting never gets exercised.
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--wall-clock", type=int, default=300)
    args = parser.parse_args()

    # config is a frozen dataclass read from the environment at import time, so
    # these have to be set before the package is imported.
    os.environ["CAPTURE_MODE"] = args.mode
    os.environ["WINDOW_SECONDS"] = str(args.window_seconds)
    os.environ["MAX_WINDOWS"] = str(args.windows)
    os.environ["MAX_WALL_CLOCK_SECONDS"] = str(args.wall_clock)
    os.environ.setdefault("ANALYSIS_CONCURRENCY", "2")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-28s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("cctv_audit").setLevel(logging.DEBUG)

    from computer_use_agent.config import config
    from computer_use_agent.pipeline import AuditPipeline, AuditRequest
    from computer_use_agent.store import AuditStore

    problems = config.validate()
    if problems:
        for problem in problems:
            print(f"config: {problem}", file=sys.stderr)
        return 2

    store = AuditStore()
    token_log: list[tuple[int, int]] = []
    original_record = store.record

    async def record(outcome):
        token_log.append((outcome.clip.index, outcome.input_tokens))
        return await original_record(outcome)

    store.record = record

    def on_status(event, payload):
        print(f"  [{event}] {payload}")

    pipeline = AuditPipeline(store=store, on_status=on_status)
    summary = asyncio.run(pipeline.run(AuditRequest(
        target=args.url, start_seconds=args.start, duration_seconds=args.duration,
    )))

    print("\n" + "=" * 70)
    print("SUMMARY")
    for key, value in summary.items():
        print(f"  {key:24} {value}")

    print("\nINPUT TOKENS PER WINDOW  (the core claim: this must not trend up)")
    for index, tokens in token_log:
        print(f"  window {index:3}  {tokens:>8} input tokens")
    if len(token_log) >= 2:
        counts = [t for _, t in token_log]
        spread = max(counts) - min(counts)
        print(f"  min {min(counts)}  max {max(counts)}  spread {spread} "
              f"({spread / max(1, min(counts)):.1%} of the smallest)")

    # The report is what the user actually reads, so print the real thing
    # rather than a paraphrase of the summary dict.
    from computer_use_agent.agent import CctvAuditAgent

    print("\n" + "=" * 70)
    print("REPORT AS THE USER SEES IT\n")
    print(CctvAuditAgent._report(summary, store))

    violations = store.violations()
    print(f"\nVIOLATIONS: {len(violations)}")
    for record_row in violations:
        for finding in record_row["findings"]:
            if finding["status"] == "VIOLATION":
                print(f"  [{finding['timestamp']}] {finding['rule_id']} "
                      f"({finding['severity']}) -> {finding['evidence_frame']}")
                print(f"      {finding['evidence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
