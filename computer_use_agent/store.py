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

"""The one and only place audit records are written.

There used to be two writers -- the monitor client and the monitor server --
both appending to the same JSONL, which is why the shipped records file reads
`1, 1, 2, 2, 3, 3`. Ids are now issued here, under a lock, and the dashboard is
a downstream consumer rather than a second author.

Each violation also gets a still frame pulled from the exact moment the model
cited, so a human reviewer can confirm or reject it without replaying the
footage. A finding with no reviewable evidence is not worth recording.

Where "on disk" points depends on where this runs. On a workstation the JSONL
and the frames are files and that is the end of it. On Agent Runtime the
filesystem is scratch space that goes away with the instance, so the frames go
through `artifact_sink()` to a bucket and each record is handed to
`persist_record` on its way past. The local JSONL is still written either way:
it costs nothing, and when something goes wrong on a container it is the only
thing left to read.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

from .analyzer.schema import Severity, Status, WindowResult
from .analyzer.sop import SopRuleSet, load_rules
from .analyzer.video_analyzer import AnalysisOutcome
from .artifacts import LocalArtifactSink, artifact_sink
from .capture.ffmpeg_util import extract_frame
from .capture.types import Clip
from .config import config

logger = logging.getLogger("cctv_audit.store")


@dataclass
class RunStats:
    windows_analyzed: int = 0
    windows_failed: int = 0
    violations: int = 0
    red_line_violations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "windows_analyzed": self.windows_analyzed,
            "windows_failed": self.windows_failed,
            "violations": self.violations,
            "red_line_violations": self.red_line_violations,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class AuditStore:
    """Append-only JSONL plus evidence frames on disk."""

    records_path: Path = field(default_factory=lambda: config.records_path)
    evidence_dir: Path = field(default_factory=lambda: config.evidence_dir)
    # Wall-clock time the footage at video offset 0 was captured. Set this when
    # auditing a recording so findings can be reported in store-local time
    # rather than "seconds into the file".
    recording_started_at: Optional[datetime] = None
    on_record: Optional[Callable[[dict], None]] = None
    rules: Optional[SopRuleSet] = None
    # Which audit these records belong to. Empty for a local one-off run; set
    # on Agent Runtime, where one bucket and one Firestore database hold every
    # customer's audits and `w00007_hygiene_0003.jpg` is not a unique name.
    job_id: str = ""
    # Called with each finished record, for storage that outlives the instance.
    # Separate from `on_record`, which is the dashboard: that one is a
    # fire-and-forget UI ping, this one is the record actually being kept.
    persist_record: Optional[Callable[[dict], Awaitable[None]]] = None
    # Where the records really end up, in words a person can act on. Set by
    # whoever wired `persist_record`, because only they know: the store knows
    # that a coroutine is called per record, not that it writes to Firestore.
    #
    # Empty means the JSONL on this disk is the record, which is true locally
    # and false on Agent Runtime -- where `records_path` is a temp file that
    # dies with the instance. A report that prints it is telling the customer
    # to go look at something that is already gone.
    records_location: str = ""

    def __post_init__(self) -> None:
        # One directory per job when there is one, so the path relative to
        # `DATA_DIR` -- which is also the object key in the bucket -- carries
        # the job id without anything having to splice it in later.
        if self.job_id:
            self.evidence_dir = self.evidence_dir / self.job_id
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._rules = self.rules or load_rules()
        # What this run produced, as opposed to what the file has accumulated.
        self._run_records: List[dict] = []
        self._lock = asyncio.Lock()
        self._next_id = self._highest_existing_id() + 1
        self.stats = RunStats()

    def _highest_existing_id(self) -> int:
        """Continues numbering across restarts instead of colliding with old rows."""
        highest = 0
        if not self.records_path.exists():
            return highest
        try:
            with self.records_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        highest = max(highest, int(json.loads(line).get("id", 0)))
                    except (ValueError, AttributeError):
                        continue
        except OSError as exc:
            logger.warning("Could not read existing records: %s", exc)
        return highest

    async def record(self, outcome: AnalysisOutcome) -> Optional[dict]:
        """Persists one analysed window. Returns the record, or None on failure."""
        clip = outcome.clip
        if not outcome.ok:
            self.stats.windows_failed += 1
            logger.warning("Window %d produced no result: %s", clip.index, outcome.error)
            return None

        result: WindowResult = outcome.result
        findings = [await self._render_finding(clip, f) for f in result.findings]
        violations = [f for f in findings if f["status"] == Status.VIOLATION.value]

        async with self._lock:
            record_id = self._next_id
            self._next_id += 1

            record = {
                "id": record_id,
                "window_index": clip.index,
                "time_range": clip.time_range,
                "start_offset_seconds": round(clip.start_offset, 3),
                "end_offset_seconds": round(clip.end_offset, 3),
                "wall_clock": self._wall_clock(clip.start_offset),
                "captured_at": datetime.fromtimestamp(clip.wall_clock_start, timezone.utc).isoformat(),
                "source_mode": clip.source_mode,
                "scene_summary": result.scene_summary,
                # Kept alongside the verdicts on purpose: a reviewer disputing
                # a finding needs to see what the model claimed to observe, not
                # just what it concluded.
                "visual_scan": [
                    {"subject": s.subject, "observation": s.observation, "timestamps": s.timestamps}
                    for s in result.visual_scan
                ],
                "action_narrative": result.action_narrative,
                "people_count": result.people_count,
                "visibility_ok": result.visibility_ok,
                # Which written standard produced these verdicts. Denormalised
                # onto every record on purpose: "was this judged under v2 or
                # v3" is the first question asked when a finding is disputed,
                # and by then the standard has usually moved on.
                "sop_id": self._rules.sop_id,
                "sop_version": self._rules.version,
                "sop_status": self._window_status(result),
                "severity": result.worst_severity().value,
                "findings": findings,
                "violation_count": len(violations),
                "analysis": {
                    "model": config.analysis_model,
                    "media_resolution": config.media_resolution,
                    "analysis_fps": config.analysis_fps,
                    "input_tokens": outcome.input_tokens,
                    "output_tokens": outcome.output_tokens,
                    "latency_seconds": round(outcome.latency_seconds, 2),
                },
            }

            with self.records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._run_records.append(record)

            self.stats.windows_analyzed += 1
            self.stats.violations += len(violations)
            self.stats.red_line_violations += sum(
                1 for f in violations if f["severity"] == Severity.RED_LINE.value
            )
            self.stats.input_tokens += outcome.input_tokens
            self.stats.output_tokens += outcome.output_tokens

        # Outside the lock: a Firestore round trip per window would otherwise
        # serialise the analysis workers behind each other.
        if self.persist_record is not None:
            try:
                await self.persist_record(record)
            except Exception as exc:
                # The record is in the local JSONL and the frame is in the
                # bucket. Losing the remote copy of one window is bad; taking
                # down the audit over it is worse.
                logger.warning("Could not persist record %d: %s", record_id, str(exc)[:200])

        if self.on_record is not None:
            try:
                self.on_record(record)
            except Exception as exc:
                logger.debug("Dashboard notification failed: %s", exc)

        return record

    async def _render_finding(self, clip: Clip, finding) -> dict:
        video_offset = clip.clip_ts_to_video_offset(finding.offset_seconds)
        rule = self._rules.get(finding.rule_id)
        entry = {
            "rule_id": finding.rule_id,
            # Denormalised on purpose: a record has to stay readable years
            # later, after the rule file has moved on.
            "rule_name": rule.name if rule else "",
            "status": finding.status.value,
            "confidence": round(finding.confidence, 3),
            "evidence": finding.evidence,
            "severity": finding.severity.value,
            "offset_seconds": round(video_offset, 3),
            "timestamp": Clip.format_offset(video_offset),
            # False means the model did not give a usable moment and this is
            # the start of the window standing in for one. The evidence frame
            # is cut at this offset either way, so a reviewer has to be able to
            # tell "here is the moment" from "here is somewhere in the window".
            "timestamp_exact": finding.has_timestamp,
            "wall_clock": self._wall_clock(video_offset),
            "evidence_frame": None,
        }
        if finding.status is Status.VIOLATION:
            entry["evidence_frame"] = await self._save_evidence(clip, finding)
        return entry

    async def _save_evidence(self, clip: Clip, finding) -> Optional[str]:
        """Cuts the still and puts it where it can be looked at later.

        ffmpeg can only write to a real file, so the frame is always cut
        locally first and then handed to the sink. Locally the sink is that
        same directory and the write is the whole story; on a container the
        sink is a bucket and the local copy is deleted straight after, because
        a multi-hour audit that keeps every frame fills the instance's disk.

        Returns the locator to record, or None -- `_render_finding` puts that
        None in `evidence_frame`, and the report renders it as "no reviewable
        still" rather than a broken link.
        """
        name = f"w{clip.index:05d}_{finding.rule_id}_{int(finding.offset_seconds):04d}.jpg"
        out_path = self.evidence_dir / name
        # The offset is inside the clip file, which is where ffmpeg has to seek.
        # For a sped-up recording that is not the same as the video offset.
        ok = await extract_frame(clip.path, finding.offset_seconds, out_path)
        if not ok:
            return None

        try:
            key = str(out_path.relative_to(config.data_dir))
        except ValueError:
            # `evidence_dir` was pointed somewhere outside `DATA_DIR`. Fall back
            # to a key the sink will accept rather than refusing to store it.
            key = f"{self.job_id}/evidence/{name}" if self.job_id else f"evidence/{name}"

        sink = artifact_sink()
        locator = await sink.put_file(key, out_path, "image/jpeg")
        if locator is None:
            # The upload failed and said so. The frame is still on this disk,
            # which is worth naming for however long the instance lives --
            # better a locator that may expire than no evidence at all.
            logger.warning("Evidence frame %s was not stored remotely.", name)
            return str(out_path)
        if not isinstance(sink, LocalArtifactSink):
            # It is in the bucket now. A multi-hour audit that also keeps every
            # frame locally fills the instance's disk and takes the run with it.
            with contextlib.suppress(OSError):
                out_path.unlink(missing_ok=True)
        return locator

    @staticmethod
    def _window_status(result: WindowResult) -> str:
        if any(f.status is Status.VIOLATION for f in result.findings):
            return Status.VIOLATION.value
        if result.findings and all(f.status is Status.CANNOT_DETERMINE for f in result.findings):
            return Status.CANNOT_DETERMINE.value
        return Status.COMPLIANT.value

    def _wall_clock(self, video_offset: float) -> Optional[str]:
        """Maps a video offset to real-world time, when that mapping is known."""
        if self.recording_started_at is None:
            return None
        return (self.recording_started_at + timedelta(seconds=video_offset)).isoformat()

    def summary(self) -> dict:
        """The run's numbers, plus where to go and look at the detail.

        Both locations are what a reader should actually open, which on a
        container is not the same as what this object wrote to. Evidence frames
        are uploaded and then deleted from local disk; records are handed to
        `persist_record`. Printing `/tmp/checkpoints/...` in that deployment is
        not a small inaccuracy -- it is a report telling someone to inspect a
        file that no longer exists on a machine they cannot reach.
        """
        return {
            **self.stats.as_dict(),
            "records_path": self.records_location or str(self.records_path),
            "evidence_dir": artifact_sink().location(self._evidence_key())
                            or str(self.evidence_dir),
        }

    def _evidence_key(self) -> str:
        """The sink key the frames in this run were stored under.

        Derived the same way `_save_evidence` derives each frame's key, and
        with the same fallback, so the directory the report names is the
        directory the frames are actually in. Computing it a second way here
        is how the two drift apart the next time either changes.
        """
        try:
            return str(self.evidence_dir.relative_to(config.data_dir))
        except ValueError:
            return f"{self.job_id}/evidence" if self.job_id else "evidence"

    def violations(self, limit: Optional[int] = None) -> List[dict]:
        """The violations found by *this* run, in video-time order.

        Recording order is analysis-completion order, which with
        ANALYSIS_CONCURRENCY > 1 is whatever finishes first -- window 2 before
        window 1 before window 3 is normal and correct. Nobody reads an audit
        that way, so the order is restored here rather than left to each
        caller to remember.

        When `limit` cuts the list it keeps the *earliest* rows, so the report
        is a prefix of the audit rather than an arbitrary sample. Callers that
        truncate must say so -- see `_report`.

        Deliberately not a re-read of the JSONL. That file is an append-only
        audit trail spanning every run ever made, including runs against a
        different rule file, so reporting from it mixed old findings into a new
        report -- a summary saying "5 violations" above a table listing seven,
        two of them under rule ids that are not even in the active standard.
        Use `all_violations_on_disk()` if the whole history is what you want.
        """
        rows = sorted(
            (r for r in self._run_records if r.get("violation_count")),
            key=lambda r: (r.get("start_offset_seconds", 0.0), r.get("window_index", 0)),
        )
        return rows[:limit] if limit else rows

    def covered_span(self) -> Optional[tuple]:
        """(first_offset, last_offset) of footage this run actually analysed.

        The honest answer to "what did you look at". A run can end early --
        buffering, a dead player, a budget -- and without this the report has
        no way to distinguish "watched the ten minutes you asked for and found
        nothing" from "watched forty seconds of them".
        """
        if not self._run_records:
            return None
        starts = [r["start_offset_seconds"] for r in self._run_records]
        ends = [r["end_offset_seconds"] for r in self._run_records]
        return (min(starts), max(ends))

    def all_violations_on_disk(self, limit: Optional[int] = None) -> List[dict]:
        """Every violation ever recorded, across all runs. For review tooling."""
        rows: List[dict] = []
        if not self.records_path.exists():
            return rows
        with self.records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("violation_count"):
                    rows.append(record)
        return rows[-limit:] if limit else rows
