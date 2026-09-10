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

"""The three operations an audit is made of, and the seam they cut along.

    preflight(...)              seconds. Opens the video, says what it found,
                                hands back a job id. Nothing is spent yet.
    start_audit(user, job_id)   returns immediately. The audit runs detached.
    get_status(user, job_id)    how it is going, or the finished report.

The split is forced by measurement, not taste. A Gemini Enterprise turn is cut
at 602s, and work still running inside that request is cancelled at exactly
900s -- while an audit takes longer than both. So no single call can be "run
the audit": one call has to start it and hand back a number, and a later call
has to collect it. See `deploy/phase0/README.md` for the logs.

It also happens to be the shape the customer needs anyway -- "视频找到了，
是录屏方式，时间段够，确认开始吗" is exactly `preflight` -- which is the main
reason to believe the seam is in the right place.

Two rules hold throughout:

  * **Every call names a user.** `user_id` is the caller's real email, passed
    through by GE. Nothing is reachable without it -- see `jobs.py`.
  * **Nothing is guessed.** A SOP version that will not load, a video that
    will not open, a span past the end of the recording: each is reported, and
    none of them are quietly turned into a slightly different audit.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

from .analyzer import SopUnavailable, load_rules_for
from .jobs import Job, background_runner, job_store, new_job_id
from .pipeline import AuditPipeline, AuditRequest, PreflightResult
from .store import AuditStore

logger = logging.getLogger("cctv_audit.service")

# How often a running audit writes its progress back. Every window would be a
# round trip per analysed clip inside the analysis loop; never would leave the
# customer watching "running" for an hour with no way to tell it apart from a
# hang.
_PROGRESS_EVERY_SECONDS = 15.0


def _env_seconds(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        logger.warning("%s is not a number; using %s.", name, default)
        return default


# -- the watchdog's three limits ------------------------------------------
#
# Why any of this exists: job `d7680d` sat at `running` for 62 minutes having
# produced nothing, and the customer kept being told "正在跑：已分析 0 个窗口".
# The pipeline does carry a wall-clock budget, but it is only consulted between
# captured windows (`pipeline.py`), so a capture loop that never comes round
# again never reaches the check. A run that stops making progress has to be
# killed from outside itself.
#
# The numbers are set from measurement, not caution. Capture and analysis run
# concurrently through a queue, so the first window comes back one window's
# worth of footage after capture starts -- seconds on Plan A, a minute or so on
# Plan B. Later windows arrived every 25-45s in the cloud runs.

# Nothing at all yet. Generous, because this covers page load, login redirects,
# the probe, and ffmpeg finding the stream -- and killing a slow-but-working
# start is worse than waiting.
_FIRST_WINDOW_SECONDS = _env_seconds("AUDIT_FIRST_WINDOW_SECONDS", 420.0)

# Between windows, once the run has proved it can produce them. Roughly seven
# times the observed gap.
_STALL_SECONDS = _env_seconds("AUDIT_STALL_SECONDS", 300.0)

# The backstop, above the pipeline's own budget so that its graceful stop wins
# whenever the pipeline is healthy enough to take it.
_HARD_LIMIT_SECONDS = _env_seconds("AUDIT_HARD_LIMIT_SECONDS", 3900.0)

_WATCHDOG_TICK_SECONDS = _env_seconds("AUDIT_WATCHDOG_TICK_SECONDS", 15.0)

# How long a cancelled run gets to close its browser and kill its ffmpeg before
# we stop waiting and mark the job failed anyway. The job is dead either way;
# what must not happen is the watchdog itself hanging on the cleanup.
_CLEANUP_SECONDS = 30.0


# Pipeline events worth putting in front of whoever is watching the dashboard,
# and what to call them in the action banner. Deliberately short: everything the
# pipeline emits is already in the log, and a banner that narrates every step
# stops being read long before the one line that matters shows up.
#
# All three are about the *picture*, because that is the one thing the dashboard
# cannot report on itself -- a frozen feed and a quiet shop look identical, and
# on the bilibili run (job 692b53) the operator watched a still for two minutes
# while every number on the page said the audit was fine.
_ANNOUNCE: Dict[str, tuple] = {
    "picture_frozen": (
        "画面暂停",
        lambda p: str(p.get("reason", "")),
    ),
    "playback_recovering": (
        "正在恢复播放",
        lambda p: f"第 {p.get('attempt', 1)} 次尝试",
    ),
    "player_geometry_lost": (
        "播放器位置变了",
        lambda p: f"画面可能只剩一角（期望 {p.get('expected')}，现在 {p.get('actual')}）",
    ),
}


def _announce(monitor, event: str, payload: Dict[str, Any]) -> None:
    """Relays selected pipeline events to the dashboard's action banner.

    The local `adk web` path has always had this (`agent.py` wires `on_status`
    into the chat stream); the cloud path never did, so on Agent Runtime these
    events reached the log and stopped there -- and nobody demoing this is
    reading Cloud Logging.

    `monitor.note` is fire-and-forget over the network. Wrapped anyway: a
    status line is the least important thing in the process and must not be
    able to take an audit down with it.
    """
    entry = _ANNOUNCE.get(event)
    if not entry:
        return
    title, describe = entry
    try:
        monitor.note(title, describe(payload))
    except Exception as exc:
        logger.debug("Could not announce %s: %s", event, exc)


def _instance_id() -> str:
    """Which container. Diagnostic only -- the first question about a stuck
    job is whether the instance that owned it is still alive."""
    return os.environ.get("K_REVISION") or os.environ.get("HOSTNAME") or "local"


class AuditService:
    """Stateless with respect to jobs: everything it knows, it reads back.

    Deliberately holds no browser, no pipeline and no job between calls.
    Preflight and the confirmation that follows it are two separate GE turns
    and are not guaranteed to land on the same instance, so anything kept in
    memory here would be invisible to half the calls that need it.
    """

    def __init__(self, store=None, runner=None) -> None:
        self.jobs = store or job_store()
        self.runner = runner or background_runner()
        # Swapped in tests. A real pipeline wants Chromium and a real store
        # makes directories.
        self.build_pipeline: Callable[..., AuditPipeline] = AuditPipeline
        self.build_store: Callable[..., AuditStore] = AuditStore

    # -- 1. look before you spend -------------------------------------------

    async def preflight(
        self,
        user_id: str,
        request: AuditRequest,
        session_id: str = "",
        sop_id: str = "",
    ) -> Job:
        """Opens the video and records what it found. Fast enough for one turn.

        Always returns a job. A failure is a job in state `rejected` with a
        readable `preflight.problem`, not an exception -- the caller's next act
        is to tell the customer either way, and a job id means the attempt is
        on the record instead of vanishing.
        """
        job = Job(
            job_id=new_job_id(),
            user_id=user_id,
            session_id=session_id,
            state="probing",
            target=request.target,
            start_seconds=request.start_seconds,
            duration_seconds=request.duration_seconds,
            sop_id=sop_id,
            instance=_instance_id(),
        )
        await self.jobs.create(job)

        # Before the browser, on purpose: resolving the standard costs one GCS
        # read, and finding out the version is missing *after* a minute of
        # navigation wastes the minute and tells the customer nothing new.
        try:
            rules = await load_rules_for(sop_id)
        except SopUnavailable as exc:
            return await self._reject(job, str(exc))

        try:
            pipeline = self.build_pipeline(rules=rules)
        except ValueError as exc:  # configuration, from AuditPipeline.__init__
            return await self._reject(job, f"配置有误，无法启动：{exc}")

        result: PreflightResult = await pipeline.preflight(request, job_id=job.job_id)
        fields: Dict[str, Any] = {
            "preflight": result.as_dict(),
            # The resolved version, not the requested one: with no bucket
            # configured this is "", and the record should say so rather than
            # imply a version was chosen.
            "sop_id": rules.sop_id,
            "state": "ready" if result.ok else "rejected",
        }
        if not result.ok:
            fields["error"] = result.problem
        return await self.jobs.update(user_id, job.job_id, **fields) or job

    async def _reject(self, job: Job, problem: str) -> Job:
        logger.info("Job %s rejected: %s", job.job_id, problem)
        updated = await self.jobs.update(
            job.user_id, job.job_id,
            state="rejected",
            error=problem,
            preflight={"ok": False, "problem": problem, "target": job.target},
        )
        return updated or job

    # -- 2. the customer said 确认 -------------------------------------------

    async def start_audit(self, user_id: str, job_id: str) -> Job:
        """Starts the audit and returns without waiting for it.

        The `await` that is deliberately absent: work awaited inside a GE turn
        is cancelled at 900s. `runner.start` puts the audit on a task that does
        not belong to this request, so the request ending has nothing to
        propagate to.
        """
        job = await self.jobs.get(user_id, job_id)
        if job is None:
            raise LookupError(f"没有这个稽核单号：{job_id}")
        if job.state == "running":
            return job          # a repeated 确认 must not start a second audit
        if job.state != "ready":
            raise ValueError(
                f"单号 {job_id} 现在是「{job.state}」，不能开始稽核。"
                + (f"原因：{job.error}" if job.error else "")
            )

        job = await self.jobs.update(
            user_id, job_id, state="running", instance=_instance_id(),
            progress={"started_at": time.time(), "windows_analyzed": 0},
        ) or job
        self.runner.start(self._run(job))
        return job

    async def _run(self, job: Job) -> None:
        """The audit itself. Runs with no client attached and must not raise."""
        from .monitor import monitor_for_job  # local: keeps aiohttp off the import path

        # Scoped to this job, not the module singleton: `start_audit` returns
        # immediately and the work continues detached, so two audits can be in
        # flight in one container. They must not share a dashboard room.
        monitor = monitor_for_job(job.job_id)
        # Shared with the watchdog. `beat` is stamped by the record writer on
        # every window -- unthrottled, unlike the Firestore write below it --
        # because it is the only signal that separates "slow" from "wedged".
        progress = {
            "last": 0.0, "windows": 0, "violations": 0, "beat": time.monotonic(),
        }
        started = time.monotonic()
        logger.info(
            "Job %s starting: target=%s start=%s duration=%s sop=%s",
            job.job_id, job.target, job.start_seconds, job.duration_seconds, job.sop_id,
        )
        try:
            rules = await load_rules_for(job.sop_id)
            store = self.build_store(
                job_id=job.job_id,
                rules=rules,
                persist_record=self._record_writer(job, progress),
                on_record=monitor.push_record,
                # The store writes a JSONL to this container's disk and that
                # disk goes away. Whoever reads the report has to be pointed at
                # the copy that survives, which only this layer knows about.
                records_location=self.jobs.records_location(job.user_id, job.job_id),
            )
            # The dashboard is a separate service now, so every one of these is
            # a network call to MONITOR_URL. All of them are fire-and-forget:
            # an audit must not fail, or even slow down, because nobody is
            # watching it. `has_viewers` is why an unwatched run streams
            # nothing at all -- see LivePreview.
            monitor.start_session(
                f"稽核 {job.target}",
                f"单号 {job.job_id} · 标准 {rules.sop_id or rules.origin} v{rules.version}",
            )
            pipeline = self.build_pipeline(
                store=store,
                rules=rules,
                on_preview_frame=monitor.update_frame_b64,
                has_viewers=lambda: self._has_viewers(monitor),
                on_status=lambda event, payload: _announce(monitor, event, payload),
            )
            run = asyncio.ensure_future(pipeline.run(
                AuditRequest(
                    target=job.target,
                    start_seconds=job.start_seconds,
                    duration_seconds=job.duration_seconds,
                )
            ))
            stalled = await self._watch(job, run, progress, started)
            if stalled:
                logger.error("Job %s killed by the watchdog: %s", job.job_id, stalled)
                await self._fail(job, monitor, stalled)
                return
            summary = run.result()
        except Exception as exc:
            logger.exception("Job %s failed", job.job_id)
            await self._fail(job, monitor, str(exc))
            return

        # Rendered here, while the store is still in memory: a later turn may
        # land on another container with nothing but this document to read.
        report = ""
        try:
            from .agent import CctvAuditAgent  # local: avoids an import cycle

            report = CctvAuditAgent._report(summary, store)
        except Exception as exc:
            # The audit finished. Losing the pretty version of the result must
            # not leave the job stuck at "running" with nobody coming back for
            # it -- the numbers are in `summary` either way.
            logger.exception("Could not render the report for job %s", job.job_id)
            report = f"稽核已完成，但报告排版失败：{str(exc)[:200]}"

        logger.info(
            "Job %s done in %.1fs: %s windows, %s violations, mode=%s",
            job.job_id, time.monotonic() - started,
            summary.get("windows_analyzed"), summary.get("violations"),
            summary.get("capture_mode"),
        )
        monitor.finish_session(report)
        await self.jobs.update(
            job.user_id, job.job_id,
            state="done",
            result={"summary": summary, "report": report},
        )
        await self._close_monitor(monitor)

    # -- the watchdog ---------------------------------------------------------

    async def _watch(self, job: Job, run, progress: Dict[str, Any], started: float):
        """Kills a run that has stopped making progress. Returns why, or None.

        Returning a sentence rather than raising is deliberate: the reason is
        the whole point of this function. "一直在跑" told the customer nothing
        for 62 minutes; the answer they needed was that the first window never
        arrived, which names the fault (page, login, or stream) well enough to
        act on.
        """
        spoke_at = 0.0
        while True:
            done, _ = await asyncio.wait({run}, timeout=_WATCHDOG_TICK_SECONDS)
            if done:
                return None

            now = time.monotonic()
            windows = int(progress["windows"])
            silent = now - float(progress["beat"])
            elapsed = now - started
            # No windows yet means the run is still in its start-up phase, and
            # start-up is legitimately much slower than the steady state.
            allowance = _FIRST_WINDOW_SECONDS if windows == 0 else _STALL_SECONDS

            if silent < allowance and elapsed < _HARD_LIMIT_SECONDS:
                # A heartbeat in the log, once a minute. This is the line that
                # would have answered "is it working or is it stuck" without
                # anyone having to guess from the absence of other lines.
                if now - spoke_at >= 60.0:
                    spoke_at = now
                    logger.info(
                        "Job %s alive: %.0fs elapsed, %s windows, "
                        "%.0fs since the last one (allowance %.0fs)",
                        job.job_id, elapsed, windows, silent, allowance,
                    )
                continue

            if elapsed >= _HARD_LIMIT_SECONDS:
                reason = (
                    f"这单跑了 {int(elapsed / 60)} 分钟还没结束，超过了 "
                    f"{int(_HARD_LIMIT_SECONDS / 60)} 分钟的上限，已经停掉。"
                    f"停之前完成了 {windows} 个窗口。"
                    "把时间段改短一点再试。"
                )
            elif windows == 0:
                reason = (
                    f"开始 {int(silent / 60)} 分钟了，一个窗口都没产出，判定为卡住，已经停掉。"
                    "常见原因：视频页面打不开或要登录、媒体流拉不动、"
                    "或者这个源比实时还慢。换一段更短的时间、或者换个视频源再试。"
                )
            else:
                reason = (
                    f"已经 {int(silent / 60)} 分钟没有新的分析结果，判定为卡住，已经停掉。"
                    f"停之前完成了 {windows} 个窗口。"
                )

            await self._cancel(job, run)
            return reason

    @staticmethod
    async def _cancel(job: Job, run) -> None:
        """Stops the run and gives it a bounded moment to clean up.

        Bounded because the cleanup is the browser and ffmpeg, either of which
        can be the thing that was wedged in the first place. Waiting on it
        without a limit would move the hang from the audit into the watchdog.
        """
        run.cancel()
        done, _ = await asyncio.wait({run}, timeout=_CLEANUP_SECONDS)
        if not done:
            logger.error(
                "Job %s did not stop within %.0fs of being cancelled; "
                "abandoning the task. The instance may be leaking a browser.",
                job.job_id, _CLEANUP_SECONDS,
            )
            return
        # Retrieved so asyncio does not later complain that nobody looked.
        if not run.cancelled():
            with contextlib.suppress(Exception):
                run.exception()

    async def _fail(self, job: Job, monitor, reason: str) -> None:
        """One way out for every failure: say why, on the record and on screen."""
        with contextlib.suppress(Exception):
            monitor.fail_session(reason)
        await self.jobs.update(
            job.user_id, job.job_id, state="failed", error=reason[:500],
        )
        await self._close_monitor(monitor)

    @staticmethod
    async def _close_monitor(monitor) -> None:
        """Releases this job's dashboard client.

        Every event it sends is fire-and-forget, so the last few may still be
        in flight; give them a moment rather than closing the session out from
        under the final report. Failing to close is a leaked connection pool
        per audit, which a long-lived container notices eventually.
        """
        with contextlib.suppress(Exception):
            await asyncio.sleep(0.5)
            await monitor.close()

    async def _has_viewers(self, monitor) -> bool:
        """Is anyone looking at *this job's* dashboard right now?

        Takes the client rather than reaching for the singleton so the answer
        is scoped to this audit's room -- otherwise a colleague watching a
        different audit keeps this one streaming frames nobody sees. Kept as a
        method so tests can override it without a network stub.
        """
        return await monitor.viewers() > 0

    def _record_writer(self, job: Job, state: Optional[Dict[str, Any]] = None):
        """Persists each window, and the job's progress alongside it -- but not
        on every window.

        A Firestore round trip per clip would run inside the analysis loop and
        serialise the workers behind each other. Throttling the *progress*
        write and not the record keeps the audit trail complete while the
        status a customer polls stays roughly current.

        `state` is shared with the watchdog when there is one. The heartbeat in
        it is stamped on **every** window, throttling or not: a throttled
        heartbeat would read as fifteen seconds of silence that never happened.
        """
        if state is None:
            state = {"last": 0.0, "windows": 0, "violations": 0, "beat": time.monotonic()}

        async def write(record: dict) -> None:
            await self.jobs.add_record(job.user_id, job.job_id, record)
            state["windows"] += 1
            state["violations"] += int(record.get("violation_count", 0))
            state["beat"] = time.monotonic()
            now = time.time()
            if now - state["last"] < _PROGRESS_EVERY_SECONDS:
                return
            state["last"] = now
            await self.jobs.update(job.user_id, job.job_id, progress={
                "windows_analyzed": state["windows"],
                "violations_so_far": state["violations"],
                # Windows finish out of order, so this is "the most recent one
                # to come back", not "how far along we are". Named accordingly.
                "last_window_returned": record.get("time_range", ""),
                "updated_at": now,
            })

        return write

    # -- 3. how is it going --------------------------------------------------

    async def get_status(
        self,
        user_id: str,
        job_id: str = "",
        session_id: str = "",
    ) -> Optional[Job]:
        """The job, by number or by conversation.

        `session_id` is the convenience path: GE passes it on every turn, so a
        customer who asks "好了吗" in the same thread does not have to quote a
        number back. It is a lookup, never an authorisation -- the user scope
        applies exactly the same either way.
        """
        if job_id:
            return await self.jobs.get(user_id, job_id)
        if not session_id:
            return None
        for job in await self.jobs.recent(user_id, limit=20):
            if job.session_id == session_id:
                return job
        return None


_service: Optional[AuditService] = None


def audit_service() -> AuditService:
    global _service
    if _service is None:
        _service = AuditService()
    return _service


def set_audit_service(service: Optional[AuditService]) -> None:
    """For tests, and for wiring a fake into `adk web`."""
    global _service
    _service = service
