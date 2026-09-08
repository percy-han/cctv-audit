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

"""Audit jobs: state that has to outlive the request that created it.

Three measurements from the Phase 0 probe decide the shape of this file. All
three are in `deploy/phase0/README.md` with logs; the short version:

  * **A Gemini Enterprise turn is cut at 602s, and work still running inside
    that request is cancelled at exactly 900s.** An audit takes longer than
    both. So the request cannot be the thing that runs the audit -- it can only
    start one and hand back a number.
  * **A task detached from the request survives.** One ran the full 1800s it
    was given, twice the 900s line, with the client long gone.
  * **Every turn arrives with the caller's real email in `user_id`.** Several
    people share one deployment, so "whose job is this" is a real question with
    a real answer, and getting it wrong means one customer reads another's
    audit.

That last point is why the store is keyed the way it is. Rather than a flat
collection plus a `where("user_id", "==", ...)` filter that any future caller
can forget, jobs live in a per-user subcollection: reaching a job requires
naming its owner, and there is no query shape that returns somebody else's row.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Protocol

from .config import config

logger = logging.getLogger("cctv_audit.jobs")

# The lifecycle, in the order a job walks it.
#
#   probing  -- preflight is looking at the video right now
#   ready    -- preflight succeeded; waiting for the customer to say "确认"
#   rejected -- preflight found the audit cannot be run (no video, wrong span)
#   running  -- the audit is going, detached from any request
#   done     -- finished, `result` is populated
#   failed   -- blew up, `error` says how
#   cancelled-- someone stopped it
STATES = ("probing", "ready", "rejected", "running", "done", "failed", "cancelled")

# States a job never leaves.
TERMINAL_STATES = frozenset({"rejected", "done", "failed", "cancelled"})


def new_job_id() -> str:
    """Short enough for a customer to read back over chat, long enough to not collide.

    Six hex characters is 16.7M values. Collisions only matter within one user's
    open jobs, which is a handful, so this is many orders of magnitude of slack.
    """
    return uuid.uuid4().hex[:6]


@dataclass
class Job:
    """One audit request, from "what did you ask for" through to the result.

    Everything here has to survive being written to Firestore and read back on
    a different container, so it is plain data: no Paths, no coroutines, no
    live browser handles. That constraint is the whole point -- Agent Runtime
    scales instances, and the turn that asks "好了吗" is not guaranteed to land
    on the machine that started the job.
    """

    job_id: str
    user_id: str
    # Which conversation asked. Lets a follow-up in the same GE thread find the
    # job without the customer having to quote a number back.
    session_id: str = ""

    state: str = "probing"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # -- what was asked for -------------------------------------------------
    target: str = ""
    start_seconds: float = 0.0
    duration_seconds: Optional[float] = None
    # Which version of the written standard to judge against. Recorded on the
    # job, not looked up at analysis time, so a report can always say which
    # standard produced it even after the standard has moved on.
    sop_id: str = ""

    # -- what preflight found ------------------------------------------------
    preflight: Optional[Dict[str, Any]] = None

    # -- how it is going -----------------------------------------------------
    progress: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Which container is running it. Purely diagnostic: when a job looks stuck,
    # the first question is whether the instance that owned it is still alive.
    instance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Job":
        """Tolerates rows written by an older version of this dataclass.

        A stored job outlives a deploy. Dropping unknown keys and defaulting
        missing ones means adding a field is not a migration.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @property
    def finished(self) -> bool:
        return self.state in TERMINAL_STATES


class JobStore(Protocol):
    """Every read is scoped to one user. That is not a convention, it is the API.

    There is deliberately no `get(job_id)`. A caller that has a job id but not
    an owner cannot read the job, so "forgot to filter by user" is not a
    mistake this interface lets you make.
    """

    async def create(self, job: Job) -> Job: ...

    async def get(self, user_id: str, job_id: str) -> Optional[Job]: ...

    async def update(self, user_id: str, job_id: str, **fields: Any) -> Optional[Job]: ...

    async def recent(self, user_id: str, limit: int = 10) -> List[Job]: ...

    # Per-window audit records, kept under the job rather than on it: an audit
    # can produce hundreds and Firestore caps a document at 1 MiB, so appending
    # them to a field would work in testing and fail on a long run.
    async def add_record(self, user_id: str, job_id: str, record: Dict[str, Any]) -> None: ...

    async def records(self, user_id: str, job_id: str, limit: int = 500) -> List[Dict[str, Any]]: ...

    def records_location(self, user_id: str, job_id: str) -> str:
        """Where those records can be found afterwards, for a human to read.

        Goes in the report. "" means this store is not somewhere anyone can go
        and look, and the caller should say where it wrote its own copy.
        """
        ...


class MemoryJobStore:
    """In-process store. The default for local runs and for every unit test.

    Fine for `adk web` on a workstation, where there is one process and it is
    the only one. Not fine on Agent Runtime -- see the module docstring -- so
    `job_store()` picks Firestore whenever it is configured.
    """

    def __init__(self) -> None:
        self._rows: Dict[str, Dict[str, Job]] = {}
        self._records: Dict[tuple, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def create(self, job: Job) -> Job:
        async with self._lock:
            self._rows.setdefault(job.user_id, {})[job.job_id] = job
        return job

    async def get(self, user_id: str, job_id: str) -> Optional[Job]:
        async with self._lock:
            found = self._rows.get(user_id, {}).get(job_id)
            # A copy, so a caller mutating what it read cannot rewrite history
            # in the store without going through `update`. Firestore gives you
            # this for free; without it the two backends would disagree.
            return replace(found) if found else None

    async def update(self, user_id: str, job_id: str, **fields: Any) -> Optional[Job]:
        async with self._lock:
            found = self._rows.get(user_id, {}).get(job_id)
            if found is None:
                return None
            for key, value in fields.items():
                if key in Job.__dataclass_fields__:
                    setattr(found, key, value)
                else:
                    logger.debug("Ignoring unknown job field %r.", key)
            found.updated_at = time.time()
            return replace(found)

    async def recent(self, user_id: str, limit: int = 10) -> List[Job]:
        async with self._lock:
            rows = sorted(
                self._rows.get(user_id, {}).values(),
                key=lambda j: j.created_at,
                reverse=True,
            )
            return [replace(r) for r in rows[:limit]]

    async def add_record(self, user_id: str, job_id: str, record: Dict[str, Any]) -> None:
        async with self._lock:
            self._records.setdefault((user_id, job_id), []).append(dict(record))

    async def records(self, user_id: str, job_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        async with self._lock:
            rows = self._records.get((user_id, job_id), [])
            return [dict(r) for r in rows[:limit]]

    def records_location(self, user_id: str, job_id: str) -> str:
        # In this process's memory, which is not a place to send anybody.
        return ""


class FirestoreJobStore:
    """Jobs in `{root}/{user_id}/jobs/{job_id}`.

    The user id is a path segment rather than a filtered field on purpose --
    see the module docstring. It is also why it gets sanitised: an email is
    fine in a document id, but Firestore rejects `/` and a couple of reserved
    forms, and a rejected write here would lose an audit.
    """

    def __init__(self, root: Optional[str] = None, database: Optional[str] = None) -> None:
        self._root = root or config.jobs_collection
        self._database = database or config.firestore_database
        self._client = None

    def _collection(self, user_id: str):
        if self._client is None:
            # Imported here, not at module scope: the local `adk web` path and
            # the whole unit-test suite run without this package installed, and
            # a top-level import would make Firestore a hard dependency of
            # code that never touches it.
            from google.cloud import firestore

            self._client = firestore.AsyncClient(
                project=config.gcp_project or None,
                database=self._database or "(default)",
            )
        return (
            self._client.collection(self._root)
            .document(_document_safe(user_id))
            .collection("jobs")
        )

    async def create(self, job: Job) -> Job:
        await self._collection(job.user_id).document(job.job_id).set(job.to_dict())
        return job

    async def get(self, user_id: str, job_id: str) -> Optional[Job]:
        snapshot = await self._collection(user_id).document(job_id).get()
        if not snapshot.exists:
            return None
        return Job.from_dict(snapshot.to_dict() or {})

    async def update(self, user_id: str, job_id: str, **fields: Any) -> Optional[Job]:
        clean = {k: v for k, v in fields.items() if k in Job.__dataclass_fields__}
        clean["updated_at"] = time.time()
        document = self._collection(user_id).document(job_id)
        try:
            await document.update(clean)
        except Exception as exc:
            # Almost always "the document is gone". Worth one line rather than
            # an exception, because the caller is usually a progress callback
            # inside a running audit and killing the audit over a lost
            # progress write would be the wrong trade.
            logger.warning("Could not update job %s: %s", job_id, str(exc)[:200])
            return None
        return await self.get(user_id, job_id)

    async def recent(self, user_id: str, limit: int = 10) -> List[Job]:
        query = self._collection(user_id).order_by(
            "created_at", direction="DESCENDING"
        ).limit(limit)
        return [Job.from_dict(doc.to_dict() or {}) async for doc in query.stream()]

    async def add_record(self, user_id: str, job_id: str, record: Dict[str, Any]) -> None:
        # `id` counts every window ever archived by this instance, which after
        # a restart starts over. Padded so the natural document-id ordering is
        # also window order, which is what `records()` relies on.
        name = f"{int(record.get('window_index', 0)):06d}"
        await (
            self._collection(user_id)
            .document(job_id)
            .collection("records")
            .document(name)
            .set(record)
        )

    async def records(self, user_id: str, job_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        query = (
            self._collection(user_id)
            .document(job_id)
            .collection("records")
            .order_by("__name__")
            .limit(limit)
        )
        return [doc.to_dict() or {} async for doc in query.stream()]

    def records_location(self, user_id: str, job_id: str) -> str:
        # The document path, spelled the way the console and the CLI want it,
        # and with the same sanitisation the writes use -- an address that does
        # not match the one written to is worse than no address.
        return (f"Firestore[{self._database}] "
                f"{self._root}/{_document_safe(user_id)}/jobs/{job_id}/records")


def _document_safe(user_id: str) -> str:
    """Turns an email into something Firestore will accept as a document id.

    Firestore forbids `/`, forbids `.` and `..` on their own, and caps ids at
    1500 bytes. An email hits none of those in practice, but a blank user id
    would produce an empty segment and a confusing error a long way from here.
    """
    clean = (user_id or "").strip().replace("/", "_")
    return clean or "anonymous"


_store: Optional[JobStore] = None


def job_store() -> JobStore:
    """The process-wide store. Firestore when configured, memory otherwise."""
    global _store
    if _store is None:
        _store = FirestoreJobStore() if config.use_firestore else MemoryJobStore()
        logger.info("Job store: %s", type(_store).__name__)
    return _store


def set_job_store(store: Optional[JobStore]) -> None:
    """Overrides the store. For tests, and for wiring a fake in `adk web`."""
    global _store
    _store = store


# --- running the work somewhere the request cannot kill it ------------------


class BackgroundRunner:
    """Starts audits on tasks that do not belong to any request.

    Measured, and the reason this class exists rather than an `await`: work
    awaited inside a Gemini Enterprise turn is cancelled at exactly 900.0s --
    twice, on two engines, to the tenth of a second. A task created with
    `asyncio.create_task` and never awaited is not part of the request
    coroutine, so the request being cancelled has nothing to propagate to; one
    ran its full 1800s with the caller long gone.
    """

    def __init__(self) -> None:
        # Strong references. asyncio keeps only a weak one to a running task,
        # so a task nobody holds can be collected mid-flight. That failure
        # looks exactly like the platform killing the job, which would send
        # the next person debugging this straight back to the 900s question.
        self._tasks: set = set()

    def start(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        return task

    def _finished(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            # Nobody is awaiting this task, so without this the traceback goes
            # to asyncio's "exception was never retrieved" handler at
            # interpreter shutdown, if anywhere.
            logger.exception("Background audit died", exc_info=exc)

    @property
    def running(self) -> int:
        return len(self._tasks)

    async def drain(self, timeout: float = 30.0) -> None:
        """Waits for in-flight jobs. For tests and for an orderly shutdown."""
        if not self._tasks:
            return
        await asyncio.wait(set(self._tasks), timeout=timeout)


_runner: Optional[BackgroundRunner] = None


def background_runner() -> BackgroundRunner:
    global _runner
    if _runner is None:
        _runner = BackgroundRunner()
    return _runner
