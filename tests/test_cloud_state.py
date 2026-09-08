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

"""The state that has to survive a container it did not start on.

Everything here exists because of one Phase 0 measurement: a Gemini Enterprise
turn is cut at 602s and work inside it is cancelled at 900s, while an audit
takes longer than both. So the audit runs detached, its state lives outside the
process, and the turn that asks "好了吗" may be talking to a different machine
than the one doing the work.

None of these tests touch a browser, a bucket or Firestore -- same rule as the
rest of the suite.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from computer_use_agent.jobs import (
    BackgroundRunner,
    Job,
    MemoryJobStore,
    new_job_id,
)


class TestAJobCannotBeReadWithoutNamingItsOwner:
    """Several customers share one deployment; `user_id` is their real email.

    A flat collection plus a filter works right up until one caller forgets the
    filter, and the symptom is 张三 reading 李四's audit -- which nobody
    reports, because it looks like a working feature.
    """

    @pytest.mark.asyncio
    async def test_another_user_cannot_read_the_job(self):
        store = MemoryJobStore()
        await store.create(Job(job_id="abc123", user_id="zhang@example.com", target="u"))

        assert (await store.get("zhang@example.com", "abc123")) is not None
        assert (await store.get("li@example.com", "abc123")) is None

    @pytest.mark.asyncio
    async def test_another_user_cannot_write_the_job(self):
        store = MemoryJobStore()
        await store.create(Job(job_id="abc123", user_id="zhang@example.com"))

        assert (await store.update("li@example.com", "abc123", state="cancelled")) is None
        owned = await store.get("zhang@example.com", "abc123")
        assert owned.state == "probing"

    @pytest.mark.asyncio
    async def test_recent_lists_only_your_own(self):
        store = MemoryJobStore()
        await store.create(Job(job_id="a", user_id="zhang@example.com"))
        await store.create(Job(job_id="b", user_id="li@example.com"))

        assert [j.job_id for j in await store.recent("zhang@example.com")] == ["a"]

    @pytest.mark.asyncio
    async def test_the_store_has_no_way_to_ask_for_a_job_by_id_alone(self):
        # The guarantee above is only real if the unscoped call does not exist.
        # If someone adds one, this fails and they have to think about it.
        from computer_use_agent.jobs import JobStore

        for name in ("get", "update", "add_record", "records"):
            args = JobStore.__dict__[name].__code__.co_varnames
            assert "user_id" in args, f"{name} must be scoped to a user"


class TestWhatTheStoreHandsBackIsACopy:
    """Firestore returns a fresh object every read; memory must not differ.

    Otherwise a caller that mutates what it read rewrites history locally and
    not in the cloud, and the two backends disagree only in production.
    """

    @pytest.mark.asyncio
    async def test_mutating_a_read_does_not_change_the_store(self):
        store = MemoryJobStore()
        await store.create(Job(job_id="a", user_id="u", state="ready"))

        got = await store.get("u", "a")
        got.state = "done"

        assert (await store.get("u", "a")).state == "ready"

    @pytest.mark.asyncio
    async def test_update_returns_the_new_state_and_bumps_the_clock(self):
        store = MemoryJobStore()
        job = await store.create(Job(job_id="a", user_id="u"))
        before = job.updated_at

        await asyncio.sleep(0.01)
        updated = await store.update("u", "a", state="running", progress={"windows": 3})

        assert updated.state == "running"
        assert updated.progress == {"windows": 3}
        assert updated.updated_at > before

    @pytest.mark.asyncio
    async def test_an_unknown_field_is_ignored_rather_than_stored(self):
        # `update(**fields)` is called from progress callbacks deep in the
        # pipeline. A typo there must not add a phantom column to Firestore.
        store = MemoryJobStore()
        await store.create(Job(job_id="a", user_id="u"))

        updated = await store.update("u", "a", stat="running")
        assert updated.state == "probing"
        assert not hasattr(updated, "stat")


class TestAStoredJobOutlivesTheCodeThatWroteIt:
    """A row written before a deploy is read after it. Adding a field is not
    a migration, and neither is removing one."""

    def test_an_unknown_key_from_an_older_row_is_dropped(self):
        job = Job.from_dict({
            "job_id": "a", "user_id": "u", "state": "done",
            "retired_field": "written by last month's container",
        })
        assert job.job_id == "a"
        assert job.state == "done"

    def test_a_missing_key_takes_its_default(self):
        job = Job.from_dict({"job_id": "a", "user_id": "u"})
        assert job.state == "probing"
        assert job.preflight is None

    def test_round_trips_through_a_plain_dict(self):
        job = Job(job_id="a", user_id="u", target="https://x", start_seconds=60.0)
        assert Job.from_dict(job.to_dict()) == job

    def test_terminal_states_are_terminal(self):
        assert Job(job_id="a", user_id="u", state="done").finished
        assert Job(job_id="a", user_id="u", state="rejected").finished
        assert not Job(job_id="a", user_id="u", state="running").finished

    def test_ids_are_short_enough_to_read_back_over_chat(self):
        # Six hex characters: a customer can type it into a chat box, and
        # 16.7M values is far more slack than one person's open jobs need.
        # Not asserting global uniqueness -- 500 draws collide about once in a
        # hundred runs, and a test that fails one time in a hundred teaches
        # people to re-run rather than to look.
        ids = [new_job_id() for _ in range(200)]
        assert all(len(i) == 6 and int(i, 16) >= 0 for i in ids)
        assert len(set(ids)) >= 199


class TestRecordsGoUnderTheJobNotOnIt:
    """An audit produces hundreds of windows and a Firestore document caps at
    1 MiB. Appending them to a field passes every test and fails on hour two."""

    @pytest.mark.asyncio
    async def test_records_are_scoped_to_the_owner_too(self):
        store = MemoryJobStore()
        await store.create(Job(job_id="a", user_id="zhang@example.com"))
        await store.add_record("zhang@example.com", "a", {"window_index": 0})

        assert len(await store.records("zhang@example.com", "a")) == 1
        assert await store.records("li@example.com", "a") == []

    @pytest.mark.asyncio
    async def test_a_stored_record_is_a_copy(self):
        store = MemoryJobStore()
        await store.add_record("u", "a", {"window_index": 0, "findings": []})

        (await store.records("u", "a"))[0]["window_index"] = 99
        assert (await store.records("u", "a"))[0]["window_index"] == 0


class TestTheAuditOutlivesTheRequestThatStartedIt:
    """Measured: work awaited inside a GE turn dies at exactly 900.0s, and a
    detached task ran its full 1800s. This class is the detaching."""

    @pytest.mark.asyncio
    async def test_the_starter_returns_before_the_work_finishes(self):
        runner = BackgroundRunner()
        finished = asyncio.Event()

        async def slow():
            await asyncio.sleep(0.05)
            finished.set()

        runner.start(slow())
        assert not finished.is_set()   # start() did not wait
        assert runner.running == 1

        await runner.drain(timeout=1.0)
        assert finished.is_set()
        assert runner.running == 0

    @pytest.mark.asyncio
    async def test_the_task_is_held_by_a_strong_reference(self):
        # asyncio only keeps a weak reference to a running task, so one that
        # nobody holds can be collected mid-flight. That looks identical to the
        # platform killing the job -- and would send the next person debugging
        # this straight back to the 900s question.
        import gc

        runner = BackgroundRunner()
        done = asyncio.Event()

        async def work():
            await asyncio.sleep(0.05)
            done.set()

        runner.start(work())
        gc.collect()
        await runner.drain(timeout=1.0)
        assert done.is_set()

    @pytest.mark.asyncio
    async def test_a_job_that_dies_does_not_take_the_process_with_it(self, caplog):
        runner = BackgroundRunner()

        async def boom():
            raise RuntimeError("ffmpeg went away")

        runner.start(boom())
        await runner.drain(timeout=1.0)

        # Nobody awaits these tasks, so without the done-callback the traceback
        # surfaces at interpreter shutdown, if at all.
        assert "ffmpeg went away" in caplog.text
        assert runner.running == 0

    @pytest.mark.asyncio
    async def test_draining_an_idle_runner_is_free(self):
        await BackgroundRunner().drain(timeout=0.01)


class TestEvidenceGoesSomewhereItCanBeLookedAtLater:
    """On a container the local filesystem is scratch. A violation whose only
    proof is a JPG on a recycled instance is a violation nobody can check."""

    @pytest.mark.asyncio
    async def test_a_local_frame_is_recorded_by_its_key_not_its_absolute_path(self, tmp_path):
        from computer_use_agent.artifacts import LocalArtifactSink

        sink = LocalArtifactSink(root=tmp_path)
        key = await sink.put("job7/evidence/w1.jpg", b"jpegbytes", "image/jpeg")

        assert key == "job7/evidence/w1.jpg"
        assert (tmp_path / key).read_bytes() == b"jpegbytes"

    @pytest.mark.asyncio
    async def test_a_key_cannot_climb_out_of_the_data_directory(self, tmp_path):
        # The key is built from a job id and a rule id, both ours -- but it is
        # the only string in that module that becomes a filesystem path.
        from computer_use_agent.artifacts import LocalArtifactSink

        sink = LocalArtifactSink(root=tmp_path / "data")
        (tmp_path / "data").mkdir()

        with pytest.raises(ValueError):
            await sink.put("../escaped.jpg", b"x", "image/jpeg")
        assert not (tmp_path / "escaped.jpg").exists()

    @pytest.mark.asyncio
    async def test_putting_a_file_already_in_place_does_not_copy_it(self, tmp_path):
        from computer_use_agent.artifacts import LocalArtifactSink

        sink = LocalArtifactSink(root=tmp_path)
        frame = tmp_path / "evidence" / "w1.jpg"
        frame.parent.mkdir()
        frame.write_bytes(b"original")

        assert await sink.put_file("evidence/w1.jpg", frame, "image/jpeg") == "evidence/w1.jpg"
        assert frame.read_bytes() == b"original"

    @pytest.mark.asyncio
    async def test_a_missing_frame_reports_nothing_rather_than_a_broken_link(self, tmp_path):
        from computer_use_agent.artifacts import LocalArtifactSink

        sink = LocalArtifactSink(root=tmp_path)
        assert await sink.put_file("e/w1.jpg", tmp_path / "nope.jpg", "image/jpeg") is None

    @pytest.mark.asyncio
    async def test_a_failed_upload_loses_the_frame_not_the_finding(self, tmp_path, caplog):
        # `record()` puts this None straight into `evidence_frame`, which the
        # report renders as "no reviewable still". An exception here would
        # instead lose the violation the frame was evidence for.
        from computer_use_agent.artifacts import GcsArtifactSink

        sink = GcsArtifactSink(bucket="b", prefix="audits")

        def explode(_key):
            raise RuntimeError("403 from GCS")

        sink._blob = explode
        with pytest.raises(RuntimeError):
            # The lazy client build is outside the try on purpose -- a broken
            # deployment should be loud. Only the upload itself is forgiving.
            await sink.put("k.jpg", b"x", "image/jpeg")

        class Blob:
            def upload_from_string(self, *a, **k):
                raise RuntimeError("503 backend error")

        sink._blob = lambda key: (Blob(), f"gs://b/audits/{key}")
        assert await sink.put("k.jpg", b"x", "image/jpeg") is None
        assert "503" in caplog.text


class TestTheReportNamesPlacesThatStillExist:
    """The last line of every report says where the detail is. On a container
    the honest answer is a bucket and a Firestore path, not `/tmp`.

    This is not cosmetic. `/tmp/checkpoints/audit_records.jsonl` is a real file
    right up until the instance is recycled, and then it is an instruction to
    go and look at something that is gone, on a machine the customer cannot
    reach -- and the report looks exactly as authoritative either way.
    """

    def _store(self, tmp_path, monkeypatch, **kwargs):
        from computer_use_agent.analyzer.sop import parse_rules
        from computer_use_agent.store import AuditStore
        from computer_use_agent import store as store_mod

        # `_evidence_key` measures the evidence directory against `DATA_DIR`,
        # so the two have to be set together or the test proves nothing.
        monkeypatch.setattr(store_mod, "config", SimpleNamespace(data_dir=tmp_path))
        rules = parse_rules("version: 1\nrules:\n  - {id: A, name: n, description: d}\n",
                            origin="test", sop_id="")
        return AuditStore(
            records_path=tmp_path / "r.jsonl",
            evidence_dir=tmp_path / "evidence",
            rules=rules,
            **kwargs,
        )

    def test_locally_the_paths_are_the_answer(self, tmp_path, monkeypatch):
        # Nothing was wired up, so this disk really is where the records are.
        # The report should keep saying so rather than inventing a URI.
        from computer_use_agent.artifacts import LocalArtifactSink, set_artifact_sink

        set_artifact_sink(LocalArtifactSink(root=tmp_path))
        try:
            summary = self._store(tmp_path, monkeypatch).summary()
        finally:
            set_artifact_sink(None)

        assert summary["records_path"] == str(tmp_path / "r.jsonl")
        assert summary["evidence_dir"] == str(tmp_path / "evidence")

    def test_on_a_container_it_names_the_bucket_and_firestore(self, tmp_path, monkeypatch):
        from computer_use_agent.artifacts import GcsArtifactSink, set_artifact_sink

        set_artifact_sink(GcsArtifactSink(bucket="b", prefix="audits"))
        try:
            store = self._store(
                tmp_path, monkeypatch,
                job_id="abc123",
                records_location="Firestore[audits] cctv_jobs/zhang/jobs/abc123/records",
            )
            summary = store.summary()
        finally:
            set_artifact_sink(None)

        assert summary["records_path"].startswith("Firestore[audits] ")
        assert summary["evidence_dir"] == "gs://b/audits/evidence/abc123"
        assert "/tmp" not in summary["evidence_dir"]

    def test_the_named_directory_is_where_the_frames_went(self, tmp_path, monkeypatch):
        # The one failure mode that cannot be seen by reading the report: a
        # directory that exists, is spelled plausibly, and holds nothing.
        # `_save_evidence` and `_evidence_key` derive the key separately, so
        # this compares the two rather than trusting either.
        from computer_use_agent.artifacts import GcsArtifactSink, set_artifact_sink

        sink = GcsArtifactSink(bucket="b", prefix="audits")
        set_artifact_sink(sink)
        try:
            store = self._store(tmp_path, monkeypatch, job_id="abc123")
            frame_key = str(
                (store.evidence_dir / "w00001_A_0007.jpg").relative_to(tmp_path)
            )
            assert sink.location(frame_key).rsplit("/", 1)[0] == store.summary()["evidence_dir"]
        finally:
            set_artifact_sink(None)

    def test_asking_where_a_bucket_key_goes_needs_no_credentials(self):
        # `summary()` runs at the end of every audit, including one that failed
        # on the way to GCS. Building a client to answer "where would this go"
        # would turn a rendering step into a network call that can fail.
        from computer_use_agent.artifacts import GcsArtifactSink

        sink = GcsArtifactSink(bucket="b", prefix="audits")

        def explode(_key):
            raise AssertionError("location() must not build a client")

        sink._blob = explode
        assert sink.location("evidence/abc/w1.jpg") == "gs://b/audits/evidence/abc/w1.jpg"
        assert GcsArtifactSink(bucket="b", prefix="").location("k.jpg") == "gs://b/k.jpg"

    def test_a_store_nobody_can_visit_says_so_instead_of_guessing(self):
        # Memory is not an address. Returning "" makes `summary()` fall back to
        # the JSONL path, which locally is the truth -- a made-up locator here
        # would override a correct answer with a wrong one.
        assert MemoryJobStore().records_location("zhang@example.com", "abc123") == ""

    def test_firestore_names_the_path_it_actually_writes_to(self):
        # An address that does not match the writes is worse than no address:
        # the reader concludes the records were never written.
        from computer_use_agent.jobs import FirestoreJobStore, _document_safe

        store = FirestoreJobStore.__new__(FirestoreJobStore)
        store._database = "audits"
        store._root = "cctv_jobs"

        where = store.records_location("zhang@example.com", "abc123")
        assert where == (
            f"Firestore[audits] cctv_jobs/{_document_safe('zhang@example.com')}"
            "/jobs/abc123/records"
        )
        # An ordinary email needs no sanitising, which is why the two spellings
        # can drift apart unnoticed. A blank user id is where they show: the
        # write goes to `anonymous`, so the address must too.
        assert store.records_location("", "abc123") == (
            "Firestore[audits] cctv_jobs/anonymous/jobs/abc123/records"
        )

    @pytest.mark.asyncio
    async def test_the_service_tells_the_store_where_the_records_live(self, monkeypatch):
        # The store cannot know: it is handed a coroutine, not a destination.
        # If this wiring is dropped the report silently reverts to `/tmp`.
        from computer_use_agent.audit_service import AuditService
        from computer_use_agent.monitor import monitor

        # The dashboard is a separate service now, so this would be a real POST.
        monkeypatch.setattr(monitor, "fail_session", lambda *a, **k: None)

        class Naming(MemoryJobStore):
            def records_location(self, user_id, job_id):
                return f"somewhere/{user_id}/{job_id}"

        service = AuditService(store=Naming(), runner=BackgroundRunner())
        built = {}

        def build_store(**kwargs):
            built.update(kwargs)
            raise RuntimeError("stop here -- the wiring is all this checks")

        service.build_store = build_store
        job = Job(job_id="abc123", user_id="zhang@example.com", target="u")
        await service.jobs.create(job)
        await service._run(job)          # must not raise: a failed run is a state

        assert built["records_location"] == "somewhere/zhang@example.com/abc123"
        assert (await service.jobs.get(job.user_id, job.job_id)).state == "failed"


class TestTheStandardIsNeverSubstituted:
    """A report judged against the wrong version of the SOP looks exactly like
    a real one. Nobody downstream can tell, so the miss has to be loud."""

    @pytest.fixture(autouse=True)
    def _clean_cache(self):
        from computer_use_agent.analyzer import sop

        sop.clear_sop_cache()
        yield
        sop.clear_sop_cache()

    @staticmethod
    def _settings(monkeypatch, **overrides):
        """Config is frozen, so the module-level name is what gets swapped.

        Same trick the watchdog tests use. Only the fields `sop.py` reads are
        present, so a test that starts depending on a new one fails loudly
        rather than picking up the developer's own `.env`.
        """
        import types

        from computer_use_agent.analyzer import sop

        settings = types.SimpleNamespace(
            sop_bucket="", sop_prefix="sop", default_sop_id="",
            gcp_project="p", sop_rules_path=None,
        )
        for key, value in overrides.items():
            setattr(settings, key, value)
        monkeypatch.setattr(sop, "config", settings)
        return settings

    @pytest.mark.asyncio
    async def test_naming_a_version_with_nowhere_to_store_versions_is_an_error(self, monkeypatch):
        from computer_use_agent.analyzer import sop

        self._settings(monkeypatch, sop_bucket="")
        with pytest.raises(sop.SopUnavailable, match="chagee-store-v3"):
            await sop.load_rules_for("chagee-store-v3")

    @pytest.mark.asyncio
    async def test_a_bucket_with_no_version_named_is_an_error(self, monkeypatch):
        # Picking a version on the customer's behalf is the one thing this
        # must never do -- "v2 or v3" decides whether a finding is a finding.
        from computer_use_agent.analyzer import sop

        self._settings(monkeypatch, sop_bucket="sop-bucket")
        with pytest.raises(sop.SopUnavailable, match="版本号"):
            await sop.load_rules_for(None)

    @pytest.mark.asyncio
    async def test_a_fetch_failure_raises_instead_of_falling_back(self, monkeypatch):
        from computer_use_agent.analyzer import sop

        self._settings(monkeypatch, sop_bucket="sop-bucket")

        async def missing(_sop_id):
            raise sop.SopUnavailable("取不到稽核标准 gs://sop-bucket/sop/v9.yaml：404")

        monkeypatch.setattr(sop, "_fetch_sop", missing)
        with pytest.raises(sop.SopUnavailable):
            await sop.load_rules_for("v9")

    @pytest.mark.asyncio
    async def test_a_version_id_cannot_walk_out_of_the_sop_prefix(self, monkeypatch):
        # The id arrives from a chat message via GE and becomes an object path.
        from computer_use_agent.analyzer import sop

        self._settings(monkeypatch, sop_bucket="sop-bucket")

        async def unreachable(_sop_id):
            raise AssertionError("should never have been fetched")

        monkeypatch.setattr(sop, "_fetch_sop", unreachable)
        for bad in ("../../secrets/key", "v3/../../etc/passwd", "a b", "-leading"):
            with pytest.raises(sop.SopUnavailable):
                await sop.load_rules_for(bad)

    @pytest.mark.asyncio
    async def test_a_fetched_version_is_parsed_and_tagged(self, monkeypatch):
        from computer_use_agent.analyzer import sop

        self._settings(monkeypatch, sop_bucket="sop-bucket", sop_prefix="sop")

        yaml_text = """
version: 3
rules:
  - id: CHK_GLOVE_002
    name: 手套
    description: 接触食材前必须戴手套
"""
        calls = []

        async def fetch(sop_id):
            calls.append(sop_id)
            return yaml_text

        monkeypatch.setattr(sop, "_fetch_sop", fetch)
        rules = await sop.load_rules_for("chagee-store-v3")

        assert rules.sop_id == "chagee-store-v3"
        assert rules.version == 3
        assert rules.ids == ["CHK_GLOVE_002"]
        assert rules.origin == "gs://sop-bucket/sop/chagee-store-v3.yaml"

        # Cached: an audit asks for the rules once per window, and re-reading
        # the bucket each time is hundreds of round trips inside the loop.
        await sop.load_rules_for("chagee-store-v3")
        assert calls == ["chagee-store-v3"]

    @pytest.mark.asyncio
    async def test_no_bucket_and_no_version_is_the_local_file(self, monkeypatch):
        # Not a fallback: on a workstation `SOP_RULES_PATH` *is* the standard,
        # rather than one of several the customer chooses between.
        from computer_use_agent.analyzer import sop

        self._settings(monkeypatch, sop_bucket="", default_sop_id="")

        rules = await sop.load_rules_for(None)
        assert rules.rules
        assert rules.sop_id == ""

    def test_every_record_says_which_standard_judged_it(self, tmp_path):
        # "Was this judged under v2 or v3" is the first question asked when a
        # finding is disputed, and by then the standard has moved on.
        from computer_use_agent.analyzer.sop import parse_rules
        from computer_use_agent.store import AuditStore

        rules = parse_rules(
            "version: 3\nrules:\n  - {id: A, name: n, description: d}\n",
            origin="test", sop_id="chagee-store-v3",
        )
        store = AuditStore(
            records_path=tmp_path / "r.jsonl", evidence_dir=tmp_path / "ev", rules=rules,
        )
        assert store._rules.sop_id == "chagee-store-v3"


class TestAStructuredRequestIsCheckedNotTrusted:
    """GE does the first pass, so what arrives is fields rather than a
    sentence. GE is a language model too: it sends nonsense at the same rate a
    person does, and the refusals have to be the same ones."""

    def test_clock_strings_and_numbers_mean_the_same_thing(self):
        from computer_use_agent.intent import from_fields

        by_number = from_fields("https://x/v", start=900, end=1500)
        by_clock = from_fields("https://x/v", start="15:00", end="25:00")

        assert by_number.request == by_clock.request
        assert by_number.request.start_seconds == 900.0
        assert by_number.request.duration_seconds == 600.0

    def test_a_shop_name_is_not_a_video_address(self):
        # Turning one into a search would audit whatever came back first.
        from computer_use_agent.intent import from_fields

        with pytest.raises(ValueError):
            from_fields("霸王茶姬望京店")
        with pytest.raises(ValueError):
            from_fields("")

    def test_an_end_before_the_start_is_refused(self):
        from computer_use_agent.intent import UnreadableTimeSpan, from_fields

        with pytest.raises(UnreadableTimeSpan):
            from_fields("https://x/v", start="15:00", end="14:00")

    def test_an_explicit_end_wins_over_an_inferred_duration(self):
        from computer_use_agent.intent import from_fields

        got = from_fields("https://x/v", start=0, end=300, duration=99999)
        assert got.request.duration_seconds == 300.0

    def test_no_time_at_all_means_the_whole_recording(self):
        from computer_use_agent.intent import from_fields

        got = from_fields("https://x/v")
        assert got.request.start_seconds == 0.0
        assert got.request.duration_seconds is None

    def test_the_reading_is_echoed_back_for_a_human_to_check(self):
        # A misreading has to be visible in the first line of output, not at
        # the end of a four-minute run.
        from computer_use_agent.intent import from_fields

        assert from_fields("https://x/v", start=900, end=1500).reading == "从 15:00 看到 25:00"

    def test_a_time_that_cannot_be_read_stops_the_run(self):
        from computer_use_agent.intent import UnreadableTimeSpan, from_fields

        with pytest.raises(UnreadableTimeSpan):
            from_fields("https://x/v", start="高峰期")


# --- preflight ---------------------------------------------------------------


class _FakePage:
    def __init__(self, title="霸王茶姬 望京店 监控"):
        self._title = title

    async def title(self):
        return self._title

    async def screenshot(self, **_kwargs):
        return b"jpegbytes"


class _FakeNavigator:
    def __init__(self, duration=None):
        self._duration = duration

    async def read_playback_state(self, _page):
        return {"duration": self._duration, "current_time": 0.0, "ended": False}


def _preflight(source, *, start=0.0, duration=None, player_duration=None, page=None):
    """Runs `preflight` against a fake browser session.

    Built the way the rest of this suite builds pipelines: the real method on
    an object holding only what the method reads. Nothing here opens Chromium.
    """
    from computer_use_agent.pipeline import AuditPipeline, AuditRequest, _Session

    pipeline = AuditPipeline.__new__(AuditPipeline)
    pipeline.on_status = None
    request = AuditRequest(target="https://x/v", start_seconds=start, duration_seconds=duration)

    session = _Session(
        page=page or _FakePage(),
        context=None,
        navigator=_FakeNavigator(player_duration),
        source=source,
        work_dir=Path("/tmp"),
        platform="bilibili",
    )

    @contextlib.asynccontextmanager
    async def fake_session(_request, **_kwargs):
        yield session

    pipeline._browser_session = fake_session
    return pipeline.preflight(request, job_id="job7")


class TestPreflightAnswersBeforeAnythingIsSpent:
    """The call that has to fit inside one GE turn. The audit it describes
    cannot, which is the whole reason the two are separate."""

    def _source(self, mode="stream", duration=None):
        from computer_use_agent.capture.types import CaptureSource

        return CaptureSource(
            mode=mode, url="https://x/m.m4s", duration_seconds=duration,
            reason="ffprobe opened the player's own media URL",
        )

    @pytest.fixture(autouse=True)
    def _local_sink(self, tmp_path, monkeypatch):
        from computer_use_agent.artifacts import LocalArtifactSink, set_artifact_sink

        set_artifact_sink(LocalArtifactSink(root=tmp_path))
        yield
        set_artifact_sink(None)

    @pytest.mark.asyncio
    async def test_a_span_inside_the_recording_is_approved(self):
        got = await _preflight(self._source(duration=3600.0), start=900.0, duration=600.0)

        assert got.ok is True
        assert got.span_available is True
        assert got.problem is None
        assert got.capture_mode == "stream"
        assert got.video_duration_seconds == 3600.0

    @pytest.mark.asyncio
    async def test_a_span_past_the_end_is_flagged_but_still_runnable(self):
        # There is real footage to audit, just not all of it. Refusing outright
        # would throw away the fifty minutes that do exist.
        got = await _preflight(self._source(duration=3600.0), start=3000.0, duration=1800.0)

        assert got.ok is True
        assert got.span_available is False
        assert "视频只有 01:00:00 长" in got.problem

    @pytest.mark.asyncio
    async def test_a_start_past_the_end_is_refused(self):
        # Nothing to audit at all. Starting would produce an empty report
        # rather than an error, which is the worse of the two.
        got = await _preflight(self._source(duration=3600.0), start=4000.0, duration=600.0)

        assert got.ok is False
        assert got.span_available is False
        assert "已经超出了视频末尾" in got.problem

    @pytest.mark.asyncio
    async def test_landing_seconds_short_is_not_reported_as_a_shortfall(self):
        # The last clip is cut on a window boundary. `_coverage` forgives the
        # same slack at the other end of the run; telling the customer about it
        # here would train them to ignore the warning.
        got = await _preflight(self._source(duration=600.0), start=0.0, duration=605.0)

        assert got.ok is True
        assert got.span_available is True

    @pytest.mark.asyncio
    async def test_the_player_is_asked_when_ffprobe_could_not_say(self):
        # Plan B has no container for ffprobe to read, and the <video>
        # element's duration is what the platform's own timeline is drawn from.
        got = await _preflight(
            self._source(mode="screen", duration=None), start=0.0,
            duration=600.0, player_duration=300.0,
        )
        assert got.video_duration_seconds == 300.0
        assert got.span_available is False

    @pytest.mark.asyncio
    async def test_a_live_stream_is_not_treated_as_a_zero_length_recording(self):
        # A live <video> reports Infinity. `float("inf")` compares larger than
        # every requested end, so it would silently pass every span check
        # rather than admitting we do not know.
        got = await _preflight(
            self._source(mode="screen"), start=0.0, duration=600.0,
            player_duration=float("inf"),
        )
        assert got.video_duration_seconds is None
        assert got.ok is True
        assert got.span_available is True

    @pytest.mark.asyncio
    async def test_an_unopenable_page_is_an_answer_not_a_stack_trace(self):
        from computer_use_agent.pipeline import AuditPipeline, AuditRequest

        pipeline = AuditPipeline.__new__(AuditPipeline)
        pipeline.on_status = None

        @contextlib.asynccontextmanager
        async def broken(_request, **_kwargs):
            raise RuntimeError("net::ERR_NAME_NOT_RESOLVED")
            yield  # pragma: no cover

        pipeline._browser_session = broken
        got = await pipeline.preflight(AuditRequest(target="https://nope/v"))

        assert got.ok is False
        assert "打不开这个视频" in got.problem
        assert "ERR_NAME_NOT_RESOLVED" in got.problem

    @pytest.mark.asyncio
    async def test_the_cover_still_is_stored_under_the_job(self, tmp_path):
        # So the customer can see it is their shop and not somebody else's.
        got = await _preflight(self._source(duration=3600.0), duration=60.0)

        assert got.cover_frame == "job7/cover.jpg"
        assert (tmp_path / "job7" / "cover.jpg").read_bytes() == b"jpegbytes"

    @pytest.mark.asyncio
    async def test_a_missing_cover_does_not_fail_the_preflight(self):
        # Every other question was answered. Failing over a screenshot would
        # throw all of that away.
        class NoScreenshot(_FakePage):
            async def screenshot(self, **_kwargs):
                raise RuntimeError("target closed")

        got = await _preflight(self._source(duration=3600.0), duration=60.0,
                               page=NoScreenshot())
        assert got.ok is True
        assert got.cover_frame is None

    @pytest.mark.asyncio
    async def test_the_result_survives_being_written_to_firestore(self):
        # It is stored on the job and read back on another container, so it has
        # to be plain data all the way down.
        import json

        got = await _preflight(self._source(duration=3600.0), duration=60.0)
        assert json.loads(json.dumps(got.as_dict()))["capture_mode"] == "stream"


# --- the three operations ----------------------------------------------------


class _StubPipeline:
    """Stands in for the real pipeline, which wants Chromium and Vertex.

    Recorded on the class so a test can see what the service handed it -- the
    SOP version in particular, since passing the wrong one is invisible in the
    output.
    """

    built = []

    def __init__(self, store=None, rules=None, **kwargs):
        self.store = store
        self.rules = rules
        self.kwargs = kwargs
        _StubPipeline.built.append(self)
        self.preflight_result = None
        # The shape `AuditStore.summary()` plus `_coverage()` really produce.
        # A short-cut dict here would let a missing key through to production.
        self.summary = {
            "windows_analyzed": 3, "windows_failed": 0, "violations": 1,
            "red_line_violations": 0, "input_tokens": 100, "output_tokens": 20,
            "records_path": "/tmp/r.jsonl", "evidence_dir": "/tmp/ev",
            "capture_mode": "screen", "elapsed_seconds": 42.0,
            "stopped_because": "采集结束", "complete": True,
            "incomplete_reason": None, "stopped_kind": None,
            "requested_start_seconds": 0.0, "requested_end_seconds": 600.0,
            "covered_from_seconds": 0.0, "covered_to_seconds": 600.0,
        }
        self.run_error = None
        self.ran = asyncio.Event()

    async def preflight(self, request, job_id="preflight"):
        from computer_use_agent.pipeline import PreflightResult

        return self.preflight_result or PreflightResult(
            ok=True, target=request.target, platform="bilibili",
            capture_mode="screen", capture_reason="录屏",
            video_duration_seconds=3600.0,
            requested_start_seconds=request.start_seconds,
            requested_end_seconds=request.end_seconds,
        )

    async def run(self, request):
        self.ran.set()
        if self.run_error:
            raise self.run_error
        return dict(self.summary)


@pytest.fixture
def service(monkeypatch, tmp_path):
    """A service on an in-memory store, with the browser stubbed out."""
    import functools

    from computer_use_agent import audit_service as svc_mod
    from computer_use_agent.jobs import BackgroundRunner
    from computer_use_agent.store import AuditStore

    _StubPipeline.built = []
    built = _StubPipeline

    async def local_rules(sop_id=None):
        from computer_use_agent.analyzer.sop import parse_rules

        return parse_rules(
            "version: 3\nrules:\n  - {id: A, name: n, description: d}\n",
            origin="test", sop_id=sop_id or "",
        )

    monkeypatch.setattr(svc_mod, "load_rules_for", local_rules)
    service = svc_mod.AuditService(store=MemoryJobStore(), runner=BackgroundRunner())
    service.build_pipeline = built
    # A real AuditStore, but writing under tmp_path rather than the repo's own
    # checkpoints directory -- which holds real evidence frames.
    service.build_store = functools.partial(
        AuditStore, records_path=tmp_path / "records.jsonl", evidence_dir=tmp_path / "ev",
    )
    return service


def _request(target="https://x/v", start=0.0, duration=600.0):
    from computer_use_agent.pipeline import AuditRequest

    return AuditRequest(target=target, start_seconds=start, duration_seconds=duration)


class TestNothingIsSpentBeforeTheCustomerSaysYes:
    """`preflight` looks; `start_audit` spends. A customer who never confirms
    is billed for one page load."""

    @pytest.mark.asyncio
    async def test_preflight_leaves_the_job_waiting_for_confirmation(self, service):
        job = await service.preflight("zhang@example.com", _request(), session_id="s1")

        assert job.state == "ready"
        assert job.preflight["capture_mode"] == "screen"
        assert len(_StubPipeline.built) == 1
        # Looked at the video; did not audit it.
        assert not _StubPipeline.built[0].ran.is_set()

    @pytest.mark.asyncio
    async def test_a_job_that_was_never_confirmed_never_runs(self, service):
        await service.preflight("zhang@example.com", _request())
        await service.runner.drain(timeout=0.1)
        assert service.runner.running == 0

    @pytest.mark.asyncio
    async def test_a_refused_preflight_is_still_on_the_record(self, service):
        from computer_use_agent.pipeline import PreflightResult

        class Refusing(_StubPipeline):
            async def preflight(self, request, job_id="preflight"):
                return PreflightResult(
                    ok=False, target=request.target, platform="bilibili",
                    problem="打不开这个视频：net::ERR_NAME_NOT_RESOLVED",
                )

        service.build_pipeline = Refusing
        job = await service.preflight("zhang@example.com", _request())

        assert job.state == "rejected"
        assert job.finished
        assert "打不开" in job.error
        # A job id means the attempt is on the record rather than vanishing.
        assert await service.jobs.get("zhang@example.com", job.job_id) is not None

    @pytest.mark.asyncio
    async def test_a_missing_sop_version_is_caught_before_the_browser_opens(
        self, service, monkeypatch
    ):
        # Finding out after a minute of navigation wastes the minute and tells
        # the customer nothing they could not have been told immediately.
        from computer_use_agent import audit_service as svc_mod
        from computer_use_agent.analyzer import SopUnavailable

        async def missing(_sop_id=None):
            raise SopUnavailable("取不到稽核标准 gs://b/sop/v9.yaml：404")

        monkeypatch.setattr(svc_mod, "load_rules_for", missing)
        job = await service.preflight("zhang@example.com", _request(), sop_id="v9")

        assert job.state == "rejected"
        assert "v9" in job.error
        assert _StubPipeline.built == []

    @pytest.mark.asyncio
    async def test_the_standard_that_was_resolved_is_the_one_recorded(self, service):
        job = await service.preflight("z@example.com", _request(), sop_id="chagee-store-v3")

        assert job.sop_id == "chagee-store-v3"
        assert _StubPipeline.built[0].rules.sop_id == "chagee-store-v3"


class TestConfirmingStartsTheAuditAndReturns:
    """The `await` that must not be there: work awaited inside a GE turn is
    cancelled at 900s, and an audit outlives that."""

    @pytest.mark.asyncio
    async def test_start_returns_before_the_audit_finishes(self, service):
        ready = await service.preflight("z@example.com", _request())
        job = await service.start_audit("z@example.com", ready.job_id)

        assert job.state == "running"
        assert service.runner.running == 1

        await service.runner.drain(timeout=2.0)
        assert (await service.jobs.get("z@example.com", job.job_id)).state == "done"

    @pytest.mark.asyncio
    async def test_the_finished_report_survives_the_instance(self, service):
        ready = await service.preflight("z@example.com", _request())
        await service.start_audit("z@example.com", ready.job_id)
        await service.runner.drain(timeout=2.0)

        done = await service.jobs.get("z@example.com", ready.job_id)
        assert done.result["summary"]["windows_analyzed"] == 3
        # Rendered now, while the store is still in memory. A later turn may
        # land on another container with nothing but this document.
        assert isinstance(done.result["report"], str) and done.result["report"]

    @pytest.mark.asyncio
    async def test_the_completion_line_names_the_keys_the_summary_really_has(
        self, service, caplog
    ):
        # It shipped reading `violation_count`, which the summary does not
        # have, so the one line saying how the audit went logged "None
        # violations". A `.get` miss is invisible until somebody reads the log
        # for the reason the log exists.
        import logging

        ready = await service.preflight("z@example.com", _request())
        with caplog.at_level(logging.INFO, logger="cctv_audit.service"):
            await service.start_audit("z@example.com", ready.job_id)
            await service.runner.drain(timeout=2.0)

        line = [m for m in caplog.messages if "done in" in m][0]
        assert "None" not in line
        assert "3 windows, 1 violations" in line

    @pytest.mark.asyncio
    async def test_saying_确认_twice_does_not_start_two_audits(self, service):
        ready = await service.preflight("z@example.com", _request())
        await service.start_audit("z@example.com", ready.job_id)
        await service.start_audit("z@example.com", ready.job_id)

        assert service.runner.running == 1
        await service.runner.drain(timeout=2.0)
        assert sum(1 for p in _StubPipeline.built if p.ran.is_set()) == 1

    @pytest.mark.asyncio
    async def test_a_rejected_job_cannot_be_confirmed(self, service):
        from computer_use_agent.pipeline import PreflightResult

        class Refusing(_StubPipeline):
            async def preflight(self, request, job_id="preflight"):
                return PreflightResult(ok=False, target=request.target,
                                       platform="x", problem="视频不存在")

        service.build_pipeline = Refusing
        job = await service.preflight("z@example.com", _request())

        with pytest.raises(ValueError, match="视频不存在"):
            await service.start_audit("z@example.com", job.job_id)

    @pytest.mark.asyncio
    async def test_another_user_cannot_confirm_your_job(self, service):
        ready = await service.preflight("zhang@example.com", _request())

        with pytest.raises(LookupError):
            await service.start_audit("li@example.com", ready.job_id)

    @pytest.mark.asyncio
    async def test_an_audit_that_dies_says_so_instead_of_hanging_at_running(self, service):
        # Nobody is awaiting the task. Without this the customer polls
        # "running" forever and the traceback goes nowhere.
        class Exploding(_StubPipeline):
            async def run(self, request):
                self.ran.set()
                raise RuntimeError("ffmpeg exited 1")

        ready = await service.preflight("z@example.com", _request())
        service.build_pipeline = Exploding
        await service.start_audit("z@example.com", ready.job_id)
        await service.runner.drain(timeout=2.0)

        failed = await service.jobs.get("z@example.com", ready.job_id)
        assert failed.state == "failed"
        assert "ffmpeg exited 1" in failed.error


def _impatient(monkeypatch, *, first=0.05, stall=0.05, hard=100.0, cleanup=0.2):
    """Shrinks the watchdog's limits so a wedge can be reproduced in a test.

    The limits are minutes in production, which is the right size for a real
    audit and the wrong size for a test suite. Only the numbers move; the
    decisions being tested are the production ones.
    """
    from computer_use_agent import audit_service as svc_mod

    monkeypatch.setattr(svc_mod, "_WATCHDOG_TICK_SECONDS", 0.01)
    monkeypatch.setattr(svc_mod, "_FIRST_WINDOW_SECONDS", first)
    monkeypatch.setattr(svc_mod, "_STALL_SECONDS", stall)
    monkeypatch.setattr(svc_mod, "_HARD_LIMIT_SECONDS", hard)
    monkeypatch.setattr(svc_mod, "_CLEANUP_SECONDS", cleanup)


class TestAWedgedAuditFailsInsteadOfRunningForever:
    """Job `d7680d` sat at `running` for 62 minutes having produced nothing,
    telling the customer "已分析 0 个窗口" the whole time.

    The pipeline does carry a wall-clock budget, but it is only consulted
    between captured windows -- so a capture that never comes round again never
    reaches the check. Being stuck is exactly the state that cannot report
    itself, which is why the watchdog has to sit outside the run.
    """

    @pytest.mark.asyncio
    async def test_a_run_that_never_produces_a_window_is_killed_and_says_why(
        self, service, monkeypatch
    ):
        _impatient(monkeypatch)

        class Wedged(_StubPipeline):
            async def run(self, request):
                self.ran.set()
                await asyncio.sleep(3600)

        ready = await service.preflight("z@example.com", _request())
        service.build_pipeline = Wedged
        await service.start_audit("z@example.com", ready.job_id)
        await service.runner.drain(timeout=5.0)

        job = await service.jobs.get("z@example.com", ready.job_id)
        assert job.state == "failed"
        # The reason is the entire point. "一直在跑" is what it replaced.
        assert "一个窗口都没产出" in job.error
        assert "登录" in job.error  # names something the customer can act on

    @pytest.mark.asyncio
    async def test_the_run_is_actually_stopped_not_just_marked_failed(
        self, service, monkeypatch
    ):
        # A job marked failed while its browser and ffmpeg keep running would
        # leave the next audit on this instance fighting for the same CPU.
        _impatient(monkeypatch)
        stopped = asyncio.Event()

        class Wedged(_StubPipeline):
            async def run(self, request):
                self.ran.set()
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    stopped.set()
                    raise

        ready = await service.preflight("z@example.com", _request())
        service.build_pipeline = Wedged
        await service.start_audit("z@example.com", ready.job_id)
        await service.runner.drain(timeout=5.0)

        assert stopped.is_set()

    @pytest.mark.asyncio
    async def test_a_run_making_progress_is_left_alone(self, service, monkeypatch):
        # The failure mode of a watchdog is killing healthy work. A slow source
        # that is still returning windows is slow, not stuck.
        _impatient(monkeypatch, first=1.0, stall=1.0)
        job = Job(job_id="j1", user_id="z@example.com", state="running")
        progress = {"windows": 0, "beat": time.monotonic()}

        async def working():
            for _ in range(5):
                await asyncio.sleep(0.05)
                progress["windows"] += 1
                progress["beat"] = time.monotonic()
            return "finished"

        task = asyncio.ensure_future(working())
        reason = await service._watch(job, task, progress, time.monotonic())

        assert reason is None
        assert task.result() == "finished"

    @pytest.mark.asyncio
    async def test_progress_that_stops_halfway_is_caught_too(
        self, service, monkeypatch
    ):
        # Different sentence from the no-windows case on purpose: "got 4
        # windows then stopped" points somewhere else than "never started".
        _impatient(monkeypatch, first=10.0, stall=0.05)
        job = Job(job_id="j2", user_id="z@example.com", state="running")
        progress = {"windows": 4, "beat": time.monotonic()}
        task = asyncio.ensure_future(asyncio.sleep(3600))

        reason = await service._watch(job, task, progress, time.monotonic())

        assert reason and "没有新的分析结果" in reason
        assert "4 个窗口" in reason
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_steady_progress_still_ends_at_the_hard_limit(
        self, service, monkeypatch
    ):
        # A source slow enough to produce a window every few minutes forever
        # would never trip the stall check and would bill for an hour.
        _impatient(monkeypatch, first=10.0, stall=10.0, hard=0.05)
        job = Job(job_id="j3", user_id="z@example.com", state="running")
        progress = {"windows": 2, "beat": time.monotonic()}
        task = asyncio.ensure_future(asyncio.sleep(3600))

        reason = await service._watch(job, task, progress, time.monotonic())

        assert reason and "上限" in reason
        assert "把时间段改短" in reason

    @pytest.mark.asyncio
    async def test_a_run_that_ignores_cancellation_does_not_hang_the_watchdog(
        self, service, monkeypatch
    ):
        # The cleanup being waited on is the browser and ffmpeg -- either of
        # which may be the thing that was wedged. An unbounded wait there would
        # move the hang from the audit into the watchdog.
        _impatient(monkeypatch, cleanup=0.05)
        job = Job(job_id="j4", user_id="z@example.com", state="running")

        async def stubborn():
            for _ in range(3):
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(0.3)
            return "escaped"

        task = asyncio.ensure_future(stubborn())
        await asyncio.sleep(0.01)
        began = time.monotonic()
        await service._cancel(job, task)

        assert time.monotonic() - began < 0.25
        assert not task.done()
        await task  # let it finish rather than leaking it into the next test

    @pytest.mark.asyncio
    async def test_the_heartbeat_is_stamped_on_every_window_not_on_the_throttle(
        self, service, monkeypatch
    ):
        # The Firestore progress write is throttled to 15s. If the heartbeat
        # rode along with it, a healthy run would look silent for 15 seconds at
        # a time and the stall limit would be measuring the throttle.
        ready = await service.preflight("z@example.com", _request())
        job = await service.jobs.get("z@example.com", ready.job_id)
        state = {"last": time.time(), "windows": 0, "violations": 0, "beat": 0.0}
        writer = service._record_writer(job, state)

        await writer({"time_range": "00:00-00:15", "violation_count": 0})

        assert state["windows"] == 1
        assert state["beat"] > 0.0        # stamped despite the throttle


class TestAskingHowItIsGoing:
    @pytest.mark.asyncio
    async def test_a_follow_up_in_the_same_thread_needs_no_job_number(self, service):
        # GE passes `session_id` on every turn, so "好了吗" can be answered
        # without making the customer quote a six-character id back.
        ready = await service.preflight("z@example.com", _request(), session_id="thread-1")

        found = await service.get_status("z@example.com", session_id="thread-1")
        assert found.job_id == ready.job_id

    @pytest.mark.asyncio
    async def test_a_thread_with_no_job_returns_nothing_rather_than_someone_elses(
        self, service
    ):
        await service.preflight("z@example.com", _request(), session_id="thread-1")
        assert await service.get_status("z@example.com", session_id="thread-9") is None

    @pytest.mark.asyncio
    async def test_the_conversation_shortcut_is_not_an_authorisation(self, service):
        # Same scope as every other read: knowing the thread id is not enough.
        await service.preflight("zhang@example.com", _request(), session_id="thread-1")
        assert await service.get_status("li@example.com", session_id="thread-1") is None

    @pytest.mark.asyncio
    async def test_an_unknown_number_is_not_an_error(self, service):
        assert await service.get_status("z@example.com", job_id="ffffff") is None

    @pytest.mark.asyncio
    async def test_progress_is_written_as_windows_come_back(self, service, monkeypatch):
        from computer_use_agent import audit_service as svc_mod

        monkeypatch.setattr(svc_mod, "_PROGRESS_EVERY_SECONDS", 0.0)
        ready = await service.preflight("z@example.com", _request())

        writer = service._record_writer(await service.jobs.get("z@example.com", ready.job_id))
        await writer({"window_index": 0, "time_range": "00:00 - 00:15", "violation_count": 1})
        await writer({"window_index": 1, "time_range": "00:15 - 00:30", "violation_count": 2})

        job = await service.jobs.get("z@example.com", ready.job_id)
        assert job.progress["windows_analyzed"] == 2
        assert job.progress["violations_so_far"] == 3
        assert len(await service.jobs.records("z@example.com", ready.job_id)) == 2

    @pytest.mark.asyncio
    async def test_every_window_is_kept_even_when_progress_is_throttled(self, service):
        # The audit trail must be complete; only the status a customer polls
        # is allowed to lag.
        ready = await service.preflight("z@example.com", _request())
        writer = service._record_writer(await service.jobs.get("z@example.com", ready.job_id))
        for i in range(20):
            await writer({"window_index": i, "time_range": f"w{i}", "violation_count": 0})

        assert len(await service.jobs.records("z@example.com", ready.job_id)) == 20

    @pytest.mark.asyncio
    async def test_a_broken_report_does_not_strand_the_job_at_running(self, service, monkeypatch):
        # The audit finished and the numbers are in `summary`. Losing the
        # rendered version must not leave a customer polling forever.
        from computer_use_agent.agent import CctvAuditAgent

        def explode(_summary, _store):
            raise KeyError("windows_failed")

        monkeypatch.setattr(CctvAuditAgent, "_report", staticmethod(explode))
        ready = await service.preflight("z@example.com", _request())
        await service.start_audit("z@example.com", ready.job_id)
        await service.runner.drain(timeout=2.0)

        job = await service.jobs.get("z@example.com", ready.job_id)
        assert job.state == "done"
        assert job.result["summary"]["windows_analyzed"] == 3
        assert "排版失败" in job.result["report"]


class TestBothEntrancesJudgeByTheSameStandard:
    """`adk web` locally and Gemini Enterprise in the cloud run the same audit.

    They resolve the standard through the same call for one reason: a local run
    that quietly reads the on-disk YAML while the cloud run errors on a missing
    version is two verdicts on the same footage with nothing on screen saying
    they were measured differently.
    """

    def test_the_local_entrance_has_no_way_to_load_rules_unscoped(self):
        # `load_rules()` ignores SOP_BUCKET and DEFAULT_SOP_ID entirely. If it
        # comes back into agent.py the divergence comes back with it, so the
        # import is what this guards -- the behaviour is untestable without ADK.
        import computer_use_agent.agent as agent_mod

        assert not hasattr(agent_mod, "load_rules"), (
            "agent.py must resolve the standard through load_rules_for()"
        )
        assert hasattr(agent_mod, "load_rules_for")

    def test_the_banner_names_where_the_standard_actually_came_from(self):
        # Not `config.sop_rules_path`: once the standard is fetched from a
        # bucket, the local filename setting is no longer what is in force, and
        # a banner that keeps printing it is confidently wrong.
        from computer_use_agent.agent import CctvAuditAgent
        from computer_use_agent.analyzer.sop import SopRule, SopRuleSet
        from computer_use_agent.intent import Intent
        from computer_use_agent.pipeline import AuditRequest

        rules = SopRuleSet(
            version=3,
            rules=[SopRule(id="CHK_GLOVE_002", name="戴手套", description="x")],
            sop_id="chagee-store-v3",
            origin="gs://example-bucket/sop/chagee-store-v3.yaml",
        )
        intent = Intent(request=AuditRequest(target="https://x/v"), reading="整段")

        banner = CctvAuditAgent._start_banner(intent, rules)

        assert "gs://example-bucket/sop/chagee-store-v3.yaml" in banner
        assert "v3" in banner and "CHK_GLOVE_002" in banner


class TestTheOperatorIsToldWhenThePictureStops:
    """On Agent Runtime the pipeline's status events reached the log and
    stopped there.

    `agent.py` has always wired `on_status` into the local chat stream; the
    cloud path never did. That gap is exactly how job 692b53 went wrong in
    front of someone: bilibili's login nag paused the player, the dashboard
    showed one still for two minutes, and the only place that said so was Cloud
    Logging -- which nobody demoing this has open.
    """

    class _Monitor:
        def __init__(self, explode=False):
            self.notes = []
            self.explode = explode

        def note(self, action, detail=""):
            if self.explode:
                raise RuntimeError("dashboard is down")
            self.notes.append((action, detail))

    def _announce(self, monitor, event, payload):
        from computer_use_agent.audit_service import _announce

        _announce(monitor, event, payload)

    def test_a_frozen_picture_reaches_the_action_banner(self):
        monitor = self._Monitor()
        self._announce(monitor, "picture_frozen", {"reason": "网页画面卡在 01:19 不动了"})

        assert monitor.notes == [("画面暂停", "网页画面卡在 01:19 不动了")]

    def test_recovery_attempts_are_numbered(self):
        monitor = self._Monitor()
        self._announce(monitor, "playback_recovering", {"at_seconds": 79.0, "attempt": 2})

        assert monitor.notes == [("正在恢复播放", "第 2 次尝试")]

    def test_the_routine_events_are_not_narrated(self):
        # Every event is in the log already. A banner that reports each window
        # is a banner nobody is reading by the time it matters.
        monitor = self._Monitor()
        for event in ("window", "navigating", "preflight", "finished"):
            self._announce(monitor, event, {})

        assert monitor.notes == []

    def test_an_unreachable_dashboard_cannot_take_the_audit_down(self):
        # `note` is a network call to another Cloud Run service. A status line
        # is the least important thing in the process.
        self._announce(self._Monitor(explode=True), "picture_frozen", {"reason": "x"})

    @pytest.mark.asyncio
    async def test_the_running_audit_is_given_a_status_hook_at_all(self, service):
        # The mapping above is worth nothing if nobody passes it in -- which
        # was the actual bug.
        job = await service.preflight("zhang@example.com", _request(), session_id="s1")
        await service.start_audit("zhang@example.com", job.job_id)
        await service.runner.drain(timeout=5)

        assert callable(_StubPipeline.built[-1].kwargs.get("on_status"))
