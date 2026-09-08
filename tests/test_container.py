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

"""The container's half of the conversation.

Gemini Enterprise sends one method and a sentence; everything about which of
the three operations runs is decided here. These tests pin the decisions that
are expensive to get wrong -- starting a forty-minute audit when the customer
asked for the status, or answering a stranger's job.

No model is called: `read_intent` is the seam, and it is faked. What is under
test is the routing and the wording, not Gemini's reading comprehension.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from computer_use_agent import server
from computer_use_agent.jobs import BackgroundRunner, Job, MemoryJobStore
from computer_use_agent.turn import _Decision, Turn, read_turn


# -- helpers -----------------------------------------------------------------

def _payload(text: str, *, session="s1", user="zhang@example.com", events=None):
    """What GE puts on the wire: a JSON string inside the JSON body."""
    request = {
        "message": {"role": "user", "parts": [{"text": text}]},
        "session_id": session,
        "user_id": user,
    }
    if events is not None:
        request["events"] = events
    return {"request_json": json.dumps(request, ensure_ascii=False)}


async def _collect(payload, service, decision=None, monkeypatch=None):
    """Runs one turn and returns the sentences it said."""
    if decision is not None:
        async def fake_read_intent(_turn):
            return decision
        monkeypatch.setattr(server, "read_intent", fake_read_intent)

    said = []
    async for chunk in server.serve_turn(payload, service=service):
        assert list(chunk.keys())[0] == "events", "envelope must lead with events"
        said.append(chunk["events"][0]["content"]["parts"][0]["text"])
    return said


class _Service:
    """An AuditService with the browser and the model taken out.

    Only the three methods the server calls, so a change in their signatures
    shows up here rather than in production.
    """

    def __init__(self):
        self.jobs = MemoryJobStore()
        self.runner = BackgroundRunner()
        self.started = []
        self.preflight_result = None

    async def preflight(self, user_id, request, session_id="", sop_id=""):
        job = self.preflight_result or Job(
            job_id="abc123", user_id=user_id, session_id=session_id,
            state="ready", target=request.target,
            preflight={
                "ok": True, "capture_mode": "stream", "title": "某某门店",
                "video_duration_seconds": 3600.0,
                "requested_start_seconds": 60.0, "requested_end_seconds": 90.0,
                "span_available": True, "problem": None,
            },
        )
        job.user_id, job.session_id = user_id, session_id
        await self.jobs.create(job)
        return job

    async def start_audit(self, user_id, job_id):
        self.started.append((user_id, job_id))
        return await self.jobs.update(user_id, job_id, state="running")

    async def get_status(self, user_id, job_id="", session_id=""):
        if job_id:
            return await self.jobs.get(user_id, job_id)
        for job in await self.jobs.recent(user_id, limit=20):
            if job.session_id == session_id:
                return job
        return None


@pytest.fixture
def service():
    return _Service()


# -- what GE puts on the wire -------------------------------------------------

class TestReadingWhatGeSends:
    """`request_json` is double-encoded and undocumented. It was read off the
    wire once; these tests are what keeps that reading from rotting."""

    def test_the_payload_is_a_json_string_inside_the_json(self):
        turn = read_turn(_payload("查一下 14:00 到 15:00"))
        assert turn.text == "查一下 14:00 到 15:00"
        assert turn.user_id == "zhang@example.com"
        assert turn.session_id == "s1"

    def test_a_broken_payload_is_an_empty_turn_not_an_exception(self):
        # The customer is waiting on a sentence either way; a 500 gives them a
        # platform error page instead of one.
        turn = read_turn({"request_json": "{not json"})
        assert turn.text == "" and turn.user_id == ""

    def test_history_includes_rounds_this_container_never_served(self):
        # GE replays the whole thread, including what its own assistant
        # answered. That is how a URL named before the customer @-ed us is
        # still available to read.
        turn = read_turn(_payload("看 14:00 到 15:00", events=[
            {"author": "user", "content": {"parts": [{"text": "https://x/v 这个店"}]}},
            {"author": "root_agent", "content": {"parts": [{"text": "好的"}]}},
        ]))
        assert [h["role"] for h in turn.history] == ["user", "assistant"]
        assert "https://x/v" in turn.transcript
        assert turn.transcript.endswith("[user] 看 14:00 到 15:00")


# -- the routing ---------------------------------------------------------------

class TestANewRequestLooksBeforeItSpends:

    @pytest.mark.asyncio
    async def test_preflight_runs_and_asks_for_a_yes(self, service, monkeypatch):
        said = await _collect(
            _payload("稽核 https://x/v 第1分钟到第1分半"), service,
            _Decision(action="audit", target_url="https://x/v",
                      start_seconds=60.0, end_seconds=90.0),
            monkeypatch,
        )
        joined = "\n".join(said)
        assert "某某门店" in joined
        assert "抓流" in joined                    # capture_mode rendered in 人话
        assert "abc123" in joined                  # the number to quote back
        assert "确认" in joined
        assert service.started == [], "nothing may start before the customer says yes"

    @pytest.mark.asyncio
    async def test_a_rejected_preflight_says_why_and_does_not_ask_for_a_yes(
        self, service, monkeypatch
    ):
        service.preflight_result = Job(
            job_id="dead01", user_id="zhang@example.com", state="rejected",
            error="打不开这个视频",
            preflight={"ok": False, "problem": "打不开这个视频：HTTP 412 风控"},
        )
        said = await _collect(
            _payload("稽核 https://x/v"), service,
            _Decision(action="audit", target_url="https://x/v"),
            monkeypatch,
        )
        joined = "\n".join(said)
        assert "412" in joined
        assert "确认" not in joined, "a dead target must not be offered for confirmation"

    @pytest.mark.asyncio
    async def test_a_short_recording_is_offered_anyway_with_the_caveat(
        self, service, monkeypatch
    ):
        # ready + span_available False: there IS something to audit, just less
        # than was asked for. The customer decides, not us.
        service.preflight_result = Job(
            job_id="short1", user_id="zhang@example.com", state="ready",
            preflight={
                "ok": True, "capture_mode": "screen", "title": "",
                "video_duration_seconds": 180.0,
                "requested_start_seconds": 150.0, "requested_end_seconds": 270.0,
                "span_available": False, "problem": "视频只有 03:00 长，最多只能稽核到 03:00",
            },
        )
        said = await _collect(
            _payload("稽核 https://x/v 从 02:30 看两分钟"), service,
            _Decision(action="audit", target_url="https://x/v",
                      start_seconds=150.0, end_seconds=270.0),
            monkeypatch,
        )
        joined = "\n".join(said)
        assert "只能稽核到 03:00" in joined
        assert "确认" in joined, "a partial span is still worth offering"


class TestSayingYesInWhateverWordsTheCustomerUses:
    """The reason this module uses a model and not a keyword list."""

    @pytest.mark.asyncio
    async def test_confirming_starts_the_audit_and_returns_at_once(
        self, service, monkeypatch
    ):
        await service.jobs.create(Job(
            job_id="abc123", user_id="zhang@example.com", session_id="s1",
            state="ready",
        ))
        said = await _collect(
            _payload("行吧那就跑"), service, _Decision(action="confirm"), monkeypatch,
        )
        assert service.started == [("zhang@example.com", "abc123")]
        assert "abc123" in "\n".join(said)

    @pytest.mark.asyncio
    async def test_confirming_twice_does_not_start_two_audits(
        self, service, monkeypatch
    ):
        await service.jobs.create(Job(
            job_id="abc123", user_id="zhang@example.com", session_id="s1",
            state="running",
        ))
        said = await _collect(
            _payload("确认"), service, _Decision(action="confirm"), monkeypatch,
        )
        assert service.started == [], "a repeated yes must not re-run the audit"
        assert "已经在跑" in "\n".join(said)

    @pytest.mark.asyncio
    async def test_confirming_with_nothing_pending_says_so(self, service, monkeypatch):
        said = await _collect(
            _payload("确认"), service, _Decision(action="confirm"), monkeypatch,
        )
        assert service.started == []
        assert "没找到待确认" in "\n".join(said)


class TestAskingHowItIsGoing:

    @pytest.mark.asyncio
    async def test_progress_comes_back_without_quoting_a_number(
        self, service, monkeypatch
    ):
        await service.jobs.create(Job(
            job_id="abc123", user_id="zhang@example.com", session_id="s1",
            state="running",
            progress={"windows_analyzed": 7, "violations_so_far": 2,
                      "last_window_returned": "01:00 - 01:15"},
        ))
        said = await _collect(
            _payload("好了吗"), service, _Decision(action="status"), monkeypatch,
        )
        joined = "\n".join(said)
        assert "7 个窗口" in joined and "2 处违规" in joined

    @pytest.mark.asyncio
    async def test_a_finished_audit_hands_back_the_report(self, service, monkeypatch):
        await service.jobs.create(Job(
            job_id="abc123", user_id="zhang@example.com", session_id="s1",
            state="done", result={"report": "### 📊 稽核完成\n- 分析窗口：3 个"},
        ))
        said = await _collect(
            _payload("结果呢"), service, _Decision(action="status"), monkeypatch,
        )
        assert "稽核完成" in "\n".join(said)

    @pytest.mark.asyncio
    async def test_a_failed_audit_says_so_rather_than_reporting_nothing(
        self, service, monkeypatch
    ):
        await service.jobs.create(Job(
            job_id="abc123", user_id="zhang@example.com", session_id="s1",
            state="failed", error="ffmpeg 退出码 1",
        ))
        said = await _collect(
            _payload("好了吗"), service, _Decision(action="status"), monkeypatch,
        )
        assert "ffmpeg" in "\n".join(said)

    @pytest.mark.asyncio
    async def test_another_users_job_number_is_not_readable(self, service, monkeypatch):
        await service.jobs.create(Job(
            job_id="abc123", user_id="li@example.com", session_id="other",
            state="done", result={"report": "李四的报告"},
        ))
        said = await _collect(
            _payload("单号 abc123 好了吗"), service,
            _Decision(action="status", job_id="abc123"), monkeypatch,
        )
        joined = "\n".join(said)
        assert "李四的报告" not in joined
        assert "没找到" in joined


class TestWhenTheTurnCannotBeRead:
    """Every one of these ends in a sentence, never an exception."""

    @pytest.mark.asyncio
    async def test_unclear_asks_back_and_spends_nothing(self, service, monkeypatch):
        said = await _collect(
            _payload("那个店昨天怎么样"), service,
            _Decision(action="unclear", question="要看哪个视频？把链接发我。"),
            monkeypatch,
        )
        assert "把链接发我" in "\n".join(said)
        assert service.started == []

    @pytest.mark.asyncio
    async def test_the_model_being_down_does_not_become_a_guess(
        self, service, monkeypatch
    ):
        from computer_use_agent.turn import ModelUnavailable

        async def boom(_turn):
            raise ModelUnavailable("Vertex 连不上")

        monkeypatch.setattr(server, "read_intent", boom)
        said = await _collect(_payload("确认"), service)

        assert service.started == [], "an unreadable turn must not start an audit"
        assert "再说一遍" in "\n".join(said)

    @pytest.mark.asyncio
    async def test_a_turn_with_no_caller_identity_is_refused(self, service, monkeypatch):
        said = await _collect(_payload("稽核 https://x/v", user=""), service)
        assert "身份" in "\n".join(said)
        assert service.started == []

    @pytest.mark.asyncio
    async def test_an_unexpected_error_still_answers_in_words(
        self, service, monkeypatch
    ):
        async def boom(*_a, **_k):
            raise RuntimeError("Firestore 挂了")

        monkeypatch.setattr(service, "get_status", boom)
        said = await _collect(
            _payload("好了吗"), service, _Decision(action="status"), monkeypatch,
        )
        assert "Firestore 挂了" in "\n".join(said)


# -- what the model is allowed to decide ---------------------------------------

class TestTheModelReadsButDoesNotInvent:

    def test_a_url_nobody_typed_is_refused(self):
        from computer_use_agent.turn import _validate

        turn = Turn(text="稽核一下那个店", session_id="s", user_id="u")
        out = _validate(
            _Decision(action="audit", target_url="https://www.bilibili.com/video/BV1made"),
            turn,
        )
        # A plausible id navigates somewhere real and audits a stranger's video.
        assert out.action == "unclear"
        assert out.target_url == ""

    def test_a_url_from_an_earlier_turn_is_allowed(self):
        from computer_use_agent.turn import _validate

        turn = Turn(
            text="看 14:00 到 15:00", session_id="s", user_id="u",
            history=[{"role": "user", "text": "https://x/v 这家店"}],
        )
        out = _validate(
            _Decision(action="audit", target_url="https://x/v",
                      span_stated=True, start_seconds=50400, end_seconds=54000),
            turn,
        )
        assert out.action == "audit" and out.target_url == "https://x/v"

    def test_an_unreadable_span_becomes_a_question_not_a_default(self):
        from computer_use_agent.turn import _validate

        turn = Turn(text="https://x/v 最后五分钟", session_id="s", user_id="u")
        out = _validate(
            _Decision(action="audit", target_url="https://x/v", span_understood=False,
                      reading="「最后五分钟」换算不出确定的秒数。"),
            turn,
        )
        assert out.action == "unclear"
        assert "最后五分钟" in out.question
        # The reading is a sentence of its own and brings its own full stop.
        assert "。。" not in out.question

    def test_a_missing_time_span_is_a_question_not_the_whole_video(self):
        from computer_use_agent.turn import _validate

        # There used to be a rule telling the model to read "no time mentioned"
        # as start 0, end -1. That is not a smaller audit -- it is the entire
        # recording, running until the hour-long wall-clock budget stops it,
        # which is how a customer got an hour of silence and no report.
        turn = Turn(text="稽核一下 https://x/v", session_id="s", user_id="u")
        out = _validate(
            _Decision(action="audit", target_url="https://x/v", span_stated=False),
            turn,
        )

        assert out.action == "unclear"
        assert "哪一段" in out.question
        # The question has to carry an example, or the next turn is the same
        # question again in different words.
        assert "05:00" in out.question

    def test_a_start_with_no_end_is_asked_about_too(self):
        from computer_use_agent.turn import _validate

        # Same disease as above: "从 05:00 开始看" has no end either, and
        # `from_fields` turns a missing end into "watch to the end of the tape".
        turn = Turn(text="https://x/v 从 05:00 开始看", session_id="s", user_id="u")
        out = _validate(
            _Decision(action="audit", target_url="https://x/v",
                      span_stated=True, start_seconds=300.0, end_seconds=-1.0),
            turn,
        )

        assert out.action == "unclear"
        assert "05:00" in out.question and "看到哪为止" in out.question

    def test_a_stated_span_still_goes_straight_through(self):
        from computer_use_agent.turn import _validate

        # The other half of the same rule: the chips customers actually click
        # ("05:00 到 07:00") must not acquire a confirmation step.
        turn = Turn(text="https://x/v 05:00 到 07:00", session_id="s", user_id="u")
        out = _validate(
            _Decision(action="audit", target_url="https://x/v",
                      span_stated=True, start_seconds=300.0, end_seconds=420.0),
            turn,
        )

        assert out.action == "audit"

    def test_an_action_outside_the_four_becomes_unclear(self):
        from computer_use_agent.turn import _validate

        turn = Turn(text="whatever", session_id="s", user_id="u")
        out = _validate(_Decision(action="cancel_everything"), turn)
        assert out.action == "unclear"

    def test_there_is_nowhere_for_an_audit_standard_to_enter(self):
        # The whole reason a verdict stays traceable to a written version.
        fields = set(_Decision.model_fields)
        assert fields == {
            "action", "target_url", "start_seconds", "end_seconds",
            "span_stated", "span_understood", "sop_id", "job_id",
            "reading", "question",
        }


# -- the platform contract ------------------------------------------------------

class TestTheEnvelopeThePlatformRequires:
    """Bare events are dropped in silence. This shape cost a deploy cycle."""

    def test_an_event_carries_the_fields_adk_dumps(self):
        event = server._adk_event("你好", "inv123")
        assert event["content"] == {"parts": [{"text": "你好"}], "role": "model"}
        assert event["author"] == server.AUTHOR
        assert set(event["actions"]) == {
            "state_delta", "artifact_delta",
            "requested_auth_configs", "requested_tool_confirmations",
        }
        assert "partial" not in event, "exclude_none omits it unless it is true"

    def test_partial_is_keyword_only(self):
        # It landed as a positional once and silently became the invocation id,
        # shipping `"partial": "d84079096b6e"`.
        with pytest.raises(TypeError):
            server._adk_event("hi", "inv", True)   # type: ignore[misc]

    def test_the_envelope_names_the_session(self):
        out = server._envelope({"id": "e"}, "sess-9")
        assert out == {"events": [{"id": "e"}], "session_id": "sess-9"}

    def test_ge_calls_exactly_one_method(self):
        # If this constant drifts, GE's turns fall through to the direct-call
        # branch and every customer sees "unknown method".
        assert server.GE_METHOD == "streaming_agent_run_with_events"


# -- being able to see what the container did -----------------------------------

class TestTheContainerCanBeSeen:
    """Every INFO the audit logged used to go nowhere.

    `server.py` created loggers and never gave the root logger a handler, so
    Python's handler-of-last-resort emitted WARNING and above only. Cloud
    Logging showed uvicorn's access lines and nothing else -- on failed and
    successful runs alike. A job could then sit at `running` for an hour with
    no evidence available except the absence of evidence, which is not the same
    as evidence of a hang.
    """

    def test_an_info_line_from_the_pipeline_reaches_a_handler(self, capsys, monkeypatch):
        import logging

        from computer_use_agent.logsetup import setup_logging

        monkeypatch.setenv("LOG_FORMAT", "text")
        setup_logging(force=True)
        logging.getLogger("cctv_audit.pipeline").info("captured window 3")

        assert "captured window 3" in capsys.readouterr().out

    def test_logs_go_to_stdout_because_cloud_run_calls_stderr_an_error(
        self, capsys, monkeypatch
    ):
        import logging

        from computer_use_agent.logsetup import setup_logging

        monkeypatch.setenv("LOG_FORMAT", "text")
        setup_logging(force=True)
        logging.getLogger("cctv_audit.service").info("job 6b5ba9 alive")

        captured = capsys.readouterr()
        assert "job 6b5ba9 alive" in captured.out
        assert "job 6b5ba9 alive" not in captured.err

    def test_on_the_platform_each_line_is_json_with_a_severity(
        self, capsys, monkeypatch
    ):
        # Plain text arrives as a DEFAULT-severity entry, which sorts below
        # INFO and hides in the console's default view; `severity>=WARNING`
        # then matches nothing. The key has to be in the payload.
        import json as _json
        import logging

        from computer_use_agent.logsetup import setup_logging

        monkeypatch.delenv("LOG_FORMAT", raising=False)
        monkeypatch.setenv("K_REVISION", "agent-v12-abc")
        setup_logging(force=True)
        logging.getLogger("cctv_audit.service").warning("watchdog killed job d7680d")

        line = _json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert line["severity"] == "WARNING"
        assert line["message"] == "watchdog killed job d7680d"
        assert line["instance"] == "agent-v12-abc"

    def test_a_traceback_stays_in_one_entry(self, capsys, monkeypatch):
        # Split across forty lines it is forty entries, interleaved with every
        # other request the instance was serving.
        import json as _json
        import logging

        from computer_use_agent.logsetup import setup_logging

        monkeypatch.delenv("LOG_FORMAT", raising=False)
        monkeypatch.setenv("K_REVISION", "agent-v12-abc")
        setup_logging(force=True)
        try:
            raise RuntimeError("ffmpeg exited 1")
        except RuntimeError:
            logging.getLogger("cctv_audit.service").exception("Job z failed")

        out = capsys.readouterr().out.strip().splitlines()
        assert len(out) == 1
        entry = _json.loads(out[0])
        assert entry["severity"] == "ERROR"
        assert "ffmpeg exited 1" in entry["message"]
        assert "Traceback" in entry["message"]

    def test_calling_it_twice_does_not_double_every_line(self, capsys, monkeypatch):
        import logging

        from computer_use_agent.logsetup import setup_logging

        monkeypatch.setenv("LOG_FORMAT", "text")
        setup_logging(force=True)
        setup_logging(force=True)
        logging.getLogger("cctv_audit.turn").info("only once")

        assert capsys.readouterr().out.count("only once") == 1


class TestKeepAliveProbe:
    """The endpoint that keeps a CPU allocated while an audit runs.

    Measured, and the reason this exists: an instance with no request in
    flight is denied a processor 79-85% of the time (`/proc/self/schedstat`,
    v16 in the cloud), which is why the dashboard was a slideshow and why one
    run stalled four minutes the moment its poller stopped.
    """

    def _service(self, running: int):
        class _Runner:
            pass

        runner = _Runner()
        runner.running = running

        class _Svc:
            pass

        svc = _Svc()
        svc.runner = runner
        return svc

    @pytest.mark.asyncio
    async def test_an_idle_container_answers_at_once(self, monkeypatch):
        import time as _time

        monkeypatch.setattr(server, "audit_service", lambda: self._service(0))
        monkeypatch.setattr(server, "KEEPALIVE_HOLD_SECONDS", 30.0)

        began = _time.monotonic()
        body = await server.is_busy()

        # Nothing to keep alive for, so holding the slot would only crowd out
        # the real traffic.
        assert _time.monotonic() - began < 0.5
        assert body["busy"] is False
        assert body["held_seconds"] == 0

    @pytest.mark.asyncio
    async def test_a_busy_container_holds_the_request_open(self, monkeypatch):
        import time as _time

        monkeypatch.setattr(server, "audit_service", lambda: self._service(1))
        monkeypatch.setattr(server, "KEEPALIVE_HOLD_SECONDS", 0.3)
        monkeypatch.setattr(server, "_HOLD_STEP_SECONDS", 0.05)

        began = _time.monotonic()
        body = await server.is_busy()

        assert _time.monotonic() - began >= 0.25
        assert body["busy"] is True
        assert body["jobs"] == 1

    @pytest.mark.asyncio
    async def test_a_job_that_finishes_releases_the_slot_early(self, monkeypatch):
        import time as _time

        svc = self._service(1)
        monkeypatch.setattr(server, "audit_service", lambda: svc)
        monkeypatch.setattr(server, "KEEPALIVE_HOLD_SECONDS", 10.0)
        monkeypatch.setattr(server, "_HOLD_STEP_SECONDS", 0.05)

        async def finish():
            await asyncio.sleep(0.15)
            svc.runner.running = 0

        began = _time.monotonic()
        await asyncio.gather(server.is_busy(), finish())

        # Ten seconds of hold, but the work took a sixth of a second. Holding
        # the rest would keep a concurrency slot for nothing.
        assert _time.monotonic() - began < 2.0

    @pytest.mark.asyncio
    async def test_zero_hold_answers_instantly_even_when_busy(self, monkeypatch):
        import time as _time

        monkeypatch.setattr(server, "audit_service", lambda: self._service(2))
        monkeypatch.setattr(server, "KEEPALIVE_HOLD_SECONDS", 0.0)

        began = _time.monotonic()
        body = await server.is_busy()

        # This is the control for the measurement: same build, no hold.
        assert _time.monotonic() - began < 0.5
        assert body["busy"] is True
        assert body["jobs"] == 2


class TestKeepAliveDeploymentSpec:
    """The hold costs a concurrency slot, so the spec has to pay for it."""

    def _spec(self):
        import importlib.util
        import pathlib
        import sys

        path = (pathlib.Path(__file__).resolve().parents[1]
                / "deploy" / "agent_runtime" / "deploy.py")
        spec = importlib.util.spec_from_file_location("_deploy_under_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_deploy_under_test"] = module
        spec.loader.exec_module(module)
        return module.DEPLOYMENT_SPEC

    def test_the_probe_points_at_the_endpoint_the_server_serves(self):
        probe = self._spec()["keepAliveProbe"]

        assert probe["httpGet"]["path"] == "/is_busy"
        assert probe["httpGet"]["port"] == 8080
        # The documented ceiling; anything larger is rejected.
        assert probe["maxSeconds"] <= 3600

    def test_concurrency_leaves_a_slot_for_real_traffic(self):
        spec = self._spec()

        # The held probe permanently occupies one. At 1 it would crowd out
        # every Gemini Enterprise turn and they would come back 429.
        assert spec["containerConcurrency"] >= 2
