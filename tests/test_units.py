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

"""Unit tests for the pure logic -- no browser, no network, no Vertex AI.

Concentrated on the time arithmetic, because that is what decides whether an
evidence frame actually shows the moment a finding cites. A wrong timestamp
produces a confidently-wrong audit record, which is worse than no record.

Run: .venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from computer_use_agent.agent import UnreadableTimeSpan, _explain_failure, parse_request
from computer_use_agent.analyzer.schema import Severity, Status, WindowResult, response_schema
from computer_use_agent.analyzer.sop import load_rules
from computer_use_agent.analyzer.video_analyzer import (
    VideoAnalyzer,
    build_prompt,
    build_system_instruction,
)
from computer_use_agent.capture.screen_recorder import content_box, normalise_crop
from computer_use_agent.capture.types import Clip


# The active standard is chosen by SOP_RULES_PATH, so tests that assert on
# specific rule ids must name their file rather than take whatever is
# configured -- otherwise switching standards breaks the suite.
ANALYZER_DIR = Path(__file__).resolve().parents[1] / "computer_use_agent" / "analyzer"
CCTV_RULES = ANALYZER_DIR / "sop_rules.yaml"
SHIPPED_RULE_FILES = sorted(ANALYZER_DIR.glob("sop_rules*.yaml"))


def make_clip(**overrides) -> Clip:
    kwargs = dict(
        index=0, path=Path("/tmp/x.mp4"), start_offset=0.0, end_offset=15.0,
        wall_clock_start=0.0, source_mode="stream", time_scale=1.0,
    )
    kwargs.update(overrides)
    return Clip(**kwargs)


class TestClipTimeMapping:
    def test_offsets_are_absolute_video_seconds(self):
        clip = make_clip(index=3, start_offset=36.0, end_offset=51.0)
        assert clip.duration == 15.0
        assert clip.time_range == "00:36 - 00:51"

    def test_hours_render_as_hh_mm_ss(self):
        assert Clip.format_offset(3725) == "01:02:05"
        assert Clip.format_offset(65) == "01:05"

    def test_realtime_clip_maps_one_to_one(self):
        clip = make_clip(start_offset=100.0, end_offset=115.0)
        assert clip.clip_ts_to_video_offset(7.0) == pytest.approx(107.0)

    def test_sped_up_clip_scales_the_timestamp(self):
        # 15 video seconds recorded at 4x is a 3.75s file. A model timestamp of
        # 2.0s inside that file is 8.0s into the window.
        clip = make_clip(start_offset=100.0, end_offset=115.0, source_mode="screen", time_scale=4.0)
        assert clip.clip_ts_to_video_offset(2.0) == pytest.approx(108.0)

    def test_timestamp_beyond_the_clip_is_clamped(self):
        # The model sometimes reports a time past the end. Clamping keeps the
        # evidence frame inside the clip instead of failing extraction.
        clip = make_clip(start_offset=100.0, end_offset=115.0, time_scale=4.0)
        assert clip.clip_ts_to_video_offset(999.0) == pytest.approx(115.0)
        assert clip.clip_ts_to_video_offset(-5.0) == pytest.approx(100.0)


class TestWindowAssemblerMath:
    def _assembler(self, tmp_path, window, overlap, time_scale=1.0, segment=None):
        from computer_use_agent.capture.window_assembler import WindowAssembler
        return WindowAssembler(
            segment_dir=tmp_path / "seg", out_dir=tmp_path / "out",
            window_seconds=window, overlap_seconds=overlap,
            source_mode="stream", time_scale=time_scale, segment_seconds=segment,
        )

    def test_step_and_span(self, tmp_path):
        a = self._assembler(tmp_path, window=15, overlap=3)
        assert a.step_seconds == 12
        # A 15s window spans two 12s segments.
        assert a.segments_per_window == 2

    def test_no_overlap_means_one_segment_per_window(self, tmp_path):
        a = self._assembler(tmp_path, window=15, overlap=0)
        assert a.segments_per_window == 1

    def test_long_window_with_small_overlap(self, tmp_path):
        a = self._assembler(tmp_path, window=600, overlap=30)
        assert a.step_seconds == 570
        assert a.segments_per_window == 2

    def test_overlap_at_or_above_window_is_rejected(self, tmp_path):
        # Windows would never advance; failing loudly beats hanging.
        with pytest.raises(ValueError):
            self._assembler(tmp_path, window=15, overlap=15)

    def test_finer_segments_still_land_on_window_starts(self, tmp_path):
        a = self._assembler(tmp_path, window=15, overlap=3, segment=2)
        assert a.segments_per_step == 6, "window N starts at segment 6N"
        assert a.segments_per_window == 8, "15s of 2s pieces, rounded up"

    def test_a_segment_that_does_not_divide_the_step_is_rejected(self, tmp_path):
        # Window 1 would start 5s into a segment, and every timestamp from
        # there on would be wrong -- quietly, which is the problem.
        with pytest.raises(ValueError):
            self._assembler(tmp_path, window=15, overlap=3, segment=5)


class TestFirstVerdictLatency:
    """A window cannot be assembled until every segment it touches is closed,
    so the segment size is a floor on how far behind the playhead the audit
    runs. Cut at the step -- what this used to do -- a 15s window waited 24s of
    footage, and a 10-minute window waited nearly 20 minutes."""

    def _seg(self, window, overlap, **kw):
        from computer_use_agent.capture.screen_recorder import choose_segment_seconds
        return choose_segment_seconds(window, window - overlap, **kw)

    def test_the_default_window_buffers_less_than_it_used_to(self):
        seg = self._seg(15, 3)
        assert seg < 12, "cutting at the step is what made the first verdict late"
        assert 15 + seg <= 18

    def test_a_ten_minute_window_no_longer_waits_twenty(self):
        # The configuration a real store audit runs at. 1197s -> ~626s.
        seg = self._seg(600, 3)
        assert 600 + seg < 700

    def test_the_segment_always_divides_the_step(self):
        # The assembler rejects anything else, so this is load-bearing.
        for window, overlap in [(15, 3), (15, 0), (10, 3), (60, 5), (600, 3), (600, 60), (5, 1)]:
            step = window - overlap
            k = step / self._seg(window, overlap)
            assert abs(k - round(k)) < 1e-9, f"{window}/{overlap} gives a ragged segment"

    def test_pieces_stay_big_enough_and_few_enough(self):
        # Every segment opens on a forced keyframe and every window is a
        # concat of its pieces, so "as small as possible" is not free.
        import math
        for window, overlap in [(15, 3), (60, 5), (600, 3), (3600, 60)]:
            seg = self._seg(window, overlap)
            assert seg >= 2.0
            assert math.ceil(window / seg) <= 24

    def test_a_window_shorter_than_the_floor_is_not_cut_finer(self):
        # 4s window, 3s step: there is nothing sensible below 2s here, and
        # returning something that does not divide the step would be worse.
        seg = self._seg(4, 1)
        assert seg == 3.0


class TestRetryClassification:
    """Which failures are worth retrying, decided on the status code.

    The old test was the code itself: `"500" in str(exc)`. Anything with those
    three digits anywhere in its message got three attempts, and the backoff
    made a permanent failure take eight seconds to report.
    """

    @staticmethod
    def _exc(message, **attrs):
        exc = RuntimeError(message)
        for key, value in attrs.items():
            setattr(exc, key, value)
        return exc

    @pytest.mark.parametrize("exc_kwargs", [
        {"message": "boom", "code": 503},
        {"message": "boom", "code": 429},
        {"message": "boom", "status_code": 500},
        {"message": "boom", "status": "RESOURCE_EXHAUSTED"},
        {"message": "503 UNAVAILABLE: backend is down"},
        {"message": "code: 429, quota exceeded"},
        {"message": "Computer use is not supported for this model in this region"},
    ])
    def test_transient_failures_are_retried(self, exc_kwargs):
        from computer_use_agent.gcp import is_transient
        assert is_transient(self._exc(**exc_kwargs))

    @pytest.mark.parametrize("message", [
        "INVALID_ARGUMENT: the request exceeds the 15000 token limit",
        "model gemini-3.5-flash-0429 is not available to this project",
        "PERMISSION_DENIED: caller lacks aiplatform.endpoints.predict",
        "the clip is 18500000 bytes, over the inline limit",
        "NOT_FOUND: publishers/google/models/gemini-1.5-pro-500k",
    ])
    def test_permanent_failures_are_not_retried(self, message):
        # Every one of these contains 500, 429 or 503 as a substring and was
        # retried three times by the previous implementation.
        from computer_use_agent.gcp import is_transient
        assert not is_transient(self._exc(message))

    def test_a_code_of_400_is_not_a_transient_500(self):
        from computer_use_agent.gcp import is_transient
        assert not is_transient(self._exc("bad request", code=400))


class TestSchema:
    def test_vertex_schema_has_no_refs(self):
        # Vertex rejects $ref/$defs, which pydantic emits for nested models.
        blob = json.dumps(response_schema())
        assert "$ref" not in blob and "$defs" not in blob

    @staticmethod
    def _finding(raw):
        result = WindowResult.model_validate({
            "scene_summary": "s",
            "findings": [{"rule_id": "CHK_ATTIRE_001", "status": "COMPLIANT",
                          "confidence": 0.5, "timestamp_in_clip": raw}],
        })
        return result.findings[0]

    @pytest.mark.parametrize("raw,expected", [
        ("0:03", "00:03"),
        ("00:01:07", "01:07"),      # HH:MM:SS collapsed into the clip's MM:SS
        ("大约 01:12 处", "01:12"),  # model prose around the timestamp
        ("9:59", "09:59"),
    ])
    def test_timestamp_normalisation(self, raw, expected):
        finding = self._finding(raw)
        assert finding.timestamp_in_clip == expected
        assert finding.has_timestamp

    @pytest.mark.parametrize("raw", ["", "nonsense", "开头", "第 3 秒"])
    def test_an_unreadable_timestamp_is_not_quietly_turned_into_00_00(self, raw):
        # It used to be rewritten to "00:00", which is not a null -- it is a
        # moment. The evidence frame is cut at exactly that offset, so a
        # violation described at the end of the window came back with a picture
        # of its first frame and nothing said the moment was invented.
        finding = self._finding(raw)
        assert not finding.has_timestamp
        # The frame still has to be cut somewhere; what changed is that the
        # record now says the offset is a stand-in.
        assert finding.offset_seconds == 0.0

    def test_offset_seconds(self):
        result = WindowResult.model_validate({
            "scene_summary": "s",
            "findings": [{"rule_id": "CHK_ATTIRE_001", "status": "VIOLATION",
                          "confidence": 0.9, "timestamp_in_clip": "01:07"}],
        })
        assert result.findings[0].offset_seconds == 67


class TestSanitise:
    def setup_method(self):
        self.rules = load_rules(CCTV_RULES)
        self.analyzer = VideoAnalyzer(self.rules)

    def _result(self, findings):
        return WindowResult.model_validate({"scene_summary": "s", "findings": findings})

    def test_unknown_rule_ids_are_dropped(self):
        result = self.analyzer._sanitise(self._result([
            {"rule_id": "MADE_UP_999", "status": "VIOLATION", "confidence": 0.9},
            {"rule_id": "CHK_ATTIRE_001", "status": "COMPLIANT", "confidence": 0.9},
        ]))
        assert [f.rule_id for f in result.findings] == ["CHK_ATTIRE_001"]

    def test_severity_comes_from_the_rule_not_the_model(self):
        # CHK_BEHAVIOR_004 is RED_LINE in sop_rules.yaml; the model said NORMAL.
        result = self.analyzer._sanitise(self._result([
            {"rule_id": "CHK_BEHAVIOR_004", "status": "VIOLATION",
             "confidence": 0.8, "severity": "NORMAL"},
        ]))
        assert result.findings[0].severity is Severity.RED_LINE

    def test_non_violations_carry_no_severity(self):
        result = self.analyzer._sanitise(self._result([
            {"rule_id": "CHK_BEHAVIOR_004", "status": "COMPLIANT",
             "confidence": 0.8, "severity": "RED_LINE"},
        ]))
        assert result.findings[0].severity is Severity.NONE

    def test_unreadable_footage_cannot_yield_a_violation(self):
        # "I could not see" and "I saw a violation" are contradictory claims.
        result = WindowResult.model_validate({
            "scene_summary": "黑屏", "visibility_ok": False,
            "findings": [{"rule_id": "CHK_HYGIENE_002", "status": "VIOLATION", "confidence": 0.9}],
        })
        result = self.analyzer._sanitise(result)
        assert result.findings[0].status is Status.CANNOT_DETERMINE
        assert result.violations == []


class TestSopRules:
    def test_the_shipped_standards_are_all_loadable(self):
        # Every sop_rules*.yaml in the repo is something SOP_RULES_PATH can be
        # pointed at, so every one of them has to survive loading.
        assert len(SHIPPED_RULE_FILES) >= 2
        for path in SHIPPED_RULE_FILES:
            assert load_rules(path).rules, path.name

    @pytest.mark.parametrize("path", SHIPPED_RULE_FILES, ids=lambda p: p.name)
    def test_every_rule_is_complete(self, path):
        rules = load_rules(path)
        for rule in rules.rules:
            assert rule.severity in ("RED_LINE", "NORMAL")
            assert rule.detection_type in ("presence", "absence")
            assert rule.description.strip()

    @pytest.mark.parametrize("path", SHIPPED_RULE_FILES, ids=lambda p: p.name)
    def test_absence_rules_demand_full_context(self, path):
        # An absence claim from a partial view is exactly the failure mode this
        # rewrite exists to prevent, so the two flags must agree.
        for rule in load_rules(path).rules:
            if rule.detection_type == "absence":
                assert rule.requires_full_context, f"{rule.id} must require full context"

    @pytest.mark.parametrize("path", SHIPPED_RULE_FILES, ids=lambda p: p.name)
    def test_rendered_rules_reach_the_prompt(self, path):
        rules = load_rules(path)
        rendered = rules.render()
        for rule_id in rules.ids:
            assert rule_id in rendered

    def test_the_cctv_standard_still_carries_its_five_rules(self):
        assert len(load_rules(CCTV_RULES).rules) >= 5


class TestPrompt:
    def test_sped_up_clips_are_flagged_to_the_model(self):
        rules = load_rules(CCTV_RULES)
        fast = build_prompt(make_clip(time_scale=4.0, source_mode="screen"), rules)
        assert "4.0 倍速" in fast
        normal = build_prompt(make_clip(), rules)
        assert "倍速" not in normal


class TestRequestParsing:
    """The keyword fallback, used only when the model cannot be reached."""

    def test_bare_url(self):
        request = parse_request("https://www.bilibili.com/video/BV1xx411c7mD")
        assert request.target == "https://www.bilibili.com/video/BV1xx411c7mD"
        assert request.start_seconds == 0.0
        assert request.duration_seconds is None

    def test_start_and_duration(self):
        request = parse_request("稽核 https://example.com/v 从 12:30 开始，分析 10 分钟")
        assert request.start_seconds == 750.0
        assert request.duration_seconds == 600.0

    def test_trailing_punctuation_is_not_part_of_the_url(self):
        request = parse_request("请看 https://example.com/v。")
        assert request.target == "https://example.com/v"

    def test_no_url_means_no_request(self):
        assert parse_request("帮我看看门店视频") is None

    def test_the_way_people_actually_write_a_range(self):
        # Reported from a real run: this parsed as nothing at all, so the audit
        # started at 00:00 and ran past 05:00 to the end of the recording --
        # both ends wrong, with no warning and the bill to match.
        request = parse_request(
            "稽核https://www.bilibili.com/video/BV1URpRzCEXK/?spm_id_from="
            "333.337.search-card.all.click ，第1分钟到第5分钟的内容")
        assert request.start_seconds == 60.0
        assert request.duration_seconds == 240.0

    @pytest.mark.parametrize("text,start,duration", [
        ("第1分钟到第5分钟", 60.0, 240.0),
        ("01:00 到 05:00", 60.0, 240.0),
        ("1分钟-5分钟", 60.0, 240.0),
        ("00:30~02:00", 30.0, 90.0),
        ("from 1:00 to 5:00", 60.0, 240.0),
        ("第1小时到第2小时", 3600.0, 3600.0),
        ("从第3分钟开始", 180.0, None),
        ("看 90 秒", 0.0, 90.0),
        ("从 12:30 开始，分析 10 分钟", 750.0, 600.0),
    ])
    def test_range_and_offset_spellings(self, text, start, duration):
        request = parse_request(f"稽核 https://example.com/v {text}")
        assert (request.start_seconds, request.duration_seconds) == (start, duration)

    def test_digits_in_the_url_are_not_a_time(self):
        # `spm_id_from=333.337.search-card...` is on every bilibili share link.
        request = parse_request(
            "稽核 https://www.bilibili.com/video/BV1x/?spm_id_from=333.337.search-card.all.click")
        assert request.start_seconds == 0.0
        assert request.duration_seconds is None

    def test_a_time_we_cannot_read_stops_the_run(self):
        # The old behaviour -- silently auditing the whole video from the top --
        # is the one answer guaranteed to be wrong, and the expensive one.
        with pytest.raises(UnreadableTimeSpan):
            parse_request("稽核 https://example.com/v 这个5分钟的视频")

    def test_a_backwards_range_is_a_typo_not_a_guess(self):
        with pytest.raises(UnreadableTimeSpan):
            parse_request("稽核 https://example.com/v 第5分钟到第1分钟")

    def test_an_explicit_zero_start_is_not_a_parse_failure(self):
        # "从 0:00 开始" is perfectly clear and happens to equal the default.
        request = parse_request("稽核 https://example.com/v 从 0:00 开始")
        assert request.start_seconds == 0.0


class TestModelReading:
    """Reading the request with the model instead of a list of keywords.

    The model's answer is not taken on trust: it is checked back against the
    message it read, and a reading it is unsure of stops the run.
    """

    @staticmethod
    def _reading(**kwargs):
        from computer_use_agent.intent import _Reading
        return _Reading(**{"understood": True, **kwargs})

    def test_a_span_becomes_a_start_and_a_duration(self):
        from computer_use_agent.intent import _to_intent
        intent = _to_intent(
            self._reading(target_url="https://example.com/v", start_seconds=60,
                          end_seconds=300, reading="从 01:00 看到 05:00"),
            "稽核 https://example.com/v 第1分钟到第5分钟")
        assert intent.request.start_seconds == 60.0
        assert intent.request.duration_seconds == 240.0
        assert intent.reading == "从 01:00 看到 05:00"
        assert intent.source == "model"

    def test_no_end_means_watch_to_the_end(self):
        from computer_use_agent.intent import _to_intent
        intent = _to_intent(
            self._reading(target_url="https://example.com/v", end_seconds=-1),
            "稽核 https://example.com/v")
        assert intent.request.duration_seconds is None

    def test_a_url_the_message_does_not_contain_is_not_used(self):
        # A made-up video id navigates somewhere real and audits the wrong
        # shop. The message is the only source for the address.
        from computer_use_agent.intent import _to_intent
        intent = _to_intent(
            self._reading(target_url="https://www.bilibili.com/video/BV1invented"),
            "稽核 https://example.com/real 整段")
        assert intent.request.target == "https://example.com/real"

    def test_no_url_anywhere_means_no_request(self):
        from computer_use_agent.intent import _to_intent
        assert _to_intent(self._reading(), "帮我看看门店视频") is None

    def test_an_unsure_reading_stops_the_run(self):
        from computer_use_agent.intent import UnreadableTimeSpan, _to_intent
        with pytest.raises(UnreadableTimeSpan, match="总长度"):
            _to_intent(
                self._reading(understood=False, target_url="https://example.com/v",
                              problem="「最后五分钟」不知道视频总长度"),
                "https://example.com/v 看最后五分钟")

    def test_a_backwards_span_is_refused_even_if_the_model_is_happy(self):
        from computer_use_agent.intent import UnreadableTimeSpan, _to_intent
        with pytest.raises(UnreadableTimeSpan):
            _to_intent(
                self._reading(target_url="https://example.com/v",
                              start_seconds=300, end_seconds=60),
                "https://example.com/v 第5分钟到第1分钟")

    def test_the_schema_has_nowhere_to_put_a_rule(self):
        # The standing guarantee: chat text decides which video and which
        # slice of it, never what counts as a violation. That is enforced by
        # the shape of the reply, not by asking the model nicely.
        from computer_use_agent.intent import _SCHEMA
        assert set(_SCHEMA["properties"]) == {
            "understood", "target_url", "start_seconds", "end_seconds",
            "reading", "problem",
        }

    @pytest.mark.asyncio
    async def test_an_unreachable_model_falls_back_to_keywords(self, monkeypatch):
        from computer_use_agent import intent as intent_mod

        async def boom(_text):
            raise RuntimeError("no credentials")

        monkeypatch.setattr(intent_mod, "_ask_model", boom)
        result = await intent_mod.interpret_request(
            "稽核 https://example.com/v 从 12:30 开始，分析 10 分钟")
        assert result.source == "regex"
        assert (result.request.start_seconds, result.request.duration_seconds) == (750.0, 600.0)

    @pytest.mark.asyncio
    async def test_a_slow_model_does_not_hang_the_start(self, monkeypatch):
        from computer_use_agent import intent as intent_mod

        async def slow(_text):
            await asyncio.sleep(5)

        monkeypatch.setattr(intent_mod, "_ask_model", slow)
        monkeypatch.setattr(intent_mod, "_TIMEOUT_SECONDS", 0.05)
        result = await intent_mod.interpret_request("稽核 https://example.com/v")
        assert result.source == "regex"

    @pytest.mark.asyncio
    async def test_a_refusal_is_not_downgraded_into_a_guess(self, monkeypatch):
        # The fallback would read "看最后五分钟" as "duration 5 minutes from
        # the top", which is a different five minutes. A refusal stands.
        from computer_use_agent import intent as intent_mod

        async def refuse(_text):
            raise intent_mod.UnreadableTimeSpan("不知道视频总长度")

        monkeypatch.setattr(intent_mod, "_ask_model", refuse)
        with pytest.raises(intent_mod.UnreadableTimeSpan):
            await intent_mod.interpret_request("https://example.com/v 看最后五分钟")


class TestStore:
    @pytest.mark.asyncio
    async def test_ids_are_unique_and_continue_across_restarts(self, tmp_path):
        from computer_use_agent.analyzer.video_analyzer import AnalysisOutcome
        from computer_use_agent.store import AuditStore

        records = tmp_path / "records.jsonl"
        store = AuditStore(records_path=records, evidence_dir=tmp_path / "ev")
        result = WindowResult.model_validate({
            "scene_summary": "s",
            "findings": [{"rule_id": "CHK_ATTIRE_001", "status": "COMPLIANT", "confidence": 0.9}],
        })
        for i in range(3):
            await store.record(AnalysisOutcome(clip=make_clip(index=i), result=result))

        # A restart must not reuse ids -- the old code produced 1,1,2,2,3,3.
        reopened = AuditStore(records_path=records, evidence_dir=tmp_path / "ev")
        await reopened.record(AnalysisOutcome(clip=make_clip(index=3), result=result))

        ids = [json.loads(line)["id"] for line in records.read_text().splitlines()]
        assert ids == [1, 2, 3, 4]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_failed_analysis_writes_nothing(self, tmp_path):
        from computer_use_agent.analyzer.video_analyzer import AnalysisOutcome
        from computer_use_agent.store import AuditStore

        records = tmp_path / "records.jsonl"
        store = AuditStore(records_path=records, evidence_dir=tmp_path / "ev")
        written = await store.record(
            AnalysisOutcome(clip=make_clip(), result=None, error="boom")
        )
        assert written is None
        assert not records.exists() or records.read_text() == ""
        assert store.stats.windows_failed == 1


class FakePage:
    """A page whose visible text the test controls.

    The default is a phrase a page only shows when it really is challenging
    you. A bare "captcha" no longer qualifies -- see TestChallengeDetection.
    """

    def __init__(self, text: str = "请完成安全验证"):
        self.text = text

    async def evaluate(self, _script, _arg=None):
        # The widget scan passes selector specs; it finds nothing here.
        return None if _arg is not None else self.text


class TestHumanGate:
    @pytest.mark.asyncio
    async def test_dismissing_the_dashboard_banner_releases_the_gate(self):
        from computer_use_agent.navigator.base import HumanGate

        polls = iter([(True, "captcha"), (True, "captcha"), (True, None)])

        async def read_intervention():
            return next(polls)

        gate = HumanGate(read_intervention=read_intervention, timeout_seconds=10, poll_interval=0.01)
        await asyncio.wait_for(gate.wait_for_human(FakePage(), "captcha"), timeout=10)

    @pytest.mark.asyncio
    async def test_an_empty_dashboard_is_not_mistaken_for_consent(self):
        # The banner is raised by a fire-and-forget POST. If the gate read the
        # state before that landed, an unarmed "no banner" would look like the
        # operator had already pressed 继续 and the challenge would be skipped.
        from computer_use_agent.navigator.base import HumanGate, HumanInterventionRequired

        async def never_armed():
            return True, None

        gate = HumanGate(read_intervention=never_armed, timeout_seconds=0.1, poll_interval=0.01)
        with pytest.raises(HumanInterventionRequired):
            await gate.wait_for_human(FakePage(), "captcha")

    @pytest.mark.asyncio
    async def test_an_unreachable_dashboard_does_not_release_the_gate(self):
        from computer_use_agent.navigator.base import HumanGate, HumanInterventionRequired

        async def unreachable():
            return False, None

        gate = HumanGate(read_intervention=unreachable, timeout_seconds=0.1, poll_interval=0.01)
        with pytest.raises(HumanInterventionRequired):
            await gate.wait_for_human(FakePage(), "captcha")

    @pytest.mark.asyncio
    async def test_the_challenge_going_away_releases_the_gate(self):
        from computer_use_agent.navigator.base import HumanGate

        gate = HumanGate(timeout_seconds=10, poll_interval=0.01, mode="wait")
        await asyncio.wait_for(
            gate.wait_for_human(FakePage("正常播放页面"), "captcha"), timeout=10
        )


class TestUnattendedGate:
    """Production is Agent Engine: no display, no dashboard, nobody watching.

    Waiting there is not caution -- it is fifteen minutes of billed browser
    time ending in the same failure, with a timeout message instead of an
    actionable one.
    """

    @pytest.mark.asyncio
    async def test_no_dashboard_fails_immediately_instead_of_waiting(self):
        import time as _time
        from computer_use_agent.navigator.base import HumanGate, HumanInterventionRequired

        gate = HumanGate(timeout_seconds=900, poll_interval=0.01)  # mode defaults to auto
        started = _time.monotonic()
        with pytest.raises(HumanInterventionRequired) as caught:
            await gate.wait_for_human(FakePage(), "captcha")
        assert _time.monotonic() - started < 1.0, "it waited; on Agent Engine nobody answers"
        assert "storage_state" in str(caught.value), "the error must name the fix"

    @pytest.mark.asyncio
    async def test_a_dead_dashboard_is_not_an_operator(self):
        from computer_use_agent.navigator.base import HumanGate, HumanInterventionRequired

        async def unreachable():
            return False, None

        gate = HumanGate(read_intervention=unreachable, timeout_seconds=900, poll_interval=0.01)
        with pytest.raises(HumanInterventionRequired):
            await gate.wait_for_human(FakePage(), "captcha")

    @pytest.mark.asyncio
    async def test_off_refuses_even_with_a_live_dashboard(self):
        from computer_use_agent.navigator.base import HumanGate, HumanInterventionRequired

        async def live():
            return True, "captcha"

        gate = HumanGate(read_intervention=live, timeout_seconds=900,
                         poll_interval=0.01, mode="off")
        with pytest.raises(HumanInterventionRequired):
            await gate.wait_for_human(FakePage(), "captcha")

    @pytest.mark.asyncio
    async def test_a_live_dashboard_still_gets_to_answer(self):
        # The fail-fast path must not break the attended workflow it protects.
        from computer_use_agent.navigator.base import HumanGate

        polls = iter([(True, "captcha"), (True, "captcha"), (True, None)])

        async def read_intervention():
            return next(polls)

        gate = HumanGate(read_intervention=read_intervention, timeout_seconds=10,
                         poll_interval=0.01)
        await asyncio.wait_for(gate.wait_for_human(FakePage(), "captcha"), timeout=10)

    def test_the_mode_is_validated(self):
        from computer_use_agent.config import Config

        assert any("HUMAN_GATE_MODE" in p for p in Config(human_gate_mode="sometimes").validate())


class TestBudget:
    def test_each_limit_stops_the_run(self):
        from computer_use_agent.pipeline import Budget

        assert Budget(max_wall_clock_seconds=0, max_tokens=None, max_windows=None).reason_to_stop() is None

        tokens = Budget(max_wall_clock_seconds=0, max_tokens=100, max_windows=None)
        tokens.tokens_used = 100
        assert "token budget" in tokens.reason_to_stop()

        windows = Budget(max_wall_clock_seconds=0, max_tokens=None, max_windows=2)
        windows.windows_started = 2
        assert "window budget" in windows.reason_to_stop()


class TestCropNormalisation:
    """A crop is computed from a live page and handed straight to ffmpeg, so
    every way the measurement can be wrong has to be caught here rather than
    surfacing later as 'capture produced nothing'."""

    def test_a_player_box_is_rounded_to_even_edges(self):
        # Odd width or height makes libx264 with yuv420p fail outright.
        crop = normalise_crop((1201.4, 675.7, 320.2, 148.9), 1920, 1080)
        assert crop == (1200, 676, 320, 149)
        width, height, _, _ = crop
        assert width % 2 == 0 and height % 2 == 0

    def test_a_box_covering_the_whole_frame_is_dropped(self):
        # Cropping to the full frame is a no-op filter that costs CPU per frame.
        assert normalise_crop((1920, 1080, 0, 0), 1920, 1080) is None

    def test_an_oversized_box_is_clamped_inside_the_frame(self):
        # ffmpeg errors at runtime on a crop that hangs off the edge.
        assert normalise_crop((3000, 3000, 100, 100), 1920, 1080) == (1820, 980, 100, 100)

    def test_a_collapsed_player_falls_back_to_the_full_frame(self):
        # A placeholder element measures tiny; cropping to it would throw the
        # footage away and leave the audit looking at nothing.
        assert normalise_crop((100, 100, 10, 10), 1920, 1080) is None

    def test_garbage_is_ignored_rather_than_crashing_capture(self):
        assert normalise_crop(("x", 1, 2, 3), 1920, 1080) is None
        assert normalise_crop(None, 1920, 1080) is None


class TestSopRulesAreTheOnlySource:
    def test_text_in_the_message_never_becomes_a_rule(self):
        # Pasting a standard into the chat box is a natural mistake; the parser
        # must keep ignoring it rather than half-applying it.
        request = parse_request(
            "稽核 https://www.bilibili.com/video/BV1URpRzCEXK/ 从 0:00 开始。"
            "SOP：必须佩戴一次性手套，禁止徒手接触食材。"
        )
        assert request is not None
        assert request.target == "https://www.bilibili.com/video/BV1URpRzCEXK/"
        assert not hasattr(request, "rules")
        assert "手套" not in str(request)

    def test_the_rule_file_location_is_configurable(self, tmp_path):
        path = tmp_path / "custom.yaml"
        path.write_text(
            "version: 7\nrules:\n"
            "  - id: CHK_TEST_001\n    name: 测试规则\n"
            "    description: 必须佩戴一次性手套。\n    severity: RED_LINE\n",
            encoding="utf-8",
        )
        rules = load_rules(path)
        assert rules.version == 7
        assert rules.ids == ["CHK_TEST_001"]
        assert "手套" in rules.render()


class TestVisualScan:
    def test_scan_targets_are_parsed_and_rendered(self, tmp_path):
        path = tmp_path / "custom.yaml"
        path.write_text(
            "version: 2\n"
            "visual_scan:\n"
            "  - subject: 手部\n    question: 手是肉色还是被包裹？\n"
            "rules:\n"
            "  - id: CHK_TEST_001\n    name: 测试规则\n"
            "    description: 必须佩戴手套。\n    severity: RED_LINE\n",
            encoding="utf-8",
        )
        rules = load_rules(path)
        assert rules.subjects == ["手部"]
        assert "手是肉色还是被包裹？" in rules.render_scan()

    def test_a_standard_without_a_scan_section_still_loads(self):
        # The CCTV standard predates the forced-observation step.
        assert load_rules(CCTV_RULES).scan_targets == []

    def test_the_drink_standard_scans_before_it_judges(self):
        rules = load_rules(ANALYZER_DIR / "sop_rules.drink_making.yaml")
        assert rules.subjects, "the drink-making standard relies on forced observation"
        instruction = build_system_instruction(rules)
        # The observation step has to be stated before the rules it feeds.
        assert instruction.index("视觉证据锚定") < instruction.index(rules.ids[0])

    def test_observations_are_generated_before_verdicts(self):
        # Field order in the schema is the mechanism: the model fills the JSON
        # top-down, so visual_scan must be declared ahead of findings.
        properties = list(response_schema()["properties"])
        assert properties.index("visual_scan") < properties.index("findings")
        assert properties.index("action_narrative") < properties.index("findings")


class TestReportTable:
    def _record(self, **overrides):
        record = {
            "time_range": "00:00 - 00:15",
            "violation_count": 1,
            "findings": [{
                "rule_id": "CHK_GLOVE_002",
                "rule_name": "蓝色卫生手套",
                "status": "VIOLATION",
                "severity": "RED_LINE",
                "confidence": 0.92,
                "evidence": "员工右手为人类肤色，无任何包裹物",
                "timestamp": "00:04",
                "evidence_frame": "evidence/w00001.jpg",
            }],
        }
        record.update(overrides)
        return record

    def _summary(self):
        return {
            "windows_analyzed": 1, "windows_failed": 0, "violations": 1,
            "red_line_violations": 1, "capture_mode": "screen", "elapsed_seconds": 20,
            "stopped_because": "capture ended", "input_tokens": 1, "output_tokens": 1,
            "records_path": "r.jsonl", "evidence_dir": "ev",
        }

    def test_violations_render_as_the_requested_table(self):
        from computer_use_agent.agent import CctvAuditAgent

        class Store:
            def violations(self, limit=None):
                return [TestReportTable()._record()]

        report = CctvAuditAgent._report(self._summary(), Store())
        assert "| 违规规则 | 相关时间范围 | 判定依据 | 严重程度 | 证据帧 |" in report
        assert "CHK_GLOVE_002 蓝色卫生手套" in report
        # RED_LINE is an internal token; the report speaks the auditor's words.
        assert "致命缺陷" in report and "RED_LINE" not in report

    def test_a_pipe_in_the_evidence_cannot_break_the_table(self):
        from computer_use_agent.agent import CctvAuditAgent

        record = self._record()
        record["findings"][0]["evidence"] = "左手|右手 均为裸手"

        class Store:
            def violations(self, limit=None):
                return [record]

        row = [l for l in CctvAuditAgent._report(self._summary(), Store()).splitlines()
               if "CHK_GLOVE_002" in l][0]
        assert row.count("|") == 6

    def test_a_clean_run_says_so(self):
        from computer_use_agent.agent import CctvAuditAgent

        class Store:
            def violations(self, limit=None):
                return []

        assert "未发现违反 SOP 的行为" in CctvAuditAgent._report(self._summary(), Store())


class TestContentBox:
    def test_a_portrait_video_loses_its_pillarbox(self):
        # 9:16 footage in a 16:9 fullscreen player is two thirds black; those
        # pixels cost the same tokens as the footage and carry no evidence.
        box = content_box({
            "x": 0, "y": 0, "width": 1920, "height": 1080,
            "intrinsic_width": 1080, "intrinsic_height": 1920,
        })
        width, height, x, y = box
        assert (round(width), round(height)) == (608, 1080)
        assert round(x) == 656 and y == 0

    def test_a_matching_aspect_ratio_is_left_alone(self):
        assert content_box({
            "x": 10, "y": 20, "width": 1280, "height": 720,
            "intrinsic_width": 1920, "intrinsic_height": 1080,
        }) is None

    def test_an_unknown_intrinsic_size_is_not_guessed(self):
        # videoWidth is 0 until metadata loads; cropping on that would be a
        # divide-by-zero at best and a blank recording at worst.
        assert content_box({"x": 0, "y": 0, "width": 1920, "height": 1080,
                            "intrinsic_width": 0, "intrinsic_height": 0}) is None
        assert content_box(None) is None

    def test_the_box_survives_normalisation(self):
        box = content_box({
            "x": 0, "y": 0, "width": 1920, "height": 1080,
            "intrinsic_width": 1080, "intrinsic_height": 1920,
        })
        assert normalise_crop(box, 1920, 1080) == (608, 1080, 656, 0)


class TestFailureMessages:
    def test_a_tag_in_the_message_is_not_eaten_by_the_chat_ui(self):
        from computer_use_agent.navigator.base import NavigationError

        # The UI renders markdown, so a literal tag used to be parsed away --
        # taking the rest of the sentence with it. The reader saw "中断：No".
        rendered = _explain_failure(NavigationError("No <video> element appeared on x"))
        assert "<video>" not in rendered
        assert "&lt;video&gt;" in rendered

    def test_playwright_noise_is_demoted_not_shown_first(self):
        from computer_use_agent.navigator.base import NavigationError

        rendered = _explain_failure(NavigationError(
            "页面已打开但 30 秒内没有出现视频播放器：https://x\n"
            'waiting for locator("video")\nnavigated to "https://x"'
        ))
        headline = rendered.splitlines()[0]
        assert "没有出现视频播放器" in headline
        assert "locator" not in headline
        assert "locator" in rendered  # kept, but folded away

    def test_a_dead_link_says_what_to_do(self):
        from computer_use_agent.navigator.base import TargetUnavailable

        rendered = _explain_failure(TargetUnavailable("这个链接打不开（HTTP 404）：https://x"))
        assert "HTTP 404" in rendered
        assert "占位符" in rendered

    def test_an_empty_exception_still_names_itself(self):
        assert "RuntimeError" in _explain_failure(RuntimeError())


class TestDeadTargetsAreNotWorthAnAgentTurn:
    @pytest.mark.asyncio
    async def test_a_404_never_reaches_the_fallback(self):
        from computer_use_agent.navigator.base import NavigationError, TargetUnavailable
        from computer_use_agent.navigator.resilient import ResilientNavigator

        class DeadInner:
            name = "dead"

            async def open_target(self, page, target):
                raise TargetUnavailable(f"这个链接打不开（HTTP 404）：{target}")

        class CountingFallback:
            turns = 0

            async def run(self, page, goal, success_check=None):
                CountingFallback.turns += 1
                return False, "nope"

        navigator = ResilientNavigator(DeadInner(), fallback=CountingFallback())
        with pytest.raises(TargetUnavailable):
            await navigator.open_target(FakePage(""), "https://x/BVxxx")
        # A Computer Use turn costs real money to be told what HTTP already said.
        assert CountingFallback.turns == 0

    @pytest.mark.asyncio
    async def test_a_broken_selector_still_gets_rescued(self):
        from computer_use_agent.navigator.base import NavigationError
        from computer_use_agent.navigator.resilient import ResilientNavigator

        class BrokenInner:
            name = "broken"

            async def open_target(self, page, target):
                raise NavigationError("selector moved")

        class Fallback:
            turns = 0

            async def run(self, page, goal, success_check=None):
                Fallback.turns += 1
                return True, "recovered"

        navigator = ResilientNavigator(BrokenInner(), fallback=Fallback())
        await navigator.open_target(FakePage(""), "https://x")
        assert Fallback.turns == 1

    @pytest.mark.asyncio
    async def test_a_dismissable_nag_is_cleared_before_anyone_is_paged(self):
        # The overlay that blocks the step is usually the same overlay the
        # detector is looking at. Paging a human to close a box we can close
        # ourselves is the expensive way to solve it.
        from computer_use_agent.navigator.base import NavigationError
        from computer_use_agent.navigator.resilient import ResilientNavigator

        class NaggedInner:
            name = "nagged"
            cleared = False

            async def open_target(self, page, target):
                raise NavigationError("player covered by a login modal")

            async def keep_clear(self, page):
                NaggedInner.cleared = True
                page.text = "正常播放页面"

        class Gate:
            paged = 0

            async def wait_for_human(self, page, reason):
                Gate.paged += 1

        class Fallback:
            turns = 0

            async def run(self, page, goal, success_check=None):
                Fallback.turns += 1
                return True, "recovered"

        page = FakePage("请完成安全验证")
        navigator = ResilientNavigator(NaggedInner(), fallback=Fallback(), gate=Gate())
        await navigator.open_target(page, "https://x")
        assert NaggedInner.cleared
        assert Gate.paged == 0, "the nag was dismissable; nobody should have been paged"
        assert Fallback.turns == 1


class TestTheFallbackIsCheckedNotBelieved:
    """A rescue counts as successful only if the page agrees.

    Whether the Computer Use loop had really got anywhere used to be decided by
    `not reasoning.upper().startswith("BLOCKED")` -- one English keyword
    against the model's free prose. Every other way of saying "I failed"
    counted as success, and the pipeline went on to record a login wall as
    footage. Where an objective signal exists, the caller now hands one over.
    """

    class _Recorder:
        """A fallback that reports success and remembers what it was given."""

        def __init__(self):
            self.check = None

        async def run(self, page, goal, success_check=None):
            self.check = success_check
            return True, "DONE: looks fine to me"

    @pytest.mark.asyncio
    async def test_a_seek_is_checked_against_the_playhead(self):
        from computer_use_agent.navigator.base import NavigationError
        from computer_use_agent.navigator.resilient import ResilientNavigator

        class Inner:
            name = "inner"
            playhead = 12.0  # the drag landed nowhere near the 300s asked for

            async def seek_to(self, page, seconds):
                raise NavigationError("progress bar not found")

            async def read_player_time(self, page):
                return Inner.playhead

        recorder = self._Recorder()
        navigator = ResilientNavigator(Inner(), fallback=recorder)
        await navigator.seek_to(FakePage(""), 300.0)

        assert recorder.check is not None, "the seek rescue handed over no way to check itself"
        assert await recorder.check() is False, "a playhead at 00:12 is not a seek to 05:00"
        Inner.playhead = 299.0  # within a keyframe of the mark
        assert await recorder.check() is True

    @pytest.mark.asyncio
    async def test_opening_a_page_is_checked_for_a_player(self):
        from computer_use_agent.navigator.base import NavigationError
        from computer_use_agent.navigator.resilient import ResilientNavigator

        class Inner:
            name = "inner"
            has_player = False

            async def open_target(self, page, target):
                raise NavigationError("selector moved")

            async def read_player_time(self, page):
                return 0.0 if Inner.has_player else None

        recorder = self._Recorder()
        navigator = ResilientNavigator(Inner(), fallback=recorder)
        await navigator.open_target(FakePage(""), "https://x")

        assert recorder.check is not None
        assert await recorder.check() is False, "no readable player is not an opened page"
        Inner.has_player = True
        assert await recorder.check() is True

    @pytest.mark.asyncio
    async def test_an_unverifiable_result_is_a_failure_not_a_pass(self):
        # When the self-check itself cannot run, "unknown" must not resolve to
        # "fine" -- that is how a blocked page becomes recorded footage.
        from computer_use_agent.navigator.generic_agent import ComputerUseFallback

        fallback = ComputerUseFallback()

        async def explode(*_args, **_kwargs):
            raise RuntimeError("no credentials")

        fallback._screenshot = explode
        assert await fallback._verify(FakePage(""), "goal", None, "DONE, all good") is False


class TestAnUnknownPlatformIsReadNotGuessed:
    """The keyword lists are bilibili's wording. Customers do not run bilibili.

    `_DEAD_PAGE_MARKERS` and `_CHALLENGE_HINTS` are exact Chinese phrases
    lifted off one site. A Hikvision console saying 「录像已过期」or a SaaS
    slider labelled 「向右滑动完成校验」matches none of them, and the run used
    to die with a generic sentence that guessed at three possible causes. When
    they miss, the screen gets read instead.
    """

    @staticmethod
    def _navigator(gate=None, fallback=None):
        from computer_use_agent.navigator.base import NavigationError
        from computer_use_agent.navigator.resilient import ResilientNavigator

        class Inner:
            name = "unknown-vendor"

            async def open_target(self, page, target):
                raise NavigationError("selector moved")

            async def read_player_time(self, page):
                return None

        return ResilientNavigator(Inner(), fallback=fallback, gate=gate)

    @pytest.mark.asyncio
    async def test_an_expired_recording_stops_the_run_instead_of_being_retried(self, monkeypatch):
        from computer_use_agent.navigator import resilient
        from computer_use_agent.navigator.base import TargetUnavailable

        async def reads_expired(_page):
            return {"blocker": "unavailable", "description": "页面显示「录像已过期，超出保存期限」"}

        monkeypatch.setattr(resilient, "diagnose_block", reads_expired)

        class Fallback:
            turns = 0

            async def run(self, page, goal, success_check=None):
                Fallback.turns += 1
                return True, "ok"

        navigator = self._navigator(fallback=Fallback())
        with pytest.raises(TargetUnavailable, match="录像已过期"):
            await navigator.open_target(FakePage(""), "https://vendor/x")
        assert Fallback.turns == 0, "no agent can click a recording back into retention"

    @pytest.mark.asyncio
    async def test_an_unrecognised_slider_pages_a_human(self, monkeypatch):
        from computer_use_agent.navigator import resilient

        async def reads_slider(_page):
            return {"blocker": "challenge", "description": "页面要求「向右滑动完成校验」"}

        monkeypatch.setattr(resilient, "diagnose_block", reads_slider)

        class Gate:
            paged = []

            async def wait_for_human(self, page, reason):
                Gate.paged.append(reason)

        class Fallback:
            turns = 0

            async def run(self, page, goal, success_check=None):
                Fallback.turns += 1
                return True, "ok"

        navigator = self._navigator(gate=Gate(), fallback=Fallback())
        await navigator.open_target(FakePage(""), "https://vendor/x")
        assert Gate.paged and "向右滑动" in Gate.paged[0]
        assert Fallback.turns == 0, "Computer Use would burn its whole budget on a slider"

    @pytest.mark.asyncio
    async def test_an_unreadable_page_still_falls_through_to_the_agent(self, monkeypatch):
        # Not being able to diagnose must not become its own failure mode.
        from computer_use_agent.navigator import resilient

        async def cannot_tell(_page):
            return None

        monkeypatch.setattr(resilient, "diagnose_block", cannot_tell)

        class Fallback:
            turns = 0

            async def run(self, page, goal, success_check=None):
                Fallback.turns += 1
                return True, "recovered"

        navigator = self._navigator(fallback=Fallback())
        await navigator.open_target(FakePage(""), "https://vendor/x")
        assert Fallback.turns == 1

    @pytest.mark.asyncio
    async def test_a_page_that_cannot_be_screenshotted_returns_no_diagnosis(self):
        from computer_use_agent.navigator.base import diagnose_block
        assert await diagnose_block(FakePage("")) is None


class TestDeadPageClassification:
    """Gone vs blocked. Only one of them is worth telling the user to fix."""

    def _nav(self):
        from computer_use_agent.navigator.bilibili import BilibiliNavigator
        return BilibiliNavigator()

    class _Response:
        def __init__(self, status):
            self.status = status

    class _Page:
        def __init__(self, text=""):
            self._text = text

        async def title(self):
            return ""

        async def evaluate(self, script):
            return self._text

    @pytest.mark.asyncio
    async def test_404_is_permanent(self):
        from computer_use_agent.navigator.base import TargetUnavailable

        with pytest.raises(TargetUnavailable):
            await self._nav()._reject_dead_page(
                self._Page(), "https://x", self._Response(404))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403, 412, 429])
    async def test_anti_crawl_responses_are_not_called_dead(self, status):
        # bilibili answers 412 to a crawler-looking request for a perfectly
        # good video. Reporting that as "the video does not exist" sends
        # someone hunting for a URL bug that is not there.
        from computer_use_agent.navigator.base import NavigationError, TargetUnavailable

        with pytest.raises(NavigationError) as caught:
            await self._nav()._reject_dead_page(
                self._Page(), "https://x", self._Response(status))
        assert not isinstance(caught.value, TargetUnavailable)
        assert "不代表视频不存在" in str(caught.value)

    @pytest.mark.asyncio
    async def test_a_soft_404_is_read_off_the_page(self):
        from computer_use_agent.navigator.base import TargetUnavailable

        with pytest.raises(TargetUnavailable) as caught:
            await self._nav()._reject_dead_page(
                self._Page("啊叻？视频不见了"), "https://x", self._Response(200))
        assert "视频不见了" in str(caught.value)

    @pytest.mark.asyncio
    async def test_a_healthy_page_passes_through(self):
        await self._nav()._reject_dead_page(
            self._Page("正常的视频页面"), "https://x", self._Response(200))


class TestChallengeDetection:
    """False positives here suspend a multi-hour audit and page a human."""

    class _Page:
        """A page whose widget scan and body text the test controls."""

        def __init__(self, widget=None, text=""):
            self._widget = widget
            self._text = text

        async def evaluate(self, script, arg=None):
            if arg is not None:  # the widget scan takes the selector specs
                # Honour the specs actually passed: the detector runs two
                # scans with different lists, and a fake that answers both
                # identically would hide the whole conclusive/ambiguous split.
                selectors = {spec[0] for spec in arg}
                if self._widget and self._widget["selector"] in selectors:
                    return self._widget
                return None
            return self._text

    @pytest.mark.asyncio
    async def test_a_collapsed_login_widget_is_not_a_challenge(self):
        # bilibili ships .captcha-img__img and friends in its login panel at
        # all times. Blocking on those halts every run that sees a login nag.
        from computer_use_agent.navigator.base import detect_challenge

        assert await detect_challenge(self._Page(widget=None)) is None

    @pytest.mark.asyncio
    async def test_a_real_widget_still_fires(self):
        from computer_use_agent.navigator.base import detect_challenge

        page = self._Page(widget={"selector": ".geetest_panel", "width": 300, "height": 200})
        reason = await detect_challenge(page)
        assert reason and "geetest_panel" in reason and "300x200" in reason

    @pytest.mark.asyncio
    async def test_an_ordinary_sms_form_label_is_not_a_challenge(self):
        # "验证码" is the label next to every SMS login input in China. The old
        # hint list treated it as proof of a CAPTCHA.
        from computer_use_agent.navigator.base import detect_challenge

        assert await detect_challenge(
            self._Page(text="短信验证码\n请输入手机号\n登录")) is None

    @pytest.mark.asyncio
    async def test_an_explicit_instruction_is_a_challenge(self):
        from computer_use_agent.navigator.base import detect_challenge

        reason = await detect_challenge(self._Page(text="请完成安全验证后继续"))
        assert reason and "请完成安全验证" in reason

    def test_the_loose_wildcard_is_not_conclusive_on_its_own(self):
        # The vendor selectors are unambiguous, so they may be small and stand
        # alone. The catch-all may not: it matches the picture-captcha slot in
        # every Chinese login form.
        from computer_use_agent.navigator.base import _AMBIGUOUS_WIDGETS, _CHALLENGE_WIDGETS

        conclusive = dict((sel, (w, h)) for sel, w, h in _CHALLENGE_WIDGETS)
        assert "[class*='captcha' i]" not in conclusive
        assert conclusive[".gt_slider_knob"][0] < 120
        assert "[class*='captcha' i]" in dict((sel, (w, h)) for sel, w, h in _AMBIGUOUS_WIDGETS)

    @pytest.mark.asyncio
    async def test_a_full_size_login_panel_captcha_slot_is_not_a_challenge(self):
        # Measured on bilibili: ~60-90s into playback the site raises
        # .bili-mini-mask, which pauses the video and carries a 559x284
        # `captcha`-classed slot. That is a login nag -- keep_clear dismisses it
        # and playback resumes -- but it used to page a human on every run
        # longer than a minute, on a platform where no human can answer.
        from computer_use_agent.navigator.base import detect_challenge

        page = self._Page(
            widget={"selector": "[class*='captcha' i]", "width": 559, "height": 284},
            text="登录\n扫码登录\n短信验证码\n换一张",
        )
        assert await detect_challenge(page) is None

    @pytest.mark.asyncio
    async def test_the_same_widget_counts_once_the_page_says_so(self):
        from computer_use_agent.navigator.base import detect_challenge

        page = self._Page(
            widget={"selector": "[class*='captcha' i]", "width": 559, "height": 284},
            text="请完成安全验证\n拖动滑块完成拼图",
        )
        reason = await detect_challenge(page)
        assert reason and "请完成安全验证" in reason and "559x284" in reason

    @pytest.mark.asyncio
    async def test_a_scan_failure_does_not_invent_a_challenge(self):
        from computer_use_agent.navigator.base import detect_challenge

        class Broken:
            async def evaluate(self, script, arg=None):
                raise RuntimeError("page closed")

        assert await detect_challenge(Broken()) is None


class TestReportIsAboutThisRunOnly:
    """The JSONL spans every run ever made; a report must not.

    Reporting from the file produced a summary saying "5 violations" above a
    table with seven rows, two of them graded against a rule file that was
    swapped out weeks earlier.
    """

    @staticmethod
    def _outcome(index, status):
        from computer_use_agent.analyzer.video_analyzer import AnalysisOutcome

        result = WindowResult.model_validate({
            "scene_summary": "s",
            "findings": [{
                "rule_id": "CHK_GLOVE_002", "status": status,
                "confidence": 0.9, "severity": "RED_LINE" if status == "VIOLATION" else "NONE",
            }],
        })
        return AnalysisOutcome(clip=make_clip(index=index), result=result)

    @pytest.mark.asyncio
    async def test_an_earlier_runs_findings_stay_out_of_the_table(self, tmp_path):
        from computer_use_agent.store import AuditStore

        records = tmp_path / "records.jsonl"
        records.write_text(json.dumps({
            "id": 1, "time_range": "14:32 - 14:47", "violation_count": 1,
            "start_offset_seconds": 872.0, "end_offset_seconds": 887.0,
            "findings": [{"rule_id": "CHK_SANITATION_003", "status": "VIOLATION"}],
        }, ensure_ascii=False) + "\n", encoding="utf-8")

        store = AuditStore(records_path=records, evidence_dir=tmp_path / "ev")
        await store.record(self._outcome(0, "VIOLATION"))

        rule_ids = [f["rule_id"] for r in store.violations() for f in r["findings"]]
        assert rule_ids == ["CHK_GLOVE_002"]
        assert store.stats.violations == len(store.violations())

        # The trail itself is untouched -- it is the archive, not the report.
        assert len(store.all_violations_on_disk()) == 2

    @pytest.mark.asyncio
    async def test_covered_span_reports_what_was_actually_watched(self, tmp_path):
        from computer_use_agent.store import AuditStore

        store = AuditStore(records_path=tmp_path / "r.jsonl", evidence_dir=tmp_path / "ev")
        assert store.covered_span() is None
        for i in range(3):
            await store.record(self._outcome(i, "COMPLIANT"))
        start, end = store.covered_span()
        assert start == 0.0 and end == pytest.approx(make_clip(index=2).end_offset)


class TestCoverageHonesty:
    """A run that delivered 48s of a requested 10 minutes must say so."""

    class _Stub:
        """Just the attributes `_coverage` reads."""

        def __init__(self, span, stop_kind=None, footage_ends_at=None):
            self._stop_kind = stop_kind
            self._footage_ends_at = footage_ends_at
            self.store = type("S", (), {"covered_span": lambda _self: span})()

    def _coverage(self, span, *, start=120.0, duration=600.0, stop_kind=None, ends_at=None):
        from computer_use_agent.pipeline import AuditPipeline, AuditRequest

        request = AuditRequest(target="x", start_seconds=start, duration_seconds=duration)
        return AuditPipeline._coverage(self._Stub(span, stop_kind, ends_at), request)

    def test_a_video_shorter_than_the_request_is_not_an_early_stop(self):
        # Asking for ten minutes from 14:40 of a 15:15 video is the caller's
        # arithmetic, not a pipeline failure -- but coverage is still short.
        from computer_use_agent.agent import CctvAuditAgent

        out = self._coverage(
            (892.7, 907.7), start=880.0, stop_kind="video_ended", ends_at=915.2)
        assert out["complete"] is False
        assert "视频本身在 15:15 就结束了" in out["incomplete_reason"]

        class Store:
            def violations(self, limit=None):
                return []

        summary = {
            "windows_analyzed": 1, "windows_failed": 0, "violations": 0,
            "red_line_violations": 0, "capture_mode": "screen", "elapsed_seconds": 57,
            "stopped_because": "视频已播放完毕（15:15）", "input_tokens": 1,
            "output_tokens": 1, "records_path": "r", "evidence_dir": "e", **out,
        }
        headline = CctvAuditAgent._report(summary, Store()).splitlines()[0]
        assert headline == "### 📊 稽核完成（视频比请求的时间段短）"

    def test_a_short_run_is_flagged_incomplete(self):
        out = self._coverage((120.0, 194.0))
        assert out["complete"] is False
        assert "02:00" not in out["incomplete_reason"]  # it names where it *got to*
        assert "03:14" in out["incomplete_reason"] and "12:00" in out["incomplete_reason"]

    def test_landing_one_window_short_is_not_a_shortfall(self):
        # The last clip is cut on a window boundary, so exact arithmetic here
        # would flag every healthy run as incomplete.
        assert self._coverage((120.0, 715.0))["complete"] is True

    def test_analysing_nothing_is_never_reported_as_complete(self):
        out = self._coverage(None)
        assert out["complete"] is False and "没有采集到" in out["incomplete_reason"]

    def test_a_stall_is_flagged_even_with_no_duration_requested(self):
        out = self._coverage((0.0, 179.0), duration=None, stop_kind="playback_stalled")
        assert out["complete"] is False and "卡住" in out["incomplete_reason"]

    def test_a_full_run_with_no_duration_requested_is_complete(self):
        out = self._coverage((0.0, 910.0), duration=None, stop_kind="video_ended")
        assert out["complete"] is True and out["incomplete_reason"] is None

    def test_the_report_leads_with_the_shortfall(self):
        from computer_use_agent.agent import CctvAuditAgent

        summary = {
            "windows_analyzed": 5, "windows_failed": 0, "violations": 5,
            "red_line_violations": 2, "capture_mode": "screen", "elapsed_seconds": 90,
            "stopped_because": "播放在 02:59 卡住 30 秒不再前进", "input_tokens": 1,
            "output_tokens": 1, "records_path": "r.jsonl", "evidence_dir": "ev",
            "requested_start_seconds": 120.0, "requested_end_seconds": 720.0,
            "covered_from_seconds": 120.0, "covered_to_seconds": 194.0,
            "complete": False, "incomplete_reason": "只覆盖到 03:14，请求的是到 12:00，还差约 526 秒未稽核",
        }

        class Store:
            def violations(self, limit=None):
                return []

        report = CctvAuditAgent._report(summary, Store())
        # The headline is the part a reader trusts without scrolling.
        assert report.splitlines()[0].startswith("### ⚠️ 稽核提前结束")
        assert "稽核区间：02:00 - 03:14（请求 02:00 - 12:00）" in report
        assert "还差约 526 秒未稽核" in report
        # "no violations found" must not be read as "the footage was clean".
        assert report.index("未覆盖的时间段") < report.index("未发现违反 SOP 的行为")


class TestStallRecovery:
    """A pause at 03:00 of a 15-minute video is not the end of the video.

    The watchdog used to call any 30s stall end-of-footage, so one login nag
    turned a ten-minute audit into a 48-second one that still reported "稽核完成".
    """

    class _Runner:
        """Only the parts of AuditPipeline that the watchdog touches."""

        from computer_use_agent.pipeline import AuditPipeline as _P
        stop = _P.stop
        _watch_page = _P._watch_page
        # The real one: the watchdog calls it every poll, and a stub that
        # silently succeeded would hide a crash in it behind the loop's
        # catch-all.
        _hold_geometry = _P._hold_geometry
        del _P

        def __init__(self):
            self._stop = asyncio.Event()
            self._stop_reason = None
            self._stop_kind = None
            self._footage_ends_at = None
            self._player_rect = None
            self._geometry_warned = False
            self.events = []

        def _emit(self, event, payload):
            self.events.append((event, payload))

        def kinds(self):
            return [e for e, _ in self.events]

    class _Navigator:
        def __init__(self, current, duration, *, recoverable, budget=40):
            self.current, self.duration = current, duration
            self.recoverable = recoverable
            self.plays = 0
            self._left = budget
            self.runner = None

        async def keep_clear(self, page):
            pass

        async def ensure_playing(self, page):
            self.plays += 1
            if self.recoverable:
                self.moving = True

        async def read_playback_state(self, page):
            self._left -= 1
            if self._left <= 0:  # end the test, not the audit
                self.runner._stop.set()
            if getattr(self, "moving", False):
                self.current += 2.0
            return {"ended": False, "current_time": self.current, "duration": self.duration}

    async def _run(self, navigator, monkeypatch):
        import types as _types

        from computer_use_agent import pipeline as pipeline_module

        # Config is frozen, and the poll interval is the only thing standing
        # between this test and half a minute of real sleeping.
        monkeypatch.setattr(pipeline_module, "config", _types.SimpleNamespace(
            page_watch_seconds=0.001, stop_on_video_end=True,
        ))
        runner = self._Runner()
        navigator.runner = runner
        await asyncio.wait_for(runner._watch_page(object(), navigator), timeout=10)
        return runner

    @pytest.mark.asyncio
    async def test_a_recoverable_stall_does_not_end_the_run(self, monkeypatch):
        navigator = self._Navigator(179.0, 916.0, recoverable=True)
        runner = await self._run(navigator, monkeypatch)

        assert "playback_recovering" in runner.kinds()
        assert "playback_stalled" not in runner.kinds()
        assert runner._stop_reason is None  # the test stopped it, not the watchdog
        assert navigator.current > 179.0

    @pytest.mark.asyncio
    async def test_an_unrecoverable_stall_gives_up_only_after_trying(self, monkeypatch):
        navigator = self._Navigator(179.0, 916.0, recoverable=False)
        runner = await self._run(navigator, monkeypatch)

        assert navigator.plays == 3
        assert runner._stop_kind == "playback_stalled"
        assert "02:59" in runner._stop_reason and "3 次恢复播放" in runner._stop_reason
        assert runner._footage_ends_at == 179.0

    @pytest.mark.asyncio
    async def test_a_stall_in_the_last_seconds_is_still_confirmed_fast(self, monkeypatch):
        # Nothing left to buffer, so poking the player would only add delay.
        navigator = self._Navigator(910.0, 916.0, recoverable=False)
        runner = await self._run(navigator, monkeypatch)

        assert navigator.plays == 0
        assert runner._stop_kind == "playback_stalled"

    @pytest.mark.asyncio
    async def test_reaching_the_end_is_reported_as_finished_not_stalled(self, monkeypatch):
        navigator = self._Navigator(915.5, 916.0, recoverable=False)
        runner = await self._run(navigator, monkeypatch)

        assert runner._stop_kind == "video_ended"
        assert "播放完毕" in runner._stop_reason


class TestPreviewChannel:
    """The dashboard feed must never be able to degrade the evidence feed."""

    def test_preview_defaults_are_cheaper_than_the_evidence_feed(self):
        from computer_use_agent.config import config

        # Measured on a 1080p video page: a full-size q85 frame is ~280 KB of
        # base64, a 960x540 q60 frame ~58 KB. The point of the second stream is
        # that 15 fps of the cheap one costs less than 4 fps of the dear one.
        assert config.preview_width < config.screen_width
        assert config.preview_height < config.screen_height
        assert config.preview_quality < 85
        assert config.preview_fps > config.capture_fps

    def test_the_preview_is_not_the_recorder_s_job(self):
        # It used to be, and that was invisible for as long as every run was a
        # screen recording. The day Plan A started working the dashboard went
        # black: under Plan A no ScreenRecorder is ever constructed, so nothing
        # started a screencast and the operator watched a frozen frame while
        # the audit ran fine. The preview has to belong to the run, not to one
        # of the two capture strategies.
        import inspect

        from computer_use_agent.capture.screen_recorder import ScreenRecorder

        params = inspect.signature(ScreenRecorder.__init__).parameters
        assert not [p for p in params if "preview" in p], \
            "the recorder must not own the dashboard feed"

    @pytest.mark.asyncio
    async def test_the_preview_forwards_the_latest_frame_and_drops_the_rest(self):
        from computer_use_agent.capture.preview import LivePreview

        sent: list[str] = []
        page = self._FakePage()
        preview = LivePreview(page=page, on_frame=sent.append, fps=50)
        await preview.start()

        assert page.cdp.started["quality"] == 60
        await page.cdp.emit("frame-1")
        await page.cdp.emit("frame-2")   # arrives before the pump next ticks
        await asyncio.sleep(0.1)
        await preview.aclose()

        # Whatever it forwarded, it forwarded the newest frame and never
        # queued the stale one behind it.
        assert sent and set(sent) == {"frame-2"}
        assert page.cdp.stopped

    @pytest.mark.asyncio
    async def test_closing_a_preview_that_never_started_is_safe(self):
        # The pipeline's shutdown path runs whether or not navigation got
        # far enough to open anything.
        from computer_use_agent.capture.preview import LivePreview

        await LivePreview(page=None, on_frame=lambda _: None).aclose()

    class _FakeCdp:
        def __init__(self):
            self.started = None
            self.stopped = False
            self._handler = None

        def on(self, event, handler):
            assert event == "Page.screencastFrame"
            self._handler = handler

        async def send(self, method, params=None):
            if method == "Page.startScreencast":
                self.started = params
            elif method == "Page.stopScreencast":
                self.stopped = True

        async def emit(self, data):
            await self._handler({"data": data, "sessionId": 1})

    class _FakePage:
        def __init__(self):
            self.cdp = TestPreviewChannel._FakeCdp()
            outer = self

            class _Context:
                async def new_cdp_session(self, _page):
                    return outer.cdp

            self.context = _Context()

    def test_a_slow_link_drops_frames_instead_of_queueing_them(self):
        from computer_use_agent.monitor import BrowserMonitorClient

        client = BrowserMonitorClient()
        sent = []
        client._fire_and_forget = lambda payload, on_done=None: sent.append(on_done)

        client.update_frame_b64("a")
        # Second frame arrives before the first POST completes.
        client.update_frame_b64("b")
        assert len(sent) == 1 and client._frames_dropped == 1

        sent[0]()  # the first POST finally lands
        client.update_frame_b64("c")
        assert len(sent) == 2 and client._frames_dropped == 1

    def test_a_failed_send_does_not_wedge_the_channel_shut(self):
        # The in-flight flag is only safe if it is always released. A send that
        # cannot even be scheduled must not stop the preview forever.
        from computer_use_agent.monitor import BrowserMonitorClient

        client = BrowserMonitorClient()
        client.update_frame_b64("a")  # no running loop -> _fire_and_forget bails
        assert client._frame_in_flight is False
        client.update_frame_b64("b")
        assert client._frames_dropped == 0


class TestPlayerGeometryIsHeld:
    """ffmpeg's crop is fixed for the life of the recording, so the player has
    to be held still under it. Measured on bilibili: the login modal that
    appears a minute in drops the player out of web fullscreen, from (0,0)
    1920x1080 to (62,172) 1354x762, and dismissing the modal does not put it
    back. Nothing errors -- the frames stay the right size, and the crop starts
    framing the page header and half the picture."""

    class _Runner:
        from computer_use_agent.pipeline import AuditPipeline as _P
        _hold_geometry = _P._hold_geometry
        del _P

        def __init__(self, player_rect):
            self._player_rect = player_rect
            self._geometry_warned = False
            self.events = []

        def _emit(self, event, payload):
            self.events.append((event, payload))

        def kinds(self):
            return [e for e, _ in self.events]

    class _Navigator:
        def __init__(self, rects, *, fullscreen_restores):
            self._rects = list(rects)
            self._fullscreen_restores = fullscreen_restores
            self.fullscreen_calls = 0

        async def video_rect(self, page):
            return self._rects[0] if len(self._rects) == 1 else self._rects.pop(0)

        async def enter_fullscreen(self, page):
            self.fullscreen_calls += 1
            return self._fullscreen_restores

    @staticmethod
    def _box(x, y, w, h):
        return {"x": x, "y": y, "width": w, "height": h}

    @pytest.fixture
    def _cfg(self, monkeypatch):
        import types as _types

        from computer_use_agent import pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "config",
                            _types.SimpleNamespace(fullscreen_player=True))

    @pytest.mark.asyncio
    async def test_a_player_knocked_out_of_fullscreen_is_put_back(self, _cfg):
        full = self._box(0, 0, 1920, 1080)
        navigator = self._Navigator(
            [self._box(62, 172, 1354, 762), full], fullscreen_restores=True)
        runner = self._Runner(dict(full))

        await runner._hold_geometry(object(), navigator)

        assert navigator.fullscreen_calls == 1
        assert runner.kinds() == ["player_geometry_restored"]

    @pytest.mark.asyncio
    async def test_a_player_that_never_moved_costs_nothing(self, _cfg):
        full = self._box(0, 0, 1920, 1080)
        navigator = self._Navigator([dict(full)], fullscreen_restores=True)
        runner = self._Runner(dict(full))

        await runner._hold_geometry(object(), navigator)

        assert navigator.fullscreen_calls == 0, "re-entering fullscreen every 5s would fight the page"
        assert runner.events == []

    @pytest.mark.asyncio
    async def test_sub_pixel_jitter_is_not_a_move(self, _cfg):
        navigator = self._Navigator([self._box(0.4, 0, 1919.7, 1080)],
                                    fullscreen_restores=True)
        runner = self._Runner(self._box(0, 0, 1920, 1080))

        await runner._hold_geometry(object(), navigator)

        assert runner.events == []

    @pytest.mark.asyncio
    async def test_an_unrestorable_move_is_reported_once(self, _cfg):
        # Silence here means a clean-looking report built on mis-framed
        # footage, which is worse than an ugly one that says so.
        moved = self._box(62, 172, 1354, 762)
        navigator = self._Navigator([moved], fullscreen_restores=False)
        runner = self._Runner(self._box(0, 0, 1920, 1080))

        await runner._hold_geometry(object(), navigator)
        await runner._hold_geometry(object(), navigator)

        assert runner.kinds() == ["player_geometry_lost"], "once, not every 5s poll"
        assert runner.events[0][1]["actual"] == [62, 172, 1354, 762]

    @pytest.mark.asyncio
    async def test_a_video_that_needs_no_crop_is_still_held(self, _cfg):
        # The bug this class was written for did not fire on the one video it
        # was verified against. A 16:9 video fullscreened into a 16:9 viewport
        # has no letterbox bars, so content_box returns None, so no crop filter
        # is built -- and the baseline rect used to be recorded only when one
        # was. Nothing held the player, the login modal dropped it out of
        # fullscreen, and every evidence frame after that was the whole
        # bilibili page: sidebar, recommendations, comments, with the footage
        # in one corner. Confirmed against saved evidence frames.
        from computer_use_agent.capture.screen_recorder import content_box
        full_frame = {"x": 0, "y": 0, "width": 1920, "height": 1080,
                      "intrinsic_width": 1920, "intrinsic_height": 1080}
        assert content_box(full_frame) is None, "no bars to cut"

        # So the baseline has to come from the measurement, not from the crop,
        # and holding it has to work with no crop in play.
        navigator = self._Navigator(
            [self._box(62, 172, 1354, 762), dict(full_frame)], fullscreen_restores=True)
        runner = self._Runner(dict(full_frame))

        await runner._hold_geometry(object(), navigator)

        assert runner.kinds() == ["player_geometry_restored"]


class TestTheRunStartsWhereItWasAsked:
    """Setup -- probing for a stream, going fullscreen, measuring the player --
    takes ten-odd seconds, and the video plays throughout."""

    def _seek_needed(self, playhead, requested):
        # Mirrors the guard in _build_producer.
        return abs((playhead or 0.0) - requested) > 1.0

    def test_from_the_beginning_still_seeks_back(self):
        # The old guard was `if request.start_seconds > 0`, which skipped the
        # single most common request there is. The first window came out at
        # 00:14 and the opening fourteen seconds -- for a store audit, the
        # whole hand-washing step -- were never looked at, with nothing in the
        # report to say so.
        assert self._seek_needed(playhead=14.2, requested=0.0)

    def test_a_mid_video_start_still_seeks_back(self):
        assert self._seek_needed(playhead=112.0, requested=100.0)

    def test_an_already_correct_playhead_is_left_alone(self):
        assert not self._seek_needed(playhead=100.4, requested=100.0)
        assert not self._seek_needed(playhead=0.3, requested=0.0)


class TestPlanAIsNotLeftOnTheTable:
    """Which capture mode gets picked, without a browser or a network.

    Written after a live check showed bilibili -- the only platform the whole
    pipeline had ever been run against -- falling back to screen recording
    while ffprobe could open its media URL end to end. Plan A had never
    actually executed once.
    """

    class _Page:
        url = "https://example.com/watch"

        def on(self, *_):
            pass

        def remove_listener(self, *_):
            pass

    def _probe(self, monkeypatch, *, playlists=(), streams=(), segments=(), answers=None):
        from computer_use_agent.capture import probe as probe_mod

        async def fake_probe_stream(url, headers=None, timeout=15.0):
            return (answers or {}).get(url)

        monkeypatch.setattr(probe_mod, "probe_stream", fake_probe_stream)
        probe = probe_mod.StreamProbe(self._Page())
        probe._playlists = list(playlists)
        probe._streams = list(streams)
        probe._segments = list(segments)
        # Header building wants real cookies off a real context.
        probe._build_headers = lambda: asyncio.sleep(0, result={})
        return probe

    @staticmethod
    def _info(duration, width=1920, height=1080):
        return {
            "format": {"duration": str(duration), "format_name": "mov,mp4"},
            "streams": [{"codec_type": "video", "width": width, "height": height,
                         "codec_name": "h264"}],
        }

    @pytest.mark.asyncio
    async def test_a_whole_track_disguised_as_a_segment_wins_plan_a(self, monkeypatch):
        # bilibili's exact shape: no .m3u8 and no .mpd is ever requested (the
        # manifest is embedded in page JSON), and the player byte-ranges one
        # .m4s that is the entire 519-second recording.
        url = "https://cdn.example.com/v/33214631912-1-100023.m4s?token=x"
        probe = self._probe(monkeypatch, segments=[url], answers={url: self._info(519.33)})

        source = await probe.decide("auto", wait_seconds=0.0)

        assert source.mode == "stream"
        assert source.url == url

    @pytest.mark.asyncio
    async def test_a_real_short_slice_still_falls_back(self, monkeypatch):
        # An HLS .ts really is a few seconds. Grabbing one would audit six
        # seconds of footage and report it as the whole window.
        url = "https://cdn.example.com/hls/seg-00042.ts"
        probe = self._probe(monkeypatch, segments=[url], answers={url: self._info(6.0)})

        source = await probe.decide("auto", wait_seconds=0.0)

        assert source.mode == "screen"

    @pytest.mark.asyncio
    async def test_the_best_quality_track_is_chosen_not_the_first_seen(self, monkeypatch):
        # A DASH player fetches several quality ladders and whichever lands
        # first is arbitrary. Auditing 640x360 when 1920x1080 cost the same is
        # a worse audit for no saving.
        low = "https://cdn.example.com/v/1-100022.m4s"
        high = "https://cdn.example.com/v/1-100026.m4s"
        probe = self._probe(monkeypatch, segments=[low, high], answers={
            low: self._info(519.0, 640, 360),
            high: self._info(519.0, 1920, 1080),
        })

        source = await probe.decide("auto", wait_seconds=0.0)

        assert source.url == high

    @pytest.mark.asyncio
    async def test_a_playlist_still_beats_a_whole_track(self, monkeypatch):
        # One URL covering the recording is simpler and seekable; only reach
        # for the segment bucket when there is no playlist at all.
        playlist = "https://cdn.example.com/index.m3u8"
        segment = "https://cdn.example.com/v/1-100026.m4s"
        probe = self._probe(monkeypatch, playlists=[playlist], segments=[segment], answers={
            playlist: self._info(519.0, 640, 360),
            segment: self._info(519.0, 1920, 1080),
        })

        source = await probe.decide("auto", wait_seconds=0.0)

        assert source.url == playlist

    @pytest.mark.asyncio
    async def test_forcing_stream_mode_raises_rather_than_pretending(self, monkeypatch):
        # CAPTURE_MODE=stream is how an operator tests Plan A. Silently
        # recording the screen would make that test always pass.
        url = "https://cdn.example.com/hls/seg-00042.ts"
        probe = self._probe(monkeypatch, segments=[url], answers={url: self._info(6.0)})

        with pytest.raises(RuntimeError, match="no ffprobe-readable media URL"):
            await probe.decide("stream", wait_seconds=0.0)

    @pytest.mark.asyncio
    async def test_a_canvas_player_falls_back_without_probing_anything(self, monkeypatch):
        # Hikvision/Dahua web clients decode in WASM into a canvas. Nothing to
        # grab, and the reason has to say so rather than blame ffprobe.
        probe = self._probe(monkeypatch)

        source = await probe.decide("auto", wait_seconds=0.0)

        assert source.mode == "screen"
        assert "canvas" in source.reason


class TestTheReportReadsAsATimeline:
    """Analysis runs concurrently, so windows finish out of order.

    Every read path has to put them back. Recorded order is correct data and
    unreadable presentation -- a table that goes 01:12, 01:00, 01:24 makes a
    reviewer distrust the whole document.
    """

    def _record(self, index, start, findings):
        return {
            "id": 100 + index,
            "window_index": index,
            "start_offset_seconds": float(start),
            "end_offset_seconds": float(start) + 15.0,
            "time_range": f"{Clip.format_offset(start)} - {Clip.format_offset(start + 15)}",
            "violation_count": sum(1 for f in findings if f["status"] == "VIOLATION"),
            "findings": findings,
        }

    def _finding(self, offset, rule_id="CHK_GLOVE_002", status="VIOLATION"):
        return {"rule_id": rule_id, "rule_name": "手套", "status": status,
                "confidence": 1.0, "evidence": f"@{offset}", "severity": "RED_LINE",
                "offset_seconds": float(offset), "timestamp": Clip.format_offset(offset),
                "timestamp_exact": True, "evidence_frame": "e.jpg"}

    def _store(self, records):
        from computer_use_agent.store import AuditStore

        store = AuditStore.__new__(AuditStore)
        store._run_records = records
        return store

    def test_windows_come_back_in_video_order_not_completion_order(self):
        # The exact shape seen in a live Plan A run: worker 2 finished first.
        store = self._store([
            self._record(1, 72, [self._finding(84)]),
            self._record(0, 60, [self._finding(68)]),
            self._record(2, 84, [self._finding(86)]),
        ])
        assert [r["window_index"] for r in store.violations()] == [0, 1, 2]

    def test_findings_inside_one_window_are_ordered_too(self):
        from computer_use_agent.agent import CctvAuditAgent

        store = self._store([
            self._record(0, 60, [self._finding(71, "B"), self._finding(62, "A")]),
        ])
        report = CctvAuditAgent._report(self._summary(), store)
        assert report.index("| A ") < report.index("| B ")

    def test_a_truncated_table_says_it_is_truncated(self):
        # Keeping the earliest 50 silently reads as "these are all of them".
        from computer_use_agent.agent import CctvAuditAgent, _MAX_TABLE_ROWS

        store = self._store([
            self._record(i, i * 15, [self._finding(i * 15 + 3)])
            for i in range(_MAX_TABLE_ROWS + 7)
        ])
        report = CctvAuditAgent._report(self._summary(), store)

        assert f"共 {_MAX_TABLE_ROWS + 7} 个窗口有违规" in report
        # And it is a prefix of the audit, not an arbitrary sample of it.
        assert "00:00 - 00:15" in report
        assert "12:30 - 12:45" not in report

    def test_a_complete_table_says_nothing_about_truncation(self):
        from computer_use_agent.agent import CctvAuditAgent

        store = self._store([self._record(0, 60, [self._finding(62)])])
        assert "个窗口有违规" not in CctvAuditAgent._report(self._summary(), store)

    def test_the_dashboard_slots_a_late_window_into_place(self):
        # A slow window must not land at the bottom of the sidebar minutes
        # after the windows that follow it.
        from computer_use_agent.monitor_server import _segment_order

        arrived = [
            {"id": 2, "start_seconds": 72.0},
            {"id": 3, "start_seconds": 84.0},
            {"id": 1, "start_seconds": 60.0},   # the straggler
        ]
        assert [s["id"] for s in sorted(arrived, key=_segment_order)] == [1, 2, 3]

    def test_records_from_an_older_jsonl_keep_their_insertion_order(self):
        # `start_seconds` postdates some records on disk. Missing must not mean
        # "time zero", or replayed history piles up above the live run.
        from computer_use_agent.monitor_server import _segment_order

        old = [{"id": 3}, {"id": 1}, {"id": 2}]
        assert [s["id"] for s in sorted(old, key=_segment_order)] == [1, 2, 3]

    @staticmethod
    def _summary():
        return {
            "windows_analyzed": 3, "windows_failed": 0, "violations": 3,
            "red_line_violations": 3, "input_tokens": 1, "output_tokens": 1,
            "records_path": "/tmp/r.jsonl", "evidence_dir": "/tmp/e",
            "capture_mode": "stream", "elapsed_seconds": 1.0,
            "stopped_because": "done", "complete": True,
            "requested_start_seconds": 0.0, "requested_end_seconds": None,
            "covered_from_seconds": 0.0, "covered_to_seconds": 15.0,
        }
