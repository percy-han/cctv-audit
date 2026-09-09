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
import contextlib
import json
import logging
import time
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


def _confirm_reply(*, capture_mode: str, **preflight) -> str:
    """What the customer is told when they say 「确认」, as one string.

    Drives `_do_confirm` against a service stub rather than asserting on the
    source, because the branch under test is about what reaches the customer.
    """
    import asyncio as _asyncio

    from computer_use_agent.jobs import Job
    from computer_use_agent.server import Turn, _do_confirm

    job = Job(
        user_id="u", job_id="j0b", target="t", state="ready",
        preflight={"ok": True, "capture_mode": capture_mode, **preflight},
    )

    class Svc:
        async def get_status(self, user_id, **kw):
            return job

        async def start_audit(self, user_id, job_id):
            return job

    async def go():
        turn = Turn(user_id="u", session_id="s", text="确认")
        said = [
            e["events"][0]["content"]["parts"][0]["text"]
            async for e in _do_confirm(Svc(), turn, _plain_say)
        ]
        return "\n".join(said)

    return _asyncio.run(go())


async def _async_none(*args, **kwargs):
    """Stands in for a best-effort coroutine that produced nothing."""
    return None


def _plain_say(text: str) -> dict:
    """The shape `_do_confirm` expects back from `say`, minus the ADK plumbing."""
    return {"events": [{"content": {"parts": [{"text": text}]}}]}


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


class TestInlineSizeCap:
    """Vertex's real inline limits, measured 2026-09-08 by walking MP4s upwards.

    240 MB went through; 360 MB came back with "exceeds the maximum allowed
    inline size of 256000000 bytes"; 458 MB tripped a second, outer limit --
    "Request payload size exceeds the limit: 524288000 bytes" -- because the
    body is base64 and inflates by about a third.

    The constant used to be 18 MB against a guessed "~20 MB", which was the
    stated reason WINDOW_SECONDS could not go to 300. It was out by more than
    a factor of ten.
    """

    def test_the_cap_stays_under_the_limit_vertex_actually_enforces(self):
        from computer_use_agent.analyzer.video_analyzer import _MAX_INLINE_BYTES

        assert _MAX_INLINE_BYTES <= 256_000_000
        # And under the whole-request cap once base64 has had its way with it.
        assert _MAX_INLINE_BYTES * 4 / 3 <= 524_288_000

    def test_a_five_minute_window_fits_a_normal_store_camera(self):
        from computer_use_agent.analyzer.video_analyzer import _MAX_INLINE_BYTES

        # The number that decides whether WINDOW_SECONDS=300 needs a GCS upload
        # path: any source below this bitrate does not.
        mbps = _MAX_INLINE_BYTES * 8 / 300 / 1e6
        assert mbps > 6.0, f"a 300s window only fits {mbps:.1f} Mbps"

    def test_an_oversized_clip_is_refused_before_the_request_is_built(
        self, monkeypatch, tmp_path
    ):
        from computer_use_agent.analyzer import video_analyzer as va

        # Shrink the cap rather than write 250 MB: the guard is a comparison,
        # and the real number is pinned by the two tests above.
        monkeypatch.setattr(va, "_MAX_INLINE_BYTES", 1024)
        clip_path = tmp_path / "huge.mp4"
        clip_path.write_bytes(b"\x00" * 1025)

        def explode(*a, **kw):
            raise AssertionError("the guard let an oversized clip reach Vertex")

        monkeypatch.setattr(va, "generate_content_with_retry", explode)
        analyzer = VideoAnalyzer(load_rules(CCTV_RULES))
        outcome = asyncio.run(analyzer.analyze(make_clip(path=clip_path)))

        assert outcome.result is None
        assert "inline limit" in outcome.error


class TestMediaProcessingSwitch:
    """MEDIA_PROCESSING=agentic hands the window to the model's own video tool.

    Three things about that mode are not guessable from the API surface and
    were measured on 2026-09-08 against Vertex (project study-project-496907,
    location=global, gemini-3.8-flash):

      * the frames it fetches are billed to `tool_use_prompt_token_count`, not
        `prompt_token_count` -- 12,671 vs 2,723 on a 300s window, so reading
        only the latter under-reports a run by 85% and MAX_COST_TOKENS stops
        tracking the spend;
      * it ignores `VideoMetadata`, so setting fps alongside it is a lie to the
        next reader;
      * gemini-3.5-flash rejects it outright, with a message that names neither
        the setting nor a model that works.
    """

    def _capture_call(self, monkeypatch, tmp_path, processing):
        import dataclasses

        from computer_use_agent.analyzer import video_analyzer as va
        from computer_use_agent.config import config as real_config

        clip_path = tmp_path / "w.mp4"
        clip_path.write_bytes(b"\x00" * 64)
        monkeypatch.setattr(
            va, "config", dataclasses.replace(real_config, media_processing=processing)
        )

        seen = {}

        async def fake_call(*, model, contents, generate_config, **kw):
            seen["contents"] = contents
            raise RuntimeError("stop here -- the request is what this test is about")

        monkeypatch.setattr(va, "generate_content_with_retry", fake_call)
        analyzer = VideoAnalyzer(load_rules(CCTV_RULES))
        asyncio.run(analyzer.analyze(make_clip(path=clip_path)))
        return seen["contents"][0]

    def test_static_sends_a_frame_ladder_and_no_agentic_flag(self, monkeypatch, tmp_path):
        from computer_use_agent.config import config

        part = self._capture_call(monkeypatch, tmp_path, "static")
        # Not a hardcoded number: a developer's .env may set ANALYSIS_FPS.
        assert part.video_metadata is not None
        assert part.video_metadata.fps == config.analysis_fps
        assert part.media_processing is None

    def test_agentic_sets_the_flag_and_claims_no_fps_it_cannot_honour(
        self, monkeypatch, tmp_path
    ):
        from google.genai.types import MediaProcessing

        part = self._capture_call(monkeypatch, tmp_path, "agentic")
        assert part.media_processing is MediaProcessing.AGENTIC
        assert part.video_metadata is None

    def test_the_bill_counts_the_frames_the_tool_fetched(self, monkeypatch, tmp_path):
        from computer_use_agent.analyzer import video_analyzer as va

        clip_path = tmp_path / "w.mp4"
        clip_path.write_bytes(b"\x00" * 64)

        class Usage:
            prompt_token_count = 2723
            tool_use_prompt_token_count = 12671
            candidates_token_count = 1018

        class Response:
            usage_metadata = Usage()
            text = json.dumps({"scene_summary": "s", "findings": []})

        async def fake_call(**kw):
            return Response()

        monkeypatch.setattr(va, "generate_content_with_retry", fake_call)
        analyzer = VideoAnalyzer(load_rules(CCTV_RULES))
        outcome = asyncio.run(analyzer.analyze(make_clip(path=clip_path)))
        assert outcome.input_tokens == 2723 + 12671

    def test_the_wrong_model_is_reported_as_the_wrong_model(self):
        analyzer = VideoAnalyzer(load_rules(CCTV_RULES))
        message = analyzer._explain(
            RuntimeError("400 INVALID_ARGUMENT. Video understanding tool is not "
                         "enabled for this model")
        )
        # The raw API text says none of this.
        assert "MEDIA_PROCESSING" in message
        assert "gemini-3.8-flash" in message

    def test_an_unknown_mode_is_refused_at_startup(self):
        import dataclasses

        from computer_use_agent.config import config as real_config

        bad = dataclasses.replace(real_config, media_processing="agentix")
        assert any("MEDIA_PROCESSING" in p for p in bad.validate())

    def test_agentic_on_a_model_that_refuses_it_is_caught_at_startup(self):
        # Otherwise the first sign is window 1 coming back empty, with the
        # browser open and the clips already cut -- and then every window
        # after it, one at a time, for the whole length of the run.
        import dataclasses

        from computer_use_agent.config import config as real_config

        bad = dataclasses.replace(
            real_config, media_processing="agentic",
            analysis_model="gemini-3.5-flash",
        )
        problems = [p for p in bad.validate() if "video understanding" in p]
        assert len(problems) == 1
        # Naming the way out matters more than naming the fault: the raw API
        # message mentions neither the setting nor a model that works.
        assert "gemini-3.8-flash" in problems[0]
        assert "ANALYSIS_MODEL" in problems[0]

    def test_the_working_pairing_passes_and_so_does_static_on_either_model(self):
        import dataclasses

        from computer_use_agent.config import config as real_config

        for processing, model in (
            ("agentic", "gemini-3.8-flash"),
            ("static", "gemini-3.5-flash"),
            ("static", "gemini-3.8-flash"),
        ):
            cfg = dataclasses.replace(
                real_config, media_processing=processing, analysis_model=model,
            )
            assert not [p for p in cfg.validate() if "video understanding" in p], (
                f"{processing} + {model} must not be flagged"
            )

    def test_a_model_that_merely_shares_the_prefix_is_not_assumed_broken(self):
        # `gemini-3.5-flash-lite` is a different model, and the blog lists it as
        # supporting agentic. We have not tried it, and guessing "no" would
        # block a pairing that may well work -- the deny-list only names what
        # was actually tried and refused.
        import dataclasses

        from computer_use_agent.config import config as real_config

        cfg = dataclasses.replace(
            real_config, media_processing="agentic",
            analysis_model="gemini-3.5-flash-lite",
        )
        assert not [p for p in cfg.validate() if "video understanding" in p]


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

    @pytest.mark.asyncio
    async def test_a_fallback_that_crashes_does_not_erase_the_diagnosis(self, monkeypatch):
        """Job 994608, 2026-09-08: the customer was told the wrong thing.

        bilibili answered 412 and the deterministic step said so, in plain
        language, in the log. Computer Use took over, its own screenshot timed
        out after 30s, and that exception flew straight past the composition
        below -- so what came back was "Page.screenshot: Timeout 30000ms
        exceeded" and nothing about risk control at all. A worse message than
        none: it points at a screenshot bug that does not exist.
        """
        from computer_use_agent.navigator import resilient
        from computer_use_agent.navigator.base import NavigationError

        async def cannot_tell(_page):
            return None

        monkeypatch.setattr(resilient, "diagnose_block", cannot_tell)

        class ExplodingFallback:
            async def run(self, page, goal, success_check=None):
                raise TimeoutError("Page.screenshot: Timeout 30000ms exceeded")

        navigator = self._navigator(fallback=ExplodingFallback())
        with pytest.raises(NavigationError) as caught:
            await navigator.open_target(FakePage(""), "https://vendor/x")

        message = str(caught.value)
        assert "selector moved" in message, "the original diagnosis must survive"
        assert "Page.screenshot" in message, "and so must the reason the rescue failed"


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


class TestClosingTheLoginNag:
    """Closing the dialog is only half the job -- bilibili pauses the video.

    Job 692b53: the nag appeared ~70s in, `keep_clear` closed it on the next
    poll, and the player stayed paused. Chromium repaints nothing, so the
    screencast delivered 0 fps and the dashboard showed one still for the rest
    of the run while the audit itself carried on perfectly well.
    """

    class _Element:
        def __init__(self, present):
            self.present, self.clicked = present, 0

        async def count(self):
            return 1 if self.present else 0

        async def is_visible(self, timeout=None):
            return self.present

        async def click(self, timeout=None, force=False):
            self.clicked += 1
            self.present = False

    class _Locator:
        def __init__(self, element):
            self.first = element

    class _Page:
        def __init__(self, *, modal, hidden=0):
            self.modal = TestClosingTheLoginNag._Element(modal)
            self.hidden = hidden
            self.evaluated = 0

        def locator(self, selector):
            # Only the first closer in the list matches, as on the real page.
            present = selector == ".bili-mini-mask .bili-mini-close-icon"
            return TestClosingTheLoginNag._Locator(
                self.modal if present else TestClosingTheLoginNag._Element(False))

        async def evaluate(self, script, *args):
            self.evaluated += 1
            # The overlay-hiding pass is the only one that takes an argument.
            return self.hidden if args else None

    def _nav(self):
        from computer_use_agent.navigator.bilibili import BilibiliNavigator
        return BilibiliNavigator()

    @pytest.mark.asyncio
    async def test_dismissing_a_modal_also_restarts_the_player(self, monkeypatch):
        navigator = self._nav()
        played = []

        async def ensure_playing(page):
            played.append(page)

        monkeypatch.setattr(navigator, "ensure_playing", ensure_playing)
        page = self._Page(modal=True)
        await navigator.keep_clear(page)

        assert page.modal.clicked == 1
        assert len(played) == 1

    @pytest.mark.asyncio
    async def test_a_clean_page_is_left_alone(self, monkeypatch):
        # keep_clear runs every few seconds for the whole audit. Calling play()
        # unconditionally would override a human working the gate, and would
        # make "we restarted the player" meaningless in the log.
        navigator = self._nav()
        played = []

        async def ensure_playing(page):
            played.append(page)

        monkeypatch.setattr(navigator, "ensure_playing", ensure_playing)
        page = self._Page(modal=False)
        await navigator.keep_clear(page)

        assert played == []

    @pytest.mark.asyncio
    async def test_a_player_that_will_not_restart_does_not_break_the_poll(self, monkeypatch):
        # The caller does its remaining housekeeping after this returns, and
        # will come round again in a few seconds anyway.
        from computer_use_agent.navigator.base import NavigationError

        navigator = self._nav()

        async def refuses(page):
            raise NavigationError("播放始终没有推进")

        monkeypatch.setattr(navigator, "ensure_playing", refuses)
        await navigator.keep_clear(self._Page(modal=True))

    @pytest.mark.asyncio
    async def test_the_count_separates_never_saw_it_from_could_not_fix_it(self):
        # This number is why the logging is INFO rather than DEBUG: DEBUG does
        # not reach Cloud Logging, and without it "no modal appeared" and "we
        # closed the modal and the player stayed paused" look identical.
        navigator = self._nav()
        assert await navigator._dismiss_overlays(self._Page(modal=False)) == 0
        assert await navigator._dismiss_overlays(self._Page(modal=True)) == 1
        assert await navigator._dismiss_overlays(self._Page(modal=True, hidden=2)) == 3


class TestAPageScriptCannotHangTheRun:
    """Job fcfd3c, 2026-09-08: a YouTube link, 900 seconds of silence.

    The `<video>` on a YouTube page opened headless exists but has no source
    (`readyState` 0, `networkState` 0), and `play()` on it returns a promise
    that is never settled -- not resolved, not rejected. `page.evaluate` is the
    one Playwright call with no timeout of its own, so `await v.play()` inside
    it waited until Agent Runtime cancelled the request at 900s. GE had already
    given up at 602s. The job sat in `probing` with an empty error field and no
    log line naming the step it died in.

    Three things had to be true at once for that, so there are three defences
    here, and each of these tests kills exactly one of them.
    """

    class _NeverSettles:
        """A page whose scripts return a promise the page keeps forever."""

        def __init__(self):
            self.calls = 0

        async def evaluate(self, _script, *_args):
            self.calls += 1
            await asyncio.Event().wait()  # exactly as unresolvable as v.play()

    @pytest.mark.asyncio
    async def test_a_script_that_never_returns_becomes_an_ordinary_failure(self):
        from computer_use_agent.browser_actions import PageScriptTimeout, evaluate_bounded

        with pytest.raises(PageScriptTimeout) as caught:
            await asyncio.wait_for(
                evaluate_bounded(self._NeverSettles(), "() => 1", budget=0.05,
                                 what="start playback"),
                timeout=5,
            )
        # The message has to name the step: "a page script hung" sends whoever
        # reads it back through the whole preflight to find out which one.
        assert "start playback" in str(caught.value)

    @pytest.mark.asyncio
    async def test_the_playback_script_does_not_await_the_page(self):
        # The bounded wait above is the backstop. This is the actual fix: the
        # promise is fired and dropped, because whether playback started is
        # answered by watching the playhead, not by the promise.
        from computer_use_agent.navigator.bilibili import BilibiliNavigator

        scripts = []

        class _Page:
            async def evaluate(self, script, *args):
                scripts.append(script)
                return True

            def locator(self, _selector):
                raise AssertionError("should not need a click")

        navigator = BilibiliNavigator()
        page = _Page()
        # Playback is advancing, so ensure_playing returns after one script.
        navigator._is_advancing = lambda _p: _true()

        async def _true():
            return True

        await navigator.ensure_playing(page)
        assert len(scripts) == 1
        assert "await" not in scripts[0], (
            "awaiting a page promise inside evaluate is what hung job fcfd3c")
        assert ".catch(" in scripts[0], "the rejection still has to be swallowed"

    @pytest.mark.asyncio
    async def test_no_source_is_reported_as_no_source(self):
        # A screenshot of this cannot tell it apart from an ad or a slow load:
        # all three are a black rectangle. `diagnose_block` would say "nothing
        # is blocking it", which is true and useless. The element's own state
        # is the only place the answer exists.
        from computer_use_agent.navigator.bilibili import BilibiliNavigator

        class _Page:
            async def evaluate(self, _script, *_args):
                return {"src": "", "ready_state": 0, "network_state": 0}

        why = await BilibiliNavigator()._why_not_playing(_Page())
        assert "没有加载任何视频源" in why
        assert "适配器" in why, "the reader needs to know what to do next"

    @pytest.mark.asyncio
    async def test_opening_the_page_gives_up_inside_its_budget(self, monkeypatch):
        # Every step already had a timeout when fcfd3c hung. So the guard that
        # matters is the one that does not depend on having guessed which step
        # hangs -- a wall clock around the whole opening sequence.
        import dataclasses

        from computer_use_agent import pipeline as pipeline_mod
        from computer_use_agent.config import config
        from computer_use_agent.intent import AuditRequest
        from computer_use_agent.pipeline import AuditPipeline

        monkeypatch.setattr(
            pipeline_mod, "config",
            dataclasses.replace(config, navigation_budget_seconds=0.05),
        )

        class _Playwright:
            class chromium:
                @staticmethod
                async def launch(**_kwargs):
                    await asyncio.Event().wait()  # hangs where fcfd3c hung

            async def stop(self):
                return None

        class _Starter:
            async def start(self):
                return _Playwright()

        monkeypatch.setattr(pipeline_mod, "async_playwright", lambda: _Starter())

        pipeline = AuditPipeline.__new__(AuditPipeline)
        pipeline._emit = lambda *_a, **_k: None
        pipeline.on_preview_frame = None
        pipeline.gate = None
        pipeline.has_viewers = None

        request = AuditRequest(target="https://example.invalid/watch", start_seconds=0)
        result = await asyncio.wait_for(pipeline.preflight(request), timeout=5)
        assert result.ok is False
        assert "秒内没能打开" in result.problem

    @pytest.mark.asyncio
    async def test_the_audit_path_is_guarded_too_not_just_preflight(self):
        # `run()` opens the page the same way, and `start_audit` runs it as a
        # detached background task. A hang there is a job that reads `running`
        # forever with no request left for anyone to notice dying.
        import inspect

        from computer_use_agent.pipeline import AuditPipeline

        source = inspect.getsource(AuditPipeline._browser_session)
        assert "navigation_budget_seconds" in source, (
            "the guard has to sit in the shared opening path, not in preflight")


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


class TestZeroViolationsHasTwoOppositeCauses:
    """"Nobody misbehaved" and "we could not see" must not print the same tick.

    Job `ad345a`, 2026-09-09: the model called the footage
    「无法判读的无效监控录像」, answered CANNOT_DETERMINE on all five rules and set
    `visibility_ok=false`. The report said ✅ 未发现违反 SOP 的行为, because the
    summary only counted rows whose status was VIOLATION.

    Plan C is what turned this from a rounding error into a demo risk: the whole
    video is one window, so one unreadable window is the entire audit.
    """

    @staticmethod
    def _outcome(index, statuses, *, visible=True):
        from computer_use_agent.analyzer.video_analyzer import AnalysisOutcome

        result = WindowResult.model_validate({
            "scene_summary": "s",
            "visibility_ok": visible,
            "findings": [
                {"rule_id": f"CHK_{n}", "status": s, "confidence": 0.9,
                 "severity": "RED_LINE" if s == "VIOLATION" else "NONE"}
                for n, s in enumerate(statuses)
            ],
        })
        return AnalysisOutcome(clip=make_clip(index=index), result=result)

    async def _summary(self, tmp_path, statuses, *, visible=True):
        from computer_use_agent.store import AuditStore

        store = AuditStore(records_path=tmp_path / "r.jsonl", evidence_dir=tmp_path / "ev")
        await store.record(self._outcome(0, statuses, visible=visible))
        return store.stats.as_dict(), store

    @pytest.mark.asyncio
    async def test_an_unreadable_window_is_counted_not_swallowed(self, tmp_path):
        summary, _ = await self._summary(
            tmp_path, ["CANNOT_DETERMINE"] * 5, visible=False)

        assert summary["checks_total"] == 5
        assert summary["checks_undetermined"] == 5
        assert summary["windows_unreadable"] == 1
        assert summary["violations"] == 0

    @pytest.mark.asyncio
    async def test_a_clean_window_reads_as_fully_judged(self, tmp_path):
        summary, _ = await self._summary(tmp_path, ["COMPLIANT"] * 3)

        assert summary["checks_undetermined"] == 0
        assert summary["windows_unreadable"] == 0

    @pytest.mark.asyncio
    async def test_the_report_refuses_to_pass_footage_it_could_not_read(self, tmp_path):
        from computer_use_agent.agent import CctvAuditAgent

        summary, store = await self._summary(
            tmp_path, ["CANNOT_DETERMINE"] * 5, visible=False)
        summary.update({
            "complete": True, "capture_mode": "file", "elapsed_seconds": 22.5,
            "stopped_because": "采集结束", "records_path": "x", "evidence_dir": "y",
            "covered_from_seconds": 0.0, "covered_to_seconds": 1200.0,
        })
        report = CctvAuditAgent._report(summary, store)

        assert "✅" not in report, "a tick on footage nobody could read is the bug"
        assert "没有得出结论" in report
        assert "全部无法判定" in report

    @pytest.mark.asyncio
    async def test_a_partly_readable_run_says_how_much_was_missed(self, tmp_path):
        from computer_use_agent.agent import CctvAuditAgent

        summary, store = await self._summary(
            tmp_path, ["COMPLIANT", "COMPLIANT", "CANNOT_DETERMINE"])
        summary.update({
            "complete": True, "capture_mode": "stream", "elapsed_seconds": 30.0,
            "stopped_because": "采集结束", "records_path": "x", "evidence_dir": "y",
            "covered_from_seconds": 0.0, "covered_to_seconds": 60.0,
        })
        report = CctvAuditAgent._report(summary, store)

        # Not the blanket refusal -- two rules really were judged.
        assert "没有得出结论" not in report
        assert "1 项没能判读" in report
        assert "3 项检查中 1 项无法判定" in report

    @pytest.mark.asyncio
    async def test_a_genuinely_clean_run_still_gets_its_tick(self, tmp_path):
        from computer_use_agent.agent import CctvAuditAgent

        summary, store = await self._summary(tmp_path, ["COMPLIANT"] * 4)
        summary.update({
            "complete": True, "capture_mode": "stream", "elapsed_seconds": 30.0,
            "stopped_because": "采集结束", "records_path": "x", "evidence_dir": "y",
            "covered_from_seconds": 0.0, "covered_to_seconds": 60.0,
        })
        report = CctvAuditAgent._report(summary, store)

        # The point is not to make every report hedge. A run that read
        # everything and found nothing is a pass, and must still read as one.
        assert "✅ 未发现违反 SOP 的行为（全部规则均已判读）" in report
        assert "4 项检查全部判读完成" in report


class TestCoverageHonesty:
    """A run that delivered 48s of a requested 10 minutes must say so."""

    class _Stub:
        """Just the attributes `_coverage` reads."""

        def __init__(self, span, stop_kind=None, footage_ends_at=None, source=None):
            self._stop_kind = stop_kind
            self._footage_ends_at = footage_ends_at
            self.source = source
            self.store = type("S", (), {"covered_span": lambda _self: span})()

    def _coverage(
        self, span, *, start=120.0, duration=600.0, stop_kind=None, ends_at=None, source=None
    ):
        from computer_use_agent.pipeline import AuditPipeline, AuditRequest

        request = AuditRequest(target="x", start_seconds=start, duration_seconds=duration)
        return AuditPipeline._coverage(
            self._Stub(span, stop_kind, ends_at, source), request)

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
        _note_picture_frozen = _P._note_picture_frozen
        del _P

        def __init__(self):
            self._stop = asyncio.Event()
            self._stop_reason = None
            self._stop_kind = None
            self._footage_ends_at = None
            self._player_rect = None
            self._geometry_warned = False
            self._picture_frozen = False
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
            self.cleanings = getattr(self, "cleanings", 0) + 1

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

    async def _run(self, navigator, monkeypatch, *, page_is_source=True):
        import types as _types

        from computer_use_agent import pipeline as pipeline_module

        # Config is frozen, and the poll interval is the only thing standing
        # between this test and half a minute of real sleeping.
        monkeypatch.setattr(pipeline_module, "config", _types.SimpleNamespace(
            page_watch_seconds=0.001, stop_on_video_end=True,
        ))
        runner = self._Runner()
        navigator.runner = runner
        # Cancelled from outside, because that is now the only thing that ends
        # it: the loop deliberately outlives `_stop` so the page stays minded
        # while the analysis queue drains. `_stop` still marks "capture is
        # over", so waiting on it puts the assertions at the same point in the
        # run they were checked at before.
        task = asyncio.create_task(
            runner._watch_page(object(), navigator, page_is_source=page_is_source)
        )
        await asyncio.wait_for(runner._stop.wait(), timeout=10)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
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


class TestPlanAPageWatchdog:
    """Under Plan A the page is the dashboard picture, not the footage.

    Measured on bilibili (job 692b53): the login nag appeared ~70s in and
    paused the player. Chromium only emits a screencast frame on repaint, so
    the cast went to 0.0 fps while the pump kept forwarding the same 84.3 KB
    still at 11.9 fps -- an operator watching a frozen screen with every number
    on the page reporting success. ffmpeg was pulling the media directly the
    whole time and the audit finished with 25 windows and 9 violations.

    So the watchdog has to run on this plan too, and it has to do exactly two
    things differently: never end the run from what the page says, and never
    stop trying to get the picture back.
    """

    _Runner = TestStallRecovery._Runner
    _Navigator = TestStallRecovery._Navigator
    _run = TestStallRecovery._run

    @pytest.mark.asyncio
    async def test_a_paused_page_never_ends_a_plan_a_audit(self, monkeypatch):
        # The exact shape of the bilibili run: stuck at 01:19 of 05:00, and
        # nothing we do to the player brings it back.
        navigator = self._Navigator(79.0, 300.0, recoverable=False, budget=60)
        runner = await self._run(navigator, monkeypatch, page_is_source=False)

        assert runner._stop_kind is None, "a dialog box must not stop the audit"
        assert runner._stop_reason is None
        assert runner._footage_ends_at is None
        assert "playback_stalled" not in runner.kinds()

    @pytest.mark.asyncio
    async def test_it_keeps_trying_past_plan_bs_three_strikes(self, monkeypatch):
        navigator = self._Navigator(79.0, 300.0, recoverable=False, budget=60)
        await self._run(navigator, monkeypatch, page_is_source=False)

        # Plan B stops at three because three failures mean the footage is
        # over. Plan A has no such verdict to reach, and bilibili re-raises the
        # nag every couple of minutes -- giving up would forfeit the rest of
        # the run's picture.
        assert navigator.plays > 3
        # And the overlay sweep runs on every single poll, not just the ones
        # that nudge -- that sweep is what closes the dialog in the first place.
        assert navigator.cleanings >= 60

    @pytest.mark.asyncio
    async def test_the_frozen_picture_is_announced_exactly_once(self, monkeypatch):
        navigator = self._Navigator(79.0, 300.0, recoverable=False, budget=60)
        runner = await self._run(navigator, monkeypatch, page_is_source=False)

        frozen = [p for e, p in runner.events if e == "picture_frozen"]
        # Said once, not on every poll: the watchdog runs every few seconds and
        # a banner repeating itself thirty times is a banner nobody reads.
        assert len(frozen) == 1
        assert "01:19" in frozen[0]["reason"]
        assert "稽核不受影响" in frozen[0]["reason"]

    @pytest.mark.asyncio
    async def test_the_notice_is_rearmed_when_the_picture_comes_back(self, monkeypatch):
        navigator = self._Navigator(79.0, 300.0, recoverable=True, budget=60)
        runner = await self._run(navigator, monkeypatch, page_is_source=False)

        # Recovery happens at the tenth stalled poll, long before the thirtieth
        # that would trigger the notice -- so nothing is said at all, and the
        # latch is left ready for the next nag.
        assert "picture_frozen" not in runner.kinds()
        assert runner._picture_frozen is False
        assert navigator.current > 79.0

    @pytest.mark.asyncio
    async def test_a_finished_page_is_a_still_frame_not_a_finished_audit(self, monkeypatch):
        # Plan A's page can run out well before the audit does -- and nudging
        # an ended <video> restarts it from zero, which on the dashboard reads
        # as the audit jumping back to the beginning.
        navigator = self._Navigator(299.5, 300.0, recoverable=False, budget=5)
        runner = await self._run(navigator, monkeypatch, page_is_source=False)

        assert runner._stop_kind is None
        assert "video_ended" not in runner.kinds()
        assert navigator.plays == 0
        assert [p["reason"] for e, p in runner.events if e == "picture_frozen"] == [
            "网页里的视频已播完，画面停在最后一帧（稽核不受影响，仍在继续）"
        ]


class TestWatchdogOutlivesCapture:
    """Capture stopping is not the page stopping mattering.

    Job a8c236, on bilibili: ffmpeg reached the requested end at 15:17:47 and
    the watchdog was cancelled with it, but twelve windows were still in the
    analysis queue and the preview feeds the dashboard until the browser
    closes. The login nag came up two seconds later, nobody swept it, and the
    operator watched a frozen still for the remaining 56 seconds while the
    numbers ticked up. The audit was perfect and the demo was not.
    """

    _Runner = TestStallRecovery._Runner
    _Navigator = TestStallRecovery._Navigator

    class _StoppedNavigator(TestStallRecovery._Navigator):
        """Sets `_stop` early, the way `[budget_stop]` does, then keeps going.

        `ends_after_stop` moves the page to its final frame only once capture
        is over, which is the ordering that matters: a page that had already
        ended before then is a genuine end-of-footage and should be treated as
        one.
        """

        ends_after_stop = False

        async def read_playback_state(self, page):
            self._left -= 1
            if self._left == 55:  # capture reaches the requested end
                self.runner._stop.set()
                if self.ends_after_stop:
                    self.current = self.duration - 0.5
            return {"ended": False, "current_time": self.current,
                    "duration": self.duration}

    async def _drain(self, navigator, monkeypatch, *, page_is_source, polls=60):
        import types as _types

        from computer_use_agent import pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "config", _types.SimpleNamespace(
            page_watch_seconds=0.001, stop_on_video_end=True,
        ))
        runner = self._Runner()
        navigator.runner = runner
        task = asyncio.create_task(
            runner._watch_page(object(), navigator, page_is_source=page_is_source)
        )
        for _ in range(2000):  # bounded spin, not a sleep: the poll is 1ms
            if navigator._left <= navigator._budget - polls or task.done():
                break
            await asyncio.sleep(0.001)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return runner

    @pytest.mark.asyncio
    async def test_it_keeps_minding_the_page_after_capture_ends(self, monkeypatch):
        navigator = self._StoppedNavigator(79.0, 300.0, recoverable=False, budget=60)
        navigator._budget = 60
        runner = await self._drain(navigator, monkeypatch, page_is_source=False)

        # The old loop was `while not self._stop.is_set()`, so it exited on the
        # first poll after capture stopped. Both of these were 0 for the whole
        # drain, which is exactly how long the dashboard stayed frozen.
        assert runner._stop.is_set()
        assert navigator.cleanings > 20, "overlays must still be swept"
        assert navigator.plays > 3, "and the player still nudged"

    @pytest.mark.asyncio
    async def test_it_can_no_longer_end_the_run_once_capture_is_over(self, monkeypatch):
        # Plan B, page at its end, during the drain. Left ungated, this sets
        # `_footage_ends_at`, and every clip already sitting in the queue whose
        # offset is past it gets dropped unanalysed -- windows the customer
        # asked for and paid to capture, reported as "footage ended".
        navigator = self._StoppedNavigator(150.0, 300.0, recoverable=False, budget=60)
        navigator._budget = 60
        navigator.ends_after_stop = True
        runner = await self._drain(navigator, monkeypatch, page_is_source=True)

        assert runner._footage_ends_at is None
        assert "video_ended" not in runner.kinds()
        assert runner._stop_kind is None  # the budget stopped it, not the page


class TestRepaintDetectorReachesTheDashboard:
    """The pixel detector knew, and only told the container log.

    On a8c236 the playhead detector never fired -- the watchdog was already
    gone -- and the only thing that noticed was the preview pump counting zero
    screencast frames. It wrote a warning to stdout. The person looking at the
    frozen picture was in a browser.
    """

    def _runner(self):
        from computer_use_agent.pipeline import AuditPipeline

        runner = TestStallRecovery._Runner()
        runner._repaint_stalled = False
        runner._note_repaint = AuditPipeline._note_repaint.__get__(runner)
        return runner

    def test_a_page_that_stopped_repainting_is_announced(self):
        runner = self._runner()
        runner._note_repaint(False, 15.0)

        frozen = [p for e, p in runner.events if e == "picture_frozen"]
        assert len(frozen) == 1
        assert "15 秒没有重绘" in frozen[0]["reason"]
        assert "稽核不受影响" in frozen[0]["reason"]

    def test_it_says_it_once_however_long_the_freeze_lasts(self):
        runner = self._runner()
        for _ in range(4):  # a8c236 reported four windows in a row
            runner._note_repaint(False, 15.0)

        assert runner.kinds().count("picture_frozen") == 1

    def test_the_notice_re_arms_when_frames_come_back(self):
        runner = self._runner()
        runner._note_repaint(False, 15.0)
        runner._note_repaint(True, 15.0)
        runner._note_repaint(False, 15.0)

        assert runner.kinds().count("picture_frozen") == 2

    def test_it_does_not_clear_a_warning_it_did_not_raise(self):
        # The watchdog latches the same flag for something this detector cannot
        # see: a video that ended while the page still animates. Frames are
        # arriving, so `alive` is True -- and clearing on that would silently
        # cancel the other detector's warning.
        runner = self._runner()
        runner._note_picture_frozen("网页里的视频已播完，画面停在最后一帧")
        runner._note_repaint(True, 15.0)

        assert runner._picture_frozen is True


class TestCpuProbe:
    """Attribution, not arithmetic: the line has to name the culprit."""

    def test_the_ceiling_is_the_cgroup_allowance_not_the_host(self, tmp_path, monkeypatch):
        from computer_use_agent import cpuprobe

        # A four-core allowance carved out of a sixteen-core host. Reading the
        # host would make every later percentage look four times healthier.
        quota = tmp_path / "cpu.max"
        quota.write_text("400000 100000\n")
        monkeypatch.setattr(cpuprobe, "Path", lambda p: quota if "cpu.max" in p else Path(p))
        monkeypatch.setattr(cpuprobe.os, "cpu_count", lambda: 16)

        assert cpuprobe.cpu_quota() == 4.0

    def test_an_unlimited_cgroup_falls_back_to_the_core_count(self):
        from computer_use_agent.cpuprobe import _parse_cpu_max

        assert _parse_cpu_max("max 100000") is None
        assert _parse_cpu_max("nonsense") is None

    def test_chromiums_many_processes_are_counted_as_one_name(self, monkeypatch):
        from computer_use_agent import cpuprobe

        # A dozen renderers at 25% each outweigh one ffmpeg at 90%, and
        # reporting the busiest single pid would say the opposite.
        before = {i: ("chrome", 0.0) for i in range(1, 13)}
        before[99] = ("ffmpeg", 0.0)
        after = {i: ("chrome", 2.5) for i in range(1, 13)}
        after[99] = ("ffmpeg", 9.0)

        probe = _probe_over(cpuprobe, monkeypatch, before, after)
        line = probe.report(10.0)

        assert "chrome 300%" in line
        assert line.index("chrome") < line.index("ffmpeg")

    def test_a_process_that_started_mid_window_counts_all_of_its_time(self, monkeypatch):
        from computer_use_agent import cpuprobe

        # Clip ffmpegs are born and die inside a fifteen-second window. Charging
        # them only the part after a snapshot they never appeared in would hide
        # exactly the spikes worth finding.
        probe = _probe_over(cpuprobe, monkeypatch, {}, {7: ("ffmpeg", 4.0)})

        assert "ffmpeg 40%" in probe.report(10.0)

    def test_the_line_says_how_much_of_the_allowance_is_gone(self, monkeypatch):
        from computer_use_agent import cpuprobe

        probe = _probe_over(cpuprobe, monkeypatch, {}, {7: ("python", 30.0)})
        probe.quota = 4.0

        line = probe.report(10.0)
        assert "3.0 of 4 cores" in line
        assert "75% of the allowance" in line


class TestSchedulingProbe:
    """Telling 'nobody scheduled us' apart from 'we blocked our own loop'."""

    def test_a_missing_schedstat_is_said_so_not_reported_as_zero(self, monkeypatch):
        from computer_use_agent import cpuprobe

        # A fabricated 0ms reads as "definitely not throttled", which is the
        # one conclusion this probe exists to stop being drawn by accident.
        monkeypatch.setattr(cpuprobe, "runqueue_wait_seconds", lambda: None)
        probe = _probe_over(cpuprobe, monkeypatch, {}, {})

        line = probe.scheduling_report(10.0)
        assert "runqueue wait unavailable" in line
        assert "0ms (0.0%" not in line
        probe.stop()

    def test_time_spent_waiting_for_a_cpu_is_reported_as_a_share(self, monkeypatch):
        from computer_use_agent import cpuprobe

        waits = iter([1.0, 4.0])
        monkeypatch.setattr(cpuprobe, "runqueue_wait_seconds", lambda: next(waits))
        probe = _probe_over(cpuprobe, monkeypatch, {}, {})

        line = probe.scheduling_report(10.0)
        assert "runqueue wait 3000ms (30.0% of the window)" in line
        probe.stop()

    def test_the_thread_tick_keeps_time_when_nothing_is_in_the_way(self):
        from computer_use_agent.cpuprobe import ThreadTicker

        ticker = ThreadTicker(0.01)
        ticker.start()
        time.sleep(0.25)
        ticker.stop()
        ticks = ticker.drain()

        assert len(ticks) > 10
        # Generous: this asserts the thread is running and timing, not that a
        # shared CI box can hit 10ms.
        assert max(ticks) < 200

    def test_draining_the_ticker_twice_does_not_double_count(self):
        from computer_use_agent.cpuprobe import ThreadTicker

        ticker = ThreadTicker(0.01)
        ticker.start()
        time.sleep(0.1)
        first = ticker.drain()
        ticker.stop()

        assert first
        assert ticker.drain() == [] or len(ticker.drain()) < len(first)


def _probe_over(module, monkeypatch, before, after):
    """A CpuProbe whose two snapshots are the two dicts given."""
    monkeypatch.setattr(module, "_snapshot", lambda: before)
    probe = module.CpuProbe()
    monkeypatch.setattr(module, "_snapshot", lambda: after)
    return probe


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

        # q50 at 640x360: the dashboard is for watching progress, and every
        # viewer is a separate metered stream once it lives on Cloud Run. The
        # evidence frames are a different feed and stay full quality.
        assert page.cdp.started["quality"] == 50
        assert (page.cdp.started["maxWidth"], page.cdp.started["maxHeight"]) == (640, 360)
        await page.cdp.emit("frame-1")
        await page.cdp.emit("frame-2")   # arrives before the pump next ticks
        await asyncio.sleep(0.1)
        await preview.aclose()

        # Whatever it forwarded, it forwarded the newest frame and never
        # queued the stale one behind it.
        assert sent and set(sent) == {"frame-2"}
        assert page.cdp.stopped

    def test_a_cast_that_stopped_producing_is_reported(self, caplog):
        # Chromium emits on repaint, so a paused page produces nothing while
        # the pump keeps forwarding the cached frame at the full rate. Every
        # counter the dashboard has looks healthy; this is the one number that
        # does not (job 692b53: `screencast in` 0.0 fps for two minutes).
        import logging

        from computer_use_agent.capture.preview import LivePreview

        live = LivePreview(page=None, on_frame=lambda _: None)
        live._forwarding = True
        live._report_at = 1.0
        live._ticks = 180
        live._tick_ms = [83.0] * 180
        live._cast_frames = 0

        with caplog.at_level(logging.WARNING, logger="cctv_audit.preview"):
            live._pump_report(16.0)

        assert "no screencast frame" in caplog.text
        assert "still" in caplog.text

    def test_a_cast_still_producing_says_nothing(self, caplog):
        import logging

        from computer_use_agent.capture.preview import LivePreview

        live = LivePreview(page=None, on_frame=lambda _: None)
        live._forwarding = True
        live._report_at = 1.0
        live._ticks = 180
        live._tick_ms = [83.0] * 180
        live._cast_frames = 190

        with caplog.at_level(logging.WARNING, logger="cctv_audit.preview"):
            live._pump_report(16.0)

        # A warning that fires when nothing is wrong is a warning nobody reads.
        assert "no screencast frame" not in caplog.text

    @pytest.mark.asyncio
    async def test_closing_a_preview_that_never_started_is_safe(self):
        # The pipeline's shutdown path runs whether or not navigation got
        # far enough to open anything.
        from computer_use_agent.capture.preview import LivePreview

        await LivePreview(page=None, on_frame=lambda _: None).aclose()

    # -- only stream while somebody is looking ---------------------------
    #
    # Once the dashboard is its own Cloud Run service these frames are metered
    # egress -- ~1.6 Mbps per viewer, and every viewer is a separate stream.
    # Most audits run with nobody watching, so the cheapest correct answer is
    # to send nothing and to draw nothing.

    def _watched_preview(self, monkeypatch, probe):
        """A preview whose viewer probe is `probe`. Returns (preview, page, sent).

        The probe is driven by the test rather than by a timer: asserting on
        "how many polls have happened by now" makes the test measure the sleep
        schedule instead of the behaviour.
        """
        from computer_use_agent.capture import preview as preview_mod

        # 5s between checks is right in production and unusable in a test.
        monkeypatch.setattr(preview_mod, "_VIEWER_POLL_SECONDS", 0.0)

        sent: list[str] = []
        page = self._FakePage()
        live = preview_mod.LivePreview(
            page=page, on_frame=sent.append, fps=100, has_viewers=probe,
        )
        return live, page, sent

    @pytest.mark.asyncio
    async def test_nobody_watching_means_no_frame_is_ever_sent(self, monkeypatch):
        async def nobody():
            return False

        live, page, sent = self._watched_preview(monkeypatch, nobody)
        await live.start()
        await page.cdp.emit("frame-1")
        await asyncio.sleep(0.1)
        await live.aclose()

        # Nothing leaves the container. The screencast itself does run -- see
        # the next test for why it has to.
        assert sent == []

    @pytest.mark.asyncio
    async def test_the_screencast_starts_immediately_even_with_no_audience(self, monkeypatch):
        # The invariant that keeps Plan B alive. Two screencasts on one page
        # only behave if the low-resolution one starts first, and the recorder
        # is constructed after navigation -- so this cast cannot wait for a
        # viewer to turn up. Deferring it is exactly the bug that starved
        # ffmpeg of frames and produced an audit with zero windows.
        async def nobody():
            return False

        live, page, sent = self._watched_preview(monkeypatch, nobody)
        await live.start()
        assert page.cdp.started is not None, "cast must be live before any recorder exists"
        await asyncio.sleep(0.05)
        assert not page.cdp.stopped, "must not be stopped while the session runs"
        await live.aclose()

    @pytest.mark.asyncio
    async def test_the_picture_appears_when_someone_opens_the_page(self, monkeypatch):
        watching = {"now": False}

        async def probe():
            return watching["now"]

        live, page, sent = self._watched_preview(monkeypatch, probe)
        await live.start()
        await page.cdp.emit("frame-1")
        await asyncio.sleep(0.05)
        assert sent == [], "still nobody there"

        watching["now"] = True
        await asyncio.sleep(0.05)
        await live.aclose()
        # Repeats are expected: the pump ticks at `fps` and forwards whatever
        # the latest frame is, so a static page resends the same one.
        assert set(sent) == {"frame-1"}

    @pytest.mark.asyncio
    async def test_the_last_viewer_leaving_stops_the_frames_not_the_cast(self, monkeypatch):
        watching = {"now": True}

        async def probe():
            return watching["now"]

        live, page, sent = self._watched_preview(monkeypatch, probe)
        await live.start()
        await asyncio.sleep(0.05)
        await page.cdp.emit("frame-1")
        await asyncio.sleep(0.05)
        assert sent, "someone was watching, so frames should have flowed"

        watching["now"] = False
        await asyncio.sleep(0.05)
        # Egress stops; the cast does not. Calling stopScreencast here would
        # also stop the recorder's, which shares the page.
        assert not page.cdp.stopped

        before = len(sent)
        await page.cdp.emit("frame-2")
        await asyncio.sleep(0.05)
        await live.aclose()

        assert len(sent) == before, "frames must stop when the audience does"
        assert "frame-2" not in sent

    @pytest.mark.asyncio
    async def test_a_dashboard_that_cannot_be_reached_counts_as_nobody(self, monkeypatch):
        # Erring the other way would stream a live CCTV feed at a service that
        # is not answering, for as long as the audit runs.
        async def unreachable():
            raise RuntimeError("dashboard down")

        live, page, sent = self._watched_preview(monkeypatch, unreachable)
        await live.start()
        await page.cdp.emit("frame-1")
        await asyncio.sleep(0.1)
        await live.aclose()

        assert sent == []

    @pytest.mark.asyncio
    async def test_a_slow_viewer_check_does_not_slow_the_frames(self, monkeypatch):
        """The cloud bug, in miniature.

        The check used to be awaited inside the pump loop, so the frame rate
        became a function of how long the dashboard took to answer. Measured on
        job c70488: ~1s per answer against a 2s poll, and the pump delivered
        0.6-3.2 fps against a configured 12 -- with nothing dropped by the
        in-flight cap and 140ms posts. It was only ever waiting.
        """
        from computer_use_agent.capture import preview as preview_mod

        monkeypatch.setattr(preview_mod, "_VIEWER_POLL_SECONDS", 0.0)

        async def slow():
            # Longer than the whole measurement window below. If this is on the
            # pump's loop, the pump gets exactly one tick.
            await asyncio.sleep(0.5)
            return True

        sent: list[str] = []
        page = self._FakePage()
        live = preview_mod.LivePreview(
            page=page, on_frame=sent.append, fps=100, has_viewers=slow,
        )
        await live.start()
        live._forwarding = True     # as if the first check had already said yes
        await page.cdp.emit("frame-1")
        await asyncio.sleep(0.2)
        await live.aclose()

        # 100 fps for 0.2s is ~20 ticks. Ten is a wide margin that still fails
        # loudly if the network call is ever put back on the loop.
        assert len(sent) > 10, f"the pump stalled behind the viewer check ({len(sent)} frames)"

    @pytest.mark.asyncio
    async def test_one_missed_viewer_check_does_not_blank_the_screen(self, monkeypatch):
        # The dashboard's client gives up after 1.5s, and a check that gives up
        # is indistinguishable from "nobody is there". In the cloud that
        # happened constantly: the container logged paused/resumed about twice
        # a second with a viewer connected the whole time.
        from computer_use_agent.capture import preview as preview_mod

        monkeypatch.setattr(preview_mod, "_VIEWER_POLL_SECONDS", 0.0)
        answers = [True, True, RuntimeError("timed out"), True, True]

        async def flaky():
            answer = answers.pop(0) if answers else True
            if isinstance(answer, Exception):
                raise answer
            return answer

        sent: list[str] = []
        page = self._FakePage()
        live = preview_mod.LivePreview(
            page=page, on_frame=sent.append, fps=100, has_viewers=flaky,
        )
        await live.start()
        await page.cdp.emit("frame-1")
        while answers:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.02)
        await live.aclose()

        assert live._forwarding, "one blip must not pause the preview"

    @pytest.mark.asyncio
    async def test_two_misses_in_a_row_still_stop_the_stream(self, monkeypatch):
        # The other half: a dashboard that is genuinely gone must not keep
        # being sent a live CCTV feed for the rest of the audit.
        from computer_use_agent.capture import preview as preview_mod

        monkeypatch.setattr(preview_mod, "_VIEWER_POLL_SECONDS", 0.0)
        state = {"up": True}

        async def probe():
            if not state["up"]:
                raise RuntimeError("dashboard down")
            return True

        sent: list[str] = []
        page = self._FakePage()
        live = preview_mod.LivePreview(
            page=page, on_frame=sent.append, fps=100, has_viewers=probe,
        )
        await live.start()
        await asyncio.sleep(0.02)
        assert live._forwarding
        state["up"] = False
        await asyncio.sleep(0.05)
        await live.aclose()

        assert not live._forwarding

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
        from computer_use_agent.monitor import (
            _MAX_FRAMES_IN_FLIGHT, BrowserMonitorClient)

        client = BrowserMonitorClient()
        sent = []
        client._fire_and_forget = lambda payload, on_done=None: sent.append(on_done)

        # Fill every slot, then push one more before any POST has completed.
        for i in range(_MAX_FRAMES_IN_FLIGHT):
            client.update_frame_b64(f"f{i}")
        assert len(sent) == _MAX_FRAMES_IN_FLIGHT and client._frames_dropped == 0

        client.update_frame_b64("overflow")
        assert len(sent) == _MAX_FRAMES_IN_FLIGHT and client._frames_dropped == 1

        sent[0]()  # one POST finally lands, freeing exactly one slot
        client.update_frame_b64("next")
        assert len(sent) == _MAX_FRAMES_IN_FLIGHT + 1 and client._frames_dropped == 1

    def test_the_cap_is_more_than_one_so_latency_is_not_the_frame_rate(self):
        # A cap of one makes the round trip to the dashboard the frame rate: a
        # measured cloud run delivered 0.5 fps while PREVIEW_FPS said 12, and
        # nothing reported it because dropping frames is by design. Pinning
        # this so the constant cannot quietly go back to 1.
        from computer_use_agent.monitor import _MAX_FRAMES_IN_FLIGHT

        assert _MAX_FRAMES_IN_FLIGHT > 1

    def test_a_failed_send_does_not_wedge_the_channel_shut(self):
        # The in-flight count is only safe if it is always released. A send that
        # cannot even be scheduled must not stop the preview forever.
        from computer_use_agent.monitor import BrowserMonitorClient

        client = BrowserMonitorClient()
        client.update_frame_b64("a")  # no running loop -> _fire_and_forget bails
        assert client._frames_in_flight == 0
        client.update_frame_b64("b")
        assert client._frames_dropped == 0

    def test_releasing_more_than_was_sent_cannot_uncap_the_sender(self):
        # `_fire_and_forget` releases synchronously when there is no loop, so a
        # stray extra release is reachable. A counter allowed below zero would
        # turn the cap into an unbounded queue -- the exact pile-up it exists
        # to prevent.
        from computer_use_agent.monitor import (
            _MAX_FRAMES_IN_FLIGHT, BrowserMonitorClient)

        client = BrowserMonitorClient()
        for _ in range(5):
            client._frame_sent()
        assert client._frames_in_flight == 0

        sent = []
        client._fire_and_forget = lambda payload, on_done=None: sent.append(on_done)
        for i in range(_MAX_FRAMES_IN_FLIGHT + 3):
            client.update_frame_b64(f"f{i}")
        assert len(sent) == _MAX_FRAMES_IN_FLIGHT

    def test_what_happened_to_the_preview_is_said_out_loud(self, caplog):
        # The dashboard has been called a slideshow twice, and both times the
        # investigation stalled here: dropping frames is the designed behaviour
        # of the cap, so it was silent, and the counter it bumps was only ever
        # read in tests. Without this line the next investigation guesses too.
        import logging

        from computer_use_agent.monitor import (
            _MAX_FRAMES_IN_FLIGHT, BrowserMonitorClient)

        client = BrowserMonitorClient(job_id="d7680d")
        sent = []
        client._fire_and_forget = lambda payload, on_done=None: sent.append(on_done)

        # The whole exercise is inside the capture, not just the last call: a
        # dropped frame reports too, so the line can land during the loop and
        # then not again. Capturing only the tail made this test pass or fail
        # on whether some earlier test had already raised the root log level.
        with caplog.at_level(logging.INFO, logger="cctv_audit.monitor"):
            # Backdated so the report is due on the next frame, not in 15s.
            client._report_at = time.monotonic() - 20.0
            for i in range(_MAX_FRAMES_IN_FLIGHT + 5):
                client.update_frame_b64("x" * 1024)
            sent[0]()

        line = caplog.text
        assert "d7680d" in line          # which audit, when two share a container
        assert "dropped" in line and "POST median" in line
        assert "KB/frame" in line

    def test_the_report_does_not_run_on_every_frame(self, caplog):
        # At 12 fps a per-frame line is 43,000 entries an hour, which is both a
        # bill and a reason nobody reads the log.
        import logging

        from computer_use_agent.monitor import BrowserMonitorClient

        client = BrowserMonitorClient()
        client._fire_and_forget = lambda payload, on_done=None: None
        with caplog.at_level(logging.INFO, logger="cctv_audit.monitor"):
            for _ in range(50):
                client.update_frame_b64("x")

        assert "preview" not in caplog.text


class TestWhichProjectWeAreIn:
    """Agent Runtime sets GOOGLE_CLOUD_PROJECT to the project *number* and will
    not let the deployment override it. A named Firestore database is not
    addressable by number -- the client gets a 404 saying the database does not
    exist, while it plainly does -- so the deployment passes the id separately
    and that is the one that has to win."""

    @staticmethod
    def _project(monkeypatch, **env):
        from computer_use_agent.config import Config

        for name in ("GCP_PROJECT", "GOOGLE_CLOUD_PROJECT"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return Config().gcp_project

    def test_the_explicit_id_beats_the_number_the_platform_injects(self, monkeypatch):
        assert self._project(
            monkeypatch,
            GCP_PROJECT="study-project-496907",
            GOOGLE_CLOUD_PROJECT="596821501265",
        ) == "study-project-496907"

    def test_nothing_changes_for_anyone_who_only_sets_the_usual_one(self, monkeypatch):
        assert self._project(
            monkeypatch, GOOGLE_CLOUD_PROJECT="study-project-496907",
        ) == "study-project-496907"

    def test_neither_set_is_empty_rather_than_a_guess(self, monkeypatch):
        # config.validate() turns this into a readable complaint. Guessing a
        # project here would write a customer's audit into someone else's.
        assert self._project(monkeypatch) == ""


class TestNotBillingSomeoneElsesProject:
    """Agent Runtime's credentials arrive with a quota project already on them.
    Anything built from them sends `x-goog-user-project`, GCS reads that as
    "bill this project", and an agent service account without
    `serviceusage.services.use` gets a 403 that names the *storage* object and
    blames the wrong role. Stripping the quota project is what keeps the SOP
    fetch and the evidence upload working."""

    class _Creds:
        def __init__(self, quota):
            self.quota_project_id = quota

        def with_quota_project(self, quota):
            return type(self)(quota)

    def _patch_adc(self, monkeypatch, result):
        import google.auth

        def fake_default(*args, **kwargs):
            if isinstance(result, Exception):
                raise result
            return result, "some-project"

        monkeypatch.setattr(google.auth, "default", fake_default)

    def test_the_quota_project_is_taken_off(self, monkeypatch):
        from computer_use_agent.gcp import credentials_without_quota_project

        self._patch_adc(monkeypatch, self._Creds("596821501265"))
        assert credentials_without_quota_project().quota_project_id is None

    def test_credentials_that_cannot_carry_one_are_passed_through(self, monkeypatch):
        # Some credential types have no `with_quota_project`. They also never
        # send the header, so there is nothing to strip.
        from computer_use_agent.gcp import credentials_without_quota_project

        class Plain:
            pass

        plain = Plain()
        self._patch_adc(monkeypatch, plain)
        assert credentials_without_quota_project() is plain

    def test_no_credentials_at_all_is_none_not_a_crash(self, monkeypatch):
        # Locally, and in this suite, there may be no ADC. Returning None lets
        # the client fall back to its own default and fail with its own message
        # instead of this helper's.
        from computer_use_agent.gcp import credentials_without_quota_project

        self._patch_adc(monkeypatch, RuntimeError("could not automatically determine"))
        assert credentials_without_quota_project() is None


class TestTheSopErrorSaysWhatIsWrong:
    """The refusal a customer reads is the only copy of the error anyone sees --
    it is caught, stored on the job, and never re-raised. A GCS permission
    message names the missing permission around 240 characters in, so clipping
    at 200 threw away the one word that mattered and left a sentence pointing at
    the wrong role."""

    def test_the_missing_permission_survives_the_clip(self, monkeypatch):
        import asyncio
        import types

        from computer_use_agent.analyzer import sop as sop_mod

        real = (
            "403 GET https://storage.googleapis.com/download/storage/v1/b/"
            "study-project-496907-cctv-audit/o/sop%2Fchagee-store-v1.yaml?alt=media: "
            "cctv-audit-agent@study-project-496907.iam.gserviceaccount.com does not "
            "have serviceusage.services.use access to the Google Cloud project. "
            "Permission 'serviceusage.services.use' denied on resource (or it may "
            "not exist)."
        )
        monkeypatch.setattr(sop_mod, "config", types.SimpleNamespace(
            sop_bucket="study-project-496907-cctv-audit", sop_prefix="sop",
            gcp_project="study-project-496907"))

        async def boom(fn, *args, **kwargs):
            raise RuntimeError(real)

        monkeypatch.setattr(asyncio, "to_thread", boom)
        with pytest.raises(sop_mod.SopUnavailable) as caught:
            asyncio.run(sop_mod._fetch_sop("chagee-store-v1"))
        assert "serviceusage.services.use" in str(caught.value)
        assert "gs://study-project-496907-cctv-audit/sop/chagee-store-v1.yaml" in str(caught.value)

    def test_a_runaway_error_is_still_cut(self, monkeypatch):
        import asyncio
        import types

        from computer_use_agent.analyzer import sop as sop_mod

        monkeypatch.setattr(sop_mod, "config", types.SimpleNamespace(
            sop_bucket="b", sop_prefix="", gcp_project="p"))

        async def boom(fn, *args, **kwargs):
            raise RuntimeError("x" * 5000)

        monkeypatch.setattr(asyncio, "to_thread", boom)
        with pytest.raises(sop_mod.SopUnavailable) as caught:
            asyncio.run(sop_mod._fetch_sop("v1"))
        assert len(str(caught.value)) < 800


class TestOnlyNamedOriginsSeeOurIdentity:
    """The demo footage sits on a Cloud Run service that cannot be made public
    -- the org policy refuses `allUsers` -- so the container's browser and its
    ffmpeg both have to present a Google ID token to read it. The token is this
    deployment's own identity, so the rule that matters is not "does it get
    sent" but "does it get sent anywhere else". Every test here is about the
    second half."""

    @staticmethod
    def _origins(monkeypatch, raw):
        from computer_use_agent.config import Config

        monkeypatch.setenv("OIDC_ORIGINS", raw)
        return Config().oidc_origins

    def test_the_operator_can_be_sloppy_about_the_form(self, monkeypatch):
        # Trailing slash, a path, a capital host, and a bare hostname are all
        # the same origin. An allow-list that missed on a slash would fail in
        # the confusing direction: the request goes out bare and 403s.
        assert self._origins(monkeypatch, "https://demo.a.run.app/") == ("https://demo.a.run.app",)
        assert self._origins(monkeypatch, "https://demo.a.run.app/hls.html") == ("https://demo.a.run.app",)
        assert self._origins(monkeypatch, "https://DEMO.A.Run.App") == ("https://demo.a.run.app",)
        assert self._origins(monkeypatch, "demo.a.run.app") == ("https://demo.a.run.app",)

    def test_several_origins_keep_their_order_and_lose_duplicates(self, monkeypatch):
        assert self._origins(
            monkeypatch, "https://a.run.app, https://b.run.app ,https://a.run.app/"
        ) == ("https://a.run.app", "https://b.run.app")

    def test_unset_means_nobody(self, monkeypatch):
        # The default, and the only configuration that has run against
        # bilibili. Empty here is what makes every other site token-free.
        monkeypatch.delenv("OIDC_ORIGINS", raising=False)
        from computer_use_agent.config import Config

        assert Config().oidc_origins == ()
        assert self._origins(monkeypatch, "  ,  ") == ()

    @staticmethod
    def _patch_minting(monkeypatch, origins, *, record=None):
        import types

        from computer_use_agent import gcp as gcp_mod
        from google.oauth2 import id_token as google_id_token

        gcp_mod.reset_id_token_cache()
        monkeypatch.setattr(gcp_mod, "config",
                            types.SimpleNamespace(oidc_origins=tuple(origins)))

        calls = record if record is not None else []

        def fake_fetch(request, audience):
            calls.append(audience)
            return f"token-for-{audience}"

        monkeypatch.setattr(google_id_token, "fetch_id_token", fake_fetch)
        return calls

    def test_a_listed_origin_gets_a_token_audienced_at_itself(self, monkeypatch):
        from computer_use_agent.gcp import id_token_for

        calls = self._patch_minting(monkeypatch, ["https://demo.a.run.app"])
        assert id_token_for("https://demo.a.run.app/hls/index.m3u8") == \
            "token-for-https://demo.a.run.app"
        # The audience is the origin, not the full URL: Cloud Run checks the
        # token against the service, and a per-segment audience would be
        # rejected -- and would mint one token per segment besides.
        assert calls == ["https://demo.a.run.app"]

    def test_everywhere_else_gets_nothing_and_mints_nothing(self, monkeypatch):
        from computer_use_agent.gcp import id_token_for

        calls = self._patch_minting(monkeypatch, ["https://demo.a.run.app"])
        for url in (
            "https://www.bilibili.com/video/BV1",
            "https://demo.a.run.app.evil.example/hls.html",  # suffix, not the origin
            "http://demo.a.run.app/hls.html",                # scheme is part of it
            "https://cdn.example.com/seg0001.ts",
        ):
            assert id_token_for(url) is None, url
        # Not just "returned None" -- no token was ever created. A mint that
        # happened and was then discarded would still be one this deployment
        # could leak somewhere else later.
        assert calls == []

    def test_the_token_is_minted_once_and_reused(self, monkeypatch):
        from computer_use_agent.gcp import id_token_for

        calls = self._patch_minting(monkeypatch, ["https://demo.a.run.app"])
        for _ in range(50):  # roughly one HLS segment each
            id_token_for("https://demo.a.run.app/hls/seg0001.ts")
        assert calls == ["https://demo.a.run.app"]

    def test_a_stale_token_is_replaced(self, monkeypatch):
        from computer_use_agent.gcp import id_token_for

        calls = self._patch_minting(monkeypatch, ["https://demo.a.run.app"])
        url = "https://demo.a.run.app/store.mp4"
        id_token_for(url, now=0.0)
        id_token_for(url, now=2999.0)     # still inside the hour
        assert len(calls) == 1
        id_token_for(url, now=3001.0)     # past it
        assert len(calls) == 2

    def test_ffmpeg_gets_the_header_for_the_media_url_not_the_page(self, monkeypatch):
        # The page and its media are not always on the same host. Putting the
        # token in the shared header block would hand it to whichever CDN the
        # player pulls from, and the audit would succeed either way, so nothing
        # would ever surface it.
        from computer_use_agent.capture.probe import StreamProbe

        self._patch_minting(monkeypatch, ["https://demo.a.run.app"])
        base = {"Referer": "https://demo.a.run.app/hls.html", "Cookie": "a=b"}

        protected = StreamProbe._headers_for("https://demo.a.run.app/hls/index.m3u8", base)
        assert protected["Authorization"] == "Bearer token-for-https://demo.a.run.app"
        assert protected["Cookie"] == "a=b"        # the existing ones survive
        assert "Authorization" not in base         # and the shared block is untouched

        elsewhere = StreamProbe._headers_for("https://cdn.example.com/seg0001.ts", base)
        assert "Authorization" not in elsewhere
        assert elsewhere == base

    def test_the_browser_route_is_scoped_to_the_origin(self, monkeypatch):
        import asyncio
        import types

        from computer_use_agent import pipeline as pipeline_mod

        self._patch_minting(monkeypatch, ["https://demo.a.run.app"])
        monkeypatch.setattr(pipeline_mod, "config", types.SimpleNamespace(
            oidc_origins=("https://demo.a.run.app",)))

        routes = []

        class FakeContext:
            async def route(self, pattern, handler):
                routes.append((pattern, handler))

        asyncio.run(pipeline_mod._attach_oidc_headers(FakeContext()))

        # The pattern is the guard. A `**/*` route with the check inside the
        # handler would work today and would be one early return away from
        # posting our identity to bilibili.
        assert [p for p, _ in routes] == ["https://demo.a.run.app/**"]

        class FakeRoute:
            def __init__(self):
                self.request = types.SimpleNamespace(
                    headers={"referer": "https://demo.a.run.app/hls.html"})
                self.sent = None

            async def continue_(self, headers=None):
                self.sent = headers

        handler = routes[0][1]
        # Exactly one parameter. Playwright reads the arity and calls a
        # two-parameter handler as `(route, request)` -- so writing the token
        # capture as `async def h(route, _hdr=token)` binds `_hdr` to a Request
        # object, and every navigation dies sixty seconds later as a page-load
        # timeout that says nothing about routing. Measured, not theorised.
        import inspect

        assert len(inspect.signature(handler).parameters) == 1

        fake = FakeRoute()
        asyncio.run(handler(fake))
        assert fake.sent["authorization"] == "Bearer token-for-https://demo.a.run.app"
        assert fake.sent["referer"] == "https://demo.a.run.app/hls.html"

    def test_no_origins_means_no_routes_at_all(self, monkeypatch):
        import asyncio
        import types

        from computer_use_agent import pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod, "config",
                            types.SimpleNamespace(oidc_origins=()))

        class FakeContext:
            async def route(self, pattern, handler):
                raise AssertionError(f"registered a route for {pattern}")

        asyncio.run(pipeline_mod._attach_oidc_headers(FakeContext()))


class TestTalkingToARemoteDashboard:
    """Once the dashboard is its own Cloud Run service, "post to the dashboard"
    stops being a loopback write and becomes an authenticated internet call.
    Two things have to be right: the address, and the two separate credentials
    a `--no-allow-unauthenticated` service demands."""

    @staticmethod
    def _client(monkeypatch, *, url="", token="s3cret"):
        import types
        from computer_use_agent import config as config_mod
        from computer_use_agent.monitor import BrowserMonitorClient

        monkeypatch.setattr(config_mod, "config", types.SimpleNamespace(
            monitor_port=8080, monitor_token=token, monitor_url=url,
            screen_width=1920, screen_height=1080,
        ))
        return BrowserMonitorClient()

    class _Response:
        def __init__(self, status, body=None):
            self.status = status
            self._body = body or {}

        async def json(self):
            return self._body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    class _Session:
        """Records what was sent, so the assertions can be about headers."""

        def __init__(self, response):
            self._response = response
            self.calls = []

        def get(self, url, params=None, headers=None):
            self.calls.append((url, headers or {}, params or {}))
            if isinstance(self._response, Exception):
                raise self._response
            return self._response

    def test_no_monitor_url_still_means_the_process_next_door(self, monkeypatch):
        # The local `adk web` path must keep working untouched.
        client = self._client(monkeypatch)
        assert client.base_url == "http://127.0.0.1:8080"

    def test_a_configured_dashboard_is_where_everything_is_sent(self, monkeypatch):
        client = self._client(monkeypatch, url="https://cctv-monitor-x.a.run.app")
        assert client.base_url == "https://cctv-monitor-x.a.run.app"

    @pytest.mark.asyncio
    async def test_loopback_needs_no_google_token_and_asks_for_none(self, monkeypatch):
        # Fetching one locally would fail and log a scary warning about an
        # authentication problem that does not exist.
        client = self._client(monkeypatch)
        asked = []
        monkeypatch.setattr(client, "_identity_token",
                            lambda: asked.append(1) or "")

        headers = await client._headers()
        assert headers == {"X-Monitor-Token": "s3cret"}
        assert asked == [], "no identity token should be fetched for 127.0.0.1"

    @pytest.mark.asyncio
    async def test_cloud_run_gets_both_credentials(self, monkeypatch):
        # X-Monitor-Token is what our app checks; the bearer is what Cloud Run
        # checks before our app is ever reached. Neither substitutes for the other.
        client = self._client(monkeypatch, url="https://dash.run.app")

        async def fake_token():
            return "id-token-abc"

        monkeypatch.setattr(client, "_identity_token", fake_token)
        assert await client._headers() == {
            "X-Monitor-Token": "s3cret",
            "Authorization": "Bearer id-token-abc",
        }

    @pytest.mark.asyncio
    async def test_a_token_is_fetched_once_and_then_reused(self, monkeypatch):
        # One metadata-server round trip per frame would be absurd at 8 fps.
        client = self._client(monkeypatch, url="https://dash.run.app")
        fetched = []

        def fetch(_request, audience):
            fetched.append(audience)
            return "tok"

        monkeypatch.setattr(
            "google.oauth2.id_token.fetch_id_token", fetch, raising=False)
        assert await client._identity_token() == "tok"
        assert await client._identity_token() == "tok"
        assert fetched == ["https://dash.run.app"], "audience is the dashboard URL"

    @pytest.mark.asyncio
    async def test_no_credentials_warns_once_and_keeps_going(self, monkeypatch, caplog):
        # A workstation has user credentials, which cannot mint an ID token.
        # That is a normal local run, not a reason to take the audit down.
        client = self._client(monkeypatch, url="https://dash.run.app")

        def fetch(_request, _audience):
            raise RuntimeError("no service account")

        monkeypatch.setattr(
            "google.oauth2.id_token.fetch_id_token", fetch, raising=False)
        with caplog.at_level(logging.WARNING, logger="cctv_audit.monitor"):
            assert await client._identity_token() == ""
            assert await client._identity_token() == ""
        assert len([r for r in caplog.records if "identity token" in r.message]) == 1
        # And the call still goes out, carrying what it does have.
        assert await client._headers() == {"X-Monitor-Token": "s3cret"}

    @pytest.mark.asyncio
    async def test_the_viewer_count_is_read_from_the_dashboard(self, monkeypatch):
        client = self._client(monkeypatch)
        session = self._Session(self._Response(200, {"viewers": 3}))

        async def get_session():
            return session

        monkeypatch.setattr(client, "_get_session", get_session)
        assert await client.viewers() == 3
        url, headers, params = session.calls[0]
        assert url.endswith("/api/viewers")
        assert headers["X-Monitor-Token"] == "s3cret", "the count is not public"
        assert params == {}, "the shared client asks about the shared room"

    @pytest.mark.asyncio
    async def test_a_job_scoped_client_only_counts_its_own_viewers(self, monkeypatch):
        # Otherwise a colleague watching a different audit keeps this one
        # streaming frames at nobody.
        from computer_use_agent.monitor import monitor_for_job

        client = self._client(monkeypatch)
        scoped = monitor_for_job("abc123")
        scoped.base_url, scoped.token = client.base_url, client.token
        session = self._Session(self._Response(200, {"viewers": 1}))

        async def get_session():
            return session

        monkeypatch.setattr(scoped, "_get_session", get_session)
        assert await scoped.viewers() == 1
        assert session.calls[0][2] == {"job": "abc123"}

    @pytest.mark.asyncio
    async def test_a_job_scoped_client_stamps_every_event(self, monkeypatch):
        # Stamped centrally, because one event type missed would show up as a
        # card on somebody else's dashboard -- a symptom nobody would trace
        # back to a missing field.
        from computer_use_agent.monitor import monitor_for_job

        scoped = monitor_for_job("abc123")
        posted = []

        class _Post:
            async def __aenter__(self_inner):
                return None

            async def __aexit__(self_inner, *_a):
                return False

        class _S:
            def post(self_inner, url, json=None, headers=None):
                posted.append(json)
                return _Post()

        async def get_session():
            return _S()

        async def no_headers():
            return {}

        monkeypatch.setattr(scoped, "_get_session", get_session)
        monkeypatch.setattr(scoped, "_headers", no_headers)
        await scoped._post_event({"type": "frame", "frame": "xyz"})
        assert posted == [{"type": "frame", "frame": "xyz", "job_id": "abc123"}]

    @pytest.mark.asyncio
    async def test_an_unreachable_dashboard_counts_as_nobody(self, monkeypatch):
        # Not an exception, and not "assume someone is there". Guessing the
        # optimistic way streams a live CCTV feed at a service that is gone.
        client = self._client(monkeypatch)

        for outcome in (OSError("connection refused"), self._Response(404)):
            session = self._Session(outcome)

            async def get_session(_s=session):
                return _s

            monkeypatch.setattr(client, "_get_session", get_session)
            assert await client.viewers() == 0


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


class TestDashboardRooms:
    """One service, several audits, and footage that must not cross over.

    The dashboard is a single Cloud Run service that every audit posts to and
    every customer opens. Before rooms it held one global state dict and one
    set of sockets, so two concurrent audits overwrote each other's picture and
    both viewers saw whichever frame landed last -- live CCTV of identifiable
    staff, shown to whoever happened to be watching something else.
    """

    @pytest.fixture(autouse=True)
    def _fresh_rooms(self):
        from computer_use_agent import monitor_server as ms

        ms._rooms.clear()
        ms._rooms[""] = ms.Room("")
        yield
        ms._rooms.clear()
        ms._rooms[""] = ms.Room("")

    @staticmethod
    def _client():
        from fastapi.testclient import TestClient
        from computer_use_agent.monitor_server import app

        return TestClient(app)

    def test_a_frame_reaches_its_own_room_and_no_other(self):
        client = self._client()
        with client.websocket_connect("/ws?job=alice") as alice, \
             client.websocket_connect("/ws?job=bob") as bob:
            alice.receive_json()          # the state each viewer gets on join
            bob.receive_json()

            client.post("/api/event",
                        json={"type": "frame", "frame": "ALICE-FRAME", "job_id": "alice"})

            assert alice.receive_json()["frame"] == "ALICE-FRAME"

            # And Bob's socket has nothing on it. Proven by making Bob's own
            # frame arrive next: if Alice's had leaked, this would read it.
            client.post("/api/event",
                        json={"type": "frame", "frame": "BOB-FRAME", "job_id": "bob"})
            assert bob.receive_json()["frame"] == "BOB-FRAME"

    def test_two_audits_do_not_overwrite_each_others_state(self):
        client = self._client()
        for job, url in (("alice", "https://a"), ("bob", "https://b")):
            client.post("/api/event",
                        json={"type": "state", "data": {"current_url": url}, "job_id": job})

        assert client.get("/api/state", params={"job": "alice"}).json()["current_url"] == "https://a"
        assert client.get("/api/state", params={"job": "bob"}).json()["current_url"] == "https://b"

    def test_the_viewer_count_is_per_room(self):
        # The audit turns its stream off when this is zero. Counting everyone
        # on the service would keep every audit streaming for as long as any
        # one person had any dashboard open.
        client = self._client()
        with client.websocket_connect("/ws?job=alice") as alice:
            alice.receive_json()
            assert client.get("/api/viewers", params={"job": "alice"}).json()["viewers"] == 1
            assert client.get("/api/viewers", params={"job": "bob"}).json()["viewers"] == 0

    def test_a_viewer_who_arrives_late_is_caught_up(self):
        # The link goes out when the audit starts; people click it whenever.
        client = self._client()
        client.post("/api/event",
                    json={"type": "frame", "frame": "EARLIER", "job_id": "alice"})
        with client.websocket_connect("/ws?job=alice") as alice:
            assert alice.receive_json()["type"] == "state"
            assert alice.receive_json()["frame"] == "EARLIER"

    def test_no_job_id_is_the_local_single_audit_case(self):
        # `adk web`: one audit, one page, nobody passing a job id. This is the
        # behaviour the whole server had before rooms existed.
        client = self._client()
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            client.post("/api/event", json={"type": "frame", "frame": "LOCAL"})
            assert ws.receive_json()["frame"] == "LOCAL"

    def test_idle_rooms_are_evicted_before_the_cap(self):
        # `/api/event` is authenticated, but a bug upstream that stamped a
        # fresh id on every frame would otherwise be a slow memory leak.
        from computer_use_agent import monitor_server as ms

        client = self._client()
        for i in range(ms._MAX_ROOMS + 20):
            client.post("/api/event", json={"type": "frame", "frame": "x", "job_id": f"j{i}"})
        assert len(ms._rooms) <= ms._MAX_ROOMS

    def test_a_watched_room_is_never_evicted(self):
        from computer_use_agent import monitor_server as ms

        client = self._client()
        with client.websocket_connect("/ws?job=keepme") as ws:
            ws.receive_json()
            for i in range(ms._MAX_ROOMS + 20):
                client.post("/api/event", json={"type": "frame", "frame": "x", "job_id": f"j{i}"})
            assert "keepme" in ms._rooms


class TestAFinishedRunSaysSoInsteadOfShowingNothing:
    """Frames are live-only; nobody who opens the link late gets a picture.

    Job 781c96 (2026-09-09) was reported as "the dashboard's video is gone".
    The dashboard logs showed the WebSocket connecting 55 seconds *after* the
    audit had ended, so there was nothing left to stream and never had been --
    no viewer means no frame is ever pushed, and none are stored. The results
    were there because those come from room state. Only the video panel was
    blank, with nothing on it to say why, which reads as a broken dashboard.
    """

    @pytest.fixture(autouse=True)
    def _fresh_rooms(self):
        from computer_use_agent import monitor_server as ms

        ms._rooms.clear()
        ms._rooms[""] = ms.Room("")
        yield
        ms._rooms.clear()
        ms._rooms[""] = ms.Room("")

    @staticmethod
    def _client():
        from fastapi.testclient import TestClient
        from computer_use_agent.monitor_server import app

        return TestClient(app)

    @staticmethod
    def _state(client, job, status):
        client.post("/api/event",
                    json={"type": "state", "data": {"status": status}, "job_id": job})
        return client.get("/api/state", params={"job": job}).json()

    def test_a_running_audit_has_no_end_time(self):
        client = self._client()
        assert self._state(client, "alice", "RUNNING")["finished_at"] is None

    def test_the_moment_a_run_ends_is_stamped_server_side(self):
        # Stamped here rather than sent by the container so it cannot disagree
        # with the clock the page formats it against.
        client = self._client()
        self._state(client, "alice", "RUNNING")
        assert self._state(client, "alice", "COMPLETED")["finished_at"] > 0

    def test_a_run_that_died_is_stamped_too(self):
        client = self._client()
        self._state(client, "alice", "RUNNING")
        assert self._state(client, "alice", "ERROR")["finished_at"] > 0

    def test_the_stamp_is_the_first_end_not_the_last_message(self):
        # Progress payloads keep arriving after `finish_session` (a segment's
        # own `state` event, say). Re-stamping on each would walk the time
        # forward and the page would say the audit ended later than it did.
        client = self._client()
        self._state(client, "alice", "RUNNING")
        first = self._state(client, "alice", "COMPLETED")["finished_at"]
        again = self._state(client, "alice", "COMPLETED")["finished_at"]
        assert again == first

    def test_starting_again_in_the_same_room_clears_the_stamp(self):
        # Local `adk web` reuses a room when the same job id runs twice.
        client = self._client()
        self._state(client, "alice", "COMPLETED")
        assert self._state(client, "alice", "RUNNING")["finished_at"] is None

    def test_a_viewer_who_arrives_after_the_end_is_told_it_ended(self):
        # The whole point: the state pushed on connect has to carry both the
        # status and the moment, or the page has nothing to build the line from.
        client = self._client()
        self._state(client, "alice", "RUNNING")
        self._state(client, "alice", "COMPLETED")
        with client.websocket_connect("/ws?job=alice") as late:
            joined = late.receive_json()
        assert joined["type"] == "state"
        assert joined["data"]["status"] == "COMPLETED"
        assert joined["data"]["finished_at"] > 0


class TestTheAnalysisModeIsInTheAnswer:
    """The customer is asked to confirm one of two very different runs.

    agentic on 60s windows costs 3.4x static and takes 45-155s a window
    against ~20s. Both are called "开始稽核". On 2026-09-09 the only way to
    find out which one a deployment was on was to read the engine's env vars
    out of the API -- the logs said nothing and the token counts of static/60s
    and agentic/60s are close enough to be ambiguous.
    """

    def test_preflight_carries_the_mode_even_on_the_paths_that_forget_things(self):
        # `default_factory`, not a value passed at each construction site:
        # preflight returns from four places and one of them is the error path.
        from computer_use_agent.pipeline import PreflightResult

        out = PreflightResult(ok=False, target="x", platform="p").as_dict()
        assert out["analysis_mode"] in ("static", "agentic")
        assert out["analysis_window_seconds"] > 0
        assert out["analysis_model"]

    def test_the_line_names_the_mode_and_editorialises_about_neither(self):
        # It used to read "agentic（模型自己挑该看的地方，慢一些、贵一些，看得细）".
        # That is our opinion of the setting sitting in a line the customer
        # reads as a statement of fact about their job, so it is gone: the
        # mode's name and the shape of the run, nothing else.
        from computer_use_agent.server import _describe_analysis

        agentic = _describe_analysis({"analysis_mode": "agentic", "analysis_window_seconds": 60})
        assert agentic == "分析方式：agentic，60 秒一段"

        static = _describe_analysis({"analysis_mode": "static", "analysis_window_seconds": 30})
        assert static == "分析方式：static，30 秒一段"

        for word in ("慢", "贵", "快", "便宜", "看得细"):
            assert word not in agentic and word not in static

    def test_an_unrecognised_mode_says_nothing_rather_than_guessing(self):
        # A job whose preflight predates this field falls back to the running
        # config; a mode nobody knows must not be labelled as one we do.
        from computer_use_agent.server import _describe_analysis

        assert _describe_analysis({"analysis_mode": "something-new"}) == ""

    def test_it_appears_next_to_the_capture_mode_the_customer_already_gets(self):
        from computer_use_agent.jobs import Job
        from computer_use_agent.server import _describe_preflight

        job = Job(
            user_id="u", job_id="abc123", target="t", state="ready",
            preflight={
                "ok": True, "capture_mode": "stream", "video_duration_seconds": 600.0,
                "analysis_mode": "agentic", "analysis_window_seconds": 60,
            },
        )
        text = _describe_preflight(job)
        assert "采集方式：抓流" in text
        assert "分析方式：agentic" in text
        assert "视频总长：10:00" in text
        # Order matters only in that the two "方式" lines belong together.
        assert text.index("采集方式") < text.index("分析方式") < text.index("视频总长")


class TestReadingAVideoStraightOutOfABucket:
    """Plan C: the customer's own SOP recordings are files, not web pages."""

    def test_a_normal_uri_splits_into_bucket_and_object(self):
        from computer_use_agent.capture.gcs_video import parse_gs_uri

        assert parse_gs_uri("gs://my-bucket/sop/开店流程.mp4") == ("my-bucket", "sop/开店流程.mp4")
        assert parse_gs_uri("  gs://my-bucket/a.mp4  ") == ("my-bucket", "a.mp4")

    def test_the_three_things_people_actually_paste_wrong_each_say_what_is_wrong(self):
        from computer_use_agent.capture.gcs_video import BadGcsUri, parse_gs_uri

        with pytest.raises(BadGcsUri, match="没给文件"):
            parse_gs_uri("gs://my-bucket")
        with pytest.raises(BadGcsUri, match="目录"):
            parse_gs_uri("gs://my-bucket/sop/")
        with pytest.raises(BadGcsUri, match="不是一个 gs:// 地址"):
            parse_gs_uri("https://example.com/a.mp4")

    def test_a_bucket_name_that_could_escape_the_bucket_position_is_refused(self):
        # This string is about to be interpolated into a URL. A name carrying
        # an `@` or a `:` out of the host position is not a naming-rules
        # quibble, it is a request to a different server.
        from computer_use_agent.capture.gcs_video import BadGcsUri, parse_gs_uri

        for bad in ("gs://evil.com:8080/x.mp4", "gs://a@b/x.mp4", "gs://-lead/x.mp4"):
            with pytest.raises(BadGcsUri):
                parse_gs_uri(bad)

    def test_the_object_name_is_encoded_slashes_and_all(self):
        # `?alt=media` on the JSON API, because the object name has to be
        # percent-encoded including its slashes. A raw `?` or `#` in a name
        # would otherwise truncate the path and fetch a different object.
        from computer_use_agent.capture.gcs_video import media_url

        url = media_url("b", "sop/v1 final#2.mp4")
        assert "/o/sop%2Fv1%20final%232.mp4?alt=media" in url
        assert url.count("?") == 1

    def test_routing_is_loose_so_a_broken_gs_uri_still_gets_a_gs_complaint(self):
        # If `gs://bucket` fell through to the browser it would come back
        # "打不开这个视频", which sends the customer looking in the wrong place.
        from computer_use_agent.capture.gcs_video import is_gcs_uri

        assert is_gcs_uri("gs://bucket")
        assert is_gcs_uri("GS://Bucket/a.mp4")
        assert not is_gcs_uri("https://example.com/a.mp4")
        assert not is_gcs_uri("")

    def test_the_first_failure_anyone_hits_names_the_grant_that_is_missing(self):
        # "Server returned 403 Forbidden" is not actionable until somebody says
        # whose permission it is -- and on day one it is always ours, on their
        # bucket.
        from computer_use_agent.capture.gcs_video import _explain

        assert "storage.objectViewer" in _explain("b", "o.mp4", "Server returned 403 Forbidden")
        assert "不存在" in _explain("b", "o.mp4", "Server returned 404 Not Found")
        assert "gs://b/o.mp4" in _explain("b", "o.mp4", "whatever else")

    def test_a_live_or_unfinalised_recording_reports_no_duration_rather_than_a_fake_one(self):
        from computer_use_agent.capture.gcs_video import _duration_of

        assert _duration_of({"format": {"duration": "180.5"}}) == 180.5
        assert _duration_of({"format": {"duration": "inf"}}) is None
        assert _duration_of({"format": {"duration": "0"}}) is None
        assert _duration_of({"format": {}}) is None
        assert _duration_of({}) is None


class TestTheEntranceRecognisesABucketPath:
    """A `gs://` URI pasted into GE has to survive the whole way in."""

    def test_from_fields_accepts_it(self):
        from computer_use_agent.intent import from_fields

        intent = from_fields("gs://bucket/sop/a.mp4", start="05:00", end="07:00")
        assert intent.request.target == "gs://bucket/sop/a.mp4"
        assert intent.request.start_seconds == 300.0
        assert intent.request.duration_seconds == 120.0

    def test_a_half_written_uri_is_caught_while_the_customer_is_still_here(self):
        # Not a minute later inside a preflight, by which point the reply is
        # "打不开这个视频" and they have moved on.
        from computer_use_agent.capture.gcs_video import BadGcsUri
        from computer_use_agent.intent import from_fields

        with pytest.raises(BadGcsUri):
            from_fields("gs://bucket", start=0)

    def test_a_shop_name_is_still_not_an_address(self):
        from computer_use_agent.intent import from_fields

        with pytest.raises(ValueError, match="不是一个视频地址"):
            from_fields("望京店", start=0)

    def test_the_offline_fallback_finds_one_in_a_sentence(self):
        from computer_use_agent.intent import _URL_RE

        found = _URL_RE.search("按 chagee-store-v1 稽核 gs://chagee-sop/开店/v3.mp4，从 05:00 看 2 分钟")
        assert found and found.group(0) == "gs://chagee-sop/开店/v3.mp4"

    def test_a_chinese_comma_ends_the_address_even_with_no_space_after_it(self):
        # `\S+` used to swallow "，从" and hand the whole thing to ffprobe as
        # the address. Not a gs:// problem -- http has always had it, it just
        # took writing a Chinese-punctuation test to notice.
        from computer_use_agent.intent import _URL_RE

        for text, want in [
            ("稽核 https://x.com/v?a=1，从 05:00 开始", "https://x.com/v?a=1"),
            ("地址是 gs://b/a.mp4。谢谢", "gs://b/a.mp4"),
            ("（gs://b/a.mp4）这个", "gs://b/a.mp4"),
        ]:
            assert _URL_RE.search(text).group(0) == want

    def test_a_path_may_still_contain_chinese_characters(self):
        # Only the punctuation is excluded. Object names like `开店/v3.mp4` are
        # exactly what the customer's bucket looks like.
        from computer_use_agent.intent import _URL_RE

        assert _URL_RE.search("gs://桶/开店流程/第三版.mp4").group(0) == "gs://桶/开店流程/第三版.mp4"

    def test_the_ge_reply_calls_it_what_it_is(self):
        from computer_use_agent.jobs import Job
        from computer_use_agent.server import _describe_preflight

        job = Job(
            user_id="u", job_id="j", target="gs://b/a.mp4", state="ready",
            preflight={"ok": True, "capture_mode": "file", "title": "sop/a.mp4"},
        )
        text = _describe_preflight(job)
        assert "采集方式：直接读文件（GCS）" in text
        assert "sop/a.mp4" in text


class TestPlanCHandsOverTheWholeObjectAndCapturesNothing:
    """No slicing, no ffmpeg, no bytes: one clip that is the whole recording.

    This replaced a design that cut the object into 60s windows with ffmpeg.
    The measurement that killed it: under `MEDIA_PROCESSING=agentic`, the
    `video_metadata` start/end offsets are silently ignored. Asked for
    240s-300s of a 5-minute file with a burnt-in timecode, static answered
    "起=00:09:00 止=00:09:59" and agentic answered "起=00:05:00 止=00:10:00" --
    the whole file, no error, no warning. So windowing a `gs://` object under
    agentic is not merely wasteful, it is wrong.
    """

    def _source(self, mode="file", uri="gs://b/a.mp4", duration=None):
        from computer_use_agent.capture.types import CaptureSource

        return CaptureSource(
            mode=mode, url="https://storage.googleapis.com/x?alt=media",
            headers={"Authorization": "Bearer t"}, reason="r",
            object_uri=uri, duration_seconds=duration,
        )

    def _producer(self, tmp_path, *, start=300.0, duration=120.0, source=None):
        from computer_use_agent.pipeline import AuditPipeline, AuditRequest

        request = AuditRequest(
            target="gs://b/a.mp4", start_seconds=start, duration_seconds=duration)
        return asyncio.run(AuditPipeline._build_producer(
            AuditPipeline(), source or self._source(duration=600.0),
            None, None, tmp_path, request))

    def test_the_whole_object_becomes_exactly_one_clip(self, tmp_path):
        from computer_use_agent.capture.gcs_video import WholeFileProducer

        producer = self._producer(tmp_path)
        assert isinstance(producer, WholeFileProducer)

        async def collect():
            return [c async for c in producer.clips()]

        clips = asyncio.run(collect())
        assert len(clips) == 1
        clip = clips[0]
        assert clip.uri == "gs://b/a.mp4" and clip.path is None
        assert clip.whole_video is True and clip.is_remote is True
        assert clip.start_offset == 0.0 and clip.end_offset == 600.0
        assert clip.source_mode == "file"

    def test_the_requested_time_range_is_ignored_rather_than_half_honoured(self, tmp_path):
        # Asked for 05:00 + 2 minutes; the clip still starts at zero and runs
        # the whole 10 minutes. Half-honouring it -- seeking to 300 and letting
        # the model read to the end anyway -- is the outcome the probe showed
        # agentic silently producing, and it reports timestamps against the
        # wrong origin.
        clip = asyncio.run(self._one(self._producer(tmp_path, start=300.0, duration=120.0)))
        assert (clip.start_offset, clip.end_offset) == (0.0, 600.0)

    @staticmethod
    async def _one(producer):
        async for clip in producer.clips():
            return clip
        raise AssertionError("producer yielded nothing")

    def test_an_unknown_duration_reads_as_zero_rather_than_a_plausible_guess(self, tmp_path):
        # ffprobe could not say. `00:00 - 00:00` looks wrong, which is the
        # point: a made-up length would look right and be wrong.
        clip = asyncio.run(self._one(self._producer(tmp_path, source=self._source())))
        assert clip.end_offset == 0.0 and clip.time_range == "00:00 - 00:00"
        # ...and the findings from it keep their own timestamps rather than
        # being clamped to a length nobody knows.
        assert clip.clip_ts_to_video_offset(412.0) == 412.0

    def test_it_refuses_a_source_that_has_no_object_address(self):
        # `url` is the HTTPS endpoint ffmpeg reads; Vertex only accepts `gs://`.
        # Building on a source that has the first and not the second would send
        # a signed URL to the model and fail deep inside the request.
        from computer_use_agent.capture.gcs_video import WholeFileProducer

        with pytest.raises(ValueError):
            WholeFileProducer(self._source(uri=None))
        with pytest.raises(ValueError):
            WholeFileProducer(self._source(mode="stream"))

    def test_closing_it_is_a_no_op_because_nothing_was_opened(self, tmp_path):
        # The pipeline closes its producer unconditionally; a missing `aclose`
        # would be an AttributeError in a `finally` block, i.e. at the one
        # moment the real error is being reported.
        assert asyncio.run(self._producer(tmp_path).aclose()) is None

    def test_building_it_never_reaches_for_the_page(self, tmp_path):
        # page and navigator are None on this path. Anything in `_build_producer`
        # that touched them would be an AttributeError at the top of every run.
        assert self._producer(tmp_path, start=0.0, duration=None) is not None

    def test_a_whole_file_run_is_not_measured_against_a_span_it_ignored(self):
        # The customer asked for 02:00 + 10 minutes and got the whole 30. The
        # old arithmetic would have called that "只覆盖到 30:00，请求的是到
        # 12:00" -- understating a run that watched everything, which is the
        # one direction a coverage report must never err in.
        from computer_use_agent.capture.types import CaptureSource

        source = CaptureSource(mode="file", url="u", object_uri="gs://b/a.mp4",
                               duration_seconds=1800.0)
        out = TestCoverageHonesty()._coverage(
            (0.0, 1800.0), start=120.0, duration=600.0, source=source)
        assert out["complete"] is True and out["incomplete_reason"] is None
        assert out["whole_video"] is True
        assert out["requested_start_seconds"] == 0.0
        assert out["requested_end_seconds"] == 1800.0

        # Nothing analysed is still nothing analysed.
        empty = TestCoverageHonesty()._coverage(None, source=source)
        assert empty["complete"] is False and empty["incomplete_reason"]

    def test_preflight_does_not_reject_a_start_it_is_about_to_ignore(self):
        # 05:00 of a 3-minute file. On Plan A that is "nothing to audit"; here
        # the whole three minutes are going to the model regardless, so
        # refusing the job would refuse a perfectly auditable video.
        from computer_use_agent.capture.types import CaptureSource
        from computer_use_agent.pipeline import AuditPipeline, AuditRequest, _Session

        session = _Session(
            page=None, context=None, navigator=None,
            source=CaptureSource(mode="file", url="u", object_uri="gs://b/a.mp4",
                                 reason="r", duration_seconds=180.0),
            work_dir=Path("/tmp"), platform="gcs",
        )

        pipeline = AuditPipeline()

        @contextlib.asynccontextmanager
        async def fake_session(request, *, live_preview=True):
            yield session

        pipeline._capture_session = fake_session
        pipeline._cover_frame = _async_none
        request = AuditRequest(target="gs://b/a.mp4", start_seconds=300.0,
                               duration_seconds=120.0)
        result = asyncio.run(pipeline._preflight_inner(request, "job1"))
        assert result.ok is True and result.span_available is True
        assert result.problem is None
        assert result.analysis_scope == "whole_file"
        assert result.as_dict()["analysis_scope"] == "whole_file"

    def test_the_scope_of_every_other_plan_is_still_windows(self):
        from computer_use_agent.pipeline import PreflightResult

        for mode in ("stream", "screen", ""):
            assert PreflightResult(
                ok=True, target="t", platform="p", capture_mode=mode
            ).analysis_scope == "windows"

    def test_the_reply_says_whole_video_and_admits_the_span_is_unused(self):
        # The customer typed a time range and is about to get a report that
        # covers everything. Being told that before confirming is the whole
        # difference between "it ignored me" and "it told me".
        from computer_use_agent.jobs import Job
        from computer_use_agent.server import _describe_preflight

        job = Job(
            user_id="u", job_id="j", target="gs://b/a.mp4", state="ready",
            preflight={
                "ok": True, "capture_mode": "file", "analysis_scope": "whole_file",
                "analysis_mode": "agentic", "analysis_window_seconds": 60,
                "requested_start_seconds": 300.0, "requested_end_seconds": 420.0,
            },
        )
        text = _describe_preflight(job)
        assert "要稽核：整段视频，从头看到尾" in text
        assert "05:00 - 07:00 这次用不上" in text
        assert "分析方式：agentic，整段视频一次看完，不切片" in text
        assert "秒一段" not in text

    def test_a_job_that_predates_the_scope_field_is_read_off_its_capture_mode(self):
        # Firestore still holds jobs whose preflight has no `analysis_scope`.
        # Answering "windows" for one of them would promise a live picture that
        # this path has never had.
        from computer_use_agent.server import _is_whole_file

        assert _is_whole_file({"capture_mode": "file"}) is True
        assert _is_whole_file({"capture_mode": "stream"}) is False
        assert _is_whole_file({"analysis_scope": "windows", "capture_mode": "file"}) is False

    def test_evidence_stills_come_out_of_the_object_when_there_is_no_local_clip(
            self, tmp_path, monkeypatch):
        # A remote clip has `path=None`. Handing that to ffmpeg is a TypeError
        # inside the one code path whose job is to preserve proof.
        from computer_use_agent import store as store_mod

        asked = {}

        async def fake_frame_at(uri, offset, out_path, **kw):
            asked["uri"], asked["offset"] = uri, offset
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"JPEG")
            return True

        async def never(*a, **kw):
            pytest.fail("a remote clip has no file for ffmpeg to cut")

        monkeypatch.setattr(store_mod.gcs_video, "frame_at", fake_frame_at)
        monkeypatch.setattr(store_mod, "extract_frame", never)

        store = store_mod.AuditStore(
            records_path=tmp_path / "r.jsonl", evidence_dir=tmp_path / "ev")
        finding = type("F", (), {"rule_id": "R1", "offset_seconds": 91.0})()
        locator = asyncio.run(store._save_evidence(self._remote_clip(), finding))
        assert locator and asked["uri"] == "gs://b/a.mp4" and asked["offset"] == 91.0

    def _remote_clip(self):
        from computer_use_agent.capture.types import Clip

        return Clip(index=0, path=None, uri="gs://b/a.mp4", start_offset=0.0,
                    end_offset=600.0, wall_clock_start=0.0, source_mode="file",
                    whole_video=True)

    def test_a_clip_says_where_the_footage_is_exactly_once(self):
        # Both would mean two answers to "where are the bytes" and the readers
        # disagree about which wins; neither means the analyser has nothing to
        # send and finds out one request too late.
        from computer_use_agent.capture.types import Clip

        for kwargs in ({}, {"path": Path("/tmp/a.mp4"), "uri": "gs://b/a.mp4"}):
            with pytest.raises(ValueError):
                Clip(index=0, start_offset=0.0, end_offset=1.0,
                     wall_clock_start=0.0, source_mode="file",
                     path=kwargs.get("path"), uri=kwargs.get("uri"))

        local = Clip(index=0, path=Path("/tmp/a.mp4"), start_offset=0.0, end_offset=1.0,
                     wall_clock_start=0.0, source_mode="stream")
        assert local.is_remote is False and local.whole_video is False


class TestTheModelIsGivenTheAddressNotTheBytes:
    """The remote branch of `analyze`, and the prompt that goes with it."""

    def _clip(self, uri="gs://b/a.mp4", **kw):
        from computer_use_agent.capture.types import Clip

        return Clip(index=0, path=None, uri=uri, start_offset=0.0, end_offset=600.0,
                    wall_clock_start=0.0, source_mode="file",
                    whole_video=kw.pop("whole_video", True), **kw)

    def test_the_mime_type_follows_the_object_name(self):
        from computer_use_agent.capture.gcs_video import mime_for

        assert mime_for("gs://b/a.mp4") == "video/mp4"
        assert mime_for("gs://b/A.MOV") == "video/quicktime"
        # Something we have no entry for still has to be a video type: an
        # unlabelled part is rejected, and mp4 is what the customer's
        # recordings have been every time so far.
        assert mime_for("gs://b/a.weirdext").startswith("video/")

    def test_a_remote_clip_is_sent_as_an_address_and_never_read(self, monkeypatch):
        # The 250 MB inline cap is a property of putting bytes in the request
        # body, and there are no bytes in this request body -- a 1 GB object
        # goes through untouched. `read_bytes` fails the test rather than
        # returning something, because a Plan C run that reads the file works
        # right up until the file is bigger than the instance's memory.
        from computer_use_agent.analyzer import video_analyzer as va

        monkeypatch.setattr(
            Path, "read_bytes",
            lambda self: pytest.fail("Plan C must not read any bytes"))

        sent = {}

        async def fake_call(**kwargs):
            sent.update(kwargs)
            raise RuntimeError("the part is all we came for")

        monkeypatch.setattr(va, "generate_content_with_retry", fake_call)
        analyzer = VideoAnalyzer(load_rules(CCTV_RULES))

        outcome = asyncio.run(analyzer.analyze(self._clip()))
        # The call was made and blew up on our own exception, not on a size
        # check and not on a missing file.
        assert outcome.error and "came for" in outcome.error
        part = sent["contents"][0]
        assert part.file_data.file_uri == "gs://b/a.mp4"
        assert part.file_data.mime_type == "video/mp4"
        assert part.inline_data is None

    def test_the_whole_video_prompt_asks_for_every_occurrence(self):
        # A window can hold one instance of a violation; a whole recording can
        # hold five. "One finding per rule" would report one of them and the
        # report would read as if the other four never happened.
        whole = build_prompt(self._clip(), load_rules(CCTV_RULES))
        assert "整段" in whole and "通篇" in whole
        assert "每次各输出一条 finding" in whole
        assert "视频第一帧是 00:00" in whole

    def test_a_window_prompt_still_says_which_window_it_is(self):
        clip = make_clip(index=3, start_offset=180.0, end_offset=240.0)
        windowed = build_prompt(clip, load_rules(CCTV_RULES))
        assert "03:00 - 04:00" in windowed and "通篇" not in windowed


class TestAPagelessRunDoesNotPretendToHaveAPage:
    """`_Session.page` is None on Plan C, and four things used to assume it was not."""

    def _session(self, page=None, duration=None):
        from computer_use_agent.capture.types import CaptureSource
        from computer_use_agent.pipeline import _Session

        return _Session(
            page=page, context=None, navigator=None,
            source=CaptureSource(mode="file", url="u", duration_seconds=duration),
            work_dir=Path("/tmp"), platform="gcs",
        )

    def test_has_page_is_the_check_every_consumer_makes(self):
        assert self._session().has_page is False
        assert self._session(page=object()).has_page is True

    def test_an_unreadable_duration_is_none_not_an_attribute_error(self):
        # Catching the AttributeError from a None navigator would produce the
        # same None, and would also hide a real navigator fault behind it.
        from computer_use_agent.pipeline import AuditPipeline

        pipeline = AuditPipeline()
        assert asyncio.run(pipeline._video_duration(self._session())) is None
        assert asyncio.run(pipeline._video_duration(self._session(duration=180.0))) == 180.0

    def test_the_cover_still_comes_from_the_footage_when_there_is_no_screenshot(self, monkeypatch):
        # The cover is the part of the preflight reply that proves we opened
        # the customer's video and not somebody else's file. Losing it on Plan C
        # would be a silent downgrade.
        from computer_use_agent import pipeline as pipeline_mod

        asked = {}

        async def fake_grab(source, offset, **kw):
            asked["offset"] = offset
            return b"JPEGBYTES"

        class Sink:
            async def put(self, key, data, content_type):
                asked["key"], asked["data"] = key, data
                return f"gs://art/{key}"

        monkeypatch.setattr(pipeline_mod.gcs_video, "grab_frame", fake_grab)
        monkeypatch.setattr(pipeline_mod, "artifact_sink", lambda: Sink())

        out = asyncio.run(pipeline_mod.AuditPipeline()._cover_frame(
            self._session(), "job1", 300.0))
        assert out == "gs://art/job1/cover.jpg"
        assert asked["offset"] == 300.0 and asked["data"] == b"JPEGBYTES"

    def test_a_missing_cover_is_still_only_a_missing_cover(self, monkeypatch):
        from computer_use_agent import pipeline as pipeline_mod

        async def fake_grab(source, offset, **kw):
            return None

        monkeypatch.setattr(pipeline_mod.gcs_video, "grab_frame", fake_grab)
        assert asyncio.run(pipeline_mod.AuditPipeline()._cover_frame(
            self._session(), "job1", 0.0)) is None

    def test_nobody_is_pointed_at_a_dashboard_that_has_nothing_to_show(self):
        # There used to be a "no live picture on this path" announcement on the
        # dashboard itself. It went together with the link: a Plan C run has no
        # page, no windows and no preview, so the honest thing is not to offer
        # the screen at all rather than offer it with an apology attached.
        assert "实时画面" not in _confirm_reply(capture_mode="file")
        assert "实时画面" in _confirm_reply(capture_mode="screen")


class TestTheSessionForksOnceAndOnlyOnce:
    """Routing lives in `_capture_session`; nothing downstream branches again."""

    def test_a_gs_target_never_starts_a_browser(self, monkeypatch):
        from computer_use_agent import pipeline as pipeline_mod
        from computer_use_agent.capture.types import CaptureSource

        started = []
        monkeypatch.setattr(
            pipeline_mod, "async_playwright",
            lambda: started.append("browser") or (_ for _ in ()).throw(AssertionError()))

        async def fake_open(uri, **kw):
            return CaptureSource(mode="file", url="u", reason="r", duration_seconds=42.0)

        monkeypatch.setattr(pipeline_mod.gcs_video, "open_source", fake_open)

        async def go():
            pipeline = pipeline_mod.AuditPipeline()
            request = pipeline_mod.AuditRequest(target="gs://b/a.mp4")
            async with pipeline._capture_session(request) as session:
                return session

        session = asyncio.run(go())
        assert started == []
        assert session.has_page is False and session.source.mode == "file"
        assert session.platform == "gcs"

    def test_an_http_target_still_goes_to_the_browser(self, monkeypatch):
        from computer_use_agent import pipeline as pipeline_mod

        went = []

        @contextlib.asynccontextmanager
        async def fake_browser(self, request, *, live_preview=True):
            went.append(request.target)
            yield "session"

        monkeypatch.setattr(pipeline_mod.AuditPipeline, "_browser_session", fake_browser)

        async def go():
            pipeline = pipeline_mod.AuditPipeline()
            request = pipeline_mod.AuditRequest(target="https://example.com/v")
            async with pipeline._capture_session(request) as session:
                return session

        assert asyncio.run(go()) == "session"
        assert went == ["https://example.com/v"]
