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

"""The contract for a window's analysis result.

Passed to Gemini as `response_schema`, so the model must return this shape --
no prose parsing, and no keyword matching over the model's own reasoning
(which is what previously turned the sentence "check whether there is a mask
violation" into a recorded red-line violation).
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("cctv_audit.analyzer.schema")


class Status(str, Enum):
    VIOLATION = "VIOLATION"
    COMPLIANT = "COMPLIANT"
    CANNOT_DETERMINE = "CANNOT_DETERMINE"


class Severity(str, Enum):
    RED_LINE = "RED_LINE"
    NORMAL = "NORMAL"
    NONE = "NONE"


class Finding(BaseModel):
    rule_id: str = Field(description="规则编号，必须来自给定的规则清单")
    status: Status = Field(description="该规则在本窗口内的判定结果")
    confidence: float = Field(ge=0.0, le=1.0, description="判定置信度 0-1")
    timestamp_in_clip: str = Field(
        default="00:00",
        description="本片段内该证据出现的时刻，格式 MM:SS；无具体时刻时填 00:00",
    )
    evidence: str = Field(
        default="",
        description="支撑判定的客观视觉事实。只写看到了什么，不写推测、不写建议",
    )
    severity: Severity = Field(default=Severity.NONE, description="违规严重度；非违规填 NONE")

    @field_validator("timestamp_in_clip")
    @classmethod
    def _normalise_timestamp(cls, value: str) -> str:
        """Normalises MM:SS / HH:MM:SS, and leaves anything else alone.

        This used to silently rewrite an unparseable timestamp to "00:00",
        which is not a null -- it is a specific moment. The evidence frame is
        cut at exactly this offset, so a violation described as happening at
        the end of the window came back with a picture of its first frame, and
        nothing anywhere said the moment had been made up. An unreadable
        timestamp is now kept as-is so `offset_seconds` can report that it
        does not know.
        """
        value = (value or "").strip()
        if not value:
            return ""
        match = re.search(r"(\d{1,3}):(\d{2})(?::(\d{2}))?", value)
        if not match:
            logger.warning("Model gave an unreadable timestamp %r; the evidence frame "
                           "will be marked as an estimate.", value[:40])
            return value
        a, b, c = match.groups()
        if c is not None:  # HH:MM:SS
            return f"{int(a) * 60 + int(b):02d}:{int(c):02d}"
        return f"{int(a):02d}:{int(b):02d}"

    @property
    def has_timestamp(self) -> bool:
        """Whether the moment is the model's, rather than our fallback."""
        return re.fullmatch(r"\d{2,}:\d{2}", self.timestamp_in_clip or "") is not None

    @property
    def offset_seconds(self) -> float:
        """`timestamp_in_clip` as seconds from the start of the clip file.

        Falls back to 0 when there is no usable timestamp -- the frame has to
        be cut somewhere -- but `has_timestamp` is what tells a reader whether
        the number means anything.
        """
        if not self.has_timestamp:
            return 0.0
        minutes, _, seconds = self.timestamp_in_clip.partition(":")
        return int(minutes) * 60 + int(seconds)

    @property
    def is_violation(self) -> bool:
        return self.status is Status.VIOLATION


class VisualScan(BaseModel):
    """One forced observation, recorded before any rule is judged.

    Separating "what is on screen" from "does that break a rule" is the whole
    point. A model asked straight for a verdict will reason from what a tea
    shop usually looks like; asked first to state the colour of the hands, it
    has to look. The scan is also what a human reviewer reads to decide whether
    the verdict was reached honestly.
    """

    subject: str = Field(description="扫描目标，必须原样取自给定的扫描清单，例如「手部」")
    observation: str = Field(
        description="只写肉眼确切看到的：颜色、材质、遮挡关系、有无。看不清就写「画面清晰度不足，无法判定」"
    )
    timestamps: str = Field(
        default="",
        description="支撑该观察的片段内时刻，格式 MM:SS 或 MM:SS-MM:SS；看不清时留空",
    )


class WindowResult(BaseModel):
    # Declared first on purpose: the model fills the schema in order, so the
    # observations are written before the verdicts that must follow from them.
    visual_scan: List[VisualScan] = Field(
        default_factory=list,
        description="视觉证据锚定。必须为扫描清单中的每一项各输出一条，且必须先于 findings 完成",
    )
    action_narrative: str = Field(
        default="",
        description="按时间顺序完整描述员工做了什么动作、拿了什么东西，细到单个动作",
    )
    scene_summary: str = Field(description="本窗口画面内容的客观描述，2-4 句")
    people_count: int = Field(default=0, ge=0, description="画面中出现的员工人数（取窗口内最大值）")
    visibility_ok: bool = Field(
        default=True,
        description="画面是否清晰可判读。黑屏、严重遮挡、马赛克、无人画面时填 false",
    )
    findings: List[Finding] = Field(default_factory=list, description="逐条规则的判定结果")

    @property
    def violations(self) -> List[Finding]:
        return [f for f in self.findings if f.is_violation]

    def worst_severity(self) -> Severity:
        for finding in self.violations:
            if finding.severity is Severity.RED_LINE:
                return Severity.RED_LINE
        return Severity.NORMAL if self.violations else Severity.NONE


def response_schema() -> dict:
    """The JSON Schema handed to Vertex AI.

    Built from the pydantic model and then flattened: Vertex rejects `$ref`/
    `$defs`, which pydantic emits for nested models by default.
    """
    return _inline_refs(WindowResult.model_json_schema())


def _inline_refs(schema: dict) -> dict:
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, list):
            return [resolve(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            key = node["$ref"].rsplit("/", 1)[-1]
            merged = {k: v for k, v in node.items() if k != "$ref"}
            return resolve({**defs.get(key, {}), **merged})
        # Vertex ignores these and some versions reject them outright.
        return {
            k: resolve(v) for k, v in node.items()
            if k not in ("title", "default", "additionalProperties")
        }

    return resolve(schema)
