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

"""Loads the SOP rule set and renders it into a prompt.

Rules live in `sop_rules.yaml` so an auditor can revise the standard without a
code change or redeploy. This file is the *only* source of audit criteria --
text typed into the chat box is parsed for a URL and a time range and nothing
else, deliberately: a standard that changes per message cannot be audited.
Point `SOP_RULES_PATH` at your own file to override it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml

from ..config import config

logger = logging.getLogger("cctv_audit.sop")

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "sop_rules.yaml"


@dataclass(frozen=True)
class SopRule:
    id: str
    name: str
    description: str
    severity: str = "NORMAL"
    detection_type: str = "presence"  # presence | absence
    requires_full_context: bool = False
    positive_indicators: List[str] = field(default_factory=list)
    negative_indicators: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"### {self.id} — {self.name}（严重度 {self.severity}，"
            f"{'缺失类判定' if self.detection_type == 'absence' else '出现即违规'}）",
            f"判定标准：{self.description.strip()}",
        ]
        if self.positive_indicators:
            lines.append("合规表现：" + "；".join(self.positive_indicators))
        if self.negative_indicators:
            lines.append("违规表现：" + "；".join(self.negative_indicators))
        if self.requires_full_context:
            lines.append(
                "注意：本条依赖完整过程。若相关动作的完整过程未被本片段完整覆盖"
                "（人员走出画面、被遮挡、动作跨越片段边界），必须判 CANNOT_DETERMINE。"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class ScanTarget:
    """Something the model must describe before it is allowed to judge anything.

    Asked for a verdict directly, a model answers from what a tea shop usually
    looks like. Asked first to state the colour of the hands, it has to look.
    """

    subject: str
    question: str

    def render(self) -> str:
        return f"- **{self.subject}**：{self.question.strip()}"


@dataclass(frozen=True)
class SopRuleSet:
    version: int
    rules: List[SopRule]
    scan_targets: List[ScanTarget] = field(default_factory=list)

    @property
    def ids(self) -> List[str]:
        return [r.id for r in self.rules]

    @property
    def subjects(self) -> List[str]:
        return [t.subject for t in self.scan_targets]

    def render_scan(self) -> str:
        return "\n".join(t.render() for t in self.scan_targets)

    def get(self, rule_id: str) -> Optional[SopRule]:
        return next((r for r in self.rules if r.id == rule_id), None)

    def severity_of(self, rule_id: str) -> str:
        rule = self.get(rule_id)
        return rule.severity if rule else "NORMAL"

    def render(self) -> str:
        return "\n\n".join(rule.render() for rule in self.rules)


@lru_cache(maxsize=4)
def load_rules(path: Optional[Path] = None) -> SopRuleSet:
    path = Path(path) if path else (config.sop_rules_path or DEFAULT_RULES_PATH)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules = []
    for entry in raw.get("rules", []):
        try:
            rules.append(SopRule(
                id=entry["id"],
                name=entry["name"],
                description=entry["description"],
                severity=entry.get("severity", "NORMAL"),
                detection_type=entry.get("detection_type", "presence"),
                requires_full_context=bool(entry.get("requires_full_context", False)),
                positive_indicators=list(entry.get("positive_indicators", [])),
                negative_indicators=list(entry.get("negative_indicators", [])),
            ))
        except KeyError as exc:
            raise ValueError(f"SOP rule in {path} is missing required field {exc}") from exc
    if not rules:
        raise ValueError(f"No SOP rules found in {path}")

    targets = []
    for entry in raw.get("visual_scan", []):
        try:
            targets.append(ScanTarget(subject=entry["subject"], question=entry["question"]))
        except KeyError as exc:
            raise ValueError(f"visual_scan entry in {path} is missing required field {exc}") from exc

    logger.info(
        "Loaded %d SOP rules and %d scan target(s) (v%s) from %s",
        len(rules), len(targets), raw.get("version", 1), path.name,
    )
    return SopRuleSet(version=int(raw.get("version", 1)), rules=rules, scan_targets=targets)
