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

Rules live in a YAML file so an auditor can revise the standard without a code
change or redeploy. That file is the *only* source of audit criteria -- text
typed into the chat box is parsed for a URL and a time range and nothing else,
deliberately: a standard that changes per message cannot be audited.

Two ways in, and the difference matters:

  * `load_rules(path)` reads a file. `SOP_RULES_PATH` on a workstation.
  * `load_rules_for(sop_id)` reads `gs://<SOP_BUCKET>/<SOP_PREFIX>/<id>.yaml`,
    which is how a customer picks a version from a GE prompt chip.

**A named version that cannot be fetched is an error, never a substitution.**
Quietly auditing against some other version produces a report that looks
exactly like a real one and is not, and nobody downstream can tell. So a miss
raises `SopUnavailable`, and the customer is told which id failed.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..config import config

logger = logging.getLogger("cctv_audit.sop")

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "sop_rules.yaml"

# A version id becomes part of an object path, and it arrives from a chat
# message via GE. Anything outside this set could walk out of the SOP prefix.
_SOP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class SopUnavailable(RuntimeError):
    """The named standard could not be loaded. Not recoverable by guessing."""


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
    # Which stored standard this came from, e.g. "chagee-store-v3". Copied onto
    # every audit record: "which version said this was a violation" has to stay
    # answerable after the standard has moved on, and `version` alone is just an
    # integer that different files reuse.
    sop_id: str = ""
    # Where it was read from, for the log line and for error messages.
    origin: str = ""

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


def parse_rules(text: str, origin: str, sop_id: str = "") -> SopRuleSet:
    """Turns the YAML into a rule set, or says exactly what is wrong with it.

    `origin` only appears in error messages, but it is the difference between
    "missing required field 'name'" and knowing which of several stored
    versions to go and fix.
    """
    raw = yaml.safe_load(text) or {}
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
            raise ValueError(f"SOP rule in {origin} is missing required field {exc}") from exc
    if not rules:
        raise ValueError(f"No SOP rules found in {origin}")

    targets = []
    for entry in raw.get("visual_scan", []):
        try:
            targets.append(ScanTarget(subject=entry["subject"], question=entry["question"]))
        except KeyError as exc:
            raise ValueError(f"visual_scan entry in {origin} is missing required field {exc}") from exc

    logger.info(
        "Loaded %d SOP rules and %d scan target(s) (v%s) from %s",
        len(rules), len(targets), raw.get("version", 1), origin,
    )
    return SopRuleSet(
        version=int(raw.get("version", 1)),
        rules=rules,
        scan_targets=targets,
        sop_id=sop_id,
        origin=origin,
    )


@lru_cache(maxsize=4)
def load_rules(path: Optional[Path] = None) -> SopRuleSet:
    path = Path(path) if path else (config.sop_rules_path or DEFAULT_RULES_PATH)
    return parse_rules(path.read_text(encoding="utf-8"), origin=path.name)


# Keyed by sop_id. A rule set is a few KB and immutable once parsed, and an
# audit asks for it once per window, so re-reading the bucket every time would
# be hundreds of pointless round trips inside the analysis loop.
_by_id: Dict[str, SopRuleSet] = {}


async def load_rules_for(sop_id: Optional[str] = None) -> SopRuleSet:
    """The standard to judge against, by version id.

    Resolution, in order:

    1. The id the customer chose.
    2. `DEFAULT_SOP_ID`, if the operator set one. That is a deliberate choice
       made at deploy time, not a guess made at audit time.
    3. No id at all: only allowed when no bucket is configured, i.e. a local
       run, where `SOP_RULES_PATH` *is* the standard rather than one of many.

    Anything else raises. In particular, a bucket configured and no id named
    is an error -- picking a version on the customer's behalf is the one thing
    this function must never do.
    """
    wanted = (sop_id or config.default_sop_id or "").strip()

    if not wanted:
        if config.sop_bucket:
            raise SopUnavailable(
                "没有指定稽核标准的版本号。"
                f"标准存放在 gs://{config.sop_bucket}/{config.sop_prefix} 下，"
                "请在请求里带上版本号（或给部署设置 DEFAULT_SOP_ID）。"
            )
        return load_rules()

    if wanted in _by_id:
        return _by_id[wanted]

    if not _SOP_ID_RE.match(wanted):
        raise SopUnavailable(
            f"稽核标准版本号 {wanted!r} 不合法：只允许字母、数字、点、下划线和连字符。"
        )

    if not config.sop_bucket:
        # Named a version on a deployment that has nowhere to keep versions.
        # Falling through to the bundled file here would be exactly the silent
        # substitution this module exists to prevent.
        raise SopUnavailable(
            f"请求了稽核标准 {wanted}，但没有配置 SOP_BUCKET，取不到这个版本。"
        )

    text = await _fetch_sop(wanted)
    rules = parse_rules(text, origin=_sop_uri(wanted), sop_id=wanted)
    _by_id[wanted] = rules
    return rules


def _sop_uri(sop_id: str) -> str:
    prefix = (config.sop_prefix or "").strip("/")
    name = f"{prefix}/{sop_id}.yaml" if prefix else f"{sop_id}.yaml"
    return f"gs://{config.sop_bucket}/{name}"


async def _fetch_sop(sop_id: str) -> str:
    uri = _sop_uri(sop_id)

    def _read() -> str:
        # Lazy, like every other GCS import here: the local path and the test
        # suite run without this package installed.
        from google.cloud import storage

        from ..gcp import credentials_without_quota_project

        client = storage.Client(project=config.gcp_project or None,
                                credentials=credentials_without_quota_project())
        blob = client.bucket(config.sop_bucket).blob(uri.split("/", 3)[3])
        return blob.download_as_text()

    try:
        return await asyncio.to_thread(_read)
    except Exception as exc:
        # Deliberately not caught anywhere upstream that could continue: the
        # caller has to either report this to the customer or stop.
        #
        # 700, not 200. A GCS permission error names the missing permission
        # about 240 characters in, so the old limit cut the sentence off one
        # word before the only part worth reading -- and the half that survived
        # pointed at the wrong role. Long enough to keep the whole IAM message,
        # short enough that a stack-trace-shaped error still gets clipped.
        raise SopUnavailable(f"取不到稽核标准 {uri}：{str(exc)[:700]}") from exc


def clear_sop_cache() -> None:
    """Forgets cached rule sets. For tests, and after publishing a new version."""
    _by_id.clear()
    load_rules.cache_clear()
