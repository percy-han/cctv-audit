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

"""SOP judgement: one video window in, one structured verdict out."""

from .schema import Finding, Severity, Status, WindowResult, response_schema
from .sop import SopRule, SopRuleSet, SopUnavailable, load_rules, load_rules_for
from .video_analyzer import AnalysisOutcome, VideoAnalyzer

__all__ = [
    "Finding", "Severity", "Status", "WindowResult", "response_schema",
    "SopRule", "SopRuleSet", "SopUnavailable", "load_rules", "load_rules_for",
    "AnalysisOutcome", "VideoAnalyzer",
]
