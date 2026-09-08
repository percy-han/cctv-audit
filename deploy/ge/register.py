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

"""Puts the audit agent into the Gemini Enterprise app the customer opens.

Registration is one POST: an `Agent` resource whose `adkAgentDefinition
.provisionedReasoningEngine.reasoningEngine` is the resource name of our Agent
Runtime deployment. There is no publishing step and no packaging step, and the
API does not check that the name exists -- a typo produces a working-looking
agent that answers nothing.

Two things the plan assumed that turned out not to hold, both worth reading
before changing anything here:

  * **There is no Instructions field.** Measured against the v1alpha discovery
    document: `Agent` carries `displayName`, `description`, `languageCode`,
    `starterPrompts`, `customPlaceholderText`, `icon`, `sharingConfig`,
    `authorizationConfig` and one of four `*AgentDefinition` blocks. Only
    `managedAgentDefinition` -- GE's own model-driven agent -- has prompt
    fields. An `adkAgentDefinition` agent is a *route*, not a reasoner: GE
    hands the turn straight to the container.

    So the plan's "GE does the first-pass filter" cannot be done in GE for
    this agent type. It is done in the container instead, by `turn.py`, with
    a model rather than a keyword match. Nothing is lost; the filter just
    lives one hop further in.

  * **Chips cannot carry hidden parameters.** `starterPrompts` items have a
    `text` field and nothing else, so the SOP version has to be inside the
    sentence the customer can see -- and can edit. That is exactly why
    `analyzer/sop.py` refuses an unknown id instead of falling back.

    "chagee" and "v1" are visible in the chips below on purpose.

Usage:

    python deploy/ge/register.py list-apps
    python deploy/ge/register.py list
    python deploy/ge/register.py create <reasoningEngine-resource-name>
    python deploy/ge/register.py update <agent-resource-name> [engine-name]
    python deploy/ge/register.py delete <agent-resource-name>

`GE_APP_ID` picks the app. It defaults to this project's demo app; **at
delivery it must be the app the customer already opens**, because agents are
listed per app and the two lists cannot see each other.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

import google.auth
import google.auth.transport.requests

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "study-project-496907")

# Gemini Enterprise is global-only here: asking for `us` answers "The current
# endpoint can only serve traffic from global region".
LOCATION = "global"
COLLECTION = "default_collection"
ASSISTANT = "default_assistant"

APP_ID = os.environ.get("GE_APP_ID", "cctv-audit")
SOP_ID = os.environ.get("DEFAULT_SOP_ID", "chagee-store-v1")

HOST = "discoveryengine.googleapis.com"
BASE = (f"https://{HOST}/v1alpha/projects/{PROJECT}/locations/{LOCATION}"
        f"/collections/{COLLECTION}")


def agent_body(reasoning_engine: str) -> dict:
    """The Agent resource GE stores.

    `displayName` is what the customer types after `@`, so it has to be short
    and guessable in Chinese. `description` is not decoration either -- the
    schema says both "might be used by an LLM to automatically select an agent
    to respond to a user query".
    """
    return {
        "displayName": "门店视频稽核",
        "description": (
            "对门店监控录像做合规稽核。给它一个视频网址和一个时间段，它会先打开"
            "看一眼——视频在不在、能不能取到画面、这个时间段够不够——把结果说给你"
            "听并等你确认；确认之后在后台逐段分析，按指定版本的门店标准作业规范"
            f"（例如 {SOP_ID}）判定违规，出报告。凡是提到门店监控、录像、稽核、"
            "违规检查的请求都交给它。"
        ),
        "languageCode": "zh-CN",
        "adkAgentDefinition": {
            "provisionedReasoningEngine": {"reasoningEngine": reasoning_engine},
        },
        # The chips. Each is a whole sentence the customer can send as-is after
        # pasting a URL, because a chip that only sets up half a request just
        # moves the typing somewhere else.
        #
        # The version string is in the text because there is nowhere else to
        # put it (see the module docstring). A customer who edits it to
        # something that does not exist gets a refusal naming the version, not
        # an audit judged against a guess.
        #
        # The spans are two minutes, not fifteen. A chip is the sentence most
        # people will send unedited, and the measured cost of a span is roughly
        # a minute of wall clock per two minutes of footage on the stream path
        # and twice that on the screen-recording path -- so a fifteen-minute
        # chip is a chip nobody watches finish. Two minutes is ten windows,
        # which is enough to see the thing work; a real audit is the same
        # sentence with a bigger number, and the customer can type that once
        # they believe it.
        "starterPrompts": [
            {"text": f"按标准 {SOP_ID} 稽核这段录像：<在这里粘贴视频网址>，"
                     f"从 05:00 开始看 2 分钟"},
            {"text": f"按标准 {SOP_ID} 稽核：<在这里粘贴视频网址>，"
                     f"05:00 到 07:00"},
            {"text": "确认，开始稽核"},
            {"text": "刚才那单现在怎么样了？"},
        ],
        "customPlaceholderText": (
            "粘贴视频网址，说清时间段，例如：稽核 https://... 从 05:00 看 2 分钟"
        ),
    }


def credentials():
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds


def call(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {credentials().token}")
    request.add_header("Content-Type", "application/json")
    # Without this: 403 "requires a quota project, which is not set by
    # default". discoveryengine bills the caller and will not guess who.
    request.add_header("X-Goog-User-Project", PROJECT)
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read() or "{}")
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}\n{exc.read().decode(errors='replace')}",
              file=sys.stderr)
        raise SystemExit(1)


def agents_url() -> str:
    return f"{BASE}/engines/{APP_ID}/assistants/{ASSISTANT}/agents"


def show(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "list"

    if action == "list-apps":
        for engine in call("GET", f"{BASE}/engines").get("engines", []):
            print(f"{engine['name'].split('/')[-1]:40s} {engine.get('displayName')}")

    elif action == "list":
        result = call("GET", agents_url())
        for agent in result.get("agents", []):
            engine = (agent.get("adkAgentDefinition", {})
                      .get("provisionedReasoningEngine", {})
                      .get("reasoningEngine", "(not an ADK agent)"))
            print(f"{agent['name']}\n    {agent.get('displayName')}  "
                  f"state={agent.get('state')}\n    -> {engine}")
        if not result.get("agents"):
            print(f"(no agents in app {APP_ID})")

    elif action == "create":
        body = agent_body(sys.argv[2])
        print(f"POST {agents_url()}")
        show(body)
        print()
        show(call("POST", agents_url(), body))

    elif action == "update":
        name = sys.argv[2]
        current = call("GET", f"https://{HOST}/v1alpha/{name}")
        engine = sys.argv[3] if len(sys.argv) > 3 else (
            current.get("adkAgentDefinition", {})
            .get("provisionedReasoningEngine", {})
            .get("reasoningEngine", ""))
        if not engine:
            raise SystemExit("no reasoning engine on that agent; pass one")
        body = agent_body(engine)
        mask = ("displayName,description,starterPrompts,customPlaceholderText,"
                "adkAgentDefinition")
        url = f"https://{HOST}/v1alpha/{name}?updateMask={mask}"
        print(f"PATCH {url}")
        show(call("PATCH", url, body))

    elif action == "delete":
        show(call("DELETE", f"https://{HOST}/v1alpha/{sys.argv[2]}"))

    else:
        raise SystemExit(
            f"unknown action {action!r} (list-apps|list|create|update|delete)")


if __name__ == "__main__":
    main()
