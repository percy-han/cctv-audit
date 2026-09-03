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

"""Creates a Gemini Enterprise app and registers the probe agent into it.

This is Phase 0 probe #2 -- "can GE mount a custom agent that lives on Agent
Runtime, and how is it registered?". The plan listed that as undocumented. It
is: the answer is not in any prose page, it is in the v1alpha discovery
document, under

    projects.locations.collections.engines.assistants.agents.create

whose Agent resource carries `adkAgentDefinition.provisionedReasoningEngine`
-- a plain pointer at a reasoningEngines/... resource name. So a GE app links
to an Agent Runtime deployment by resource name; there is no separate
publishing or packaging step.

Two other fields on that resource matter to the plan and are worth noting
here, because they were open questions:

  * `starterPrompts` -- these are the Prompt chips. The schema says each one
    is `{"text": ...}` and nothing else. There is no slot for a hidden
    parameter, so a `sop_id` has to travel *inside the visible text*.
  * `authorizationConfig` -- `agentAuthorization` rides in the auth header,
    `toolAuthorizations` ride in the request body. Relevant once the CCTV
    console needs credentials; not used in v1.

Everything here goes through the REST API rather than a client library, for
the same reason as create_engine.py: the discovery document is the only place
these fields are described at all, so what gets sent should be checkable
against it directly.

    curl -s "https://discoveryengine.googleapis.com/\\$discovery/rest?version=v1alpha"

Usage:

    python deploy/phase0/create_ge_app.py create-app
    python deploy/phase0/create_ge_app.py list-apps
    python deploy/phase0/create_ge_app.py create-agent <reasoningEngine-name>
    python deploy/phase0/create_ge_app.py list-agents
    python deploy/phase0/create_ge_app.py delete-app
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

import google.auth
import google.auth.transport.requests

PROJECT = "study-project-496907"

# Gemini Enterprise is global-only in this project: asking for `us` comes back
# with "The current endpoint can only serve traffic from global region". Not a
# choice we get to make.
LOCATION = "global"
COLLECTION = "default_collection"

APP_ID = os.environ.get("GE_APP_ID", "cctv-audit")
APP_DISPLAY_NAME = "CCTV 视频稽核"

HOST = "discoveryengine.googleapis.com"
PARENT = f"projects/{PROJECT}/locations/{LOCATION}/collections/{COLLECTION}"
BASE = f"https://{HOST}/v1alpha/{PARENT}"

# GE apps hang their agents off an assistant. Every app gets this one for free
# when it is created; we never create it ourselves.
ASSISTANT = "default_assistant"

AGENT_ID_HINT = "cctv-phase0-probe"

# Shape copied from the app the console already created in this project, minus
# everything the server fills in on its own. The two that are not cosmetic:
# APP_TYPE_INTRANET is what makes it a Gemini Enterprise app rather than a bare
# search app, and SEARCH_ADD_ON_LLM is what turns on the assistant.
APP_BODY = {
    "displayName": APP_DISPLAY_NAME,
    "solutionType": "SOLUTION_TYPE_SEARCH",
    "appType": "APP_TYPE_INTRANET",
    "industryVertical": "GENERIC",
    "searchEngineConfig": {
        "searchTier": "SEARCH_TIER_ENTERPRISE",
        "searchAddOns": ["SEARCH_ADD_ON_LLM"],
    },
    "commonConfig": {"companyName": "CHAGEE"},
}


def agent_body(reasoning_engine: str) -> dict:
    """The Agent resource that points GE at our Agent Runtime deployment.

    `displayName` and `description` are not just labels -- the schema says both
    "might be used by an LLM to automatically select an agent to respond to a
    user query", so they are routing input. Written accordingly.
    """
    return {
        "displayName": "CCTV 稽核探针",
        "description": (
            "Phase 0 probe for the CCTV audit pipeline. Measures request "
            "timeout, two-turn confirmation and outbound network access. "
            "Runs no audit and analyses no video."
        ),
        "languageCode": "zh-CN",
        "adkAgentDefinition": {
            "provisionedReasoningEngine": {"reasoningEngine": reasoning_engine}
        },
        # Stand-ins for the real SOP chips. Phase 4 replaces these; what they
        # prove now is only whether chips can carry a machine-readable token
        # at all, since the schema gives them no field but `text`.
        "starterPrompts": [
            {"text": "探针自检：说一声 hello"},
            {"text": "挂 60 秒不返回，测超时"},
            {"text": "从容器里访问一次 bilibili，测出网"},
        ],
        "customPlaceholderText": "输入门店和时间段，例如：XX 店 昨天 14:00-15:00",
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
    # Without this the API refuses with 403 "requires a quota project, which is
    # not set by default" -- discoveryengine bills the caller's project and
    # will not guess which one that is.
    request.add_header("X-Goog-User-Project", PROJECT)
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read() or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"HTTP {exc.code}\n{detail}", file=sys.stderr)
        raise SystemExit(1)


def agents_url() -> str:
    return f"{BASE}/engines/{APP_ID}/assistants/{ASSISTANT}/agents"


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "list-apps"

    if action == "create-app":
        url = f"{BASE}/engines?engineId={urllib.parse.quote(APP_ID)}"
        print(f"POST {url}")
        print(json.dumps(APP_BODY, indent=2, ensure_ascii=False))
        print()
        print(json.dumps(call("POST", url, APP_BODY), indent=2, ensure_ascii=False))

    elif action == "list-apps":
        for engine in call("GET", f"{BASE}/engines").get("engines", []):
            print(f"{engine['name']}\n    {engine.get('displayName')}  "
                  f"{engine.get('solutionType')} {engine.get('appType')}")

    elif action == "create-agent":
        body = agent_body(sys.argv[2])
        print(f"POST {agents_url()}")
        print(json.dumps(body, indent=2, ensure_ascii=False))
        print()
        print(json.dumps(call("POST", agents_url(), body), indent=2, ensure_ascii=False))

    elif action == "list-agents":
        result = call("GET", agents_url())
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif action == "delete-agent":
        print(json.dumps(call("DELETE", f"https://{HOST}/v1alpha/{sys.argv[2]}"), indent=2))

    elif action == "delete-app":
        print(json.dumps(call("DELETE", f"{BASE}/engines/{APP_ID}"), indent=2))

    else:
        raise SystemExit(
            f"unknown action {action!r} "
            "(create-app|list-apps|create-agent|list-agents|"
            "delete-agent|delete-app)"
        )


if __name__ == "__main__":
    main()
