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

"""Creates (or deletes) the Phase 0 probe as a ReasoningEngine.

Talks to the REST API directly rather than through google-cloud-aiplatform.
Two reasons: the container-image path wants a very recent SDK that this venv
does not carry, and -- more usefully -- every field below was copied out of the
API's own discovery document, so what gets sent is checkable against the source
of truth instead of against a wrapper's idea of it:

    curl -s "https://aiplatform.googleapis.com/\\$discovery/rest?version=v1"

That document is also where `asyncQuery` turned up, which none of the prose
documentation mentions. Worth re-reading whenever something here surprises you.

    python deploy/phase0/create_engine.py create
    python deploy/phase0/create_engine.py list
    python deploy/phase0/create_engine.py delete <resource-name>
"""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import sys
import tarfile
import urllib.error
import urllib.request

import google.auth
import google.auth.transport.requests

PROJECT = "study-project-496907"

# `global` is the project-wide convention for this agent (see config.py), and it
# is what the audit pipeline already talks to. The Artifact Registry repository
# still has to live in a real region -- images have no global tier.
#
# Overridable because the first `global` deployment failed with a bare INTERNAL
# and no logs, and "does a real region behave differently?" is the cheapest way
# to tell a location restriction apart from a broken container.
LOCATION = os.environ.get("ENGINE_LOCATION", "global")
IMAGE = os.environ.get(
    "ENGINE_IMAGE",
    "us-central1-docker.pkg.dev/study-project-496907/cctv-audit/phase0-probe:v2",
)

# Overridable so a second probe can be stood up alongside the first. Worth it:
# the timeout run takes over an hour, and redeploying the engine underneath it
# would swap the container mid-measurement and quietly invalidate the numbers.
# Two engines off the same image cost nothing and keep the two experiments from
# stepping on each other.
DISPLAY_NAME = os.environ.get("ENGINE_DISPLAY_NAME", "cctv-phase0-probe")

# Host naming for Vertex: regional endpoints are <region>-aiplatform, the global
# one drops the prefix.
HOST = (
    "aiplatform.googleapis.com"
    if LOCATION == "global"
    else f"{LOCATION}-aiplatform.googleapis.com"
)
PARENT = f"projects/{PROJECT}/locations/{LOCATION}"
BASE = f"https://{HOST}/v1/{PARENT}/reasoningEngines"

# The deployment identity. Left unset, the platform uses the Reasoning Engine
# Service Agent, whose role (`roles/aiplatform.reasoningEngineServiceAgent`)
# carries no `artifactregistry.*` permission at all -- so a containerSpec
# deployment cannot pull our own image. The one reasoning engine in this
# project that actually runs sets this field, which is where the idea came
# from. `gce-automation-sa` holds roles/editor, so it can read the registry.
SERVICE_ACCOUNT = os.environ.get("ENGINE_SERVICE_ACCOUNT", "")

# From the runtime contract: `""` (or "async") is unary and lands on
# /api/reasoning_engine; "stream" lands on /api/stream_reasoning_engine. The
# playground refuses to open without a stream_query, which is the only reason
# the probe has one at all.
#
# `description` and `parameters` are annotated in "OpenAPI specification
# format". They matter for `:query` callers and the console playground.
#
# They do NOT matter to Gemini Enterprise, contrary to what this comment used
# to claim: GE never picks a method. It calls exactly one --
# `streaming_agent_run_with_events` -- and hands it the raw conversation turn.
# Measured on the wire; see the README. Routing between operations therefore
# has to happen *inside* the container.
CLASS_METHODS = [
    {
        "name": "hello",
        "api_mode": "",
        "description": "Minimal round trip. Returns the instance id and uptime.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "hang",
        "api_mode": "",
        "description": (
            "Sleeps for `seconds` before replying. Used to find where a long "
            "request gets cut, and by which layer."
        ),
        "parameters": {
            "type": "object",
            "required": ["seconds"],
            "properties": {"seconds": {"type": "number"}},
        },
    },
    {
        "name": "jobs",
        "api_mode": "",
        "description": (
            "Reads the background job table. The readout for the "
            "'does a detached task outlive the 900s request cancel' probe."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "probe_log",
        "api_mode": "",
        "description": (
            "Reads back what the server recorded. The only way to tell a "
            "dropped connection apart from a killed container."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "confirm_flow",
        "api_mode": "",
        "description": (
            "Two-turn handshake. Turn one asks a question and returns a "
            "session id; turn two supplies the answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session": {"type": "string"},
                "answer": {"type": "string"},
            },
        },
    },
    {
        "name": "egress",
        "api_mode": "",
        "description": "Fetches public URLs from inside the container.",
        "parameters": {
            "type": "object",
            "properties": {"targets": {"type": "array", "items": {"type": "string"}}},
        },
    },
    # The one Gemini Enterprise calls -- measured on the wire, not documented.
    # GE ignores every other entry in this list: it does not see preflight /
    # start_audit / get_status as separate tools, it sends one turn of chat
    # here and expects ADK events back. `request_json` is a JSON *string*
    # inside the JSON body, double-encoded.
    {
        "name": "streaming_agent_run_with_events",
        # `async_stream`, not `stream`. Taken from ADK's own deployment
        # template (google/adk/cli/cli_deploy.py), which is the list a real ADK
        # agent registers with.
        "api_mode": "async_stream",
        "description": (
            "The Gemini Enterprise entry point. Takes one conversation turn "
            "and streams ADK events back."
        ),
        "parameters": {
            "type": "object",
            "required": ["request_json"],
            "properties": {"request_json": {"type": "string"}},
        },
    },
    {
        "name": "stream_query",
        "api_mode": "stream",
        "description": "Emits a heartbeat every `interval` seconds for `seconds`.",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number"},
                "interval": {"type": "number"},
            },
        },
    },
]

BODY = {
    "displayName": DISPLAY_NAME,
    "description": (
        "Phase 0 probe: measures request timeout, two-turn confirmation and "
        "egress before the real pipeline is containerised. No business logic."
    ),
    "spec": {
        "agentFramework": "custom",
        "containerSpec": {"imageUri": IMAGE, "port": 8080},
        "classMethods": CLASS_METHODS,
        "deploymentSpec": {
            # The probe does nothing expensive; the real image will need the
            # numbers from the plan. min_instances=1 keeps a cold start from
            # polluting the timeout measurement, which is the whole point.
            "resourceLimits": {"cpu": "1", "memory": "1Gi"},
            "containerConcurrency": 4,
            "minInstances": 1,
            "maxInstances": 2,
        },
    },
}

if SERVICE_ACCOUNT:
    BODY["spec"]["serviceAccount"] = SERVICE_ACCOUNT


HERE = pathlib.Path(__file__).parent
SOURCE_FILES = ["Dockerfile", "probe.py", "requirements.txt"]


def source_archive() -> str:
    """Pack this directory into the base64 tar.gz that `inlineSource` wants.

    This is the alternative to pushing an image ourselves. `containerSpec`
    makes the platform pull from our Artifact Registry, and the identity it
    pulls with -- the Reasoning Engine Service Agent -- holds
    `roles/aiplatform.reasoningEngineServiceAgent`, which grants
    `storage.objects.get` but *no* `artifactregistry.*` permission at all.
    So the pull fails before any container starts: the deployment dies in
    about nine seconds with a bare "failed to start and cannot serve traffic"
    and not one line in Cloud Logging.

    `sourceCodeSpec` sidesteps that. The archive travels in the request, the
    platform builds it with its own Cloud Build, and nothing has to read our
    registry. The Dockerfile is the same one, so what runs is the same thing.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in SOURCE_FILES:
            archive.add(HERE / name, arcname=name)
    return base64.b64encode(buffer.getvalue()).decode()


def source_body() -> dict:
    body = json.loads(json.dumps(BODY))  # deep copy; leave BODY untouched
    spec = body["spec"]
    del spec["containerSpec"]
    # imageSpec (rather than pythonSpec) is what says "there is a Dockerfile in
    # the archive, build that". buildArgs is the only field it has and we need
    # none, but the key has to be present to select this branch.
    spec["sourceCodeSpec"] = {
        "inlineSource": {"sourceArchive": source_archive()},
        "imageSpec": {},
    }
    return body


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
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read() or "{}")
    except urllib.error.HTTPError as exc:
        # The server's own message is far more useful than the status line --
        # it names the offending field. Surface it rather than a stack trace.
        detail = exc.read().decode(errors="replace")
        print(f"HTTP {exc.code}\n{detail}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "create"

    if action in ("create", "create-src"):
        body = source_body() if action == "create-src" else BODY
        print(f"POST {BASE}")
        # The archive is a few kilobytes of base64 and drowns everything else.
        redacted = json.loads(json.dumps(body))
        source = redacted["spec"].get("sourceCodeSpec", {}).get("inlineSource")
        if source:
            source["sourceArchive"] = f"<{len(source['sourceArchive'])} b64 chars>"
        print(json.dumps(redacted, indent=2))
        print()
        operation = call("POST", BASE, body)
        print(json.dumps(operation, indent=2))
        name = operation.get("name", "")
        if name:
            print(f"\nOperation started. Poll it with:\n  python {sys.argv[0]} poll {name}")

    elif action in ("update-image", "update-methods"):
        # Swapping the tag is enough to force a redeploy; there is no separate
        # "restart". The mask has to be this specific -- a broader one wipes
        # sibling fields that are not in the patch body. `update-methods` also
        # re-declares classMethods, which is needed whenever a method is added
        # (the container knowing about it is not enough; the platform routes on
        # this list).
        name = sys.argv[2]
        mask = "spec.container_spec.image_uri"
        body = {"spec": {"containerSpec": {"imageUri": IMAGE, "port": 8080}}}
        if action == "update-methods":
            mask += ",spec.class_methods"
            body["spec"]["classMethods"] = CLASS_METHODS
        url = f"https://{HOST}/v1/{name}?updateMask={mask}"
        print(f"PATCH {url}\n{json.dumps(body, indent=2)}\n")
        print(json.dumps(call("PATCH", url, body), indent=2))

    elif action == "poll":
        print(json.dumps(call("GET", f"https://{HOST}/v1/{sys.argv[2]}"), indent=2))

    elif action == "list":
        result = call("GET", BASE)
        for engine in result.get("reasoningEngines", []):
            print(f"{engine['name']}\n    {engine.get('displayName')}")
        if not result.get("reasoningEngines"):
            print("(none)")

    elif action == "delete":
        print(json.dumps(call("DELETE", f"https://{HOST}/v1/{sys.argv[2]}"), indent=2))

    else:
        raise SystemExit(
            f"unknown action {action!r} (create|create-src|poll|list|delete)"
        )


if __name__ == "__main__":
    main()
