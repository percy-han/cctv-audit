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

"""Deploys the audit container as a ReasoningEngine.

Same shape as `deploy/phase0/create_engine.py` and for the same reason: every
field below was copied out of the API's own discovery document rather than out
of a wrapper, so it can be checked against the source of truth --

    curl -s "https://aiplatform.googleapis.com/\\$discovery/rest?version=v1"

Build the image first (there is no Docker on the dev VM, and pushing ~2 GB
from it would be slow anyway):

    gcloud builds submit \\
      --tag us-central1-docker.pkg.dev/<project>/cctv-audit/agent:v1 \\
      --region=us-central1 --timeout=1800s

Then:

    python deploy/agent_runtime/deploy.py create
    python deploy/agent_runtime/deploy.py poll <operation-name>
    ENGINE_IMAGE=v9 python deploy/agent_runtime/deploy.py update-image <engine-id>
    python deploy/agent_runtime/deploy.py list
    python deploy/agent_runtime/deploy.py delete <engine-id>

`ENGINE_IMAGE` takes a bare tag or a full image reference; `<engine-id>` takes
the bare id or the full `projects/.../reasoningEngines/<id>`. Both short forms
exist so the command fits on one line -- a wrapped paste of the long one has
already patched an engine with the wrong image.

`update-methods` re-declares classMethods as well. Needed whenever a method is
added: the container knowing about it is not enough, the platform routes on
this list.
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

# NOT `global`, even though `global` is the project-wide default for Gemini
# calls (see config.py). Measured 2026-09-03: `global` accepts a BYOC create,
# runs for about sixty seconds, and then fails the operation with
#
#     "code": 13, "message": "Please refer to our troubleshooting pages ..."
#
# and not one log line anywhere -- no `aiplatform.googleapis.com/ReasoningEngine`
# resource is ever created, so there is nothing to filter Logs Explorer by.
# Three separate deployments failed identically: with and without a custom
# service account, with and without a deploymentSpec. The one thing they had
# in common was the location. Every engine that has ever actually run in this
# project, including both Phase 0 probes, is in us-central1.
#
# The two locations are unrelated settings: this is where the engine runs, and
# `config.gcp_location` is which Gemini endpoint the container calls.
LOCATION = os.environ.get("ENGINE_LOCATION", "us-central1")
_REPO = f"us-central1-docker.pkg.dev/{PROJECT}/cctv-audit/agent"

# A bare tag is expanded to the repo above; anything with a slash is taken as a
# full image reference and left alone. The full reference is 68 characters, and
# a wrapped paste of it has already cost one failed deploy -- the shell ran the
# assignment as its own command, the variable did not survive, and the engine
# was patched with the default tag instead of the one that was asked for. The
# short form fits on a line.
# `.get(..., "v4")` is not enough: a wrapped paste can leave ENGINE_IMAGE set to
# the empty string, which would build the tagless `agent:` and fail somewhere
# far from here.
IMAGE = os.environ.get("ENGINE_IMAGE") or "v4"
if "/" not in IMAGE:
    IMAGE = f"{_REPO}:{IMAGE}"
DISPLAY_NAME = os.environ.get("ENGINE_DISPLAY_NAME", "cctv-audit-agent")

# The deployment identity. Unset, the platform uses the Reasoning Engine
# Service Agent, whose role carries no `artifactregistry.*` permission at all,
# so it cannot pull our image: the deployment dies in about nine seconds with
# "failed to start and cannot serve traffic" and not one line in Cloud Logging.
# Whichever account is used here needs `roles/artifactregistry.reader` --
# put that on the customer's deployment checklist too.
#
# It also needs, and this is the list the customer's platform team will ask for:
#   roles/datastore.user           job state in Firestore
#   roles/storage.objectAdmin      on the artifacts bucket: SOP in, evidence out
#   roles/aiplatform.user          calling Gemini
#   roles/run.invoker              on the dashboard service, else every frame
#                                  is a 403 that nothing surfaces (all
#                                  dashboard writes are fire-and-forget)
SERVICE_ACCOUNT = os.environ.get(
    "ENGINE_SERVICE_ACCOUNT",
    f"cctv-audit-agent@{PROJECT}.iam.gserviceaccount.com",
)

HOST = (
    "aiplatform.googleapis.com"
    if LOCATION == "global"
    else f"{LOCATION}-aiplatform.googleapis.com"
)
BASE = f"https://{HOST}/v1/projects/{PROJECT}/locations/{LOCATION}/reasoningEngines"


MONITOR_SERVICE = os.environ.get("MONITOR_SERVICE", "cctv-monitor")
MONITOR_REGION = os.environ.get("MONITOR_REGION", "us-central1")
MONITOR_TOKEN_FILE = os.environ.get("MONITOR_TOKEN_FILE", "auth/monitor-token.txt")
# The demo footage. Not part of the product -- it is there so the agent has
# something to open that is not a third-party site with its own rate limits and
# its own bad days. Absent in a customer deployment, which is why every use of
# it tolerates "".
DEMO_VIDEO_SERVICE = os.environ.get("DEMO_VIDEO_SERVICE", "cctv-demo-video")


# Defined up here, above the settings, because `_service_url` asks Cloud Run a
# question while the settings are still being assembled at import time.
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
        # The server names the offending field; a stack trace does not.
        print(f"HTTP {exc.code}\n{exc.read().decode(errors='replace')}", file=sys.stderr)
        raise SystemExit(1)


def _service_url(service: str) -> str:
    """Asks Cloud Run where a service is, rather than pasting a URL.

    Cloud Run URLs contain a project-hash nobody can guess, so hardcoding one
    means it is wrong in the customer's project and wrong again after a
    delete/recreate. Returns "" when the service is not deployed, which is a
    legitimate state for both callers: the audit runs perfectly well with
    nobody watching, and the demo footage is optional everywhere.

    This used to shell out to `gcloud run services describe`, and that is how
    an engine got deployed with no MONITOR_URL at all: the dev VM's gcloud
    login had expired, the subprocess failed, "" is a legal answer, and the
    variable was dropped without a word. Same ADC the rest of this file uses,
    so it cannot fail for a reason unrelated to Cloud Run -- and it says so out
    loud when it does come back empty.
    """
    url = (f"https://{MONITOR_REGION}-run.googleapis.com/apis/serving.knative.dev"
           f"/v1/namespaces/{PROJECT}/services/{service}")
    try:
        found = call("GET", url).get("status", {}).get("url", "")
    except SystemExit:      # `call` prints the HTTP body before raising
        found = ""
    if not found:
        print(f"warning: could not resolve the URL of Cloud Run service "
              f"{service!r}; anything depending on it is being left unset",
              file=sys.stderr)
    return found


def _monitor_url() -> str:
    return _service_url(MONITOR_SERVICE)


def _monitor_token() -> str:
    """The shared secret, from a gitignored file under `auth/`.

    Not Secret Manager yet, and that is a deliberate first-version limit: this
    puts the token in the engine's environment, where anyone with
    `aiplatform.reasoningEngines.get` can read it. Acceptable while the only
    thing it protects is a demo dashboard; not acceptable once a real store's
    footage is on the other end. See the delivery notes.
    """
    try:
        with open(MONITOR_TOKEN_FILE, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _redacted(body: dict) -> str:
    """The request, minus anything that must not land in a terminal scrollback."""
    printable = json.loads(json.dumps(body))
    for entry in printable.get("spec", {}).get("deploymentSpec", {}).get("env", []):
        if "TOKEN" in entry.get("name", "") and entry.get("value"):
            entry["value"] = "***"
    return json.dumps(printable, indent=2)


# ----------------------------------------------------------------------
# What the container runs with.
#
# Empty values are left OUT rather than sent as "": config.py reads an empty
# string as "stay local", and a half-configured cloud deployment that quietly
# writes to a temporary disk is the failure mode this whole phase exists to
# remove. Better to have the deployment be obviously local than subtly lossy.
# ----------------------------------------------------------------------
def env_vars() -> list:
    settings = {
        # No GOOGLE_CLOUD_PROJECT here. The platform sets it itself and rejects
        # the deployment outright if we also send one:
        #   400 FAILED_PRECONDITION -- Environment variable name
        #   'GOOGLE_CLOUD_PROJECT' is reserved.
        # Same for GOOGLE_CLOUD_REGION and the other GOOGLE_CLOUD_* names.
        #
        # What it sets is the project *number*, and a named Firestore database
        # cannot be addressed by number -- the client gets a 404 saying the
        # database does not exist while it plainly does. Hence GCP_PROJECT,
        # which config.py reads first. See the comment there.
        "GCP_PROJECT": PROJECT,
        "GOOGLE_GENAI_USE_VERTEXAI": "TRUE",
        # State that has to outlive an instance. Without these the audit still
        # runs and then evaporates when the container is recycled.
        #
        # A *named* database, not "(default)": this project's default database
        # is in Datastore mode, and the Firestore API refuses it outright --
        # "The Cloud Firestore API is not available for Firestore in Datastore
        # Mode database". A customer project may well be the same, so the
        # deployment checklist has to say "Firestore-native", not "Firestore".
        "FIRESTORE_DATABASE": os.environ.get("FIRESTORE_DATABASE", "cctv-audit"),
        "ARTIFACTS_BUCKET": os.environ.get(
            "ARTIFACTS_BUCKET", f"{PROJECT}-cctv-audit"),
        "SOP_BUCKET": os.environ.get("SOP_BUCKET", f"{PROJECT}-cctv-audit"),
        # Which standard applies when the customer names none. config.py has no
        # built-in default on purpose -- a verdict judged against an unnamed
        # standard cannot be defended later -- so the *deployment* names one,
        # and every record carries it.
        "DEFAULT_SOP_ID": os.environ.get("DEFAULT_SOP_ID", "chagee-store-v1"),
        # Nobody is watching a cloud container, so a gate that waits for a
        # human is a job that hangs until the platform kills it. `off` fails
        # the run the moment a challenge appears and says which credential was
        # missing -- see config.py. (`auto`, the default, probes the dashboard
        # first; here the answer is known in advance.)
        "HUMAN_GATE_MODE": "off",
        # Where the dashboard is. Empty means "127.0.0.1", which in a container
        # with no dashboard in it means every frame is posted into the void --
        # silently, because dashboard writes are fire-and-forget by design.
        # So: if this is unset, the deployment is *visibly* without a dashboard
        # rather than subtly so.
        "MONITOR_URL": os.environ.get("MONITOR_URL", _monitor_url()),
        # The shared secret `monitor_server` checks. Cloud Run's own IAM check
        # happens first and is the real gate; this one is what stops anything
        # that gets past it from writing to the page.
        "MONITOR_TOKEN": os.environ.get("MONITOR_TOKEN", _monitor_token()),
        # Pinned rather than inherited. The picture on the wall is the part of
        # this system a customer actually looks at, and leaving it to whatever
        # `config.py` happens to default to is how it silently shipped at
        # 640x360 for a demo on a 1080p screen. If these change, they change
        # here, in the deployment, deliberately.
        # How a window is analysed. Pinned here for the same reason the preview
        # settings below are: these are a *deployment* decision, and the three
        # of them are one decision, not three.
        #
        #   agentic  -- the model drives its own video tool and picks what to
        #               look at, instead of us handing it a 1 fps frame ladder
        #   60s      -- the window it gets to search
        #   3.8-flash -- agentic needs the video understanding tool;
        #               gemini-3.5-flash refuses outright (measured), and
        #               config.validate() now fails the boot rather than
        #               failing every window one at a time
        #
        # Cost, measured 2026-09-08 per five minutes of footage (see the table
        # in config.py): 60s agentic is 60,600 tokens against 36,920 for 60s
        # static -- 64% more. Agentic only comes out ahead at 300s windows,
        # where it is 17,800 against 23,226. This combination is chosen for
        # what it does to the *verdicts*, not to the bill; if the bill is what
        # matters, the change is WINDOW_SECONDS=300, not MEDIA_PROCESSING.
        "MEDIA_PROCESSING": os.environ.get("MEDIA_PROCESSING", "agentic"),
        "WINDOW_SECONDS": os.environ.get("WINDOW_SECONDS", "60"),
        "ANALYSIS_MODEL": os.environ.get("ANALYSIS_MODEL", "gemini-3.8-flash"),
        "PREVIEW_WIDTH": os.environ.get("PREVIEW_WIDTH", "1280"),
        "PREVIEW_HEIGHT": os.environ.get("PREVIEW_HEIGHT", "720"),
        "PREVIEW_FPS": os.environ.get("PREVIEW_FPS", "12"),
        "PREVIEW_QUALITY": os.environ.get("PREVIEW_QUALITY", "60"),
        # Origins the container may present its own Google identity to. The org
        # policy here forbids `allUsers` on anything, so the demo footage sits
        # behind Cloud Run IAM and the browser has to authenticate to read it.
        #
        # An allow-list rather than "authenticate when challenged", because the
        # same browser also opens whatever URL the customer pasted. A site that
        # 401s is not thereby entitled to this deployment's service account
        # token. Unset in a deployment with no demo service, and then nothing
        # anywhere gets one.
        "OIDC_ORIGINS": os.environ.get(
            "OIDC_ORIGINS", _service_url(DEMO_VIDEO_SERVICE)),
        # How long `/is_busy` holds its answer while an audit is running, which
        # is what keeps a request in flight and therefore a CPU allocated. See
        # that endpoint and `containerConcurrency` above.
        #
        # 25s rather than the probe's full hour: short enough that a finished
        # job frees the slot promptly and a wedged handler cannot hold it
        # forever, long enough that the gaps between probes are a rounding
        # error next to a four-minute stall. Set it to 0 to answer instantly
        # and measure the platform's own behaviour without the hold -- that
        # comparison is why this is an environment variable and not a constant.
        "KEEPALIVE_HOLD_SECONDS": os.environ.get("KEEPALIVE_HOLD_SECONDS", "25"),
    }
    return [
        {"name": key, "value": value}
        for key, value in settings.items()
        if value
    ]


# ----------------------------------------------------------------------
# The methods.
#
# GE calls exactly one of these -- `streaming_agent_run_with_events` -- and
# ignores every description below (measured on the wire in Phase 0; see
# deploy/phase0/README.md). The other three are reachable by a direct `:query`
# and exist for debugging and for a non-GE caller. They are the same code path
# either way; the container routes GE's turn to them itself.
# ----------------------------------------------------------------------
CLASS_METHODS = [
    {
        "name": "streaming_agent_run_with_events",
        # `async_stream`, not `stream` -- taken from ADK's own deployment
        # template (google/adk/cli/cli_deploy.py).
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
        "name": "preflight",
        "api_mode": "",
        "description": (
            "Opens the target and reports what an audit would actually get: "
            "whether the video exists, stream capture or screen recording, "
            "how long it is, whether the requested span is covered. Seconds, "
            "and it starts nothing."
        ),
        "parameters": {
            "type": "object",
            "required": ["user_id", "target"],
            "properties": {
                "user_id": {"type": "string"},
                "target": {"type": "string"},
                "start": {"type": "number"},
                "end": {"type": "number"},
                "duration": {"type": "number"},
                "session_id": {"type": "string"},
                "sop_id": {"type": "string"},
            },
        },
    },
    {
        "name": "start_audit",
        "api_mode": "",
        "description": (
            "Starts the audit for a job that preflight approved. Returns "
            "immediately; the work runs in the background."
        ),
        "parameters": {
            "type": "object",
            "required": ["user_id", "job_id"],
            "properties": {
                "user_id": {"type": "string"},
                "job_id": {"type": "string"},
            },
        },
    },
    {
        "name": "get_status",
        "api_mode": "",
        "description": (
            "Progress, or the finished report. Looks up by job_id, or by "
            "session_id for the most recent job in a conversation."
        ),
        "parameters": {
            "type": "object",
            "required": ["user_id"],
            "properties": {
                "user_id": {"type": "string"},
                "job_id": {"type": "string"},
                "session_id": {"type": "string"},
            },
        },
    },
]

DEPLOYMENT_SPEC = {
    # From the plan. cpu is one of 1/2/4/6/8 only.
    #   4 cpu   Chromium, ffmpeg and three concurrent analyses at once
    #   8Gi     Chromium is the appetite here; --disable-dev-shm-usage is
    #           already set and does not make it small
    "resourceLimits": {"cpu": "4", "memory": "8Gi"},
    # One browser per container, plus one slot for the keep-alive probe.
    #
    # This was 1, for the reason that still holds: a second concurrent turn
    # would fight the first for CPU and shared memory, and the loser is a
    # half-recorded audit. It bounds *turns in flight*, not audits -- audits
    # run detached, so a container serving a two-second status turn is free
    # again immediately.
    #
    # It is 2 because `/is_busy` holds its response open for the length of an
    # audit (see the endpoint's docstring: an unattended instance is denied a
    # CPU 79-85% of the time, measured off `/proc/self/schedstat`). That held
    # probe permanently occupies one slot, so at 1 it would crowd out every
    # real turn and Gemini Enterprise would get 429s. Two slots minus the
    # probe leaves exactly the one-real-turn-at-a-time guarantee this had
    # before. It is arithmetic, not a tuning preference: raise the hold and
    # this must stay at least 2.
    "containerConcurrency": 2,
    # The platform's own lever for an instance that has work to do but no
    # request in flight -- which is every audit, because they are detached to
    # survive Gemini Enterprise cancelling in-request work at 900s.
    #
    # Undocumented: the runtime-contract and optimize-and-scale pages say
    # nothing about probes or CPU allocation, and the only description is the
    # one line in the API discovery document. `/is_busy` is not our invention
    # either -- it is the example path in that schema. Whether being probed
    # restores the CPU or merely stops the instance being reclaimed is the
    # thing the hold in the endpoint is there to make true regardless.
    #
    # 3600 is the documented maximum, and it is also the pipeline's own budget,
    # so a probe cannot outlive the longest audit we will run.
    "keepAliveProbe": {
        "httpGet": {"path": "/is_busy", "port": 8080},
        "maxSeconds": 3600,
    },
    # 0 would be cheaper, but a detached background audit dies with the
    # instance that owns it. Keeping one warm also spares the customer a cold
    # start on the very first turn of a demo.
    "minInstances": 1,
    "maxInstances": 5,
    "env": env_vars(),
}

BODY = {
    "displayName": DISPLAY_NAME,
    "description": (
        "Store CCTV audit. Opens the recording, captures the requested span, "
        "and judges it against a versioned SOP."
    ),
    "spec": {
        "agentFramework": "custom",
        "containerSpec": {"imageUri": IMAGE, "port": 8080},
        "classMethods": CLASS_METHODS,
        "deploymentSpec": DEPLOYMENT_SPEC,
    },
}

if SERVICE_ACCOUNT:
    BODY["spec"]["serviceAccount"] = SERVICE_ACCOUNT


def _engine_name(argv_index: int) -> str:
    """The engine resource name from the command line, short form allowed.

    A bare id is expanded against this file's PROJECT and LOCATION; anything
    containing a slash is used as given, so a name copied out of `list` still
    works. Two engines exist in this project, so there is deliberately no
    default -- guessing which one to patch is not a favour.
    """
    try:
        name = sys.argv[argv_index]
    except IndexError:
        raise SystemExit(
            f"{sys.argv[1]} needs an engine: either the bare id or the full\n"
            f"projects/.../reasoningEngines/<id>. `list` prints both.\n"
            "\n"
            "If you just pasted a long command and got here, check it did not\n"
            "arrive split across lines -- it has to be one line."
        )
    if "/" in name:
        return name
    return f"projects/{PROJECT}/locations/{LOCATION}/reasoningEngines/{name}"


def main() -> None:
    # No default action. `create` used to be the default, so running this file
    # with no arguments -- the natural way to ask a script what it wants --
    # silently stood up a second engine. Deploying is not what "no arguments"
    # means.
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    action = sys.argv[1]

    if action == "create":
        print(f"POST {BASE}\n{_redacted(BODY)}\n")
        operation = call("POST", BASE, BODY)
        print(json.dumps(operation, indent=2))
        name = operation.get("name", "")
        if name:
            print(f"\nPoll it with:\n  python {sys.argv[0]} poll {name}")

    elif action in ("update-image", "update-methods", "update-spec"):
        name = _engine_name(2)
        mask = "spec.container_spec.image_uri"
        body: dict = {"spec": {"containerSpec": {"imageUri": IMAGE, "port": 8080}}}
        if action in ("update-methods", "update-spec"):
            mask += ",spec.class_methods"
            body["spec"]["classMethods"] = CLASS_METHODS
        if action == "update-spec":
            # Env vars live here, so this is the one to use after changing a
            # bucket name. The mask has to name the field: a broader one wipes
            # siblings that are not in the patch body.
            mask += ",spec.deployment_spec"
            body["spec"]["deploymentSpec"] = DEPLOYMENT_SPEC
        url = f"https://{HOST}/v1/{name}?updateMask={mask}"
        print(f"PATCH {url}\n{_redacted(body)}\n")
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
        print(json.dumps(call("DELETE", f"https://{HOST}/v1/{_engine_name(2)}"), indent=2))

    else:
        raise SystemExit(
            f"unknown action {action!r} "
            "(create|update-image|update-methods|update-spec|poll|list|delete)"
        )


if __name__ == "__main__":
    main()
