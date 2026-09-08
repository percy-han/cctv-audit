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

"""Puts the dashboard on Cloud Run, without gcloud.

    python deploy/dashboard/deploy.py v10

The sibling `deploy.sh` does the same thing through the CLI and is easier to
read; this exists for the same reason `agent_runtime/build_image.py` does. The
dev VM's gcloud login expires on its own schedule and cannot be renewed
non-interactively, while Application Default Credentials keep refreshing. Every
call here is ADC plus the Cloud Run Admin API, so a deploy works whether or not
anyone has run `gcloud auth login` lately.

Same image as the audit container, different entrypoint -- two images would be
two builds to keep in step, and the two have already drifted five versions
apart once, which is how a day of dashboard fixes shipped to nothing.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import google.auth
import google.auth.transport.requests

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "study-project-496907")
REGION = os.environ.get("MONITOR_REGION", "us-central1")
SERVICE = os.environ.get("MONITOR_SERVICE", "cctv-monitor")
REPO = os.environ.get("ARTIFACT_REPO", "cctv-audit")
SERVICE_ACCOUNT = os.environ.get(
    "MONITOR_SERVICE_ACCOUNT", f"cctv-monitor@{PROJECT}.iam.gserviceaccount.com")
TOKEN_FILE = os.environ.get("MONITOR_TOKEN_FILE", "auth/monitor-token.txt")

ROOT = Path(__file__).resolve().parents[2]
API = f"https://{REGION}-run.googleapis.com/apis/serving.knative.dev/v1"


def credentials():
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"])
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
        print(f"HTTP {exc.code} on {method} {url}\n"
              f"{exc.read().decode(errors='replace')}", file=sys.stderr)
        raise SystemExit(1)


def monitor_token() -> str:
    path = ROOT / TOKEN_FILE
    if not path.exists():
        raise SystemExit(
            f"No {TOKEN_FILE}. It is the shared secret the agent signs its writes\n"
            f"with, and it stays out of git:\n"
            f"  (umask 077; python -c 'import secrets;print(secrets.token_urlsafe(32))'"
            f" > {TOKEN_FILE})")
    return path.read_text(encoding="utf-8").strip()


def template(tag: str, token: str) -> dict:
    image = f"{REGION}-docker.pkg.dev/{PROJECT}/{REPO}/agent:{tag}"
    return {
        "metadata": {
            "annotations": {
                # One instance, always. The dashboard's state lives in Python
                # dicts, so a second instance means the agent posting frames to
                # one process while the viewer's websocket hangs off another --
                # a permanently blank screen with nothing in the logs to say
                # why. min=1 for the same reason from the other direction:
                # scaling to zero mid-audit throws the findings away.
                "autoscaling.knative.dev/minScale": "1",
                "autoscaling.knative.dev/maxScale": "1",
                # Frames arrive as short POSTs and leave over a websocket. With
                # the default throttling the container only gets full CPU while
                # a request is being handled, which is not the shape of this
                # workload. It is one instance that is always up, so this costs
                # what it was already costing.
                "run.googleapis.com/cpu-throttling": "false",
                # Belt and braces next to maxScale=1: if the cap is ever
                # raised, affinity at least keeps one viewer on one instance
                # instead of flipping them between two half-populated ones.
                "run.googleapis.com/sessionAffinity": "true",
                "run.googleapis.com/startup-cpu-boost": "true",
            },
        },
        "spec": {
            "containerConcurrency": 80,
            # Cloud Run counts a websocket as one long request. The default
            # 300s would drop every viewer every five minutes; the page
            # reconnects, but it flickers.
            "timeoutSeconds": 3600,
            "serviceAccountName": SERVICE_ACCOUNT,
            "containers": [{
                "image": image,
                "command": ["sh"],
                "args": ["-c",
                         "exec uvicorn computer_use_agent.monitor_server:app "
                         "--host 0.0.0.0 --port ${PORT:-8080}"],
                "ports": [{"name": "http1", "containerPort": 8080}],
                "env": [
                    {"name": "MONITOR_TOKEN", "value": token},
                    {"name": "DATA_DIR", "value": "/tmp/checkpoints"},
                    {"name": "MONITOR_REPLAY_HISTORY", "value": "false"},
                ],
                "resources": {"limits": {"cpu": "1", "memory": "1Gi"}},
            }],
        },
    }


def tag_digest(tag: str) -> str:
    """What `agent:<tag>` points at right now, as a sha256.

    Needed because Cloud Run resolves a tag to a digest at deploy time and
    reports the digest forever after, so "does the serving image end in
    :v10" can never be true and is not the question anyway -- a tag can be
    moved. Comparing digests answers the question actually being asked: is the
    thing serving the thing we just built.
    """
    url = (f"https://artifactregistry.googleapis.com/v1/projects/{PROJECT}"
           f"/locations/{REGION}/repositories/{REPO}/packages/agent/tags/{tag}")
    version = call("GET", url).get("version", "")
    return version.rsplit("/", 1)[-1]     # ".../versions/sha256:abc" -> "sha256:abc"


def wait_ready(timeout: int = 300) -> dict:
    """Polls until the *new* revision is serving, or says why it is not.

    A deploy call that returns 200 has only been accepted. Reporting success
    there is how a revision that crash-loops on boot gets called deployed.

    The subtle half is `observedGeneration`: for the first few seconds after a
    PUT the status block still describes the previous revision, Ready=True and
    all. Polling for Ready alone therefore returns immediately and reports the
    old revision's health as the new one's -- the exact failure this function
    exists to catch. Only trust the status once the controller says it has
    caught up with the generation we just wrote, and once the newest revision
    is also the newest *ready* one.
    """
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = call("GET", f"{API}/namespaces/{PROJECT}/services/{SERVICE}")
        status = last.get("status", {})
        generation = last.get("metadata", {}).get("generation")
        if status.get("observedGeneration") == generation:
            conditions = status.get("conditions", [])
            ready = next((c for c in conditions if c.get("type") == "Ready"), {})
            if ready.get("status") == "False":
                raise SystemExit(
                    f"revision not ready: {ready.get('reason')} — {ready.get('message')}")
            if (ready.get("status") == "True"
                    and status.get("latestReadyRevisionName")
                    == status.get("latestCreatedRevisionName")):
                return last
        time.sleep(5)
    raise SystemExit(f"still not ready after {timeout}s; last status:\n"
                     f"{json.dumps(last.get('status', {}), indent=2)[:2000]}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: deploy.py <image-tag>   e.g. deploy.py v10")
    tag = sys.argv[1]
    token = monitor_token()

    url = f"{API}/namespaces/{PROJECT}/services/{SERVICE}"
    service = call("GET", url)
    old = service["spec"]["template"]["spec"]["containers"][0]["image"]

    # Replace the template wholesale rather than patching fields into the old
    # one: a half-updated template is how a service ends up with settings
    # nobody chose. The revision name is dropped so Cloud Run assigns a fresh
    # one -- reusing it is rejected.
    service["spec"]["template"] = template(tag, token)
    service["spec"]["traffic"] = [{"percent": 100, "latestRevision": True}]

    print(f"{SERVICE}: {old.rsplit(':', 1)[-1]} -> {tag}")
    call("PUT", url, service)
    ready = wait_ready()

    # Read the image back off the revision that is actually serving, not off
    # the template we just sent -- checking our own request against itself
    # would pass no matter what Cloud Run did with it.
    revision_name = ready["status"]["latestReadyRevisionName"]
    revision = call(
        "GET", f"{API}/namespaces/{PROJECT}/revisions/{revision_name}")
    live = revision["spec"]["containers"][0]["image"]

    print(f"serving:  {ready['status'].get('url')}")
    print(f"revision: {revision_name}")
    print(f"image:    {live}")
    want = tag_digest(tag)
    if want and not live.endswith(want):
        raise SystemExit(f"serving {live}, but agent:{tag} is {want}")
    print()
    print("It is --no-allow-unauthenticated and the org policy forbids allUsers,")
    print("so viewing it needs one of:")
    print(f"  gcloud run services proxy {SERVICE} --project={PROJECT} "
          f"--region={REGION} --port=9090")
    print("  ...or IAP on the service, which is a separate decision.")


if __name__ == "__main__":
    main()
