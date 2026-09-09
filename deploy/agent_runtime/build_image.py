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

"""Builds the agent image, without gcloud.

    python deploy/agent_runtime/build_image.py v9

`gcloud builds submit` does exactly this and is the normal way to do it. This
script exists because the CLI's own login expires on its own schedule -- the
dev VM has been sitting at "Reauthentication failed. cannot prompt during
non-interactive execution" for a day -- while Application Default Credentials
keep refreshing fine. Everything here goes through ADC, so a build works
whether or not anyone has run `gcloud auth login` lately.

Three steps, no magic: tar the working tree minus `.gcloudignore`, put the tar
in the artifacts bucket, and ask Cloud Build to build it. Then it waits, which
is the part worth having -- a build id and no follow-up is how a stale image
ends up deployed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import google.auth
import google.auth.transport.requests

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "study-project-496907")
REGION = os.environ.get("BUILD_REGION", "us-central1")
REPO = os.environ.get("ARTIFACT_REPO", "cctv-audit")
BUCKET = os.environ.get("ARTIFACTS_BUCKET", f"{PROJECT}-cctv-audit")
ROOT = Path(__file__).resolve().parents[2]

# Long enough for a Chromium + ffmpeg image on a cold cache. Cloud Build's own
# default is 10 minutes, which this exceeds every time.
BUILD_TIMEOUT = "3600s"


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
        print(f"HTTP {exc.code}\n{exc.read().decode(errors='replace')}", file=sys.stderr)
        raise SystemExit(1)


def _ignored() -> list[str]:
    """The patterns in `.gcloudignore`, comments and blanks dropped.

    Read rather than duplicated, because the first two blocks in that file are
    live session cookies and footage of identifiable people. A second copy of
    the list here is a copy that can fall behind the real one.
    """
    lines = (ROOT / ".gcloudignore").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def _excluded(rel: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        stem = pattern.rstrip("/")
        if pattern.startswith("*."):        # *.pyc, *.md, *.env
            if rel.endswith(pattern[1:]):
                return True
        elif rel == stem or rel.startswith(f"{stem}/") or f"/{stem}/" in f"/{rel}/":
            return True
    return False


def make_tarball(dest: Path) -> int:
    patterns = _ignored()
    count = 0
    with tarfile.open(dest, "w:gz") as tar:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(ROOT))
            if _excluded(rel, patterns):
                continue
            tar.add(path, arcname=rel)
            count += 1
    return count


def upload(tarball: Path, object_name: str) -> str:
    """Puts the tar in the bucket. Uses the storage client, not gsutil."""
    from google.cloud import storage

    client = storage.Client(project=PROJECT, credentials=credentials())
    blob = client.bucket(BUCKET).blob(object_name)
    blob.upload_from_filename(str(tarball), content_type="application/gzip")
    return f"gs://{BUCKET}/{object_name}"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: build_image.py <tag>   e.g. build_image.py v9")
    tag = sys.argv[1]
    image = f"{REGION}-docker.pkg.dev/{PROJECT}/{REPO}/agent:{tag}"

    tarball = Path(f"/tmp/agent-{tag}.tgz")
    files = make_tarball(tarball)
    print(f"context: {files} files, {tarball.stat().st_size / 1e6:.1f} MB")

    object_name = f"build-source/agent-{tag}.tgz"
    print(f"upload:  {upload(tarball, object_name)}")

    base = f"https://cloudbuild.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}"
    build = call("POST", f"{base}/builds", {
        "source": {"storageSource": {"bucket": BUCKET, "object": object_name}},
        "steps": [{
            "name": "gcr.io/cloud-builders/docker",
            "args": ["build", "-t", image, "."],
        }],
        "images": [image],
        "timeout": BUILD_TIMEOUT,
        "options": {"logging": "CLOUD_LOGGING_ONLY"},
    })

    build_id = build.get("metadata", {}).get("build", {}).get("id", "")
    print(f"build:   {build_id}")
    if not build_id:
        print(json.dumps(build, indent=2))
        raise SystemExit(1)

    # Polling, and not "here is the id, go and look": an unwatched build that
    # fails leaves the previous image in the registry, and the next deploy step
    # succeeds against yesterday's code with nothing to show that it did.
    started = time.monotonic()
    while True:
        time.sleep(20)
        status = call("GET", f"{base}/builds/{build_id}").get("status", "")
        waited = time.monotonic() - started
        print(f"  [{waited:6.0f}s] {status}")
        if status in ("SUCCESS", "FAILURE", "TIMEOUT", "CANCELLED",
                      "EXPIRED", "INTERNAL_ERROR"):
            break

    print(f"\n{status}: {image}")
    if status != "SUCCESS":
        print(f"logs: gcloud builds log {build_id} --region={REGION}", file=sys.stderr)
        raise SystemExit(1)
    # Printed on one line on purpose. The two-line form this used to print,
    # with a trailing backslash, does not survive being copied out of a
    # terminal into another one: the continuation is lost, the assignment runs
    # as its own command, and the engine gets patched with the default tag.
    # `.venv/bin/python`, not `python`: the system interpreter on the dev VM
    # has none of this project's dependencies, so the bare form printed here
    # until 2026-09-09 died on `ModuleNotFoundError: google.auth`. A hint that
    # has to be edited before it runs is a hint nobody trusts.
    print("\nMove the engine onto it with (one line):\n"
          f"  ENGINE_IMAGE={tag} .venv/bin/python deploy/agent_runtime/deploy.py "
          "update-image <engine-id>")


if __name__ == "__main__":
    main()
