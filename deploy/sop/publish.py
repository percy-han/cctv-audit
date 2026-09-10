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

"""Publishes a version of the store standard, and lists what is published.

    python deploy/sop/publish.py list
    python deploy/sop/publish.py show   chagee-store-v1
    python deploy/sop/publish.py put    chagee-store-v2 path/to/rules.yaml
    python deploy/sop/publish.py render chagee-store-v2 path/to/rules.yaml

This is the whole "customer changes the standard without touching our source"
story: write a YAML, `put` it under a new id, add a prompt chip that names the
id. See README.md in this directory for the naming convention and for why a
published version is never edited in place.

Two things it refuses to do, both on purpose:

  * **It parses before it uploads.** A YAML that `analyzer/sop.py` cannot read
    is not a standard, and the place to find that out is here -- not in the
    middle of an audit the customer is watching, where the only symptom is a
    job that goes straight to `rejected`.
  * **It will not overwrite an existing id.** An audit record says "judged
    under chagee-store-v1". If v1's contents can change afterwards, that
    sentence stops meaning anything and every past report quietly becomes
    unverifiable. Use `--force` if you really are fixing a typo nobody has
    audited against yet, and know that you are rewriting history when you do.

`render` uploads nothing. It prints the prompt the model would actually be
given, which is the only way to see what a wording change did.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import google.auth  # noqa: E402
from google.cloud import storage  # noqa: E402

from cctv_audit.analyzer.sop import parse_rules  # noqa: E402

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "study-project-496907")
BUCKET = os.environ.get("SOP_BUCKET", f"{PROJECT}-cctv-audit")
PREFIX = os.environ.get("SOP_PREFIX", "sop").strip("/")


def client() -> storage.Client:
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return storage.Client(project=PROJECT, credentials=creds)


def object_name(sop_id: str) -> str:
    return f"{PREFIX}/{sop_id}.yaml" if PREFIX else f"{sop_id}.yaml"


def check(text: str, sop_id: str):
    """Parses, and prints what the deployment will see. Raises if it cannot."""
    rules = parse_rules(text, origin=f"gs://{BUCKET}/{object_name(sop_id)}",
                        sop_id=sop_id)
    print(f"  version:      {rules.version}")
    print(f"  rules:        {len(rules.rules)}  {', '.join(rules.ids)}")
    print(f"  scan targets: {len(rules.scan_targets)}")
    red = [r.id for r in rules.rules if r.severity.upper() in ("RED_LINE", "RED-LINE")]
    print(f"  red line:     {', '.join(red) if red else '(none)'}")
    return rules


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "list"

    if action == "list":
        blobs = sorted(client().list_blobs(BUCKET, prefix=f"{PREFIX}/"),
                       key=lambda b: b.name)
        for blob in blobs:
            if not blob.name.endswith(".yaml"):
                continue
            sop_id = Path(blob.name).stem
            print(f"{sop_id:28s} {blob.size:6d} B  "
                  f"{blob.updated:%Y-%m-%d %H:%M}  gs://{BUCKET}/{blob.name}")
        if not blobs:
            print(f"(nothing under gs://{BUCKET}/{PREFIX}/)")

    elif action == "show":
        sop_id = sys.argv[2]
        blob = client().bucket(BUCKET).blob(object_name(sop_id))
        if not blob.exists():
            raise SystemExit(f"no such version: gs://{BUCKET}/{object_name(sop_id)}")
        print(blob.download_as_text())

    elif action in ("put", "render"):
        sop_id, path = sys.argv[2], Path(sys.argv[3])
        text = path.read_text(encoding="utf-8")
        print(f"{path} -> {sop_id}")
        rules = check(text, sop_id)

        if action == "render":
            # Both halves, because they fail differently. A weak rule wording
            # produces a wrong verdict; a missing scan target lets the model
            # answer from what a tea shop usually looks like without ever
            # looking at this one.
            print("\n" + "=" * 72 + "\n先描述后判定（scan targets）\n" + "=" * 72)
            print(rules.render_scan() or "(none -- the model judges unanchored)")
            print("\n" + "=" * 72 + "\n判定条款（rules）\n" + "=" * 72)
            print(rules.render())
            return

        blob = client().bucket(BUCKET).blob(object_name(sop_id))
        if blob.exists() and "--force" not in sys.argv:
            raise SystemExit(
                f"\ngs://{BUCKET}/{object_name(sop_id)} already exists.\n"
                "Publish a new id instead -- past reports name this one. "
                "Pass --force only if nothing has been audited against it yet."
            )
        blob.upload_from_string(text, content_type="application/x-yaml")
        print(f"\nuploaded gs://{BUCKET}/{object_name(sop_id)}")
        print(f"Add a GE chip naming it, e.g.\n"
              f"  按标准 {sop_id} 稽核这段录像：<在这里粘贴视频网址>，从 05:00 开始看 2 分钟")

    else:
        raise SystemExit("usage: publish.py (list|show|put|render) [args]")


if __name__ == "__main__":
    main()
