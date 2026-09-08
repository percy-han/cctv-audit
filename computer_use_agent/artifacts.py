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

"""Where evidence frames and cover stills go.

On a workstation that is a directory. On Agent Runtime it has to be a bucket:
the container filesystem is temporary and the instance can be recycled between
the turn that runs the audit and the turn that asks to see it, so a violation
whose only proof is a JPG on local disk is a violation nobody can check.

Both backends return a *locator string* that goes straight into the audit
record -- a path relative to `DATA_DIR`, or a `gs://` URI. Records are read
years later by tools that do not exist yet, so the locator says on its face
which kind it is instead of relying on the reader knowing the deployment.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional, Protocol

from .config import config

logger = logging.getLogger("cctv_audit.artifacts")


class ArtifactSink(Protocol):
    async def put(self, key: str, data: bytes, content_type: str) -> Optional[str]: ...

    async def put_file(self, key: str, path: Path, content_type: str) -> Optional[str]: ...

    def location(self, key: str) -> str:
        """Where `key` lands, without storing anything.

        For telling a reader where to look. The local sink returns "" -- there
        the path the store already knows is the answer, and a second rendering
        of the same directory is only a chance to disagree with it.
        """
        ...


class LocalArtifactSink:
    """Writes under `DATA_DIR`, returning the path relative to it."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self._root = root or config.data_dir

    def _resolve(self, key: str) -> Path:
        # `key` is built from a job id and a rule id, both of ours, but it is
        # the only string in this file that ever ends up in a filesystem path.
        # Anchoring it keeps a stray "../" from writing outside DATA_DIR.
        target = (self._root / key).resolve()
        root = self._root.resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"artifact key escapes the data directory: {key!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def location(self, key: str) -> str:
        return ""  # the caller's own path is the answer here

    async def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> Optional[str]:
        target = self._resolve(key)
        await asyncio.to_thread(target.write_bytes, data)
        return key

    async def put_file(self, key: str, path: Path, content_type: str = "image/jpeg") -> Optional[str]:
        if not path.exists():
            return None
        target = self._resolve(key)
        if target == path.resolve():
            return key
        await asyncio.to_thread(lambda: target.write_bytes(path.read_bytes()))
        return key


class GcsArtifactSink:
    """Uploads to `gs://<ARTIFACTS_BUCKET>/<ARTIFACTS_PREFIX>/<key>`."""

    def __init__(self, bucket: Optional[str] = None, prefix: Optional[str] = None) -> None:
        self._bucket_name = bucket or config.artifacts_bucket
        self._prefix = (prefix if prefix is not None else config.artifacts_prefix).strip("/")
        self._bucket = None

    def location(self, key: str) -> str:
        # Deliberately does not go through `_blob`: that builds a client, and
        # asking "where would this go" must not need credentials or a network.
        name = f"{self._prefix}/{key}" if self._prefix else key
        return f"gs://{self._bucket_name}/{name}"

    def _blob(self, key: str):
        if self._bucket is None:
            # Lazy for the same reason as the Firestore client: the local path
            # and the entire test suite run without this package installed.
            from google.cloud import storage

            from .gcp import credentials_without_quota_project

            # Without the credential scrub every upload here 403s on Agent
            # Runtime -- see `credentials_without_quota_project` for why.
            client = storage.Client(project=config.gcp_project or None,
                                    credentials=credentials_without_quota_project())
            self._bucket = client.bucket(self._bucket_name)
        name = f"{self._prefix}/{key}" if self._prefix else key
        return self._bucket.blob(name), f"gs://{self._bucket_name}/{name}"

    async def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> Optional[str]:
        blob, uri = self._blob(key)
        try:
            # google-cloud-storage is synchronous, and a 60 KB upload inside
            # the analysis loop would otherwise stall every other window's
            # coroutine for the round trip.
            await asyncio.to_thread(blob.upload_from_string, data, content_type=content_type)
        except Exception as exc:
            # Losing one evidence frame must not lose the finding. The record
            # keeps `evidence_frame: null`, which the report already renders as
            # "no reviewable still" rather than pretending there is one.
            logger.warning("Could not upload %s: %s", uri, str(exc)[:200])
            return None
        return uri

    async def put_file(self, key: str, path: Path, content_type: str = "image/jpeg") -> Optional[str]:
        if not path.exists():
            return None
        blob, uri = self._blob(key)
        try:
            await asyncio.to_thread(blob.upload_from_filename, str(path), content_type=content_type)
        except Exception as exc:
            logger.warning("Could not upload %s: %s", uri, str(exc)[:200])
            return None
        return uri


_sink: Optional[ArtifactSink] = None


def artifact_sink() -> ArtifactSink:
    global _sink
    if _sink is None:
        _sink = GcsArtifactSink() if config.use_gcs else LocalArtifactSink()
        logger.info("Artifact sink: %s", type(_sink).__name__)
    return _sink


def set_artifact_sink(sink: Optional[ArtifactSink]) -> None:
    """Overrides the sink. For tests."""
    global _sink
    _sink = sink
