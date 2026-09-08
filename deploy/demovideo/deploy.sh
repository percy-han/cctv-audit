#!/usr/bin/env bash
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
#
# Puts the demo footage on Cloud Run.
#
#   ./make_assets.sh && ./deploy.sh
#
# `--no-allow-unauthenticated` is not a choice. The effective org policy
# `constraints/iam.allowedPolicyMemberDomains` allows two customer IDs and the
# organisation's own principalSet, and nothing else -- so `allUsers` cannot be
# granted `run.invoker` here however the deploy is written.
#
# The consequence lands on the browser, not on this service: whatever opens
# these pages has to present a Google OIDC token. The audit container does,
# for this one origin only (`DEMO_ORIGIN` / `OIDC_ORIGINS`); a human opens it
# with `gcloud run services proxy`.

set -euo pipefail
cd "$(dirname "$0")"

PROJECT="${PROJECT:-study-project-496907}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-cctv-demo-video}"
AGENT_SA="${AGENT_SA:-cctv-audit-agent@${PROJECT}.iam.gserviceaccount.com}"

if [[ ! -f assets/store.mp4 || ! -f assets/hls/index.m3u8 ]]; then
  echo "assets/ is missing or incomplete -- run ./make_assets.sh first." >&2
  exit 1
fi

echo "== deploying ${SERVICE} to ${REGION} =="
gcloud run deploy "$SERVICE" \
  --source=. \
  --project="$PROJECT" \
  --region="$REGION" \
  --port=8080 \
  --cpu=1 --memory=512Mi \
  --min-instances=0 --max-instances=3 \
  --no-allow-unauthenticated \
  --quiet

URL="$(gcloud run services describe "$SERVICE" --project="$PROJECT" \
        --region="$REGION" --format='value(status.url)')"

cat <<EOF

Service URL: ${URL}
  Plan A: ${URL}/hls.html
  Plan B: ${URL}/mp4.html

Two things this script deliberately does NOT do, because both are grants and a
grant should be somebody's decision rather than a side effect of a deploy:

  # let the audit container open it
  gcloud run services add-iam-policy-binding ${SERVICE} \\
    --project=${PROJECT} --region=${REGION} \\
    --member=serviceAccount:${AGENT_SA} \\
    --role=roles/run.invoker

  # look at it yourself
  gcloud run services proxy ${SERVICE} --project=${PROJECT} --region=${REGION} --port=8099
EOF
