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
# Puts the dashboard on Cloud Run.
#
# Same image as the audit container, different entrypoint. Two images would
# mean two builds to keep in step, and the dashboard's only dependency that the
# audit does not already have is uvicorn -- which it also already has.
#
#   ./deploy/dashboard/deploy.sh [image-tag]     # default: v10
#
# On the dev VM this script does not run: gcloud's login there expires and
# cannot be renewed non-interactively. Use the sibling `deploy.py`, which does
# the same deployment over ADC and the Cloud Run Admin API. This file stays
# because it is the readable statement of what gets deployed, and because it
# works fine anywhere gcloud is logged in.
#
# Afterwards, to actually look at it:
#
#   gcloud run services proxy cctv-monitor --region=us-central1 --port=8080
#   # then open http://localhost:8080/
#
# `proxy` is not a convenience here, it is the only way in. The org policy
# `constraints/iam.allowedPolicyMemberDomains` does not allow `allUsers`, so
# the service cannot be made public even if we wanted it to be -- and we do
# not: the page streams a live CCTV feed with identifiable faces.

set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-study-project-496907}"
REGION="${MONITOR_REGION:-us-central1}"
SERVICE="${MONITOR_SERVICE:-cctv-monitor}"
TAG="${1:-v10}"
IMAGE="${MONITOR_IMAGE:-${REGION}-docker.pkg.dev/${PROJECT}/cctv-audit/agent:${TAG}}"
SA="${MONITOR_SERVICE_ACCOUNT:-cctv-monitor@${PROJECT}.iam.gserviceaccount.com}"

TOKEN_FILE="${MONITOR_TOKEN_FILE:-auth/monitor-token.txt}"
if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "No $TOKEN_FILE. Make one (it is the shared secret the agent signs its" >&2
  echo "writes with) and keep it out of git:" >&2
  echo "  (umask 077; python -c 'import secrets;print(secrets.token_urlsafe(32))' > $TOKEN_FILE" >&2
  exit 1
fi
TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"

gcloud run deploy "$SERVICE" \
  --project="$PROJECT" \
  --region="$REGION" \
  --image="$IMAGE" \
  --service-account="$SA" \
  --no-allow-unauthenticated \
  --command=sh \
  --args="^@^-c@exec uvicorn computer_use_agent.monitor_server:app --host 0.0.0.0 --port \${PORT:-8080}" \
  --port=8080 \
  --cpu=1 --memory=1Gi \
  `# One instance, always. The dashboard's state lives in a Python dict, so a` \
  `# second instance would mean the agent posting frames to one process while` \
  `# the viewer's websocket hangs off another -- a permanently blank screen` \
  `# with nothing in the logs to say why. min=1 for the same reason from the` \
  `# other direction: scaling to zero mid-audit throws the findings away.` \
  --min-instances=1 --max-instances=1 \
  `# Cloud Run counts a websocket as one long request. The default 300s would` \
  `# drop every viewer every five minutes; the page reconnects, but it flickers.` \
  --timeout=3600 \
  --set-env-vars="MONITOR_TOKEN=${TOKEN},DATA_DIR=/tmp/checkpoints,MONITOR_REPLAY_HISTORY=false" \
  --quiet

URL="$(gcloud run services describe "$SERVICE" --project="$PROJECT" --region="$REGION" \
        --format='value(status.url)')"
echo
echo "Dashboard: $URL"
echo "Set MONITOR_URL to exactly that on the Agent Runtime engine."
echo
echo "Two grants are still needed and are not done here, because handing out"
echo "access is a decision, not a deployment step:"
echo "  gcloud run services add-iam-policy-binding $SERVICE --region=$REGION \\"
echo "      --member='serviceAccount:cctv-audit-agent@${PROJECT}.iam.gserviceaccount.com' \\"
echo "      --role=roles/run.invoker      # so the audit can post to it"
echo "  gcloud run services add-iam-policy-binding $SERVICE --region=$REGION \\"
echo "      --member='user:YOU@example.com' --role=roles/run.invoker   # so you can watch"
