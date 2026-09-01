#!/bin/bash
# ============================================================
# Start ADK Web Server for Computer Use Agent (Google Cloud Vertex AI)
# ============================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 1. Activate Python virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install google-adk playwright certifi
    playwright install chromium
else
    source .venv/bin/activate
fi

# 2. Setup SSL certificates (required for macOS Python 3.13 aiohttp SSL)
export SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

# 3. Load environment variables from .env if present
if [ -f "computer_use_agent/.env" ]; then
    set -a
    source computer_use_agent/.env 2>/dev/null || true
    set +a
fi

# 4. Check Google Cloud Authentication (GCloud CLI or Application Default Credentials)
echo "🔍 Checking Google Cloud authorization..."
if gcloud auth print-access-token >/dev/null 2>&1; then
    ACTIVE_ACCOUNT="$(gcloud config get-value account 2>/dev/null || echo 'Authenticated')"
    echo "✅ Google Cloud authorization verified (Account: $ACTIVE_ACCOUNT)"
elif [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "✅ Service account key detected: $GOOGLE_APPLICATION_CREDENTIALS"
else
    echo "⚠️  No valid Google Cloud authentication found."
    echo "   Please authenticate with Google Cloud by running:"
    echo "     gcloud auth login"
    echo "     gcloud auth application-default login"
    echo ""
    echo "   Or set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json"
    exit 1
fi

PROJECT="${GOOGLE_CLOUD_PROJECT:-cs-poc-hzdu6g9fvdacmw21rd6jq89}"
LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
MODEL="${ADK_MODEL:-gemini-3.5-flash}"
PORT="${PORT:-8000}"
MONITOR_PORT="${MONITOR_PORT:-8080}"

# 5. Start Browser Use Live Monitor Server in background
python3 -m computer_use_agent.monitor_server --port "$MONITOR_PORT" >/dev/null 2>&1 &
MONITOR_PID=$!

cleanup() {
    kill $MONITOR_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "🚀 Google ADK & Browser Use Live Monitor"
echo "------------------------------------------------------------"
echo "  Auth Mode:   Google Cloud Platform (No API Key)"
echo "  GCP Project: $PROJECT"
echo "  Location:    $LOCATION"
echo "  Model:       $MODEL"
echo "  💬 ADK Web:  http://127.0.0.1:$PORT/dev-ui/  (对话与指令交互)"
echo "  📺 实时监控: http://127.0.0.1:$MONITOR_PORT/      (实时操作画面/雷达标点/时间线)"
echo "============================================================"

exec adk web computer_use_agent --port "$PORT"
