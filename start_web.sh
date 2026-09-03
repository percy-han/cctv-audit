#!/bin/bash
# ============================================================
# CCTV Audit Agent — ADK web UI + live monitor dashboard
# ============================================================

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# ------------------------------------------------------------
# 1. Python environment
# ------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    playwright install chromium
else
    source .venv/bin/activate
    # Cheap guard against a venv predating a requirements.txt change.
    if ! python -c "import fastapi, uvicorn, yaml, dotenv, google.genai" 2>/dev/null; then
        echo "📦 Installing missing dependencies from requirements.txt..."
        pip install -q -r requirements.txt
    fi
fi

# ------------------------------------------------------------
# 2. ffmpeg — a hard requirement, not an optional extra
# ------------------------------------------------------------
# ffmpeg does four jobs here: pull the stream, cut it into segments, encode
# screen-captured frames into MP4, and extract evidence stills. Without it the
# pipeline cannot produce a single analysable clip.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "❌ ffmpeg not found on PATH."
    echo "   Debian/Ubuntu:  sudo apt install ffmpeg"
    echo "   macOS:          brew install ffmpeg"
    exit 1
fi
if ! command -v ffprobe >/dev/null 2>&1; then
    echo "❌ ffprobe not found on PATH (usually shipped with ffmpeg)."
    exit 1
fi

# ------------------------------------------------------------
# 3. SSL bundle (some Python builds miss the system trust store)
# ------------------------------------------------------------
SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
export SSL_CERT_FILE
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

# ------------------------------------------------------------
# 4. Environment
# ------------------------------------------------------------
if [ -f "computer_use_agent/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source computer_use_agent/.env 2>/dev/null || true
    set +a
fi

# No project fallback: a hardcoded default silently bills someone else's
# project and hides a misconfiguration until the first API call fails.
if [ -z "${GOOGLE_CLOUD_PROJECT:-}" ]; then
    echo "❌ GOOGLE_CLOUD_PROJECT is not set."
    echo "   Copy computer_use_agent/.env.example to computer_use_agent/.env and fill it in."
    exit 1
fi

# ------------------------------------------------------------
# 5. Google Cloud authentication
# ------------------------------------------------------------
echo "🔍 Checking Google Cloud authorization..."
if gcloud auth print-access-token >/dev/null 2>&1; then
    ACTIVE_ACCOUNT="$(gcloud config get-value account 2>/dev/null || echo 'authenticated')"
    echo "✅ Authorized as $ACTIVE_ACCOUNT"
elif [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "✅ Service account key: $GOOGLE_APPLICATION_CREDENTIALS"
else
    echo "⚠️  No valid Google Cloud authentication found. Run one of:"
    echo "     gcloud auth login"
    echo "     gcloud auth application-default login"
    echo "   Or set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json"
    exit 1
fi

LOCATION="global"   # pinned in config.py; see the note there before changing
MODEL="${ADK_MODEL:-gemini-3.5-flash}"
ANALYSIS_MODEL="${ANALYSIS_MODEL:-$MODEL}"
PORT="${PORT:-8000}"
MONITOR_HOST="${MONITOR_HOST:-127.0.0.1}"
MONITOR_PORT="${MONITOR_PORT:-8080}"
export MONITOR_HOST MONITOR_PORT

if [ "$MONITOR_HOST" != "127.0.0.1" ] && [ "$MONITOR_HOST" != "localhost" ] && [ -z "${MONITOR_TOKEN:-}" ]; then
    echo "⚠️  MONITOR_HOST=$MONITOR_HOST with no MONITOR_TOKEN: the live CCTV view"
    echo "    (identifiable faces) and its write endpoints are open to the network."
fi

# ------------------------------------------------------------
# 6. Live monitor, then the ADK web UI in the foreground
# ------------------------------------------------------------
python -m computer_use_agent.monitor_server --host "$MONITOR_HOST" --port "$MONITOR_PORT" \
    >/tmp/cctv_audit_monitor.log 2>&1 &
MONITOR_PID=$!

cleanup() {
    kill "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "🚀 CCTV Audit Agent"
echo "------------------------------------------------------------"
echo "  GCP project:   $GOOGLE_CLOUD_PROJECT"
echo "  Location:      $LOCATION"
echo "  Nav model:     $MODEL"
echo "  Analysis:      $ANALYSIS_MODEL @ ${ANALYSIS_FPS:-1} FPS, ${MEDIA_RESOLUTION:-low} res"
echo "  Window:        ${WINDOW_SECONDS:-15}s (overlap ${WINDOW_OVERLAP_SECONDS:-3}s), capture ${CAPTURE_MODE:-auto}"
echo "  💬 ADK web:    http://127.0.0.1:$PORT/dev-ui/"
echo "  📺 Monitor:    http://$MONITOR_HOST:$MONITOR_PORT/   (log: /tmp/cctv_audit_monitor.log)"
echo "============================================================"

exec adk web computer_use_agent --port "$PORT"
