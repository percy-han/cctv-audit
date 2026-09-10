# The Agent Runtime image.
#
# One image, two entrypoints. The audit container and the Cloud Run dashboard
# run the same code from the same build -- so the dashboard can never be
# reading records written by a version of the store it does not share. The
# entrypoint decides which one you get:
#
#   uvicorn cctv_audit.server:app          the agent (default)
#   uvicorn cctv_audit.monitor_server:app  the dashboard
#
# It is a big image (Chromium alone is ~400 MB) and there is no way around
# that: Plan B captures the screen of a real browser, and Plan A still needs
# one to find the stream URL.

FROM python:3.11-slim

# ffmpeg is the capture and clipping engine and is NOT pip-installable.
# The rest is what Chromium refuses to start without; `playwright
# install --with-deps` would fetch them too, but naming them here keeps the
# layer cacheable and makes a future base-image bump fail loudly instead of
# silently pulling a different set.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
        fonts-liberation \
        fonts-noto-cjk \
        libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
        libcups2 libdrm2 libxkbcommon0 libatspi2.0-0 libxcomposite1 \
        libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
        libcairo2 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# fonts-noto-cjk is not decoration. The store names, the page chrome and the
# on-screen clock are Chinese; without it every one of them renders as tofu
# boxes, in the screen recording that becomes the evidence.

WORKDIR /app

# Browsers land in a fixed path rather than under $HOME so they are found
# whatever uid the platform decides to run as.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install chromium

COPY cctv_audit ./cctv_audit

# The container filesystem is temporary and the instance can be replaced
# between two turns of the same conversation. Everything here is scratch;
# anything that has to survive goes to Firestore and GCS, which is what
# FIRESTORE_DATABASE and ARTIFACTS_BUCKET switch on. AUTH_STATE_DIR is set
# because config insists on one -- there are no cookies to keep yet, and
# baking any into an image would put live sessions in a registry.
ENV WORK_DIR=/tmp/work \
    DATA_DIR=/tmp/checkpoints \
    AUTH_STATE_DIR=/tmp/auth \
    BROWSER_HEADLESS=true \
    PORT=8080

# From the runtime contract: 0.0.0.0, port 8080.
EXPOSE 8080

# --timeout-keep-alive well past the platform's own 900s cut, so that a
# connection dying mid-turn is the platform's doing and not uvicorn's. Phase 0
# spent a day telling those two apart; no reason to re-run that experiment.
CMD ["sh", "-c", "exec uvicorn cctv_audit.server:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 3600"]
