# Phase 0 probe container.
#
# Slim on purpose. The real image (Phase 2) will need Chromium and ffmpeg and
# will be an order of magnitude bigger; this one exists to isolate platform
# behaviour from our own, so the fewer moving parts the better.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY probe.py .

# From the runtime contract: the container must listen on 0.0.0.0 and port 8080.
ENV PORT=8080
EXPOSE 8080

# timeout_keep_alive is set far past any plausible platform limit so that a cut
# connection during the hang probe is attributable to the platform, not to
# uvicorn giving up first. Without this the timeout measurement is worthless.
CMD ["uvicorn", "probe:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "3600"]
