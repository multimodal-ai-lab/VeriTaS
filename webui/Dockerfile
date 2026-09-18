# Build from the repository root:
#   docker compose build webui
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY webui/requirements.txt /app/webui/requirements.txt
RUN pip install --no-cache-dir -r /app/webui/requirements.txt

COPY webui /app/webui

# The service only ever reads; running it unprivileged keeps it that way even if
# a mounted volume is writable.
RUN useradd --create-home --uid 10001 viewer
USER viewer

# The port the server binds, so that `WEBUI_PORT` keeps working where there is
# no port mapping to apply it to - which is the case on a podman host, where the
# container shares the host's network namespace.
ENV WEBUI_PORT=8080

# Documentation only: EXPOSE cannot read a runtime variable. Publishing is done
# by compose, which maps `WEBUI_PORT` to itself.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request,sys; port=os.environ.get('WEBUI_PORT','8080'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=4).status == 200 else 1)"

# Shell form, so `WEBUI_PORT` is expanded at start-up; `exec` keeps uvicorn as
# PID 1 so that it still receives SIGTERM and shuts down gracefully.
CMD exec uvicorn webui.main:app --host 0.0.0.0 --port "${WEBUI_PORT:-8080}" --proxy-headers
