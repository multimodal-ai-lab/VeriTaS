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

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "webui.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
