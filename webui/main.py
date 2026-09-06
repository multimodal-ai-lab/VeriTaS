"""FastAPI application serving the Gold Evidence viewer.

Run it with

    uvicorn webui.main:app --host 0.0.0.0 --port 8080

or, more conveniently, through `docker compose up webui`.
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from webui import __version__, database, queries
from webui.config import get_settings
from webui.media import get_registry
from webui.parsing import KINDS
from webui.serialization import claim_payload, evidence_payload

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
logger = logging.getLogger("veritas-webui")

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: `bytes=start-end`, both bounds optional, as sent by video players seeking.
RANGE_REGEX = re.compile(r"bytes=(\d*)-(\d*)")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await database.connect()
    except Exception as error:  # The UI must start even if the DB is down.
        logger.error("Could not connect to the database: %s", error)
    yield
    await database.disconnect()
    get_registry().close()


app = FastAPI(
    title="VeriTaS Gold Evidence Viewer",
    version=__version__,
    description="Browse the evidence reconstructed by the Gold Evidence pipeline.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict:
    settings = get_settings()
    registry = get_registry()
    return {
        "version": __version__,
        "database": {
            "connected": await database.healthy(),
            "host": settings.db_host,
            "port": settings.db_port,
            "name": settings.db_name,
        },
        "media": {
            "registry_path": str(settings.ezmm_path),
            "registry_found": registry.available,
        },
    }


@app.get("/api/filters")
async def filters() -> dict:
    return await _guarded(queries.get_filter_options())


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@app.get("/api/stats")
async def stats() -> dict:
    return await _guarded(queries.get_overview())


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

@app.get("/api/claims")
async def list_claims(
        status: list[str] | None = Query(default=None),
        reason: str | None = None,
        q: str | None = None,
        released: bool | None = None,
        has_media: bool | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        sort: str = "updated",
        order: str = "desc",
        limit: int | None = None,
        offset: int = 0,
) -> dict:
    settings = get_settings()
    limit = min(limit or settings.page_size, settings.max_page_size)
    return await _guarded(queries.list_claims(
        statuses=status,
        reason=reason,
        query=q.strip() if q else None,
        released=released,
        has_media=has_media,
        date_from=date_from,
        date_to=date_to,
        sort=sort,
        descending=order.lower() != "asc",
        limit=max(limit, 1),
        offset=max(offset, 0),
    ))


@app.get("/api/claims/{claim_id}")
async def get_claim(claim_id: int) -> dict:
    detail = await _guarded(queries.get_claim(claim_id))
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No claim with ID {claim_id}.")
    return claim_payload(detail, get_registry())


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@app.get("/api/evidence/{evidence_id}")
async def get_evidence(evidence_id: int) -> dict:
    row = await _guarded(queries.get_evidence(evidence_id))
    if row is None:
        raise HTTPException(status_code=404, detail=f"No evidence with ID {evidence_id}.")
    return evidence_payload(row, get_registry())


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------

@app.get("/api/media/{kind}/{identifier}/meta")
async def media_meta(kind: str, identifier: int) -> dict:
    item = get_registry().get(kind, identifier)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Unknown item kind '{kind}'.")
    return item.to_dict()


@app.get("/api/media/{kind}/{identifier}")
async def media_file(kind: str, identifier: int, request: Request) -> Response:
    """Streams a medium, honouring HTTP range requests so videos can be seeked."""
    if kind not in KINDS:
        raise HTTPException(status_code=404, detail=f"Unknown item kind '{kind}'.")
    item = get_registry().get(kind, identifier)
    if item is None or not item.exists:
        raise HTTPException(
            status_code=404,
            detail=f"No file for <{kind}:{identifier}> in the ezMM registry.",
        )

    headers = {"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=86400"}
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(item.file_path, media_type=item.content_type, headers=headers)

    size = item.size or item.file_path.stat().st_size
    match = RANGE_REGEX.fullmatch(range_header.strip())
    if match is None:
        return FileResponse(item.file_path, media_type=item.content_type, headers=headers)

    raw_start, raw_end = match.groups()
    if raw_start:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    else:  # `bytes=-500` asks for the last 500 bytes.
        start = max(size - int(raw_end or 0), 0)
        end = size - 1
    end = min(end, size - 1)

    if start > end or start >= size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    headers.update({
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Content-Length": str(end - start + 1),
    })
    return StreamingResponse(
        _read_range(item.file_path, start, end, get_settings().chunk_size),
        status_code=206,
        media_type=item.content_type,
        headers=headers,
    )


def _read_range(path: Path, start: int, end: int, chunk_size: int):
    """Yields `[start, end]` of the file in chunks."""
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
@app.get("/{_path:path}")
async def index(_path: str = "") -> Response:
    """Single-page app: every non-API path serves the shell, which routes on the
    URL hash."""
    if _path.startswith("api/"):
        raise HTTPException(status_code=404, detail=f"No such endpoint: /{_path}")
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

async def _guarded(awaitable):
    """Turns a database outage into a clean 503 instead of a stack trace."""
    try:
        return await awaitable
    except HTTPException:
        raise
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}") from error


@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException) -> Response:
    """API paths get JSON; everything else falls back to the SPA shell."""
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": exc.detail}, status_code=404)
    return FileResponse(STATIC_DIR / "index.html", status_code=200)
