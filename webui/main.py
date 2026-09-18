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
from webui.serialization import (claim_list_payload, claim_payload,
                                 evidence_item_payload,
                                 evidence_list_payload, evidence_source_payload)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
logger = logging.getLogger("veritas-webui")

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: `bytes=start-end`, both bounds optional, as sent by video players seeking.
RANGE_REGEX = re.compile(r"bytes=(\d*)-(\d*)")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        # Bounded by `database.CONNECT_TIMEOUT`, so an address that drops packets
        # instead of refusing cannot hold up start-up. Requests reconnect on
        # their own, so giving up here costs nothing.
        await database.connect()
    except Exception as error:  # The UI must start even if the DB is down.
        logger.error("Could not connect to the database: %s", error)
        hint = database.connection_hint()
        if hint:
            logger.error("%s", hint)
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
    error = await database.check()
    return {
        "version": __version__,
        "database": {
            "connected": error is None,
            "error": error,
            "host": settings.db_host,
            "port": settings.db_port,
            "name": settings.db_name,
            "user": settings.db_user,
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
        language: list[str] | None = Query(default=None),
        q: str | None = None,
        released: bool | None = None,
        media: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        sort: str = "updated",
        order: str = "desc",
        limit: int | None = None,
        offset: int = 0,
) -> dict:
    settings = get_settings()
    limit = min(limit or settings.page_size, settings.max_page_size)
    page = await _guarded(queries.list_claims(
        statuses=status,
        reason=reason,
        languages=language,
        query=q.strip() if q else None,
        released=released,
        media=_media_filter(media),
        date_from=date_from,
        date_to=date_to,
        sort=sort,
        descending=order.lower() != "asc",
        limit=max(limit, 1),
        offset=max(offset, 0),
    ))
    return claim_list_payload(page, get_registry())


@app.get("/api/claims/{claim_id}")
async def get_claim(claim_id: int) -> dict:
    detail = await _guarded(queries.get_claim(claim_id))
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No claim with ID {claim_id}.")
    return claim_payload(detail, get_registry())


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@app.get("/api/evidence")
async def list_evidence(
        claim_id: int | None = None,
        admissibility: list[str] | None = Query(default=None),
        kind: list[str] | None = Query(default=None),
        proximity: list[str] | None = Query(default=None),
        role: list[str] | None = Query(default=None),
        zone: list[str] | None = Query(default=None),
        language: list[str] | None = Query(default=None),
        reason: str | None = None,
        domain: str | None = None,
        q: str | None = None,
        media: str | None = None,
        accessible: bool | None = None,
        later_event: bool | None = None,
        dismissed: bool | None = None,
        min_confidence: float | None = Query(default=None, ge=0, le=1),
        min_faithfulness: float | None = Query(default=None, ge=-1, le=1),
        max_faithfulness: float | None = Query(default=None, ge=-1, le=1),
        date_from: date | None = None,
        date_to: date | None = None,
        sort: str = "id",
        order: str = "desc",
        limit: int | None = None,
        offset: int = 0,
) -> dict:
    """A page of evidence items across all claims, filtered on the stored fields."""
    settings = get_settings()
    limit = min(limit or settings.page_size, settings.max_page_size)
    page = await _guarded(queries.list_evidence(
        claim_id=claim_id,
        admissibility=admissibility,
        kinds=kind,
        proximities=proximity,
        roles=role,
        zones=zone,
        languages=language,
        reason=reason,
        domain=domain.strip().lower() if domain else None,
        query=q.strip() if q else None,
        media=_media_filter(media),
        accessible=accessible,
        later_event=later_event,
        dismissed=dismissed,
        min_confidence=min_confidence,
        min_faithfulness=min_faithfulness,
        max_faithfulness=max_faithfulness,
        date_from=date_from,
        date_to=date_to,
        sort=sort,
        descending=order.lower() != "asc",
        limit=max(limit, 1),
        offset=max(offset, 0),
    ))
    return evidence_list_payload(page, get_registry())


# Declared before `/api/evidence/{source_id}` so the literal path wins.
@app.get("/api/evidence/options")
async def evidence_options() -> dict:
    return await _guarded(queries.get_evidence_filter_options())


@app.get("/api/evidence/items/{evidence_id}")
async def get_evidence_item(evidence_id: int) -> dict:
    """One evidence item with every stored field of it and of its sources.

    The claim view lists items in a light column set and asks for this when the
    reader expands one, so opening a claim does not have to transfer every
    scraped document the claim's sources hold."""
    row = await _guarded(queries.get_evidence_item(evidence_id))
    if row is None:
        raise HTTPException(status_code=404, detail=f"No evidence item with ID {evidence_id}.")
    return evidence_item_payload(row, get_registry())


@app.get("/api/evidence/{source_id}")
async def get_evidence_source(source_id: int) -> dict:
    """One evidence source. The browser lists sources, because every Stage-2
    judgement is a property of a source rather than of the proposition."""
    row = await _guarded(queries.get_evidence_source(source_id))
    if row is None:
        raise HTTPException(status_code=404, detail=f"No evidence source with ID {source_id}.")
    return evidence_source_payload(row, get_registry())


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

#: The shell and its modules carry no version in their URL, so a browser that
#: caches them without asking will keep running an old app against a new API -
#: the symptom is a nav entry that exists but routes nowhere. `no-cache` does
#: not forbid caching, it only forbids *reusing* without revalidating, and the
#: ETag makes that revalidation a bodyless 304.
REVALIDATE = {"Cache-Control": "no-cache"}


class RevalidatingStaticFiles(StaticFiles):
    """`StaticFiles` that asks the browser to revalidate every file it holds."""

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.update(REVALIDATE)
        return response


app.mount("/static", RevalidatingStaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
@app.get("/{_path:path}")
async def index(_path: str = "") -> Response:
    """Single-page app: every non-API path serves the shell, which routes on the
    URL hash."""
    if _path.startswith("api/"):
        raise HTTPException(status_code=404, detail=f"No such endpoint: /{_path}")
    return FileResponse(STATIC_DIR / "index.html", headers=REVALIDATE)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def _media_filter(media: str | None) -> str | None:
    """Validates the `media` query parameter against the fixed vocabulary, so a
    typo becomes a 400 rather than an unfiltered result set."""
    if not media:
        return None
    if media not in queries.MEDIA_FILTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown media filter {media!r}. "
                   f"Expected one of: {', '.join(queries.MEDIA_FILTERS)}.",
        )
    return media


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
    return FileResponse(STATIC_DIR / "index.html", status_code=200, headers=REVALIDATE)
