"""Resolution of ezMM item references to the media files on disk.

The ezMM registry is a SQLite database (`item_registry.db`) with one table per
item kind, mapping the item ID to an absolute path. Those paths were written by
the machine that ran the pipeline, so they frequently do not exist inside the
container. Whenever that happens we fall back to the canonical layout ezMM uses
itself, `<registry root>/<kind>/<id>.<ext>`.

Read-only: the SQLite connection is opened in immutable mode, so the UI can
neither modify nor lock the registry the pipeline is writing to.
"""

from __future__ import annotations

import mimetypes
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from webui.config import Settings, get_settings
from webui.parsing import KINDS, parse_reference

#: Fallback content types for the extensions ezMM produces that `mimetypes`
#: does not necessarily know on a slim base image.
_EXTRA_TYPES = {
    ".avif": "image/avif",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".m4v": "video/mp4",
    ".ogv": "video/ogg",
    ".opus": "audio/opus",
}

_DEFAULT_TYPES = {"image": "image/jpeg", "video": "video/mp4", "audio": "audio/mpeg"}


@dataclass(frozen=True)
class MediaItem:
    kind: str
    id: int
    reference: str
    file_path: Path | None
    source_url: str | None
    size: int | None
    content_type: str | None

    @property
    def exists(self) -> bool:
        return self.file_path is not None

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "kind": self.kind,
            "id": self.id,
            "exists": self.exists,
            "source_url": self.source_url,
            "size": self.size,
            "content_type": self.content_type,
            "url": f"/api/media/{self.kind}/{self.id}",
            "file_name": self.file_path.name if self.file_path else None,
        }


def content_type_for(path: Path, kind: str) -> str:
    suffix = path.suffix.lower()
    if suffix in _EXTRA_TYPES:
        return _EXTRA_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or _DEFAULT_TYPES.get(kind, "application/octet-stream")


class MediaRegistry:
    """Thread-safe, read-only view onto the ezMM item registry."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        self._checked = False
        self._cache: dict[tuple[str, int], MediaItem] = {}

    # -- registry access ----------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether a registry database was found at the configured path."""
        return self._settings.registry_db.is_file()

    def _connect(self) -> sqlite3.Connection | None:
        if self._connection is not None:
            return self._connection
        if self._checked or not self.available:
            self._checked = True
            return None
        self._checked = True
        uri = f"file:{self._settings.registry_db.as_posix()}?mode=ro&immutable=1"
        try:
            self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        except sqlite3.Error:
            self._connection = None
        return self._connection

    def _lookup(self, kind: str, identifier: int) -> tuple[str | None, str | None]:
        """`(path, source_url)` as recorded in the registry, if recorded at all."""
        connection = self._connect()
        if connection is None:
            return None, None
        try:
            with self._lock:
                # `kind` is validated against KINDS by the caller, so interpolating
                # it into the table name is safe; SQLite cannot parameterize it.
                row = connection.execute(
                    f"SELECT path, source_url FROM {kind} WHERE id = ?", (identifier,)
                ).fetchone()
        except sqlite3.Error:
            return None, None
        return (row[0], row[1]) if row else (None, None)

    # -- file resolution ----------------------------------------------------

    def _resolve_file(self, kind: str, identifier: int, recorded: str | None) -> Path | None:
        if recorded:
            path = Path(recorded)
            if path.is_file():
                return path
            # The registry was written elsewhere: re-root it at our media volume.
            relocated = self._settings.ezmm_path / kind / path.name
            if relocated.is_file():
                return relocated
        # Canonical ezMM layout, extension unknown.
        directory = self._settings.ezmm_path / kind
        if directory.is_dir():
            candidates = [path for path in directory.glob(f"{identifier}.*") if path.is_file()]
            # `42.jpg` before derivatives such as `42.thumb.jpg`, which belong to
            # the same item but are not the medium itself.
            candidates.sort(key=lambda path: (path.stem != str(identifier), path.name))
            if candidates:
                return candidates[0]
        return None

    def get(self, kind: str, identifier: int) -> MediaItem | None:
        """The media item, or None if `kind` is not an ezMM item kind."""
        if kind not in KINDS:
            return None
        cached = self._cache.get((kind, identifier))
        if cached is not None:
            if cached.file_path.is_file():
                return cached
            # The file was moved or quarantined since we resolved it; re-resolve.
            self._cache.pop((kind, identifier), None)

        recorded_path, source_url = self._lookup(kind, identifier)
        file_path = self._resolve_file(kind, identifier, recorded_path)
        item = MediaItem(
            kind=kind,
            id=identifier,
            reference=f"<{kind}:{identifier}>",
            file_path=file_path,
            source_url=source_url,
            size=file_path.stat().st_size if file_path else None,
            content_type=content_type_for(file_path, kind) if file_path else None,
        )
        # Only successful resolutions are cached: a medium may still be downloading.
        if item.exists:
            self._cache[(kind, identifier)] = item
        return item

    def get_by_reference(self, reference: str) -> MediaItem | None:
        parsed = parse_reference(reference)
        if parsed is None:
            return None
        return self.get(*parsed)

    def describe(self, references: list[str]) -> list[dict]:
        """Metadata for every reference, keeping the given order."""
        described = []
        for reference in references:
            item = self.get_by_reference(reference)
            if item is None:
                described.append({"reference": reference, "kind": None, "id": None,
                                  "exists": False, "url": None})
            else:
                described.append(item.to_dict())
        return described

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None


_registry: MediaRegistry | None = None


def get_registry() -> MediaRegistry:
    global _registry
    if _registry is None:
        _registry = MediaRegistry()
    return _registry
