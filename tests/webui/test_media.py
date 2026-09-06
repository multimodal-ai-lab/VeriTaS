"""Resolution of ezMM item references to files on disk.

Uses a temporary registry rather than the real media store, so the tests stay
pure and never read the pipeline's data.
"""

import sqlite3
from pathlib import Path

import pytest

from webui.config import Settings
from webui.media import MediaRegistry, content_type_for


@pytest.fixture
def store(tmp_path: Path) -> Path:
    for kind in ("image", "video", "audio"):
        (tmp_path / kind).mkdir()
    connection = sqlite3.connect(tmp_path / "item_registry.db")
    for kind in ("image", "video", "audio"):
        connection.execute(
            f"CREATE TABLE {kind} (id INTEGER PRIMARY KEY, path TEXT, source_url TEXT)")
    connection.commit()
    connection.close()
    return tmp_path


def register(store: Path, kind: str, identifier: int, path: str, source_url: str = "") -> None:
    connection = sqlite3.connect(store / "item_registry.db")
    connection.execute(f"INSERT INTO {kind} VALUES (?,?,?)", (identifier, path, source_url))
    connection.commit()
    connection.close()


def registry_for(store: Path) -> MediaRegistry:
    return MediaRegistry(Settings(
        db_name="x", db_user="x", db_password="", db_host="localhost", db_port=5432,
        ezmm_path=store, page_size=25, max_page_size=200, chunk_size=1024,
    ))


def test_resolves_a_file_at_the_recorded_path(store):
    path = store / "image" / "42.jpg"
    path.write_bytes(b"jpeg")
    register(store, "image", 42, str(path), "https://example.org/42.jpg")

    item = registry_for(store).get("image", 42)
    assert item.exists
    assert item.file_path == path
    assert item.source_url == "https://example.org/42.jpg"
    assert item.size == 4
    assert item.content_type == "image/jpeg"
    assert item.to_dict()["url"] == "/api/media/image/42"


def test_falls_back_to_the_canonical_layout_when_the_recorded_path_is_stale(store):
    # The registry was written on another machine, so its path does not exist here.
    (store / "image" / "42.webp").write_bytes(b"webp")
    register(store, "image", 42, "/srv/elsewhere/ezmm/image/42.webp")

    item = registry_for(store).get("image", 42)
    assert item.exists
    assert item.file_path == store / "image" / "42.webp"
    assert item.content_type == "image/webp"


def test_prefers_the_medium_over_its_derivatives(store):
    (store / "image" / "42.webp").write_bytes(b"webp")
    (store / "image" / "42.thumb.jpg").write_bytes(b"thumb")
    register(store, "image", 42, "/gone/42.webp")

    assert registry_for(store).get("image", 42).file_path.name == "42.webp"


def test_resolves_an_item_missing_from_the_registry_by_layout_alone(store):
    (store / "video" / "7.mp4").write_bytes(b"mp4")
    item = registry_for(store).get("video", 7)
    assert item.exists
    assert item.content_type == "video/mp4"
    assert item.source_url is None


def test_reports_a_reference_whose_file_is_gone(store):
    register(store, "image", 42, "/gone/42.jpg")
    item = registry_for(store).get("image", 42)
    assert item.exists is False
    assert item.to_dict()["exists"] is False
    assert item.reference == "<image:42>"


def test_unknown_kind_is_not_an_item(store):
    assert registry_for(store).get("pdf", 1) is None
    assert registry_for(store).get_by_reference("<pdf:1>") is None


def test_get_by_reference(store):
    (store / "image" / "9.png").write_bytes(b"png")
    assert registry_for(store).get_by_reference("<image:9>").id == 9


def test_describe_keeps_the_given_order_and_flags_unknown_references(store):
    (store / "image" / "1.png").write_bytes(b"png")
    described = registry_for(store).describe(["<image:1>", "<image:2>", "not-a-reference"])
    assert [entry["reference"] for entry in described] == [
        "<image:1>", "<image:2>", "not-a-reference"]
    assert [entry["exists"] for entry in described] == [True, False, False]


def test_missing_registry_database_is_not_fatal(tmp_path):
    registry = registry_for(tmp_path)
    assert registry.available is False
    assert registry.get("image", 1).exists is False


@pytest.mark.parametrize("name,kind,expected", [
    ("a.jpg", "image", "image/jpeg"),
    ("a.avif", "image", "image/avif"),
    ("a.webp", "image", "image/webp"),
    ("a.mp4", "video", "video/mp4"),
    ("a.mkv", "video", "video/x-matroska"),
    ("a", "image", "image/jpeg"),
    ("a", "video", "video/mp4"),
])
def test_content_type_for(name, kind, expected):
    assert content_type_for(Path(name), kind) == expected


def test_a_cached_item_is_re_resolved_when_its_file_disappears(store):
    path = store / "image" / "42.jpg"
    path.write_bytes(b"jpeg")
    register(store, "image", 42, str(path))
    registry = registry_for(store)

    assert registry.get("image", 42).exists is True
    path.unlink()  # e.g. quarantined by scripts/cleanup_media.py
    assert registry.get("image", 42).exists is False
