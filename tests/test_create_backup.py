"""Uploading backups larger than the server's limit in parts.

No network: the WebDAV PUT is replaced by a fake that records what was uploaded
and rejects anything above a configurable size with "413 Request Entity Too
Large", as the nginx in front of Nextcloud does.
"""

import hashlib
import json

import pytest
import pytest_asyncio

import scripts.export.create_backup as backup


@pytest_asyncio.fixture(scope="function", autouse=True)
async def db():
    """Overrides the DB fixture from the parent conftest: these tests need no DB."""
    yield None


@pytest.fixture
def server(monkeypatch):
    """A fake Nextcloud: stores uploads by path, rejects those above `limit`."""
    state = {"files": {}, "limit": None, "requests": []}

    def fake_put(remote_path, data):
        body = data if isinstance(data, bytes) else data.read()
        state["requests"].append((remote_path, len(body)))
        # The manifest is a few KB; the limit is about the data, which is what
        # the tiny limits of these tests stand in for.
        limited = not remote_path.endswith(".parts.json")
        if limited and state["limit"] is not None and len(body) > state["limit"]:
            return 413
        state["files"][remote_path] = body
        return 201

    monkeypatch.setattr(backup, "_put", fake_put)
    monkeypatch.setattr(backup, "MIN_PART_SIZE", 4)
    return state


@pytest.fixture
def dump(tmp_path):
    path = tmp_path / "db.sql"
    path.write_bytes(bytes(range(256)) * 4)  # 1024 bytes
    return path


def reassemble(files: dict, remote: str) -> bytes:
    manifest = json.loads(files[f"{remote}.parts.json"])
    directory = remote.rsplit("/", 1)[0]
    return b"".join(files[f"{directory}/{part['name']}"] for part in manifest["parts"])


# --- Pure helpers ------------------------------------------------------------------

def test_parts_cover_the_file_exactly():
    assert backup.plan_parts(10, 4) == [(0, 4), (4, 4), (8, 2)]
    assert backup.plan_parts(8, 4) == [(0, 4), (4, 4)]
    assert backup.plan_parts(0, 4) == [(0, 0)]


def test_part_names_sort_in_order():
    names = backup.part_names("2026-09-30/db.sql", 3)
    assert names == ["2026-09-30/db.sql.part001", "2026-09-30/db.sql.part002",
                     "2026-09-30/db.sql.part003"]
    many = backup.part_names("db.sql", 1200)
    assert many[0] == "db.sql.part0001" and sorted(many) == many


def test_a_file_slice_reads_only_its_bytes_and_has_a_length(dump):
    with backup.FileSlice(dump, 10, 5) as part:
        assert len(part) == 5
        assert part.read(3) == bytes([10, 11, 12])
        assert part.read() == bytes([13, 14])
        assert part.read() == b""


def test_a_file_slice_gives_requests_a_content_length(dump):
    """Otherwise requests would fall back to chunked transfer encoding."""
    from requests.utils import super_len

    with backup.FileSlice(dump, 100, 300) as part:
        assert super_len(part) == 300


# --- Uploading ------------------------------------------------------------------

def test_a_file_within_the_limit_is_uploaded_whole(server, dump):
    assert backup.upload_file(dump, "2026-09-30/db.sql", part_size=2048)
    assert list(server["files"]) == ["2026-09-30/db.sql"]
    assert server["files"]["2026-09-30/db.sql"] == dump.read_bytes()


def test_a_larger_file_is_uploaded_in_parts_with_a_manifest(server, dump):
    assert backup.upload_file(dump, "2026-09-30/db.sql", part_size=300)

    names = sorted(server["files"])
    assert names == ["2026-09-30/db.sql.part001", "2026-09-30/db.sql.part002",
                     "2026-09-30/db.sql.part003", "2026-09-30/db.sql.part004",
                     "2026-09-30/db.sql.parts.json"]
    assert reassemble(server["files"], "2026-09-30/db.sql") == dump.read_bytes()

    manifest = json.loads(server["files"]["2026-09-30/db.sql.parts.json"])
    assert manifest["size"] == 1024
    assert manifest["sha256"] == hashlib.sha256(dump.read_bytes()).hexdigest()
    assert [p["size"] for p in manifest["parts"]] == [300, 300, 300, 124]
    assert manifest["restore"].startswith("cat db.sql.part* > db.sql")


def test_a_413_halves_the_part_size_until_the_parts_fit(server, dump):
    """The limit is unknown in advance - it is usually a proxy's, not Nextcloud's."""
    server["limit"] = 200
    assert backup.upload_file(dump, "b/db.sql", part_size=2048)

    manifest = json.loads(server["files"]["b/db.sql.parts.json"])
    assert all(part["size"] <= 200 for part in manifest["parts"])
    assert reassemble(server["files"], "b/db.sql") == dump.read_bytes()
    # No part of an earlier attempt is left behind outside the manifest.
    listed = {f"b/{part['name']}" for part in manifest["parts"]} | {"b/db.sql.parts.json"}
    assert set(server["files"]) == listed


def test_the_upload_gives_up_at_the_minimum_part_size(server, dump):
    server["limit"] = 2  # below MIN_PART_SIZE (4) in this fixture
    assert backup.upload_file(dump, "b/db.sql", part_size=2048) is False


def test_other_errors_are_not_retried(server, dump, monkeypatch):
    calls = []

    def failing_put(remote_path, data):
        calls.append(remote_path)
        return 500

    monkeypatch.setattr(backup, "_put", failing_put)
    assert backup.upload_file(dump, "b/db.sql", part_size=300) is False
    assert calls == ["b/db.sql.part001"]
