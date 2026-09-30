"""Backfilling an appearance's original URL from its Archive.today snapshot.

`resolve_missing_archive_today_original_urls` (used by
`scripts/resolve_archive_today_original_urls.py`) re-resolves every appearance
whose original URL is still missing, now that scrapeMM's `identify_snapshot()`
can answer short-code Archive.today URLs that used to need Decodo.
"""

import pytest

from veritas.common import appearance as appearance_module
from veritas.common.appearance import Appearance


def make_appearance(*, archive_url: str, id: int = 1) -> Appearance:
    app = Appearance(archive_url=archive_url)
    app.id = id
    return app


@pytest.fixture(autouse=True)
def no_op_save(monkeypatch):
    saved = []

    async def fake_save(self):
        saved.append(self)

    monkeypatch.setattr(Appearance, "save_to_db", fake_save)
    return saved


@pytest.mark.asyncio
async def test_backfills_the_original_url_when_resolvable(monkeypatch, no_op_save):
    import veritas.db as db_module

    app = make_appearance(archive_url="https://archive.ph/abcde")

    class FakeDB:
        async def get_appearances_missing_original_url(self):
            return [app]

    async def fake_resolve(url):
        assert url == "https://archive.ph/abcde"
        return {"original_url": "https://example.org/article"}

    monkeypatch.setattr(db_module, "db", FakeDB())
    monkeypatch.setattr(appearance_module, "resolve_archive_today_url", fake_resolve)

    n_candidates, n_resolved = await appearance_module.resolve_missing_archive_today_original_urls()

    assert n_candidates == 1
    assert n_resolved == 1
    assert str(app.url) == "https://example.org/article"
    assert no_op_save == [app]


@pytest.mark.asyncio
async def test_leaves_the_url_empty_when_still_unresolvable(monkeypatch, no_op_save):
    import veritas.db as db_module

    app = make_appearance(archive_url="https://archive.ph/abcde")

    class FakeDB:
        async def get_appearances_missing_original_url(self):
            return [app]

    async def fake_resolve(url):
        return None

    monkeypatch.setattr(db_module, "db", FakeDB())
    monkeypatch.setattr(appearance_module, "resolve_archive_today_url", fake_resolve)

    n_candidates, n_resolved = await appearance_module.resolve_missing_archive_today_original_urls()

    assert n_candidates == 1
    assert n_resolved == 0
    assert app.url is None
    assert no_op_save == []


@pytest.mark.asyncio
async def test_one_failing_resolution_does_not_abort_the_rest(monkeypatch, no_op_save):
    import veritas.db as db_module

    failing = make_appearance(archive_url="https://archive.ph/broken", id=1)
    working = make_appearance(archive_url="https://archive.ph/ok", id=2)

    class FakeDB:
        async def get_appearances_missing_original_url(self):
            return [failing, working]

    async def fake_resolve(url):
        if "broken" in url:
            raise RuntimeError("boom")
        return {"original_url": "https://example.org/article"}

    monkeypatch.setattr(db_module, "db", FakeDB())
    monkeypatch.setattr(appearance_module, "resolve_archive_today_url", fake_resolve)

    n_candidates, n_resolved = await appearance_module.resolve_missing_archive_today_original_urls()

    assert n_candidates == 2
    assert n_resolved == 1
    assert failing.url is None
    assert str(working.url) == "https://example.org/article"
