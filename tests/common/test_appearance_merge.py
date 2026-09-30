"""`Appearance.save_to_db()` merges with an existing row on a unique-constraint
collision (same `url`/`archive_url`) instead of failing outright.

Regression test: `_merge_with_existing` used to call `db.replace_appearance()`
with keyword arguments (`to_replace=`, `replacement=`) that did not match its
actual parameters (`to_replace_id`, `replacement_id`), so every such merge
crashed with a `TypeError` right after resolving the collision.
"""

import pytest
from pydantic import HttpUrl

from veritas.common.appearance import Appearance
from veritas.common.base_model import VeritasBaseModel


@pytest.mark.asyncio
async def test_merge_calls_replace_appearance_with_matching_parameter_names(monkeypatch):
    existing = Appearance(url="https://example.org/article", id=7)
    app = Appearance(archive_url="https://archive.ph/abcde", id=3)
    app.url = HttpUrl("https://example.org/article")  # now collides with `existing`

    calls = []

    class FakeDB:
        async def get_appearance_by_url(self, url):
            return existing

        async def get_appearance_by_archive_url(self, archive_url):
            return None

        # Same signature as the real `VeritasDB.replace_appearance`: a keyword
        # mismatch here would raise TypeError, exactly as it did for real.
        async def replace_appearance(self, to_replace_id: int, replacement_id: int):
            calls.append((to_replace_id, replacement_id))

    import veritas.db as db_module
    monkeypatch.setattr(db_module, "db", FakeDB())

    saved = []

    async def fake_base_save(self):
        saved.append(self)

    monkeypatch.setattr(VeritasBaseModel, "save_to_db", fake_base_save)

    await app._merge_with_existing()

    assert calls == [(7, 3)]
    assert saved == [app]
    assert app.id == 3  # the initiating row survives; `existing` is superseded
