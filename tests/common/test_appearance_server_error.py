"""A scrapeMM server that cannot be reached is a run-level condition, not a
per-appearance failure: `appearance_from_url` converts scrapeMM's `ServerError`
into VeriTaS's own `QuotaExceededError` (imported locally to avoid a circular
import - this module sits upstream of `veritas.models`), so it aborts the run
instead of being recorded as "no original URL"."""

import pytest
from scrapemm.common.exceptions import ServerError

from veritas.common import appearance as appearance_module
from veritas.models import QuotaExceededError


@pytest.mark.asyncio
async def test_server_error_propagates_as_quota_exceeded(monkeypatch):
    async def fake_unshorten(url, session=None):
        return url

    async def fake_resolve_archiving_url(url):
        raise ServerError("Could not reach the scrapeMM server at http://localhost:8080: timeout")

    monkeypatch.setattr(appearance_module, "unshorten", fake_unshorten)
    monkeypatch.setattr(appearance_module, "is_archiving_url", lambda url: True)
    monkeypatch.setattr(appearance_module, "resolve_archiving_url", fake_resolve_archiving_url)

    with pytest.raises(QuotaExceededError):
        await appearance_module.appearance_from_url("https://archive.ph/abcde")


@pytest.mark.asyncio
async def test_a_plain_resolution_failure_still_yields_no_url(monkeypatch):
    """Regression guard: the new `except ServerError` clause must not swallow
    the ordinary "could not resolve" outcome (`resolve_archiving_url` returning
    None), which is not an error at all."""
    saved = []

    async def fake_unshorten(url, session=None):
        return url

    async def fake_resolve_archiving_url(url):
        return None

    async def fake_save(self):
        saved.append(self)
        self.id = 1

    monkeypatch.setattr(appearance_module, "unshorten", fake_unshorten)
    monkeypatch.setattr(appearance_module, "is_archiving_url", lambda url: True)
    monkeypatch.setattr(appearance_module, "resolve_archiving_url", fake_resolve_archiving_url)
    monkeypatch.setattr(appearance_module.Appearance, "save_to_db", fake_save)

    appearance = await appearance_module.appearance_from_url("https://archive.ph/abcde")

    assert appearance.url is None
    assert str(appearance.archive_url) == "https://archive.ph/abcde"
    assert saved == [appearance]
