"""Pure tests for retrieving the IFCN and EFCSN signatories. No network access:
static requests and scrapeMM are replaced by fakes."""

from types import SimpleNamespace

import pytest

from veritas.pipeline.util import signatories
from veritas.pipeline.util.signatories import IFCN_CODE_OF_PRINCIPLES_DOMAIN, ifcn_profile_urls_from_api_response
from veritas.util import scraping


def test_ifcn_profile_urls_keep_only_verified_and_in_renewal():
    data = {"organizations": [
        {"slug": "verified-org", "signatory_status": "Verified Signatory"},
        {"slug": "renewing-org", "signatory_status": "In Renewal"},
        {"slug": "expired-org", "signatory_status": "Expired Signatory"},
        {"slug": "verified-org", "signatory_status": "Verified Signatory"},  # Duplicate
        {"slug": "", "signatory_status": "Verified Signatory"},
        {"signatory_status": "Verified Signatory"},
        "garbage",
    ]}
    assert ifcn_profile_urls_from_api_response(data) == [
        f"{IFCN_CODE_OF_PRINCIPLES_DOMAIN}/profile/verified-org",
        f"{IFCN_CODE_OF_PRINCIPLES_DOMAIN}/profile/renewing-org",
    ]


@pytest.mark.parametrize("data", [{}, {"organizations": None}, {"organizations": []}])
def test_ifcn_profile_urls_of_empty_response(data):
    assert ifcn_profile_urls_from_api_response(data) == []


@pytest.mark.asyncio
async def test_ifcn_profile_urls_are_fetched_from_the_api(monkeypatch):
    requested = []

    async def fake_request_static(url, session, **kwargs):
        requested.append(url)
        return '{"organizations": [{"slug": "org", "signatory_status": "In Renewal"}]}'

    monkeypatch.setattr(signatories, "request_static", fake_request_static)
    assert await signatories.get_ifcn_signatories_profile_urls() == [
        f"{IFCN_CODE_OF_PRINCIPLES_DOMAIN}/profile/org"]
    assert requested == [signatories.IFCN_SIGNATORIES_API_URL]


@pytest.mark.parametrize("response", [None, "<html>Not JSON</html>"])
@pytest.mark.asyncio
async def test_ifcn_api_failure_yields_no_urls(monkeypatch, response):
    async def fake_request_static(url, session, **kwargs):
        return response

    monkeypatch.setattr(signatories, "request_static", fake_request_static)
    assert await signatories.get_ifcn_signatories_profile_urls() == []


@pytest.mark.asyncio
async def test_rendered_htmls_come_from_scrapemm(monkeypatch):
    calls = []

    async def fake_retrieve(urls, **kwargs):
        calls.append((urls, kwargs))
        return [SimpleNamespace(success=True, content=SimpleNamespace(html="<h1>A</h1>")),
                SimpleNamespace(success=False, content=None)]

    monkeypatch.setattr(scraping, "retrieve", fake_retrieve)
    assert await scraping.get_rendered_htmls(["https://a.example", "https://b.example"]) == ["<h1>A</h1>", None]
    assert calls[0][0] == ["https://a.example", "https://b.example"]
    assert calls[0][1]["output_format"] == "html"


@pytest.mark.asyncio
async def test_no_rendering_without_urls(monkeypatch):
    async def fake_retrieve(urls, **kwargs):
        raise AssertionError("scrapeMM must not be called")

    monkeypatch.setattr(scraping, "retrieve", fake_retrieve)
    assert await scraping.get_rendered_htmls([]) == []


@pytest.mark.asyncio
async def test_all_links_of_static_page(monkeypatch):
    async def fake_request_static(url, session, **kwargs):
        return '<a href="/organization/a">A</a><a>No link</a><a href="https://x.example">X</a>'

    monkeypatch.setattr(scraping, "request_static", fake_request_static)
    assert await scraping.get_all_links("https://members.efcsn.com/signatories") == [
        "/organization/a", "https://x.example"]


@pytest.mark.asyncio
async def test_no_links_if_page_unavailable(monkeypatch):
    async def fake_request_static(url, session, **kwargs):
        return None

    monkeypatch.setattr(scraping, "request_static", fake_request_static)
    assert await scraping.get_all_links("https://members.efcsn.com/signatories") is None
