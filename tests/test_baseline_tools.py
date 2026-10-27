"""Tests for the baseline's web_search and fetch_url tools, with scrapeMM mocked."""

import asyncio
import io
from datetime import date

import ezmm
import pytest
from ezmm.common.registry import item_registry
from PIL import Image as PILImage
from scrapemm import QuotaExceededError, RateLimitError, TargetUnavailableError
from scrapemm.common import ScrapingResponse
from scrapemm.search import SerperAnswerBox, SerperKnowledgeGraph, SerperOrganicResult, SerperResponse

from baseline_fakes import ezmm_registry, page  # noqa: F401  # ezmm_registry is a fixture

from eval.baselines.common import tools
from eval.baselines.common.prompts import build_prompt
from eval.baselines.common.media import MediaLimits
from eval.baselines.common.tools import MAX_FETCH_CHARS, SearchQuotaExceededError, ToolSession, normalize_url


def _serper_response() -> SerperResponse:
    return SerperResponse(
        organic=[
            SerperOrganicResult(title="Old article", link="https://example.com/old",
                                snippet="Before the claim.", date="Jan 3, 2024"),
            SerperOrganicResult(title="No link", link=None),
        ],
        # Not date-filtered by Google, must never reach the model
        knowledge_graph=SerperKnowledgeGraph(title="Leaky knowledge graph"),
        answer_box=SerperAnswerBox(answer="Leaky answer box", link="https://example.com/answer"),
    )


@pytest.fixture
def searches(monkeypatch):
    """Replace scrapeMM's search with a fake that records the queries."""
    queries = []

    async def fake_search(query):
        queries.append(query)
        return _serper_response()

    monkeypatch.setattr(tools, "search", fake_search)
    return queries


def _fake_retrieve(monkeypatch, response_for):
    calls = []

    async def fake_retrieve(url, **kwargs):
        calls.append((url, kwargs))
        return response_for(url)

    monkeypatch.setattr(tools, "retrieve", fake_retrieve)
    return calls


def test_web_search_filters_by_claim_day_and_formats_organic_only(searches):
    session = ToolSession(claim_date="2024-02-05T12:22:02")
    text = session.web_search("eiffel tower")

    assert searches[0].q == "eiffel tower"
    assert searches[0].before == date(2024, 2, 5)
    assert "https://example.com/old" in text
    assert "Jan 3, 2024" in text
    assert "Before the claim." in text
    assert "Leaky" not in text
    assert session.encountered_urls == ["https://example.com/old"]
    assert session.retrieved_urls == []


def test_log_lines_name_the_claim(searches, capsys):
    """Claims run in parallel, so each tool log line must say which claim it belongs to."""
    session = ToolSession(claim_date="2024-02-05", claim_id=124824)
    session.web_search("eiffel tower")
    session.fetch_url("https://unseen.org/page")

    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "    [124824] web_search: 'eiffel tower' -> 1 results",
        "    [124824] fetch_url: https://unseen.org/page -> rejected, URL not seen",
    ]


def test_web_search_limit(searches):
    session = ToolSession(claim_date="2024-02-05", max_searches=1)
    session.web_search("first")
    text = session.web_search("second")

    assert {query.q for query in searches} == {"first"}
    assert "limit" in text.lower()


def test_web_search_quota_exceeded_terminates_run(monkeypatch):
    async def fake_search(query):
        raise QuotaExceededError("Serper credits used up")

    monkeypatch.setattr(tools, "search", fake_search)
    with pytest.raises(SearchQuotaExceededError):
        ToolSession(claim_date="2024-02-05").web_search("q")


def test_web_search_retries_rate_limits(monkeypatch):
    attempts = []

    async def fake_search(query):
        attempts.append(query)
        if len(attempts) < 3:
            raise RateLimitError("slow down")
        return _serper_response()

    monkeypatch.setattr(tools, "search", fake_search)
    monkeypatch.setattr(tools.time, "sleep", lambda _: None)
    text = ToolSession(claim_date="2024-02-05").web_search("q")

    assert [query.page for query in attempts] == [1, 1, 1, 2]
    assert "https://example.com/old" in text


def _session_that_saw(*urls: str, **kwargs) -> ToolSession:
    """A session whose web_search results included `urls`, so fetch_url accepts them."""
    session = ToolSession(claim_date="2024-02-05", **kwargs)
    session.encountered_urls += urls
    return session


def test_fetch_url_truncates_content(monkeypatch):
    calls = _fake_retrieve(monkeypatch, lambda url: page("Q" * (MAX_FETCH_CHARS + 5000)))
    session = _session_that_saw("https://example.com/page", scrape_methods=["firecrawl"])
    text = session.fetch_url("https://example.com/page").text

    assert calls[0][1]["output_format"] == "multimodal"
    assert calls[0][1]["methods"] == ["firecrawl"]
    assert calls[0][1]["strip"] is True  # Main content only, without the page's UI
    assert text.count("Q") == MAX_FETCH_CHARS
    assert f"[Content truncated to {MAX_FETCH_CHARS} characters]" in text
    assert session.retrieved_urls == ["https://example.com/page"]
    assert session.linked_urls == []


def test_fetch_url_failure_returns_error_text(monkeypatch):
    _fake_retrieve(monkeypatch, lambda url: ScrapingResponse(
        url=url, content=None, output_format="multimodal",
        errors={"firecrawl": TargetUnavailableError("404 Not Found")},
    ))
    session = _session_that_saw("https://example.com/gone")
    text = session.fetch_url("https://example.com/gone").text

    assert text.startswith("Could not retrieve https://example.com/gone")
    assert "TargetUnavailableError" in text
    assert session.retrieved_urls == []


def test_fetch_url_limit(monkeypatch):
    calls = _fake_retrieve(monkeypatch, lambda url: page("content"))
    session = _session_that_saw("https://example.com/a", "https://example.com/b", max_fetches=1)
    session.fetch_url("https://example.com/a")
    text = session.fetch_url("https://example.com/b").text

    assert len(calls) == 1
    assert "limit" in text.lower()


def test_fetch_url_rejects_unseen_urls(monkeypatch):
    """Any other URL would bypass the cutoff: a search engine's results page, an archived copy,
    or a fact-check the model remembers on a site it has seen."""
    calls = _fake_retrieve(monkeypatch, lambda url: None)
    session = _session_that_saw("https://example.com/old", "https://www.snopes.com/fact-check/older-claim/")

    for url in ("https://www.bing.com/search?q=claim", "https://snopes.com/fact-check/the-claim/",
                "https://example.com/other", "https://example.com/old?page=2", "https://sub.example.com/old",
                "https://web.archive.org/web/2026/https://example.com/old", "https://[malformed"):
        assert session.fetch_url(url).text.startswith(f"Error: Cannot fetch {url}.")

    assert calls == []
    assert session.n_fetches == 0  # Rejected fetches don't use up the budget
    assert session.retrieved_urls == []


def test_fetch_url_allows_urls_from_search_results_and_fetched_pages(monkeypatch, searches):
    pages = {
        "https://www.example.com/old": (
            "Per [AFP](https://factcheck.afp.com/x), see also https://archive.example.org/y. [Home](/) "
            + "Q" * MAX_FETCH_CHARS + " https://hidden.org/past-the-truncation"
        ),
    }
    calls = _fake_retrieve(monkeypatch, lambda url: page(pages.get(url, "content")))
    session = ToolSession(claim_date="2024-02-05")
    session.web_search("eiffel tower")  # Result: https://example.com/old

    # "www." makes no difference
    assert "Content of https://www.example.com/old" in session.fetch_url("https://www.example.com/old").text
    # Relative links are saved as full URLs
    assert session.linked_urls == [
        "https://factcheck.afp.com/x", "https://archive.example.org/y", "https://www.example.com/"]

    # Linked URLs become fetchable, written as the model may write them, but not other pages of
    # their sites, nor what was cut off
    assert "Content of" in session.fetch_url("http://factcheck.afp.com/x/#top").text
    for url in ("https://factcheck.afp.com/other", "https://afp.com/x", "https://hidden.org/past-the-truncation"):
        assert session.fetch_url(url).text.startswith("Error: Cannot fetch")
    assert [url for url, _ in calls] == ["https://www.example.com/old", "http://factcheck.afp.com/x/#top"]


def test_the_claims_fact_check_is_never_shown_or_fetched(monkeypatch, capsys):
    """It would give the answer away. Google's date filter lets it through at times, and
    fact-checkers' pages link to it in their current versions."""
    fact_check = "https://www.factcheck.example.org/2026/07/the-claim-is-false/"

    async def fake_search(query):
        return SerperResponse(organic=[SerperOrganicResult(title="The claim is false", link=fact_check + "?utm_source=x"),
                                       SerperOrganicResult(title="Old article", link="https://example.com/old")])

    monkeypatch.setattr(tools, "search", fake_search)
    calls = _fake_retrieve(monkeypatch, lambda url: page(
        f'See [The claim is false]({fact_check}#top "The claim is false") and [another](https://example.com/other).'))
    session = ToolSession(claim_date="2026-07-10", blocked_urls=[fact_check], claim_id=7)

    text = session.web_search("the claim")
    assert "The claim is false" not in text and "factcheck.example.org" not in text
    assert session.encountered_urls == ["https://example.com/old"]

    # The page's link to the fact-check is gone, with its text and title
    text = session.fetch_url("https://example.com/old").text
    assert "See  and [another](https://example.com/other)." in text and "The claim is false" not in text
    assert session.linked_urls == ["https://example.com/other"]
    for url in (fact_check, "http://factcheck.example.org/2026/07/the-claim-is-false#top",
                # Pages under it are blocked, other paths of the article are not, but not seen either
                "https://factcheck.example.org/2026/08/the-claim-is-false/", "https://factcheck.example.org/the-claim-is-false/amp"):
        # As for any URL not seen: another answer would tell the model that it guessed the fact-check
        assert session.fetch_url(url).text == (f"Error: Cannot fetch {url}. fetch_url only fetches URLs that appeared "
                                               f"in web_search results or in fetched pages. Use them as they appeared.")
    assert [url for url, _ in calls] == ["https://example.com/old"]
    assert session.n_fetches == 1 and session.retrieved_urls == ["https://example.com/old"]
    log = capsys.readouterr().out
    assert "web_search: 'the claim' -> 1 results (1 blocked, 2 pages)" in log
    assert "fetch_url: https://example.com/old -> 46 chars, 0 media shown, 1 blocked links removed" in log


def test_fetch_url_removes_the_images_in_a_link_to_the_claims_fact_check(monkeypatch, tmp_path, ezmm_registry):
    """As in a fact-checker's list of its latest articles: the teaser's image shows what the fact-check is about."""
    fact_check = "https://factcheck.example.org/doc.C8PX8KE"
    teaser, other = _image(tmp_path, "teaser", "red"), _image(tmp_path, "other", "green")
    _fake_retrieve(monkeypatch, lambda url: page(
        "Latest:\n[", teaser, f"\n### The claim is false]({fact_check})\n[", other, "\n### Another](/doc.X)"))
    session = _session_that_saw("https://factcheck.example.org/doc.36L44H2", blocked_urls=[fact_check])

    result = session.fetch_url("https://factcheck.example.org/doc.36L44H2")

    assert [part for part in result.content if not isinstance(part, str)] == [other]
    assert "The claim is false" not in result.text and f"Latest:\n\n[ {other.reference}" in result.text
    assert session.linked_urls == ["https://factcheck.example.org/doc.X"]


@pytest.mark.parametrize("url, blocked", [
    ("http://thip.media/health-news-fact-check/fact-check-can-eating-dalchini/153121?utm_source=x#top", True),
    ("https://www.thip.media/Health-News-Fact-Check/fact-check-can-eating-dalchini/153121/amp/", True),
    ("https://www.thip.media/health-news-fact-check/fact-check-can-eating-dalchini-daily/153122", False),
])
def test_the_claims_fact_checks_are_blocked_with_the_pages_under_them(url, blocked):
    session = ToolSession(blocked_urls=["https://www.thip.media/health-news-fact-check/fact-check-can-eating-dalchini/153121/"])
    assert session.is_blocked(url) == blocked


LEAD_STORIES_SCOPES = ["leadstories.com/hoax-alert", "verifica.efe.com", "xn--80aa2aboqjl0g5e.leadstories.com/factcheck"]


@pytest.mark.parametrize("url, blocked", [
    ("https://www.leadstories.com/hoax-alert/2026/07/fact-check-other.html", True),
    ("http://leadstories.com/Hoax-Alert/", True),
    ("https://verifica.efe.com/anything/", True),
    (f"https://{'xn--80aa2aboqjl0g5e'.encode().decode('idna')}.leadstories.com/factcheck/x", True),  # Unicode host
    ("https://leadstories.com/about", False),
    ("https://leadstories.com/hoax-alerts/x", False),  # Not under the path
    ("https://efe.com/verifica/x", False),
    ("https://arabic.leadstories.com/hoax-alert/x", False),  # Subdomains are scopes of their own
])
def test_the_fact_check_scopes_of_the_publishers_are_blocked(url, blocked):
    """The publishers' fact-checks, but not the rest of their sites."""
    assert ToolSession(blocked_scopes=LEAD_STORIES_SCOPES).is_blocked(url) == blocked


def test_the_tools_show_no_fact_check_of_the_publisher(monkeypatch, capsys):
    snippets = [
        "Fact Check: Other claim false. leadstories.com/hoax-alert/2026/07/fact-check-other.html",  # Without scheme
        "A post on https://leadstories.com/hoa ... and the rest",  # Cut short: the start of the scope
        "Lead Stories reports on hoaxes, see leadstories.com/about",
    ]

    async def fake_search(query):
        return SerperResponse(organic=[
            SerperOrganicResult(title="In the scope", link="https://leadstories.com/hoax-alert/2026/06/x.html"),
            *(SerperOrganicResult(title=f"Post {i}", link=f"https://www.instagram.com/p/{i}/", snippet=snippet)
              for i, snippet in enumerate(snippets))])

    monkeypatch.setattr(tools, "search", fake_search)
    calls = _fake_retrieve(monkeypatch, lambda url: page(
        "Debunked by [Fact Check: It is false](https://leadstories.com/hoax-alert/2026/07/y.html), see [about](https://leadstories.com/about)."))
    session = ToolSession(claim_date="2026-07-10", blocked_scopes=LEAD_STORIES_SCOPES)

    text = session.web_search("the claim")
    assert "In the scope" not in text and "Post 0" not in text and "Post 1" not in text
    assert f"Snippet: {snippets[2]}" in text
    assert session.encountered_urls == ["https://www.instagram.com/p/2/"]

    url = "https://leadstories.com/hoax-alert/2026/06/x.html"
    assert session.fetch_url(url).text.startswith(f"Error: Cannot fetch {url}.")
    session.encountered_urls.append("https://example.com/page")
    text = session.fetch_url("https://example.com/page").text
    assert "Debunked by , see [about](https://leadstories.com/about)." in text
    assert [u for u, _ in calls] == ["https://example.com/page"]


def test_search_results_that_show_a_blocked_url_are_left_out(monkeypatch, capsys):
    """Fact-checkers' posts announce their fact-checks with the headline and a link, which Google
    may break with spaces or cut short. Such results are left out whole; the others stay as they are."""
    fact_check = "https://www.thip.media/health-news-fact-check/fact-check-can-eating-dalchini/153121/"
    shown = [
        f"Fact Check: Can eating dalchini cure diabetes? {fact_check}?utm_medium=social",
        "Read thip.media/health-news-fact-check/fact-check-can-eating-dalchini/153121 now",  # Without scheme
        "Read: https://thip. media/health-news-fact-check/fact-check-can- eating-dalchini/153121/ Thanks",  # Broken
        "Via https://www.thip.media/health-news-fact-check/fact-check-can-eat ...",  # Cut short within the page
        "Fact Check: Other claim false. leadstories.com/hoax-alert/2026/07/fact-check-other.html",
        "Via https://leadstories. com/hoax-alert/2026/07/fact-check- the-cl ...",
        "More at https://leadstories.com/hoax-al ...",  # Cut short within the scope's path
    ]
    kept = [
        "Other https://www.thip.media/health-news-fact-check/other-article-on-the-site/153999/ and https://example.org/a...",
        "Cut short https://www.thip.media/health-news-fact-check/fact-check-can-eat-more ...",
        "Lead Stories reports on hoaxes, see https://leadstories.com/about and leadstories.com/hoax-alerts ...",
    ]

    async def fake_search(query):
        return SerperResponse(organic=[SerperOrganicResult(title=f"Post {i}", link=f"https://www.instagram.com/p/{i}/",
                                                           snippet=snippet) for i, snippet in enumerate(shown + kept)])

    monkeypatch.setattr(tools, "search", fake_search)
    session = ToolSession(claim_date="2026-07-10", blocked_urls=[fact_check], blocked_scopes=LEAD_STORIES_SCOPES)

    text = session.web_search("the claim")

    assert not any(f"Post {i}" in text for i in range(len(shown)))
    assert all(f"Snippet: {snippet}" in text for snippet in kept)
    assert session.encountered_urls == [f"https://www.instagram.com/p/{i}/" for i in range(len(shown), len(shown + kept))]
    assert f"web_search: 'the claim' -> {len(kept)} results ({len(shown)} blocked, 2 pages)" in capsys.readouterr().out


@pytest.mark.parametrize("url, blocked", [
    ("https://www.dogrulukpayi.com/dogruluk-kontrolu/ucuz-mama-kedileri-gay-yapar-mi", True),
    ("https://dogrulukpayi.com/dogrulama/ucuz-mama-kedileri-gay-yapar-mi?utm_source=x", True),
    ("https://www.dogrulukpayi.com/dogrulama/ucuz-mama-kedileri", False),
    ("https://web.archive.org/web/2026/https://www.dogrulukpayi.com/dogrulama/ucuz-mama-kedileri-gay-yapar-mi", True),
    ("https://example.com/ucuz-mama-kedileri-gay-yapar-mi", True),  # A copy
    ("https://aapnews.aap.com.au/news/bird-flu-denial-sweeps-social-media-as-australian-cases-surge", True),
    ("https://www.thip.media/health-news-fact-check/fact-check-can-eating-dalchini/153121/", True),
    ("https://www.thip.media/health-news-fact-check/other-article-on-the-site/153999/", False),
    ("https://boomlive.in/n-31898", True),
    ("https://boomlive.in/other/n-31898", False),  # Slug too short to tell the article apart
])
def test_other_urls_of_the_claims_fact_check_are_blocked(url, blocked):
    """Sites serve an article under several paths and on subdomains, and archives and other sites copy it."""
    session = ToolSession(blocked_urls=[
        "https://www.dogrulukpayi.com/dogruluk-kontrolu/ucuz-mama-kedileri-gay-yapar-mi",
        "https://www.aap.com.au/factcheck/bird-flu-denial-sweeps-social-media-as-australian-cases-surge/",
        "https://www.thip.media/uncategorized/fact-check-can-eating-dalchini/153121/",
        "https://boomlive.in/n-31898"])
    assert session.is_blocked(url) == blocked


def test_search_results_with_the_slug_of_the_claims_fact_check_are_left_out(monkeypatch):
    """Fact-checkers' posts on other sites give their fact-checks' headlines and slugs without the
    host, which Google may break with spaces or cut short at either end."""
    fact_check = ("https://leadstories.com/hoax-alert/2026/07/fact-check-euronews-did-not-report-that-zelenskyys-"
                  "parents-were-detained-for-laundering-11-million.html")
    shown = [
        "Euronews didn't publish this video ... detained-for-laundering-11-million.html.",
        "... 2026/07/fact-check-euronews-did- not-report-that-zelenskyys-parents-were-detained-for-laundering-11-million.html",
        "Fact Check: Euronews Did NOT Report ... fact-check-euronews-did-not-report-that ...",
    ]
    kept = [
        "Zelenskyy's parents detained? Reports of 11-million ...",  # A piece too short to tell
        "See zelenskyys-parents-were-arrested-in-kyiv-2026 ...",  # Another article's slug
    ]

    async def fake_search(query):
        return SerperResponse(organic=[SerperOrganicResult(title=f"Post {i}", link=f"https://www.instagram.com/p/{i}/",
                                                           snippet=snippet) for i, snippet in enumerate(shown + kept)])

    monkeypatch.setattr(tools, "search", fake_search)
    session = ToolSession(claim_date="2026-07-10", blocked_urls=[fact_check])

    text = session.web_search("the claim")

    assert not any(f"Post {i}" in text for i in range(len(shown)))
    assert all(f"Snippet: {snippet}" in text for snippet in kept)


def test_the_log_says_what_was_blocked_and_by_which_rule(monkeypatch, capsys):
    fact_check = "https://leadstories.com/hoax-alert/2026/07/fact-check-the-claim-is-false.html"
    results = [
        ("https://fullfact.org/health/x/", ""),
        (fact_check + "?utm_source=x", ""),
        ("https://web.archive.org/web/2026/" + fact_check, ""),
        ("https://www.instagram.com/p/1/", "Fact Check: ... fact-check-the-claim-is ..."),
        ("https://www.instagram.com/p/2/", "Read verifica.efe.com/2026/x"),
        ("https://www.instagram.com/p/3/", "More at https://leadstories.com/hoax-al ..."),
        ("https://example.com/page", "Nothing blocked"),
    ]

    async def fake_search(query):
        return SerperResponse(organic=[SerperOrganicResult(title="Result", link=link, snippet=snippet)
                                       for link, snippet in results])

    monkeypatch.setattr(tools, "search", fake_search)
    _fake_retrieve(monkeypatch, lambda url: page(
        "See [it](https://fullfact.org/a), [again](https://fullfact.org/a) and https://verifica.efe.com/b."))
    session = ToolSession(claim_date="2026-07-10", blocked_urls=[fact_check],
                          blocked_scopes=["fullfact.org", "verifica.efe.com", "leadstories.com/hoax-alert"])

    session.web_search("q")
    session.fetch_url("https://fullfact.org/other")
    session.fetch_url("https://example.com/page")

    log = capsys.readouterr().out
    for line in [
        "web_search: 'q': dropped https://fullfact.org/health/x/, a URL in the fact-check scope fullfact.org",
        f"web_search: 'q': dropped {fact_check}?utm_source=x, a URL with the slug fact-check-the-claim-is-false.html "
        "of the claim's fact-check in its path",
        f"web_search: 'q': dropped https://web.archive.org/web/2026/{fact_check}, a URL with the slug "
        "fact-check-the-claim-is-false.html of the claim's fact-check in its path",
        "web_search: 'q': dropped https://www.instagram.com/p/1/, whose title or snippet has 'fact-check-the-claim-is', "
        "a piece of the slug fact-check-the-claim-is-false.html of the claim's fact-check",
        "web_search: 'q': dropped https://www.instagram.com/p/2/, whose title or snippet shows verifica.efe.com/2026/x, "
        "a URL in the fact-check scope verifica.efe.com",
        "web_search: 'q': dropped https://www.instagram.com/p/3/, whose title or snippet shows leadstories.com/hoax-al…, "
        "cut short, the start of a URL at or under the claim's fact-check " + fact_check,
        "web_search: 'q' -> 1 results (6 blocked, 2 pages)",
        "fetch_url: https://fullfact.org/other -> blocked, a URL in the fact-check scope fullfact.org",
        "fetch_url: https://example.com/page -> 12 chars, 0 media shown, 3 blocked links removed",
        "fetch_url: https://example.com/page: removed the links to 1 URLs in the fact-check scope fullfact.org: "
        "https://fullfact.org/a",
        "fetch_url: https://example.com/page: removed the links to 1 URLs in the fact-check scope verifica.efe.com: "
        "https://verifica.efe.com/b",
    ]:
        assert line in log
    # The results record them too, as the log has them
    assert [(entry["tool"], entry["url"]) for entry in session.blocked] == [
        *(("web_search", link) for link, _ in results[:6]), ("fetch_url", "https://fullfact.org/other")]
    assert session.blocked[0] == {"tool": "web_search", "query": "q", "url": "https://fullfact.org/health/x/",
                                  "rule": "a URL in the fact-check scope fullfact.org"}
    assert session.blocked[-1]["rule"] == "a URL in the fact-check scope fullfact.org"


def _google_pages(monkeypatch, pages: dict[int, list[str]]) -> list[int]:
    """Make the pages of Google results hold results with the URLs in `pages`, by page number.
    Returns the pages requested."""
    requested = []

    async def fake_search(query):
        requested.append(query.page)
        return SerperResponse(organic=[SerperOrganicResult(title="Result", link=url) for url in pages.get(query.page, [])])

    monkeypatch.setattr(tools, "search", fake_search)
    return requested


def test_web_search_requests_further_pages_for_the_blocked_results(monkeypatch, capsys):
    page_1 = ["https://a.org/1", "https://blocked.org/1", "https://a.org/2", "https://blocked.org/2",
              *(f"https://a.org/{i}" for i in range(3, 9))]
    page_2 = [f"https://b.org/{i}" for i in range(1, 11)]
    requested = _google_pages(monkeypatch, {1: page_1, 2: page_2, 3: ["https://c.org/1"]})
    session = ToolSession(claim_date="2026-07-10", blocked_scopes=["blocked.org"])

    session.web_search("q")

    assert requested == [1, 2]
    assert session.encountered_urls == [f"https://a.org/{i}" for i in range(1, 9)] + ["https://b.org/1", "https://b.org/2"]
    assert "web_search: 'q' -> 10 results (2 blocked, 2 pages)" in capsys.readouterr().out


def test_web_search_requests_one_page_with_ten_unblocked_results(monkeypatch, capsys):
    requested = _google_pages(monkeypatch, {1: [f"https://a.org/{i}" for i in range(10)], 2: ["https://b.org/1"]})
    session = ToolSession(claim_date="2026-07-10", blocked_scopes=["blocked.org"])

    session.web_search("q")

    assert requested == [1]
    assert len(session.encountered_urls) == tools.NUM_SEARCH_RESULTS
    assert "web_search: 'q' -> 10 results\n" in capsys.readouterr().out


def test_web_search_requests_at_most_max_search_pages(monkeypatch):
    requested = _google_pages(monkeypatch, {page: [f"https://a.org/{page}", *(f"https://blocked.org/{page}/{i}" for i in range(9))]
                                            for page in range(1, 10)})
    session = ToolSession(claim_date="2026-07-10", blocked_scopes=["blocked.org"])

    session.web_search("q")

    assert requested == list(range(1, tools.MAX_SEARCH_PAGES + 1))
    assert session.encountered_urls == [f"https://a.org/{page}" for page in requested]


@pytest.mark.parametrize("page_3", [[], ["https://a.org/1", "https://blocked.org/1", "https://a.org/2"]], ids=["empty", "repeats"])
def test_web_search_stops_at_a_page_without_new_results(monkeypatch, page_3):
    """A page with fewer than ten results is not always the last, and later pages repeat some results."""
    requested = _google_pages(monkeypatch, {1: ["https://a.org/1", "https://blocked.org/1"],
                                            2: ["https://blocked.org/1", "https://a.org/2", "https://a.org/1"],
                                            3: page_3,
                                            4: ["https://a.org/4"]})
    session = ToolSession(claim_date="2026-07-10", blocked_scopes=["blocked.org"])

    results, n_blocked, n_pages = session._search("q", max_pages=5)

    assert [r.link for r in results] == ["https://a.org/1", "https://a.org/2"]
    assert (n_blocked, n_pages) == (1, 3)
    assert requested == [1, 2, 3]


def test_normalize_url():
    assert normalize_url("https://www.Example.org/a%20b/?q=1#top") == "example.org/a b?q=1"
    assert normalize_url("http://example.org") == normalize_url("https://example.org/") == "example.org"
    assert normalize_url("https://example.org/A") != normalize_url("https://example.org/a")  # Paths are case-sensitive
    assert normalize_url("https://[malformed") == normalize_url("not a url") == ""


def _png(path, color: str, size=(40, 20)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", size, color).save(path, "PNG")
    return str(path)


def _image(tmp_path, name: str, color: str, size=(2000, 1000)) -> ezmm.Image:
    """A page's image as scrapeMM delivers it: an ezMM item. Same files make the same item."""
    path = tmp_path / f"{name}.png"
    PILImage.new("RGB", size, color).save(path, "PNG")
    return ezmm.Image(file_path=str(path))


def _truncated_jpeg(tmp_path) -> ezmm.Image:
    """A JPEG cut off halfway, as a download may leave it: the image library opens it, but
    can't decode it."""
    buffer = io.BytesIO()
    PILImage.effect_noise((400, 300), 80).convert("RGB").save(buffer, "JPEG")
    path = tmp_path / "truncated.jpg"
    path.write_bytes(buffer.getvalue()[:len(buffer.getvalue()) // 2])
    return ezmm.Image(file_path=str(path))


def test_fetch_url_shows_the_page_media_in_their_places(monkeypatch, tmp_path, ezmm_registry):
    first, second, third, hidden = (_image(tmp_path, color, color) for color in ("red", "green", "blue", "white"))
    (tmp_path / "clip.mp4").write_bytes(b"mp4")
    video = ezmm.Video(file_path=str(tmp_path / "clip.mp4"))
    (tmp_path / "logo.svg").write_text("<svg/>")
    svg = ezmm.Image(file_path=str(tmp_path / "logo.svg"))
    truncated = _truncated_jpeg(tmp_path)
    _fake_retrieve(monkeypatch, lambda url: page(
        "Intro", first, "text", svg, truncated, second, first, video, third, "Q" * MAX_FETCH_CHARS, hidden))
    session = _session_that_saw("https://example.com/page", media_limits=MediaLimits(page_images=2))

    result = session.fetch_url("https://example.com/page")

    # The media shown stay ezMM items in their places. Notes replace the SVG and the truncated
    # JPEG, which can't be read, the repeated image and the third image, beyond the limit; the
    # image past the cut is gone
    assert [part for part in result.content if not isinstance(part, str)] == [first, second, video]
    unreadable = "[image in a format that cannot be shown]"
    assert (f"Intro {first.reference} text {unreadable} {unreadable} {second.reference} [the image shown above again] "
            f"{video.reference} [image not shown] QQQ") in result.text
    assert hidden.reference not in result.text
    assert f"[Content truncated to {MAX_FETCH_CHARS} characters]" in result.text
    assert result.media == [
        {"reference": first.reference, "kind": "image", "source_url": first.source_url},
        {"reference": second.reference, "kind": "image", "source_url": second.source_url},
        {"reference": video.reference, "kind": "video", "source_url": video.source_url},
    ]


def test_fetch_url_limits_the_media_of_the_whole_fact_check(monkeypatch, tmp_path, ezmm_registry):
    """Every model call sends all pages fetched so far: the totals keep it within the APIs' limits."""
    red, green, blue = (_image(tmp_path, color, color) for color in ("red", "green", "blue"))
    pages = {"https://example.com/a": page("A", red, green), "https://example.com/b": page("B", red, blue, "end")}
    _fake_retrieve(monkeypatch, lambda url: pages[url])
    session = _session_that_saw(*pages, media_limits=MediaLimits(total_images=2))

    session.fetch_url("https://example.com/a")
    second = session.fetch_url("https://example.com/b")

    # The image shown on the first page is not shown again; the third one is over the total
    assert second.text.endswith("B [the image shown above again] [image not shown] end \n**End of content of https://example.com/b:**")
    assert second.media == []
    assert [m["reference"] for m in session.shown_media] == [red.reference, green.reference]


def test_fetch_url_keeps_each_request_within_its_image_cap(monkeypatch, tmp_path, ezmm_registry):
    """Some hosts take fewer images per request than the other limits add up to, e.g. 64 on
    OpenRouter's Near AI. The claim's images count, and a video counts as its frames."""
    red, green, blue = (_image(tmp_path, color, color) for color in ("red", "green", "blue"))
    (tmp_path / "clip.mp4").write_bytes(b"mp4")
    video = ezmm.Video(file_path=str(tmp_path / "clip.mp4"))
    pages = {"https://example.com/a": page("A", red, video), "https://example.com/b": page("B", green, blue)}
    _fake_retrieve(monkeypatch, lambda url: pages[url])
    session = _session_that_saw(*pages, media_limits=MediaLimits(request_images=5, video_frames=2), claim_images=1)

    session.fetch_url("https://example.com/a")
    second = session.fetch_url("https://example.com/b")

    # 1 of the claim, 1 + 2 of the first page, 1 of the second: the next one is over the cap
    assert second.text.endswith(f"B {green.reference} [image not shown] \n**End of content of https://example.com/b:**")
    assert [m["reference"] for m in session.shown_media] == [red.reference, video.reference, green.reference]


def test_temporary_media_store(tmp_path):
    """ezMM's store is an SQLite database: on a network file system, each registration waits for a lock."""
    before = item_registry.path
    with tools.temporary_media_store() as store:
        assert item_registry.path == store != before and store.is_dir()
        image = ezmm.Image(file_path=_png(store / "image" / "a.png", "red"))
        assert (store / "item_registry.db").exists()
    assert item_registry.path == before and not store.exists()
    assert image.file_path.is_relative_to(store)


def test_page_media_files_are_deleted_once_no_fact_check_uses_them(monkeypatch, tmp_path):
    with tools.temporary_media_store() as store:
        shared, own = (ezmm.Image(file_path=_png(store / "image" / f"{name}.png", color))
                       for name, color in (("logo", "red"), ("photo", "green")))
        # A page's medium identical to the claim's image becomes the claim's file, outside the store
        claim_image = ezmm.Image(file_path=_png(tmp_path / "claim.png", "blue"))
        pages = {"https://a.org/x": page("A", shared, own, claim_image), "https://b.org/y": page("B", shared)}
        _fake_retrieve(monkeypatch, lambda url: pages[url])
        first, second = _session_that_saw("https://a.org/x"), _session_that_saw("https://b.org/y")
        first.fetch_url("https://a.org/x")
        second.fetch_url("https://b.org/y")

        first.close()
        assert not own.file_path.exists()
        assert shared.file_path.exists()  # The second fact-check still uses it
        second.close()
        assert not shared.file_path.exists()
        assert claim_image.file_path.exists()


def test_page_media_files_are_kept_outside_a_temporary_store(monkeypatch, tmp_path, ezmm_registry):
    image = _image(tmp_path, "red", "red")
    _fake_retrieve(monkeypatch, lambda url: page("A", image))
    session = _session_that_saw("https://a.org/x")
    session.fetch_url("https://a.org/x")
    session.close()
    assert image.file_path.exists()


def test_fetch_url_gives_up_on_a_page_that_hangs(monkeypatch, capsys):
    """A fetch can hang on the scrapeMM server, and with it the fact-check and the end of the run."""
    async def hanging_retrieve(url, **kwargs):
        await asyncio.sleep(60)

    monkeypatch.setattr(tools, "retrieve", hanging_retrieve)
    monkeypatch.setattr(tools, "FETCH_TIMEOUT", 0.1)
    session = _session_that_saw("https://example.com/slow", claim_id=3)

    result = session.fetch_url("https://example.com/slow")

    assert result.text == "Could not retrieve https://example.com/slow: the page took too long to load."
    assert session.n_fetches == 1 and session.retrieved_urls == []
    assert "[3] fetch_url: https://example.com/slow -> timed out after 0.1 s" in capsys.readouterr().out


def test_text_that_looks_like_a_reference_is_just_text(monkeypatch):
    """scrapeMM's client fails on a PDF's file item with this message. Read as an ezMM
    sequence, the reference in it would fail the claim."""
    async def failing_retrieve(url, **kwargs):
        raise ValueError("Item with reference <file:63> does not exist.")

    monkeypatch.setattr(tools, "retrieve", failing_retrieve)
    session = _session_that_saw("https://example.com/report.pdf")

    result = session.fetch_url("https://example.com/report.pdf")
    assert result.text == "Could not retrieve https://example.com/report.pdf: Item with reference <file:63> does not exist."
    assert result.content == [result.text]


def test_call_dispatch_and_invalid_input(monkeypatch):
    calls = _fake_retrieve(monkeypatch, lambda url: None)
    session = ToolSession(claim_date="2024-02-05")

    assert "Unknown tool" in session.call("delete_everything", {}).text
    assert "Invalid arguments" in session.call("fetch_url", None).text
    assert "not a valid URL" in session.call("fetch_url", {"url": "example.com"}).text
    assert "must not be empty" in session.call("web_search", {"query": "  "}).text
    assert calls == []


def test_tool_formats():
    assert [t["name"] for t in tools.responses_tools()] == ["web_search", "fetch_url"]
    assert all(t["type"] == "function" and t["strict"] is False and "parameters" in t for t in tools.responses_tools())
    assert [t["function"]["name"] for t in tools.chat_completions_tools()] == ["web_search", "fetch_url"]
    assert [t["name"] for t in tools.anthropic_tools()] == ["web_search", "fetch_url"]
    assert all("input_schema" in t for t in tools.anthropic_tools())


def test_prompt_renders_the_claim_like_the_pipeline():
    """Like veritas' Claim.__str__: the text in quotes and the publication date."""
    prompt, = build_prompt("The Earth is flat", date(2024, 2, 5))
    assert '# The Claim\n"The Earth is flat"\nPublished on: February 05, 2024\n\n# Rules' in prompt
    assert '"integrity": "compromised" | "intact" | "unknown",' in prompt
    assert "{{ claim }}" not in prompt

    assert '# The Claim\n"The Earth is flat"\n\n# Rules' in build_prompt("The Earth is flat")[0]


def test_prompt_says_what_day_it_is():
    prompt, = build_prompt("The Earth is flat", date(2024, 2, 5), today=date(2026, 10, 19))
    assert "\n\nToday is October 19, 2026.\n\n# Task Details" in prompt
    assert "{{ today }}" not in prompt
    # The actual day by default
    assert f"Today is {date.today().strftime('%B %d, %Y')}." in build_prompt("The Earth is flat")[0]
