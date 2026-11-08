import asyncio
import time

import aiohttp
import pytest
from scrapemm.search import SerperOrganicResult, SerperResponse

from baseline_fakes import page

from eval.baselines.common import tools
from eval.baselines.common.google_links import (GoogleLinkResolver, is_google_redirect, is_hidden_redirect,
                                                plain_target)
from eval.baselines.common.tools import UNRESOLVED_REDIRECT, ToolSession

URL_A = "https://www.google.com/url?opi=89978449&q=CAESaaa&sa=U&usg=x"
GOTO_B = "https://www.google.com/goto?url=CAESbbb"
PLAIN = "https://example.org/a"
FACT_CHECK = "https://factcheck.example.org/2026/07/the-claim-is-false/"
# How the block rules name it: the slug rule comes first (see ToolSession.blocked_by)
BLOCKED = "with the slug the-claim-is-false of the claim's fact-check in its path"


def organic(*links: str) -> dict:
    return {"organic": [{"link": link} for link in links]}


class ScriptedResolver(GoogleLinkResolver):
    """Answers Serper's requests in turn with `answers` (an exception is raised), recording them."""

    def __init__(self, *answers, **kwargs):
        super().__init__(api_key="test-key", **kwargs)
        self.answers = list(answers)
        self.requests = []

    async def _request(self, links, timeout):
        self.requests.append(links)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class MapResolver(GoogleLinkResolver):
    """Resolves the links in `targets`, leaves the rest, without retries."""

    def __init__(self, targets: dict[str, str]):
        super().__init__(api_key="test-key")
        self.targets = targets
        self.requests = []

    async def _request(self, links, timeout):
        self.requests.append(links)
        return organic(*(self.targets.get(link, link) for link in links))


def resolve(resolver, links, log=None):
    return asyncio.run(resolver.resolve(links, log=log or (lambda message: None)))



@pytest.mark.parametrize("url", [URL_A, GOTO_B, "https://google.com/url?q=x", "https://www.google.de/goto?url=x",
                                 "https://www.google.co.in/url?sa=i&url=https://example.org/a"])
def test_google_redirects(url):
    assert is_google_redirect(url)


@pytest.mark.parametrize("url", [None, "", PLAIN, "https://www.google.com/search?q=goto", "https://notgoogle.com/url?q=x",
                                 "https://example.org/goto?url=x", "https://www.google.com/url/more?q=x", "https://[x"])
def test_other_links(url):
    assert not is_google_redirect(url)


@pytest.mark.parametrize("url, target", [
    ("https://www.google.com/url?q=https://example.org/a%3Fx%3D1&sa=U", "https://example.org/a?x=1"),
    # Image results have the page in `url` and an empty `q`
    ("https://www.google.co.in/url?sa=i&rct=j&q=&url=https%3A%2F%2Fexample.org%2Fb&usg=x", "https://example.org/b"),
    (URL_A, None),
    (GOTO_B, None),
    (PLAIN, None),
])
def test_plain_target(url, target):
    assert plain_target(url) == target
    assert is_hidden_redirect(url) == (is_google_redirect(url) and target is None)



def test_links_without_hidden_redirects_send_no_request():
    resolver = ScriptedResolver()
    plain = "https://www.google.com/url?q=https://example.org/b&sa=U"

    assert resolve(resolver, [PLAIN, None, plain]) == [PLAIN, None, "https://example.org/b"]
    assert resolver.requests == []


def test_only_hidden_redirects_are_sent_and_the_order_is_kept():
    resolver = ScriptedResolver(organic("https://a.org/x", "https://b.org/y"))

    assert resolve(resolver, [URL_A, PLAIN, GOTO_B]) == ["https://a.org/x", PLAIN, "https://b.org/y"]
    assert resolver.requests == [[URL_A, GOTO_B]]


def test_links_google_holds_back_are_asked_for_again():
    resolver = ScriptedResolver({**organic("https://a.org/x", GOTO_B), "retryAfter": 30}, organic("https://b.org/y"),
                                retry_seconds=0)

    assert resolve(resolver, [URL_A, GOTO_B]) == ["https://a.org/x", "https://b.org/y"]
    assert resolver.requests == [[URL_A, GOTO_B], [GOTO_B]]


def test_without_retry_after_an_unresolved_link_stays():
    resolver = ScriptedResolver(organic("https://a.org/x", GOTO_B))

    assert resolve(resolver, [URL_A, GOTO_B]) == ["https://a.org/x", GOTO_B]
    assert len(resolver.requests) == 1


def test_the_budget_ends_the_retries():
    class ForeverPending(ScriptedResolver):
        async def _request(self, links, timeout):
            self.requests.append(links)
            await asyncio.sleep(0.02)
            return {**organic(*links), "retryAfter": 30}

    resolver = ForeverPending(budget_seconds=0.1, retry_seconds=0.01)
    start = time.monotonic()

    assert resolve(resolver, [URL_A]) == [URL_A]
    assert time.monotonic() - start < 0.5
    assert 1 <= len(resolver.requests) <= 6


@pytest.mark.parametrize("failure", [TimeoutError(), aiohttp.ClientError("refused"), ValueError("not json")])
def test_a_failed_request_leaves_the_links(failure):
    resolver = ScriptedResolver(failure)
    log = []

    assert resolve(resolver, [URL_A, PLAIN], log.append) == [URL_A, PLAIN]
    assert (log == []) == isinstance(failure, TimeoutError)


@pytest.mark.parametrize("answer", [organic("https://a.org/x"), {"error": "nope"}, organic("not a url", "https://b.org/y")])
def test_a_malformed_answer_leaves_the_links(answer):
    resolved = resolve(ScriptedResolver(answer), [URL_A, GOTO_B])

    assert resolved[0] == URL_A


def test_large_lists_go_in_batches_serper_accepts():
    links = [f"{URL_A}{n}" for n in range(150)]
    resolver = MapResolver({link: f"https://r.org/{n}" for n, link in enumerate(links)})

    assert resolve(resolver, links) == [f"https://r.org/{n}" for n in range(150)]
    assert [len(batch) for batch in resolver.requests] == [100, 50]


def test_without_an_api_key_nothing_is_sent(monkeypatch):
    monkeypatch.setattr("eval.baselines.common.google_links.serper_api_key", lambda: None)
    resolver = GoogleLinkResolver()
    log = []

    assert resolve(resolver, [URL_A], log.append) == [URL_A]
    assert "no Serper API key" in log[0]



def _search_returns(monkeypatch, *results: tuple[str, str]):
    async def fake_search(query):
        return SerperResponse(organic=[SerperOrganicResult(title=title, link=link) for title, link in results])

    monkeypatch.setattr(tools, "search", fake_search)


def _fake_retrieve(monkeypatch, text_for):
    fetched = []

    async def fake_retrieve(url, **kwargs):
        fetched.append(url)
        return page(text_for(url))

    monkeypatch.setattr(tools, "retrieve", fake_retrieve)
    return fetched


def test_search_results_show_the_redirects_targets_and_drop_the_blocked_and_unresolved(monkeypatch, capsys):
    hidden_unresolved = "https://www.google.com/goto?url=CAESccc"
    _search_returns(monkeypatch, ("Plain", PLAIN), ("Resolved", URL_A), ("The claim is false", GOTO_B),
                    ("Unresolved", hidden_unresolved))
    resolver = MapResolver({URL_A: "https://news.example.com/story", GOTO_B: FACT_CHECK})
    session = ToolSession(claim_date="2026-07-10", blocked_urls=[FACT_CHECK], claim_id=7, link_resolver=resolver)

    text = session.web_search("the claim")

    assert "URL: https://news.example.com/story" in text
    assert "google.com" not in text and "The claim is false" not in text and "Unresolved" not in text
    assert session.encountered_urls == [PLAIN, "https://news.example.com/story"]
    assert [(entry["url"], entry["rule"]) for entry in session.blocked] == [
        (FACT_CHECK, f"a URL {BLOCKED}"),
        (hidden_unresolved, f"a URL {UNRESOLVED_REDIRECT}"),
    ]
    assert "web_search: 'the claim': resolved 2 of 3 Google redirects" in capsys.readouterr().out
    fetched = _fake_retrieve(monkeypatch, lambda url: "story")
    assert "Content of https://news.example.com/story" in session.fetch_url("https://news.example.com/story").text
    assert fetched == ["https://news.example.com/story"]


def test_a_redirect_with_its_target_in_the_clear_is_blocked_as_its_target():
    session = ToolSession(claim_date="2026-07-10", blocked_urls=[FACT_CHECK], link_resolver=MapResolver({}))

    assert session.blocked_by(f"https://www.google.com/url?q={FACT_CHECK}&sa=U") == (
        f"{BLOCKED}, behind a Google redirect")
    assert session.blocked_by("https://www.google.com/url?q=https://example.org/other&sa=U") is None
    # A hidden one is no URL in a blocked scope: it is resolved when it is fetched
    assert session.blocked_by(URL_A) is None


def test_fetched_pages_lose_their_links_through_a_redirect_to_the_fact_check(monkeypatch):
    _fake_retrieve(monkeypatch, lambda url: f"Read [the debunk](https://www.google.com/url?q={FACT_CHECK}) and "
                                            f"[more](https://www.google.com/url?q=https://example.org/more).")
    session = ToolSession(claim_date="2026-07-10", blocked_urls=[FACT_CHECK], link_resolver=MapResolver({}))
    session.encountered_urls.append(PLAIN)

    text = session.fetch_url(PLAIN).text

    assert "the debunk" not in text and "factcheck.example.org" not in text
    assert "[more](https://www.google.com/url?q=https://example.org/more)" in text


@pytest.mark.parametrize("redirect, targets, fetched_url", [
    (URL_A, {URL_A: "https://news.example.com/story"}, "https://news.example.com/story"),
    ("https://www.google.com/url?q=https://example.org/more&sa=U", {}, "https://example.org/more"),
])
def test_fetch_url_fetches_a_redirects_target(monkeypatch, redirect, targets, fetched_url):
    fetched = _fake_retrieve(monkeypatch, lambda url: "story")
    resolver = MapResolver(targets)
    session = ToolSession(claim_date="2026-07-10", link_resolver=resolver)
    session.linked_urls.append(redirect)

    assert f"Content of {fetched_url}" in session.fetch_url(redirect).text
    assert fetched == [fetched_url] and session.retrieved_urls == [fetched_url]
    assert len(resolver.requests) == (1 if targets else 0)


@pytest.mark.parametrize("targets, rule", [
    ({URL_A: FACT_CHECK}, f"a URL {BLOCKED}, behind a Google redirect"),
    ({}, f"a URL {UNRESOLVED_REDIRECT}"),
])
def test_fetch_url_refuses_a_redirect_to_the_fact_check_or_one_not_resolved(monkeypatch, targets, rule):
    fetched = _fake_retrieve(monkeypatch, lambda url: "the claim is false")
    session = ToolSession(claim_date="2026-07-10", blocked_urls=[FACT_CHECK], link_resolver=MapResolver(targets))
    session.linked_urls.append(URL_A)

    # As for any URL not seen: another answer would tell the model what is behind the redirect
    assert session.fetch_url(URL_A).text == (f"Error: Cannot fetch {URL_A}. fetch_url only fetches URLs that appeared "
                                             f"in web_search results or in fetched pages. Use them as they appeared.")
    assert fetched == [] and session.n_fetches == 0
    assert session.blocked == [{"tool": "fetch_url", "url": URL_A, "rule": rule}]


def test_fetch_url_resolves_no_redirect_the_model_has_not_seen(monkeypatch):
    resolver = MapResolver({URL_A: "https://news.example.com/story"})
    session = ToolSession(claim_date="2026-07-10", link_resolver=resolver)

    assert session.fetch_url(URL_A).text.startswith("Error: Cannot fetch")
    assert resolver.requests == []
