"""Tests for the per-claim evidence URLs: encountered, retrieved, linked and quoted."""

import pytest
from baseline_fakes import ScriptedFactChecker, fake_retrieve, fake_search

from eval.baselines.common.parsing import extract_urls, remove_links
from eval.baselines.providers.base import Step, Text, ToolCall


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Per [AP](https://apnews.com/article/x).", ["https://apnews.com/article/x"]),
        # Parentheses inside a markdown link URL stay intact
        ("See [wiki](https://en.wikipedia.org/wiki/Foo_(bar)) here.", ["https://en.wikipedia.org/wiki/Foo_(bar)"]),
        ("[AP](https://apnews.com/article/a_(b),c)", ["https://apnews.com/article/a_(b),c"]),
        # Bare URLs lose trailing punctuation and unbalanced parentheses
        ("Confirmed by https://reut.rs/x?y=1&z=2.", ["https://reut.rs/x?y=1&z=2"]),
        ("(see https://example.org/page).", ["https://example.org/page"]),
        ("<https://example.org/auto>", ["https://example.org/auto"]),
        # Order of appearance, no duplicates, links and bare URLs mixed
        (
            "Both [A](https://a.org/1) and https://b.org/2, again [A](https://a.org/1).",
            ["https://a.org/1", "https://b.org/2"],
        ),
        ("No links here.", []),
        ("", []),
    ],
)
def test_extract_urls(text, expected):
    assert extract_urls(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("[Movies](/movies/) and [x](news/a.html)",
         ["https://www.news18.com/movies/", "https://www.news18.com/movies/bollywood/news/a.html"]),
        # Protocol-relative links get the page's scheme
        ("![img](//im.rediff.com/a.jpg)", ["https://im.rediff.com/a.jpg"]),
        ("[up](../tv/) [q](?page=2)",
         ["https://www.news18.com/movies/tv/", "https://www.news18.com/movies/bollywood/x?page=2"]),
        # No web pages
        ("[top](#top) [mail](mailto:a@b.org) [js](javascript:;)", []),
        # Absolute links and bare URLs as without a base URL, in order of appearance
        ("[A](https://a.org/1) then [B](/b), see https://c.org/3.",
         ["https://a.org/1", "https://www.news18.com/b", "https://c.org/3"]),
    ],
)
def test_extract_urls_resolves_relative_links_against_the_base_url(text, expected):
    assert extract_urls(text, base_url="https://www.news18.com/movies/bollywood/x") == expected


def test_extract_urls_leaves_out_relative_links_without_a_base_url():
    assert extract_urls("[Movies](/movies/) and [x](/redirect?u=https://other.org/p)") == ["https://other.org/p"]


def test_extract_urls_takes_links_with_a_title():
    assert extract_urls('[A](/a "The title") and [B](https://b.org/2 \'Another\')', base_url="https://x.org/") == [
        "https://x.org/a", "https://b.org/2"]


FACT_CHECK = "https://fc.org/2026/07/the-claim-is-false"


@pytest.mark.parametrize(
    "text, expected",
    [
        (f"See [the fact-check]({FACT_CHECK}).", "See ."),
        # With its title and an image, over several lines, as in a fact-checker's list of articles
        (f'[Image\n<image:7>\nPublished on 21/09/2026\n### The claim is false]({FACT_CHECK} "The claim is false") Next',
         " Next"),
        ("[The claim is false](/2026/07/the-claim-is-false/#top)", ""),  # Relative
        # A text with brackets, or too long, may begin anywhere before: only the target and title go
        (f'[![thumbnail](https://cdn.org/t.jpg)]({FACT_CHECK}/ "The claim is false") Next',
         "[![thumbnail](https://cdn.org/t.jpg) Next"),
        (f"[[S1] The claim is false]({FACT_CHECK})", "[[S1] The claim is false"),
        ("[" + "x" * 1001 + f"]({FACT_CHECK})", "[" + "x" * 1001),
        # Bare URLs, also in HTML links
        (f'Link: {FACT_CHECK}. <{FACT_CHECK}> <a href="{FACT_CHECK}">The claim</a>',
         'Link: . <> <a href="">The claim</a>'),
        # Other links stay, also on the same site
        (f"[Other](https://fc.org/2026/07/other), [it]({FACT_CHECK}), https://b.org/x",
         "[Other](https://fc.org/2026/07/other), , https://b.org/x"),
        ("No links here.", "No links here."),
    ],
)
def test_remove_links(text, expected):
    def is_fact_check(url):
        return url.split("?")[0].split("#")[0].rstrip("/") == FACT_CHECK

    removed, n = remove_links(text, is_fact_check, base_url="https://fc.org/latest")
    assert removed == expected
    assert n == text.count("fc.org/2026/07/the-claim-is-false") + text.count("](/2026/07/the-claim")


def test_result_keeps_encountered_retrieved_linked_and_quoted_urls_apart(monkeypatch):
    fake_search(monkeypatch, ["https://a.org/1", "https://b.org/2", "https://c.org/3"])
    fake_retrieve(monkeypatch, lambda url: "The photo is old, see [source](https://e.org/5).")
    final_text = (
        '{"integrity": "compromised", "certainty": "certain", "justification": '
        '"Per [A](https://a.org/1) the photo is from 2019; a [blog](https://d.org/4) agrees."}'
    )
    checker = ScriptedFactChecker([
        Step([ToolCall("1", "web_search", {"query": "photo"})]),
        Step([ToolCall("2", "fetch_url", {"url": "https://a.org/1"}),
              ToolCall("3", "fetch_url", {"url": "https://b.org/2"})]),
        Step([Text(final_text)]),
    ])

    result = checker.check_claim("claim", claim_date="2024-02-05")

    assert result.verdict == "Compromised (certain)"
    assert result.encountered_urls == ["https://a.org/1", "https://b.org/2", "https://c.org/3"]
    assert result.retrieved_urls == ["https://a.org/1", "https://b.org/2"]
    assert result.linked_urls == ["https://e.org/5"]
    # Quoted URLs come from the justification only, including ones never retrieved
    assert result.quoted_urls == ["https://a.org/1", "https://d.org/4"]
    assert result.citations == ["https://a.org/1", "https://b.org/2", "https://c.org/3", "https://d.org/4"]

    as_dict = result.to_dict()
    for field in ("encountered_urls", "retrieved_urls", "linked_urls", "quoted_urls"):
        assert as_dict[field] == getattr(result, field)


def test_result_without_tools_or_justification_has_empty_url_lists():
    answer = '{"integrity": "unknown", "certainty": null, "justification": "Nothing found."}'
    result = ScriptedFactChecker([Step([Text(answer)])]).check_claim("claim", claim_date="2024-02-05")

    assert result.encountered_urls == []
    assert result.retrieved_urls == []
    assert result.linked_urls == []
    assert result.quoted_urls == []
