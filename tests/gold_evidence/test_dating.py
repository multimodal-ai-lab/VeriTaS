"""Reading an evidence source's publication time off its raw HTML (Spec §3.1).

Only markup that *explicitly* denotes the publication time counts; modification,
update and access times never do, and ambiguous or implausible values are dropped
rather than completed (DESIGN_DECISIONS §10). The priority order is: JSON-LD >
Open Graph / article meta > microdata (+ `<time pubdate>`, microformats) > other
publication meta tags > publication keys in embedded script JSON.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from veritas.gold_evidence.dating import (
    extract_publication_time,
    is_plausible,
    parse_publication_value,
    publication_date_hints,
)

NOW = datetime(2026, 1, 1)


def page(head: str = "", body: str = "") -> str:
    return f"<html><head>{head}</head><body>{body}</body></html>"


def jsonld(data: object) -> str:
    return f'<script type="application/ld+json">{json.dumps(data)}</script>'


def meta(attr: str, key: str, content: str) -> str:
    return f'<meta {attr}="{key}" content="{content}">'


def extract(html: str) -> datetime | None:
    return extract_publication_time(html, now=NOW)


# --- Nothing to read -------------------------------------------------------------

@pytest.mark.parametrize("html", [None, "", page(), page(body="<p>Hello 2024</p>")])
def test_pages_without_dating_markup_stay_undated(html):
    assert extract(html) is None


# --- 1. JSON-LD --------------------------------------------------------------------

def test_jsonld_date_published_is_read_and_converted_to_naive_utc():
    html = page(jsonld({"@context": "https://schema.org", "@type": "NewsArticle",
                        "datePublished": "2024-05-10T10:00:00+02:00"}))
    assert extract(html) == datetime(2024, 5, 10, 8, 0)


def test_jsonld_graph_prefers_the_article_over_the_web_page():
    """Yoast-style @graph: the WebPage comes first in the document, but the article
    node describes the content itself."""
    html = page(jsonld({"@context": "https://schema.org", "@graph": [
        {"@type": "Organization", "name": "Publisher"},
        {"@type": "WebPage", "datePublished": "2024-05-09T00:00:00Z"},
        {"@type": "NewsArticle", "datePublished": "2024-05-10T00:00:00Z"},
    ]}))
    assert extract(html) == datetime(2024, 5, 10)


def test_jsonld_web_page_is_used_when_there_is_no_article():
    html = page(jsonld({"@graph": [{"@type": "WebPage", "datePublished": "2024-05-09"}]}))
    assert extract(html) == datetime(2024, 5, 9)


def test_jsonld_top_level_list():
    html = page(jsonld([{"@type": "BreadcrumbList"},
                        {"@type": "Article", "datePublished": "2024-05-10"}]))
    assert extract(html) == datetime(2024, 5, 10)


def test_jsonld_main_entity_is_searched():
    html = page(jsonld({"@type": "WebPage", "mainEntity": {
        "@type": "BlogPosting", "datePublished": "2024-05-10"}}))
    assert extract(html) == datetime(2024, 5, 10)


def test_jsonld_type_lists_and_iris_are_understood():
    html = page(jsonld([
        {"@type": "WebPage", "datePublished": "2024-05-09"},
        {"@type": ["https://schema.org/NewsArticle", "Thing"], "datePublished": "2024-05-10"},
    ]))
    assert extract(html) == datetime(2024, 5, 10)


def test_jsonld_date_published_given_as_a_list():
    html = page(jsonld({"@type": "Article", "datePublished": ["2024-05-10"]}))
    assert extract(html) == datetime(2024, 5, 10)


def test_jsonld_video_upload_date():
    html = page(jsonld({"@type": "VideoObject", "uploadDate": "2024-05-10T08:00:00Z"}))
    assert extract(html) == datetime(2024, 5, 10, 8, 0)


def test_jsonld_date_created_counts_for_posts():
    html = page(jsonld({"@type": "SocialMediaPosting", "dateCreated": "2024-05-10"}))
    assert extract(html) == datetime(2024, 5, 10)


@pytest.mark.parametrize("type_", ["NewsArticle", "ImageObject", "VideoObject"])
def test_jsonld_date_created_does_not_count_for_other_types(type_):
    """On an article it may be the draft's creation, on media the capture time."""
    html = page(jsonld({"@type": type_, "dateCreated": "2024-05-10"}))
    assert extract(html) is None


def test_jsonld_modified_date_is_never_used():
    html = page(jsonld({"@type": "NewsArticle", "dateModified": "2024-05-12"}))
    assert extract(html) is None


def test_jsonld_prefers_date_published_over_modified():
    html = page(jsonld({"@type": "NewsArticle", "dateModified": "2024-05-12",
                        "datePublished": "2024-05-10"}))
    assert extract(html) == datetime(2024, 5, 10)


def test_jsonld_related_items_are_not_searched():
    """Dates of the articles an ItemList points to are not this page's."""
    html = page(jsonld({"@type": "ItemList", "itemListElement": [
        {"@type": "ListItem", "item": {"@type": "NewsArticle",
                                       "datePublished": "2024-05-10"}}]}))
    assert extract(html) is None


def test_jsonld_nested_is_part_of_is_not_searched():
    html = page(jsonld({"@type": "NewsArticle", "isPartOf": {
        "@type": "PublicationIssue", "datePublished": "2024-05-01"}}))
    assert extract(html) is None


def test_malformed_jsonld_does_not_hide_other_sources():
    head = ('<script type="application/ld+json">{"@type": "NewsArticle", '
            '"datePublished": "2024-05-10",}</script>'
            + meta("property", "article:published_time", "2024-05-11"))
    assert extract(page(head)) == datetime(2024, 5, 11)


def test_a_later_valid_jsonld_block_is_still_read():
    head = ('<script type="application/ld+json">{not json</script>'
            + jsonld({"@type": "Article", "datePublished": "2024-05-10"}))
    assert extract(page(head)) == datetime(2024, 5, 10)


def test_jsonld_wrapped_in_comment_or_with_trailing_semicolon():
    body = json.dumps({"@type": "Article", "datePublished": "2024-05-10"})
    assert extract(page(f'<script type="application/ld+json"><!-- {body} --></script>')) \
        == datetime(2024, 5, 10)
    assert extract(page(f'<script type="application/ld+json">{body};</script>')) \
        == datetime(2024, 5, 10)


def test_jsonld_non_string_values_are_skipped():
    html = page(jsonld({"@type": "Article", "datePublished": {"@value": 5}}))
    assert extract(html) is None


# --- 2. Open Graph / article meta --------------------------------------------------

@pytest.mark.parametrize("attr,key", [
    ("property", "article:published_time"),
    ("name", "article:published_time"),     # property vs. name is mixed up in the wild
    ("property", "Article:Published_Time"),  # attribute values compared case-insensitively
    ("property", "og:published_time"),
    ("property", "og:article:published_time"),
    ("property", "og:video:release_date"),
    ("property", "video:release_date"),
    ("property", "og:pubdate"),
    ("name", "article.published"),
])
def test_open_graph_publication_tags(attr, key):
    assert extract(page(meta(attr, key, "2024-05-10T08:00:00Z"))) == datetime(2024, 5, 10, 8, 0)


@pytest.mark.parametrize("attr,key", [
    ("property", "article:modified_time"),
    ("property", "og:updated_time"),
    ("name", "last-modified"),
    ("http-equiv", "last-modified"),
    ("name", "lastmod"),
    ("name", "dc.modified"),
    ("name", "dcterms.modified"),
    ("name", "revised"),
])
def test_modification_tags_are_never_used(attr, key):
    assert extract(page(meta(attr, key, "2024-05-12T08:00:00Z"))) is None


def test_the_first_unparseable_tag_does_not_stop_the_search():
    """stage 3 gave up at the first matching tag it could not parse."""
    head = meta("property", "article:published_time", "sometime") \
        + meta("name", "pubdate", "2024-05-10")
    assert extract(page(head)) == datetime(2024, 5, 10)


# --- 3. Microdata, <time pubdate>, microformats -----------------------------------

def test_time_element_with_itemprop_date_published():
    body = ('<article itemscope itemtype="https://schema.org/NewsArticle">'
            '<time itemprop="datePublished" datetime="2024-05-10T10:00:00+02:00">'
            'May 10</time></article>')
    assert extract(page(body=body)) == datetime(2024, 5, 10, 8, 0)


def test_itemprop_is_matched_case_insensitively_and_per_token():
    body = '<span itemprop="DatePublished dateCreated" content="2024-05-10"></span>'
    assert extract(page(body=body)) == datetime(2024, 5, 10)


def test_itemprop_on_any_element_falls_back_to_its_text():
    body = '<span itemprop="datePublished">Published May 10, 2024</span>'
    assert extract(page(body=body)) == datetime(2024, 5, 10)


@pytest.mark.parametrize("prop", ["datePublished", "uploadDate"])
def test_meta_itemprop_as_on_video_platforms(prop):
    head = f'<meta itemprop="{prop}" content="2024-05-10T01:00:00-07:00">'
    assert extract(page(head)) == datetime(2024, 5, 10, 8, 0)


def test_itemprop_date_modified_is_never_used():
    body = '<time itemprop="dateModified" datetime="2024-05-12">May 12</time>'
    assert extract(page(body=body)) is None


def test_html5_pubdate_attribute():
    body = '<time pubdate datetime="2024-05-10">May 10</time>'
    assert extract(page(body=body)) == datetime(2024, 5, 10)


def test_wordpress_published_class():
    """Most WordPress themes mark the post's time as `published` (hAtom) - together
    with `updated` as long as the post was never edited."""
    body = ('<time class="entry-date published updated" '
            'datetime="2024-05-10T08:00:00+00:00">May 10, 2024</time>')
    assert extract(page(body=body)) == datetime(2024, 5, 10, 8, 0)


def test_h_entry_dt_published_class():
    body = '<time class="dt-published" datetime="2024-05-10">May 10</time>'
    assert extract(page(body=body)) == datetime(2024, 5, 10)


def test_updated_class_alone_is_never_used():
    body = '<time class="updated" datetime="2024-05-12">May 12</time>'
    assert extract(page(body=body)) is None


def test_plain_time_element_is_not_attributed_to_the_publication():
    """A bare <time> may be anything - an event, a comment. It is only a hint for
    the LLM, never read as the publication time directly."""
    body = '<time datetime="2024-05-10">May 10</time>'
    assert extract(page(body=body)) is None


# --- 4. Other publication meta tags ----------------------------------------------

@pytest.mark.parametrize("attr,key,value,expected", [
    ("name", "citation_publication_date", "2024/05/10", datetime(2024, 5, 10)),
    ("name", "citation_online_date", "2024/05/10", datetime(2024, 5, 10)),
    ("name", "citation_date", "2024-05-10", datetime(2024, 5, 10)),
    ("name", "DC.date.issued", "2024-05-10", datetime(2024, 5, 10)),
    ("name", "DCTERMS.issued", "2024-05-10", datetime(2024, 5, 10)),
    ("name", "dcterms.created", "2024-05-10", datetime(2024, 5, 10)),
    ("name", "DC.date", "2024-05-10", datetime(2024, 5, 10)),
    ("name", "parsely-pub-date", "2024-05-10T08:00:00Z", datetime(2024, 5, 10, 8, 0)),
    ("name", "sailthru.date", "2024-05-10 08:00:00", datetime(2024, 5, 10, 8, 0)),
    ("name", "pubdate", "20240510", datetime(2024, 5, 10)),
    ("name", "publishdate", "2024-05-10", datetime(2024, 5, 10)),
    ("name", "publish-date", "2024-05-10", datetime(2024, 5, 10)),
    ("name", "publish_date", "2024-05-10", datetime(2024, 5, 10)),
    ("name", "publication_date", "Fri, 10 May 2024 08:00:00 GMT", datetime(2024, 5, 10, 8, 0)),
    ("name", "date", "2024-05-10", datetime(2024, 5, 10)),
    ("name", "Date", "10 May 2024", datetime(2024, 5, 10)),
])
def test_other_publication_meta_tags(attr, key, value, expected):
    assert extract(page(meta(attr, key, value))) == expected


# --- Priority order ---------------------------------------------------------------

def test_jsonld_beats_open_graph():
    head = (meta("property", "article:published_time", "2024-05-11")
            + jsonld({"@type": "NewsArticle", "datePublished": "2024-05-10"}))
    assert extract(page(head)) == datetime(2024, 5, 10)


def test_open_graph_beats_microdata():
    head = meta("property", "article:published_time", "2024-05-10")
    body = '<time itemprop="datePublished" datetime="2024-05-11">May 11</time>'
    assert extract(page(head, body)) == datetime(2024, 5, 10)


def test_microdata_beats_other_meta_tags():
    head = meta("name", "citation_publication_date", "2024/05/11")
    body = '<time itemprop="datePublished" datetime="2024-05-10">May 10</time>'
    assert extract(page(head, body)) == datetime(2024, 5, 10)


def test_specific_meta_tag_beats_generic_date_regardless_of_order():
    head = meta("name", "date", "2024-05-11") + meta("name", "citation_publication_date", "2024/05/10")
    assert extract(page(head)) == datetime(2024, 5, 10)


def test_meta_tags_beat_embedded_script_json():
    head = meta("name", "date", "2024-05-10")
    body = '<script>var s = {"publishDate":"2024-05-11"};</script>'
    assert extract(page(head, body)) == datetime(2024, 5, 10)


def test_an_implausible_higher_priority_value_falls_through():
    head = (jsonld({"@type": "NewsArticle", "datePublished": "1970-01-01T00:00:00Z"})
            + meta("property", "article:published_time", "2024-05-10"))
    assert extract(page(head)) == datetime(2024, 5, 10)


# --- 5. Embedded script JSON (video platforms, app state) --------------------------

def test_youtube_style_embedded_json():
    body = ('<script>var ytInitialPlayerResponse = {"microformat": '
            '{"playerMicroformatRenderer": {"publishDate": "2024-05-10T01:00:00-07:00", '
            '"uploadDate": "2024-05-10T01:00:00-07:00"}}};</script>')
    assert extract(page(body=body)) == datetime(2024, 5, 10, 8, 0)


def test_escaped_embedded_json():
    body = r'<script>self.__next_f.push([1,"{\"datePublished\":\"2024-05-10T08:00:00Z\"}"])</script>'
    assert extract(page(body=body)) == datetime(2024, 5, 10, 8, 0)


def test_disagreeing_embedded_json_dates_are_not_used():
    """The blob also describes related items; which date is the page's is unknown."""
    body = ('<script>var s = {"datePublished": "2024-05-10", '
            '"related": [{"datePublished": "2024-04-01"}]};</script>')
    assert extract(page(body=body)) is None


def test_embedded_json_modified_dates_are_not_used():
    body = '<script>var s = {"dateModified": "2024-05-12", "updateDate": "2024-05-12"};</script>'
    assert extract(page(body=body)) is None


# --- Parsing and plausibility -----------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("2024-05-10", datetime(2024, 5, 10)),
    ("2024-05-10T08:00:00Z", datetime(2024, 5, 10, 8, 0)),
    ("2024-05-10T08:00:00.123Z", datetime(2024, 5, 10, 8, 0, 0, 123000)),
    ("2024-05-10T23:30:00-05:00", datetime(2024, 5, 11, 4, 30)),  # crosses the day in UTC
    ("2024-05-10T08:00:00", datetime(2024, 5, 10, 8, 0)),  # no offset: taken as stated
    ("Fri, 10 May 2024 08:00:00 GMT", datetime(2024, 5, 10, 8, 0)),
    ("May 10, 2024", datetime(2024, 5, 10)),
    ("2024/05/10", datetime(2024, 5, 10)),
    ("25/10/2024", datetime(2024, 10, 25)),  # unambiguous: 25 is no month
    ("10/10/2024", datetime(2024, 10, 10)),  # unambiguous: either order is the same
    ("20240510", datetime(2024, 5, 10)),
    ("1715328000", datetime(2024, 5, 10, 8, 0)),       # epoch seconds
    ("1715328000000", datetime(2024, 5, 10, 8, 0)),    # epoch milliseconds
    ("  2024-05-10  ", datetime(2024, 5, 10)),
])
def test_parse_publication_value(value, expected):
    assert parse_publication_value(value, now=NOW) == expected


@pytest.mark.parametrize("value", [
    None, "", "   ", "sometime", "yesterday", "3 hours ago",
    "May 2024",           # no day - completing it would be guessing
    "2024-05",            # ditto
    "2024",               # no month and day
    "May 10",             # no year
    "05/10/2024",         # May 10th or October 5th?
    "05.10.2024",
    "Updated May 12, 2024",
    "Last modified: 2024-05-12",
    "123",
    "12345678901",
    "x" * 100,
])
def test_unreadable_or_ambiguous_values_are_rejected(value):
    assert parse_publication_value(value, now=NOW) is None


@pytest.mark.parametrize("value", [
    "1989-12-31",
    "1970-01-01T00:00:00Z",
    "2026-01-03",          # after NOW + tolerance
    "2999-01-01",
])
def test_implausible_dates_are_rejected(value):
    assert parse_publication_value(value, now=NOW) is None


def test_future_tolerance_covers_timezone_offsets():
    assert is_plausible(datetime(2026, 1, 1, 12, 0), now=NOW)
    assert not is_plausible(datetime(2026, 1, 2, 0, 1), now=NOW)
    assert is_plausible(datetime(1990, 1, 1), now=NOW)


def test_implausible_future_date_without_explicit_now():
    assert extract_publication_time(page(meta("name", "date", "2999-01-01"))) is None


# --- Hints for the LLM fallback ----------------------------------------------------

def test_hints_show_time_elements_with_their_context():
    body = '<p>By Jane Doe | Published</p><time datetime="2024-05-10T08:00:00Z">May 10</time>'
    hints = publication_date_hints(page(body=body))
    assert hints == ['<time datetime="2024-05-10T08:00:00Z">May 10</time> '
                     '- preceded by: "By Jane Doe | Published"']


def test_hints_include_time_elements_without_datetime_attribute():
    hints = publication_date_hints(page(body="<time>May 10, 2024</time>"))
    assert hints == ["<time>May 10, 2024</time>"]


@pytest.mark.parametrize("body", [
    '<span>Updated:</span> <time datetime="2024-05-12">May 12</time>',
    '<p>Last modified</p><time datetime="2024-05-12">May 12</time>',
    '<time datetime="2024-05-12">Updated May 12</time>',
    '<time class="updated" datetime="2024-05-12">May 12</time>',
    '<time itemprop="dateModified" datetime="2024-05-12">May 12</time>',
    '<p>Accessed</p><time datetime="2024-05-12">May 12</time>',
])
def test_hints_never_offer_update_or_access_times(body):
    assert publication_date_hints(page(body=body)) == []


def test_hints_keep_a_published_and_updated_wordpress_time():
    body = '<time class="entry-date published updated" datetime="2024-05-10">May 10</time>'
    assert len(publication_date_hints(page(body=body))) == 1


def test_hints_include_unknown_date_like_meta_tags():
    head = meta("name", "release_date", "2024-05-10")
    assert publication_date_hints(page(head)) == ['<meta release_date="2024-05-10">']


@pytest.mark.parametrize("head", [
    meta("property", "og:updated_time", "2024-05-12"),        # excluded
    meta("property", "article:modified_time", "2024-05-12"),  # excluded
    meta("name", "date", "2024-05-10"),                       # already tried directly
    meta("name", "description", "Report on 2024 elections"),  # not a date key
    meta("name", "release_date", "soon"),                     # not a date value
])
def test_hints_skip_irrelevant_meta_tags(head):
    assert publication_date_hints(page(head)) == []


def test_hints_ignore_script_text_in_the_context():
    body = '<script>var x = "Updated";</script><time datetime="2024-05-10">May 10</time>'
    hints = publication_date_hints(page(body=body))
    assert hints == ['<time datetime="2024-05-10">May 10</time>']


def test_hints_are_deduplicated_and_capped_in_page_order():
    body = "".join(f'<li><time datetime="2024-05-{d:02d}">May {d}</time></li>'
                   for d in range(1, 21))
    hints = publication_date_hints(page(body=body), limit=3)
    assert len(hints) == 3
    assert 'datetime="2024-05-01"' in hints[0]
    assert 'datetime="2024-05-03"' in hints[2]


@pytest.mark.parametrize("html", [None, "", page()])
def test_no_hints_without_markup(html):
    assert publication_date_hints(html) == []
