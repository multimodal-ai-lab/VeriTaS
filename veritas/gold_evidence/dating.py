"""Stage 2.1 - Reading an evidence source's publication time off its raw HTML (Spec §3.1).

Pure functions: no I/O, no LLM. `retrieval` hands the raw HTML that scrapeMM returned
alongside the multimodal content to `extract_publication_time`; only if that finds
nothing does a cheap LLM read a stated date off the content, assisted by the
`publication_date_hints` collected here.

Why this exists instead of reusing `stage_3.extract_date_meta`: that function checks
six ``<meta>`` tags with case-sensitive attribute values, parses them with
``datetime.fromisoformat`` only, and gives up on the first unparseable match. Most
publishers, however, state the publication time in JSON-LD (``datePublished``), in
microdata (``itemprop="datePublished"``) or in meta tags stage 3 does not know
(``citation_publication_date``, ``parsely-pub-date``, ``DC.date.issued``, ...), often
in non-ISO formats. Stage 3 stays untouched because the main pipeline depends on it.

The rules of DESIGN_DECISIONS §10 hold throughout: only a field that *explicitly*
denotes the publication (or upload/release) time is read. Modification/update,
access and crawl times are never read, and a value is dropped rather than completed
when it is ambiguous (no year, no day, day/month order unclear) or implausible.

Priority order (the first source yielding a plausible date wins):

1. JSON-LD ``datePublished`` > ``uploadDate`` > ``dateCreated`` of the page's primary
   node - article/post/video types before web-page types before anything else.
2. Open Graph / article meta tags: ``article:published_time``, ``og:published_time``,
   ``og:video:release_date``, ... .
3. Microdata: any element with ``itemprop="datePublished"`` (or ``uploadDate``) -
   ``content`` / ``datetime`` attribute, else its text - HTML5 ``<time pubdate>``, and
   the microformat classes ``published`` (hAtom, WordPress) / ``dt-published``.
4. Other publication meta tags: scholarly (``citation_publication_date``), Dublin Core
   (``DC.date.issued``, ``dcterms.created``), Parse.ly, Sailthru, ``pubdate``, ``date``.
5. Publication keys in embedded JSON of inline scripts (e.g. YouTube's ``uploadDate`` /
   ``publishDate``) - only if every occurrence agrees on the same day, because such
   blobs often also describe related items.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

from bs4 import BeautifulSoup, Comment, Tag
from dateutil import parser as date_parser

from veritas.gold_evidence.models import to_naive

#: Anything before this is a parsing artefact (or a described event) rather than
#: the publication time of a web source.
EARLIEST_PLAUSIBLE = datetime(1990, 1, 1)

#: Slack for timezone offsets and clock skew when rejecting future dates.
FUTURE_TOLERANCE = timedelta(days=1)

#: Names that denote anything but the first publication. The extraction reads
#: whitelisted keys only; this guards the hints, which are collected generically.
_EXCLUDED_KEY = re.compile(r"modif|updat|revis|lastmod|(?<![a-z])edit|expir|access|crawl"
                           r"|(?<![a-z])index|retriev", re.IGNORECASE)

# --- 1. JSON-LD ------------------------------------------------------------------

#: Keys of a JSON-LD node, most specific first. ``dateCreated`` is last and only read
#: on posting types (`_DATE_CREATED_TYPES`): there it is the post time, whereas on an
#: article it may be the draft's creation and on a photo or video the capture time -
#: the time of the described event, not of the publication.
_JSONLD_KEYS = ("datePublished", "uploadDate", "dateCreated")

_DATE_CREATED_TYPES = {"socialmediaposting", "discussionforumposting", "blogposting",
                       "liveblogposting", "comment"}

#: @type values describing the page's own content (rank 0).
_PRIMARY_TYPES = {
    "article", "newsarticle", "reportagenewsarticle", "analysisnewsarticle",
    "opinionnewsarticle", "backgroundnewsarticle", "reviewnewsarticle",
    "askpublicnewsarticle", "blogposting", "liveblogposting", "socialmediaposting",
    "discussionforumposting", "scholarlyarticle", "techarticle", "report",
    "videoobject", "audioobject", "podcastepisode", "episode", "claimreview",
    "review", "creativework", "medicalscholarlyarticle", "satiricalarticle",
    "advertisercontentarticle", "qapage", "question", "comment", "dataset",
}

#: @type values of the surrounding page (rank 1). Their dates usually mirror the
#: article's, but are less specific.
_PAGE_TYPES = {"webpage", "itempage", "collectionpage", "profilepage", "aboutpage",
               "searchresultspage", "mediagallery", "imagegallery", "videogallery",
               "website"}

# --- 2.-4. Meta tags and microdata ----------------------------------------------

#: Meta keys (``property``, ``name`` or ``itemprop``, compared case-insensitively)
#: that denote the publication time explicitly - the Open Graph family.
_OG_META_KEYS = (
    "article:published_time",
    "og:article:published_time",
    "og:published_time",
    "published_time",
    "og:video:release_date",
    "video:release_date",
    "og:pubdate",
    "article.published",
    "article:published",
)

#: Microdata properties of the page's own content.
_ITEMPROPS = ("datepublished", "uploaddate")

#: Microformat class names of the publication time (matched per class token).
_MICROFORMAT_CLASSES = ["published", "dt-published"]

#: Further meta keys, in decreasing specificity. The generic ``date`` / ``dc.date``
#: come last; stage 3 reads them as publication dates too.
_OTHER_META_KEYS = (
    "citation_publication_date",
    "citation_online_date",
    "citation_date",
    "dc.date.issued",
    "dcterms.issued",
    "dc.date.created",
    "dcterms.created",
    "dc.date.published",
    "prism.publicationdate",
    "parsely-pub-date",
    "sailthru.date",
    "article_date_original",
    "original-publish-date",
    "publication_date",
    "publication-date",
    "publish-date",
    "publish_date",
    "publishdate",
    "publishdatetime",
    "pubdate",
    "datepublished",
    "date_published",
    "cxenseparse:recs:publishtime",
    "dc.date",
    "dcterms.date",
    "date",
)

# --- 5. Embedded JSON ------------------------------------------------------------

_EMBEDDED_JSON_DATE = re.compile(
    r'\\?"(datePublished|uploadDate|publishDate)\\?"\s*:\s*\\?"([^"\\]{8,40})\\?"')


# ================================================================================

def extract_publication_time(html: str | None, now: datetime | None = None) -> datetime | None:
    """Returns the publication time the page's markup states explicitly, as a naive
    UTC datetime, or None. See the module docstring for the priority order.

    :param now: Reference for rejecting future dates (naive UTC); defaults to now."""
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for candidate in _candidates(soup, now):
        parsed = parse_publication_value(candidate, now=now)
        if parsed is not None:
            return parsed
    return None


def _candidates(soup: BeautifulSoup, now: datetime | None) -> Iterator[str]:
    """All publication-time values of the page, best first. Lazy, so the cheaper and
    more reliable sources settle most pages without the later scans."""
    yield from _jsonld_candidates(soup)
    yield from _meta_candidates(soup, _OG_META_KEYS)
    yield from _microdata_candidates(soup)
    yield from _meta_candidates(soup, _OTHER_META_KEYS)
    embedded = _embedded_json_candidate(soup, now)
    if embedded is not None:
        yield embedded


def _jsonld_candidates(soup: BeautifulSoup) -> Iterator[str]:
    nodes = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        data = _load_json(script.string or script.get_text())
        if data is not None:
            nodes.extend(_primary_nodes(data))

    # Stable sort: document order within the same rank.
    for node in sorted(nodes, key=_node_rank):
        for key in _JSONLD_KEYS:
            if key == "dateCreated" and not _node_types(node) & _DATE_CREATED_TYPES:
                continue
            value = node.get(key)
            if isinstance(value, list) and value:
                value = value[0]
            if isinstance(value, str) and value.strip():
                yield value


def _load_json(text: str | None) -> object | None:
    """Parses a JSON-LD block. Tolerates the common defects - HTML comments / CDATA
    wrappers and trailing semicolons - and gives up silently on anything else, since
    a malformed block must not hide the remaining sources."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^(<!--|//\s*<!\[CDATA\[|<!\[CDATA\[)", "", text).strip()
    text = re.sub(r"(-->|//\s*\]\]>|\]\]>)$", "", text).strip().rstrip(";")
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def _primary_nodes(data: object) -> list[dict]:
    """The nodes describing the page itself: top-level objects, members of
    ``@graph`` and top-level lists, and their ``mainEntity``. Deeper nodes (related
    articles, ``itemListElement``, authors, ``isPartOf``) belong to *other* content
    and are deliberately not searched."""
    if isinstance(data, list):
        return [n for item in data for n in _primary_nodes(item)]
    if not isinstance(data, dict):
        return []
    nodes = []
    if "@graph" in data:
        nodes.extend(_primary_nodes(data["@graph"]))
    nodes.append(data)
    main = data.get("mainEntity")
    if isinstance(main, (dict, list)):
        nodes.extend(n for n in _primary_nodes(main) if isinstance(n, dict))
    return nodes


def _node_types(node: dict) -> set[str]:
    """Lower-cased @type names, also for full IRIs like ``https://schema.org/NewsArticle``."""
    types = node.get("@type")
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, list):
        return set()
    return {str(t).rsplit("/", 1)[-1].lower() for t in types}


def _node_rank(node: dict) -> int:
    types = _node_types(node)
    if types & _PRIMARY_TYPES:
        return 0
    if types & _PAGE_TYPES:
        return 1
    return 2


def _meta_key(tag: Tag) -> list[str]:
    return [str(tag.get(attr)).strip().lower()
            for attr in ("property", "name", "itemprop", "http-equiv")
            if tag.get(attr)]


def _meta_candidates(soup: BeautifulSoup, keys: tuple[str, ...]) -> Iterator[str]:
    """Values of the given meta keys, in the order of `keys` (not document order), so
    the more specific key wins wherever both are present."""
    by_key: dict[str, list[str]] = {}
    for tag in soup.find_all("meta"):
        content = tag.get("content")
        if not content or not str(content).strip():
            continue
        for key in _meta_key(tag):
            by_key.setdefault(key, []).append(str(content).strip())
    for key in keys:
        yield from by_key.get(key, [])


def _microdata_candidates(soup: BeautifulSoup) -> Iterator[str]:
    for element in soup.find_all(attrs={"itemprop": True}):
        props = str(element.get("itemprop")).lower().split()
        if not any(p in _ITEMPROPS for p in props):
            continue
        value = element.get("content") or element.get("datetime")
        if not value and element.name != "meta":
            value = element.get_text(" ", strip=True)
        if value and str(value).strip():
            yield str(value).strip()

    # HTML5-draft `pubdate` attribute: marks the article's own publication time.
    for element in soup.find_all("time", attrs={"pubdate": True}):
        value = element.get("datetime") or element.get_text(" ", strip=True)
        if value:
            yield str(value).strip()

    # Microformats: hAtom's `published` and h-entry's `dt-published` classes (the
    # former is what most WordPress themes emit on the post's `<time>`).
    for element in soup.find_all(class_=_MICROFORMAT_CLASSES):
        value = (element.get("datetime") or element.get("title") or element.get("content")
                 or element.get_text(" ", strip=True))
        if value and str(value).strip():
            yield str(value).strip()


def _embedded_json_candidate(soup: BeautifulSoup, now: datetime | None) -> str | None:
    """Publication keys inside the JSON blobs of inline scripts (YouTube's
    ``ytInitialPlayerResponse``, Next.js/Nuxt state, ...). Those blobs frequently
    describe related items too, so a value is only returned if all occurrences fall
    on the same day - otherwise it cannot be told which one is the page's."""
    values = []
    for script in soup.find_all("script"):
        type_ = str(script.get("type") or "").lower()
        if "ld+json" in type_:
            continue  # handled with structure in step 1
        text = script.string or ""
        values.extend(m.group(2) for m in _EMBEDDED_JSON_DATE.finditer(text))
    if not values:
        return None
    days = set()
    for value in values:
        parsed = parse_publication_value(value, now=now)
        if parsed is None:
            return None
        days.add(parsed.date())
    return values[0] if len(days) == 1 else None


# --- Parsing ---------------------------------------------------------------------

_HAS_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
#: Labels of anything but the first publication in visible text.
_EXCLUDED_CONTEXT = re.compile(r"\b(updated?|modified|edited|revised|accessed|retrieved)\b",
                               re.IGNORECASE)
_NUMERIC_DMY = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b")

# Two distinct defaults reveal which components the string did not state.
_DEFAULT_A = datetime(2001, 1, 1)
_DEFAULT_B = datetime(2002, 2, 2)


def parse_publication_value(value: str | None, now: datetime | None = None) -> datetime | None:
    """Parses one stated date into a naive UTC datetime, or None if it cannot be
    read *unambiguously* or is implausible.

    Accepted: ISO 8601 (with or without offset/``Z``), RFC 2822 and other textual
    dates with explicit day, month and year, compact ``YYYYMMDD``, and UNIX epoch
    seconds/milliseconds. A missing time of day becomes midnight; a missing day,
    month or year, or an all-numeric ``dd/mm/yyyy`` whose order cannot be told, is
    rejected rather than completed - completing it would be guessing (§10). So is a
    text labelled as an update ("Updated May 12, 2024"). Values without offset are
    taken as stated, since the page's timezone is unknown."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > 64:
        return None

    parsed = _parse_numeric(text) if text.isdigit() else _parse_textual(text)
    if parsed is None:
        return None
    return parsed if is_plausible(parsed, now=now) else None


def _parse_numeric(text: str) -> datetime | None:
    try:
        if len(text) == 8:  # YYYYMMDD
            return datetime.strptime(text, "%Y%m%d")
        if len(text) == 10:  # epoch seconds
            return to_naive(datetime.fromtimestamp(int(text), tz=timezone.utc))
        if len(text) == 13:  # epoch milliseconds
            return to_naive(datetime.fromtimestamp(int(text) / 1000, tz=timezone.utc))
    except (ValueError, OverflowError, OSError):
        return None
    return None  # a bare number of any other length is not a date


def _parse_textual(text: str) -> datetime | None:
    if not _HAS_YEAR.search(text) or _EXCLUDED_CONTEXT.search(text):
        return None
    numeric = _NUMERIC_DMY.match(text)
    if numeric:
        first, second = int(numeric.group(1)), int(numeric.group(2))
        if first <= 12 and second <= 12 and first != second:
            return None  # 05/10/2024: May 10th or October 5th?

    try:
        return to_naive(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        pass
    # Fuzzy, to read visible microdata like "Published May 10, 2024"; the two
    # defaults still catch any component the text leaves out.
    try:
        a = date_parser.parse(text, default=_DEFAULT_A, fuzzy=True)
        b = date_parser.parse(text, default=_DEFAULT_B, fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None
    if (a.year, a.month, a.day) != (b.year, b.month, b.day):
        return None  # some date component was not stated
    return to_naive(a)


def is_plausible(value: datetime, now: datetime | None = None) -> bool:
    """A web publication time lies between 1990 and now (plus a day of slack)."""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    return EARLIEST_PLAUSIBLE <= value <= now + FUTURE_TOLERANCE


# --- Hints for the LLM fallback ----------------------------------------------------

_HINT_CONTEXT_CHARS = 60


def publication_date_hints(html: str | None, limit: int = 12) -> list[str]:
    """Date-bearing markup that `extract_publication_time` could not attribute to the
    publication with certainty - ``<time>`` elements and date-like meta tags of
    unknown meaning - each with the text right before it, for the LLM to judge.

    The dateline often lives only in the page's markup and not in scrapeMM's
    extracted text (it is part of the chrome that content extraction strips), so
    without these the LLM regularly sees no date at all. Hints that are labelled as
    modification/update/access times are left out entirely, so that the LLM is
    never offered them (§10). Ordered by document position - the page's own dateline
    usually precedes lists of related articles - and capped at `limit`."""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    hints: list[str] = []
    seen: set[str] = set()

    def add(hint: str) -> None:
        if hint not in seen:
            seen.add(hint)
            hints.append(hint)

    known = {k.lower() for k in _OG_META_KEYS + _OTHER_META_KEYS}
    for tag in soup.find_all("meta"):
        content = str(tag.get("content") or "").strip()
        keys = _meta_key(tag)
        if not content or not keys or len(content) > 64:
            continue
        key = keys[0]
        if key in known or _EXCLUDED_KEY.search(key):
            continue  # known keys were already tried; excluded ones never count
        if not re.search(r"date|time|publish|issued|created", key):
            continue
        if not _HAS_YEAR.search(content) and not content.isdigit():
            continue
        add(f'<meta {key}="{content}">')

    for element in soup.find_all("time"):
        attrs = " ".join(f"{k}={' '.join(v) if isinstance(v, list) else v}"
                         for k, v in element.attrs.items() if k != "datetime")
        text = element.get_text(" ", strip=True)
        before = _text_before(element)
        # WordPress marks a never-edited post's `<time>` as "published updated", so
        # an update class only disqualifies an element not also marked published.
        excluded_attrs = _EXCLUDED_KEY.search(attrs) and "publish" not in attrs.lower()
        if excluded_attrs or _EXCLUDED_CONTEXT.search(text) \
                or _EXCLUDED_CONTEXT.search(before[-25:]):
            continue
        stamp = element.get("datetime")
        if not stamp and not text:
            continue
        rendered = f'<time datetime="{stamp}">{text}</time>' if stamp else f"<time>{text}</time>"
        add(f'{rendered} - preceded by: "{before}"' if before else rendered)

    return hints[:limit]


def _text_before(element: Tag) -> str:
    """Up to `_HINT_CONTEXT_CHARS` of visible text immediately preceding `element`,
    e.g. "By Jane Doe | Published". Enough to tell a dateline from an update line."""
    pieces = []
    length = 0
    for string in element.find_all_previous(string=True, limit=12):
        if isinstance(string, Comment) or (
                string.parent is not None and string.parent.name in ("script", "style", "title")):
            continue
        piece = " ".join(str(string).split())
        if not piece:
            continue
        pieces.append(piece)
        length += len(piece) + 1
        if length >= _HINT_CONTEXT_CHARS:
            break
    return " ".join(reversed(pieces))[-_HINT_CONTEXT_CHARS:].strip()
