"""Normalized keys identifying a review independently of cosmetic differences
in how the sources spell its URL and claim.

The same review regularly arrives in different spellings: with and without
`www.`, over `http` and `https`, with a trailing slash, with curly instead of
straight quotes, or with `<br/>` line breaks. Matching on the exact
`(url, claim)` pair then treats each spelling as a new review. The keys below collapse such variants
while keeping reviews apart that genuinely differ, e.g., the several claims
checked within one single article. The wording of the claim is never
altered (e.g., a verdict label like "FALSE:" is kept); only its typography is.
The keys serve matching only, they are never stored in place of the
original URL or claim."""

import html
import re
import unicodedata
from hashlib import blake2b
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

#: Query parameters that only track the visitor and never identify the page.
TRACKING_PARAMS = {"fbclid", "gclid", "dclid", "msclkid", "igshid", "mc_cid", "mc_eid", "ref", "amp"}
TRACKING_PARAM_PREFIXES = ("utm_",)

DOUBLE_QUOTES = "\"“”„‟«»‹›″〝〞＂"
SINGLE_QUOTES = "‘’‚‛′`´"

HTML_TAG_REGEX = re.compile(r"</?[a-zA-Z][^>]*>")
WHITESPACE_REGEX = re.compile(r"\s+")

_QUOTE_TABLE = str.maketrans({**{c: None for c in DOUBLE_QUOTES}, **{c: "'" for c in SINGLE_QUOTES}})


def normalize_review_url(url: str) -> str:
    """Returns a scheme-less, canonical form of the URL: lowercase host without
    `www.` and default port, percent-decoded path without trailing slashes,
    sorted query without tracking parameters, and no fragment. The path keeps
    its case because many sites treat paths case-sensitively."""
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        url = "https://" + url
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:  # Malformed URL (e.g., invalid port), compare it as is
        return url

    host = (parts.hostname or "").lower().rstrip(".")
    host = host.removeprefix("www.")
    if port and port not in (80, 443):
        host = f"{host}:{port}"

    path = unquote(parts.path).rstrip("/")

    params = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS and not key.lower().startswith(TRACKING_PARAM_PREFIXES)
    ]
    query = urlencode(sorted(params))

    return f"{host}{path}?{query}" if query else f"{host}{path}"


def normalize_claim(claim: str) -> str:
    """Returns a canonical form of the claim text: HTML entities decoded, HTML
    tags (like `<br/>`) removed, Unicode compatibility-normalized, double quotes
    dropped, single quotes unified, whitespace collapsed, and case folded. The
    wording itself stays untouched."""
    text = html.unescape(claim)
    text = HTML_TAG_REGEX.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTE_TABLE)
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    return text.casefold()


def hash_int64(text: str) -> int:
    """Deterministic hash returning a 64-bit signed integer (fits into BIGINT)."""
    digest = blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


def review_match_key(url: str, claim: str) -> tuple[str, int]:
    """Returns the key under which two reviews count as the same one: the
    normalized URL and a hash of the normalized claim."""
    return normalize_review_url(url), hash_int64(normalize_claim(claim))
