"""Parsing of model output: the JSON answer with the integrity, certainty and justification, and
URLs and links in text."""

import json
import re
from typing import Callable
from urllib.parse import urljoin

from .types import LABELS_7

# Tolerates line breaks inside strings, which models sometimes write into the justification
_JSON_DECODER = json.JSONDecoder(strict=False)


def extract_answer(response: str) -> dict:
    """The JSON object the model answered with: the last one in the response that has an
    "integrity" field, in a ```json block or not ({} if there is none)."""
    response = response or ""
    for match in reversed(list(re.finditer(r"\{", response))):
        try:
            answer, _ = _JSON_DECODER.raw_decode(response, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(answer, dict) and "integrity" in answer:
            return answer
    return {}


def extract_verdict(response: str) -> str | None:
    """The label of the model's answer in the 7-class scheme, made of its integrity and
    certainty: "Unknown" if the answer says so (whatever its certainty). None if there is no
    valid answer: none at all, or one without a valid integrity and certainty."""
    answer = extract_answer(response)
    integrity = str(answer.get("integrity") or "").strip().lower()
    certainty = str(answer.get("certainty") or "").strip().lower()
    if integrity == "unknown":
        return "Unknown"
    label = f"{integrity.capitalize()} ({certainty})"
    return label if label in LABELS_7 else None


def extract_justification(response: str) -> str:
    """The justification of the model's answer ("" if there is none)."""
    justification = extract_answer(response).get("justification")
    return justification.strip() if isinstance(justification, str) else ""


# The target of a markdown link "[text](target)", an absolute URL or a relative one, with the
# link's optional title ('[text](target "title")'). One level of parentheses inside the target is
# allowed, as in Wikipedia's "https://en.wikipedia.org/wiki/Foo_(bar)".
_MARKDOWN_LINK_TARGET = re.compile(
    r"\]\(\s*<?((?:[^()\s<>]|\([^()\s<>]*\))+)>?(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^()\n]*\)))?\s*\)")
# A whole markdown link whose text holds no brackets, with the target as group 1. Its text may
# span lines: a teaser of an article, with its image, date and headline, is one link. A text of
# more than 1000 characters is none.
_MARKDOWN_LINK = re.compile(r"\[[^\[\]]{0,1000}" + _MARKDOWN_LINK_TARGET.pattern)
_BARE_URL = re.compile(r"https?://[^\s<>\"'\[\]]+")


def _link_url(target: str, base_url: str | None) -> str | None:
    """The URL a markdown link's target points to: the target itself if it is absolute,
    resolved against `base_url` if it is relative ("/about", "//cdn.example.org/a.jpg"). None
    if it is relative and there is no `base_url`, or if it points to no web page ("#top",
    "mailto:...", "javascript:;")."""
    if target.startswith(("http://", "https://")):
        return target
    if base_url is None or target.startswith("#"):
        return None
    url = urljoin(base_url, target)
    return url if url.startswith(("http://", "https://")) else None


def extract_urls(text: str, base_url: str | None = None) -> list[str]:
    """Extract the URLs in `text`, in order of appearance and without duplicates.

    Markdown links are parsed first, so URLs containing parentheses stay intact. With
    `base_url`, the URL of the page `text` is from, relative links are resolved against
    it; without, they are left out. Bare URLs lose trailing punctuation and unbalanced
    closing parentheses ("(see https://x.org).").
    """
    found = []

    def take_link(m: re.Match) -> str:
        url = _link_url(m.group(1), base_url)
        if url is None:
            return m.group(0)
        found.append((m.start(1), url))
        # Blank out the link (keeping positions) so its URL is not found again as a bare URL
        return " " * len(m.group(0))

    rest = _MARKDOWN_LINK_TARGET.sub(take_link, text or "")
    for m in _BARE_URL.finditer(rest):
        found.append((m.start(), _trim_bare_url(m.group(0))))
    return list(dict.fromkeys(url for _, url in sorted(found)))


def _trim_bare_url(url: str) -> str:
    """A bare URL without the punctuation and unbalanced closing parentheses that follow it
    in text ("(see https://x.org/a)." -> "https://x.org/a")."""
    while True:
        stripped = url.rstrip(".,;:!?*_`")
        if stripped.endswith(")") and stripped.count(")") > stripped.count("("):
            stripped = stripped[:-1]
        if stripped == url:
            return url
        url = stripped


def remove_links(text: str, is_target: Callable[[str], bool], base_url: str | None = None) -> tuple[str, int]:
    """Remove the links to the URLs that `is_target` accepts from `text`, a web page in
    markdown: a markdown link with its text and title, which may say what the linked page says,
    as a headline does, or, if its text holds brackets, only its target and title, as where
    its text begins is uncertain then; and bare URLs, also those in HTML links. With
    `base_url`, the URL of the page, relative links are resolved against it.

    Returns the text and the number of links removed.
    """
    n_removed = 0

    def remove_link(m: re.Match) -> str:
        nonlocal n_removed
        url = _link_url(m.group(1), base_url)
        if url and is_target(url):
            n_removed += 1
            return ""
        return m.group(0)

    def remove_url(m: re.Match) -> str:
        nonlocal n_removed
        url = _trim_bare_url(m.group(0))
        if is_target(url):
            n_removed += 1
            return m.group(0)[len(url):]
        return m.group(0)

    text = _MARKDOWN_LINK.sub(remove_link, text)
    text = _MARKDOWN_LINK_TARGET.sub(remove_link, text)  # Of the links whose text holds brackets
    text = _BARE_URL.sub(remove_url, text)
    return text, n_removed
