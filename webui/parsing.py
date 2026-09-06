"""Pure helpers for turning stored strings into something the UI can render.

Nothing here touches the database, the filesystem or the network, so the media
reference handling - the part the multimodal rendering depends on - is fully
unit-testable.
"""

from __future__ import annotations

import re

#: The item kinds ezMM knows. Mirrors `ezmm.common.items.KINDS`; kept local so
#: the UI does not have to import the (heavy) pipeline dependency tree.
KINDS = ("image", "video", "audio")

#: Matches a full ezMM item reference, e.g. `<image:12345>`.
ITEM_REF_REGEX = re.compile(rf"<(?:{'|'.join(KINDS)}):[0-9]+>")

#: Same, but capturing kind and identifier separately.
ITEM_KIND_ID_REGEX = re.compile(rf"<({'|'.join(KINDS)}):([0-9]+)>")


def parse_reference(reference: str) -> tuple[str, int] | None:
    """Splits `<image:42>` into `("image", 42)`. None if it is not a reference."""
    match = ITEM_KIND_ID_REGEX.fullmatch(reference.strip())
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def find_references(text: str | None) -> list[str]:
    """All item references in the text, in order of appearance, with duplicates."""
    if not text:
        return []
    return ITEM_REF_REGEX.findall(text)


def unique_references(text: str | None) -> list[str]:
    """All *distinct* item references, in order of first appearance."""
    seen: list[str] = []
    for reference in find_references(text):
        if reference not in seen:
            seen.append(reference)
    return seen


def has_media(text: str | None) -> bool:
    return bool(find_references(text))


def segment(text: str | None) -> list[dict]:
    """Splits the text into an ordered list of `text` and `media` segments.

    The reference itself is kept as a segment of its own (rather than being
    dropped or replaced), so the UI can render it inline as an interactive chip
    that points at the medium rendered below the text.
    """
    if not text:
        return []

    segments: list[dict] = []
    position = 0
    #: Occurrence index per reference, so repeated references stay distinguishable.
    for match in ITEM_KIND_ID_REGEX.finditer(text):
        if match.start() > position:
            segments.append({"type": "text", "text": text[position:match.start()]})
        segments.append({
            "type": "media",
            "reference": match.group(0),
            "kind": match.group(1),
            "id": int(match.group(2)),
        })
        position = match.end()

    if position < len(text):
        segments.append({"type": "text", "text": text[position:]})

    return segments


def describe_multimodal(text: str | None) -> dict:
    """Compact summary of the media a piece of text refers to."""
    references = unique_references(text)
    counts = {kind: 0 for kind in KINDS}
    for reference in references:
        parsed = parse_reference(reference)
        if parsed:
            counts[parsed[0]] += 1
    return {
        "references": references,
        "n_media": len(references),
        "counts": counts,
        "is_multimodal": bool(references),
    }


def domain_of(url: str | None) -> str | None:
    """The bare host of a URL, without a leading `www.`. None if there is none."""
    if not url:
        return None
    match = re.match(r"^\s*[a-zA-Z][a-zA-Z0-9+.\-]*://([^/?#\s]+)", url)
    host = match.group(1) if match else None
    if host is None:
        # Tolerate locators stored without a scheme, e.g. "example.com/a".
        match = re.match(r"^\s*([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})(?:[/?#]|$)", url)
        host = match.group(1) if match else None
    if not host:
        return None
    host = host.split("@")[-1].split(":")[0].lower()
    return host[4:] if host.startswith("www.") else host or None
