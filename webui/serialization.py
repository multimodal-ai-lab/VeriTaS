"""Turns database rows into the payloads the frontend renders.

Two principles:

1. *Nothing is dropped.* Every column of an evidence row and every key of its
   `full_evidence` blob is forwarded, so the UI can show all stored details even
   for fields added to the model after this file was written.
2. *Media references stay in the text.* A proposition is additionally returned
   as an ordered list of text/media segments, so the UI can keep the reference
   visible inline and link it to the medium rendered beneath it.
"""

from __future__ import annotations

from typing import Any, Iterable

from webui.media import MediaRegistry
from webui.parsing import domain_of, segment, unique_references

#: Columns that are represented in a nicer, structured way elsewhere in the
#: payload. They are still included in `columns` for the raw field inspector.
_SOURCE_COLUMNS = ("source_name", "source_kind", "source_locator",
                   "source_proximity", "source_raw_content")


def render_text(text: str | None, registry: MediaRegistry) -> dict:
    """A piece of possibly multimodal text, ready to render.

    `segments` keeps the references inline (as their own segment) and `media`
    lists each distinct referenced item once, in order of first appearance."""
    references = unique_references(text)
    return {
        "text": text,
        "segments": segment(text),
        "media": registry.describe(references),
        "n_media": len(references),
        "is_multimodal": bool(references),
    }


def evidence_payload(row: dict, registry: MediaRegistry) -> dict:
    """One evidence item with everything the detail view needs."""
    full: dict = row.get("full_evidence") or {}
    columns = {key: value for key, value in row.items() if key != "full_evidence"}

    source_blob = full.get("source") or {}
    locator = row.get("source_locator") or source_blob.get("locator")

    return {
        "id": row.get("id"),
        "claim_id": row.get("claim_id"),
        "review_id": row.get("review_id"),
        "article_id": row.get("article_id"),

        "proposition": render_text(row.get("proposition"), registry),

        "source": {
            "name": row.get("source_name") or source_blob.get("name"),
            "kind": row.get("source_kind") or source_blob.get("kind"),
            "locator": locator,
            "domain": domain_of(locator),
            "proximity": row.get("source_proximity") or source_blob.get("proximity"),
            "content": render_text(row.get("source_raw_content"), registry),
        },

        "available_since": row.get("available_since"),
        "role": row.get("role"),
        "accessed_at": row.get("accessed_at"),

        "extraction": {
            "reasoning": row.get("extraction_reasoning"),
            "confidence": row.get("extraction_confidence"),
        },
        "accessible": row.get("accessible"),
        "faithfulness": _faithfulness(row, full),
        "temporal_validation": _temporal(row, full),

        "admissible": row.get("admissible"),
        "inadmissibility_reason": row.get("inadmissibility_reason"),
        "dismissed": row.get("dismissed"),
        "dismissed_reason": row.get("dismissed_reason"),
        # Set while the item waits out a rate limit on its source; it is neither
        # admissible nor rejected until the window has passed.
        "deferred_until": row.get("deferred_until"),

        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),

        # The complete stored state, for the raw field inspector.
        "columns": columns,
        "full_evidence": full,
    }


def _faithfulness(row: dict, full: dict) -> dict | None:
    """The faithfulness judgement, preferring the flat columns over the blob."""
    blob = full.get("faithfulness") or {}
    assessment = _first(row.get("faithfulness_assessment"), blob.get("assessment"))
    if assessment is None and not blob:
        return None
    return {
        "assessment": assessment,
        "reasoning": _first(row.get("faithfulness_reasoning"), blob.get("reasoning")),
        "justification": _first(row.get("faithfulness_justification"),
                                blob.get("justification")),
        "rater": blob.get("rater"),
    }


def _temporal(row: dict, full: dict) -> dict | None:
    """The temporal validation. Older blobs may carry retired fields; only the
    ones the model still defines are surfaced."""
    blob = full.get("temporal_validation") or {}
    if not blob and row.get("before_fact_check") is None:
        return None
    return {
        "before_claim": _first(row.get("before_claim"), blob.get("before_claim")),
        "before_fact_check": _first(row.get("before_fact_check"),
                                    blob.get("before_fact_check")),
        "later_event": _first(row.get("later_event"), blob.get("later_event")),
        "reasoning": _first(row.get("temporal_reasoning"), blob.get("reasoning")),
        "justification": _first(row.get("temporal_justification"),
                                blob.get("justification")),
        "rater": blob.get("rater"),
    }


def _first(*values: Any) -> Any:
    """The first value that is not None."""
    for value in values:
        if value is not None:
            return value
    return None


def claim_payload(detail: dict, registry: MediaRegistry) -> dict:
    """The claim detail bundle: claim, gold verdict, reviews, evidence, results."""
    claim = dict(detail["claim"])
    claim["content"] = render_text(claim.get("data"), registry)

    evidence = [evidence_payload(row, registry) for row in detail["evidence"]]
    verdict = detail.get("verdict")

    return {
        "claim": claim,
        "verdict": _verdict_payload(verdict, registry) if verdict else None,
        "reviews": [dict(review) for review in detail.get("reviews") or []],
        "evidence": evidence,
        "results": [dict(result) for result in detail.get("results") or []],
        "neighbours": detail.get("neighbours") or {},
        "timeline": build_timeline(claim, evidence),
    }


def _verdict_payload(verdict: dict, registry: MediaRegistry) -> dict:
    """The gold verdict, with each medium's rating resolved to a real file."""
    payload = dict(verdict)
    full = payload.get("full_verdict") or {}
    media_verdicts = []
    for medium in full.get("media_verdicts") or []:
        reference = medium.get("reference")
        item = registry.get_by_reference(reference) if reference else None
        media_verdicts.append({
            **medium,
            "media": item.to_dict() if item else {"reference": reference, "exists": False},
        })
    payload["media_verdicts"] = media_verdicts
    return payload


def build_timeline(claim: dict, evidence: Iterable[dict]) -> dict:
    """Positions of t_c, t_f and every dated evidence item on one axis.

    The UI draws the studied interval `t_c < t_e <= t_f` from this, so the
    classification is computed once here rather than in three places in JS."""
    t_c, t_f = claim.get("t_c"), claim.get("t_f")
    points = []
    for item in evidence:
        available_since = item.get("available_since")
        if available_since is None:
            continue
        temporal = item.get("temporal_validation") or {}
        before_claim = temporal.get("before_claim")
        before_fact_check = temporal.get("before_fact_check")
        if before_claim is None and t_c is not None:
            before_claim = available_since <= t_c
        if before_fact_check is None and t_f is not None:
            before_fact_check = available_since <= t_f

        if before_claim:
            zone = "before_claim"
        elif before_fact_check:
            zone = "in_window"
        else:
            zone = "after_fact_check"

        points.append({
            "evidence_id": item.get("id"),
            "available_since": available_since,
            "zone": zone,
            "admissible": item.get("admissible"),
            "role": item.get("role"),
        })

    points.sort(key=lambda point: point["available_since"])
    return {
        "t_c": t_c,
        "t_f": t_f,
        "points": points,
        "n_undated": sum(1 for item in evidence if item.get("available_since") is None),
    }
