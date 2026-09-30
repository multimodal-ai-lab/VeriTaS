"""Turns database rows into the payloads the frontend renders.

Two principles:

1. *Nothing is dropped.* Every column of a row and every key of its JSONB blob is
   forwarded, so the UI can show all stored details even for fields added to the
   model after this file was written.
2. *Media references stay in the text.* A proposition is additionally returned
   as an ordered list of text/media segments, so the UI can keep the reference
   visible inline and link it to the medium rendered beneath it.
"""

from __future__ import annotations

from typing import Any, Iterable

from webui.media import MediaRegistry
from webui.parsing import domain_of, segment, unique_references

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


def evidence_item_payload(row: dict, registry: MediaRegistry) -> dict:
    """One evidence item with every source that reports its proposition.

    The item carries the proposition and its role; each source carries what Stage 2
    decided about it. The item survives as long as one of its sources does, which is
    what `admissible` states here."""
    full: dict = row.get("full_evidence") or {}
    columns = {key: value for key, value in row.items()
               if key not in ("full_evidence", "sources")}

    return {
        "id": row.get("id"),
        "claim_id": row.get("claim_id"),
        "review_id": row.get("review_id"),
        "article_id": row.get("article_id"),

        "proposition": render_text(row.get("proposition"), registry),
        "role": row.get("role"),

        "sources": [evidence_source_payload(source, registry)
                    for source in row.get("sources") or []],
        "n_sources": row.get("n_sources"),
        "n_admissible_sources": row.get("n_admissible_sources"),

        "extraction": {
            "reasoning": row.get("extraction_reasoning"),
            "confidence": row.get("extraction_confidence"),
        },
        "later_event": _later_event(row, full),

        # t_e of the item: the earliest time any of its sources made the
        # proposition available.
        "available_since": row.get("available_since"),
        "admissible": row.get("admissible"),
        "inadmissibility_reason": row.get("inadmissibility_reason"),
        "dismissed": row.get("dismissed"),
        "dismissed_reason": row.get("dismissed_reason"),

        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),

        # The complete stored state, for the raw field inspector.
        "columns": columns,
        "full_evidence": full,
    }


def evidence_source_payload(row: dict, registry: MediaRegistry) -> dict:
    """One evidence source with everything the detail view needs.

    Rows come from `citation_rows` - a citation joined with the source it cites -
    and the item that owns them, so the proposition and role are available as
    context even when a single source is fetched on its own."""
    full: dict = row.get("full_source") or {}
    item_blob: dict = row.get("full_evidence") or {}
    columns = {key: value for key, value in row.items()
               if key not in ("full_source", "full_evidence")}
    locator = row.get("source_locator") or full.get("locator")

    return {
        "id": row.get("id"),
        "evidence_id": row.get("evidence_id"),
        "claim_id": row.get("claim_id"),
        "review_id": row.get("review_id"),
        "article_id": row.get("article_id"),

        # The item this source belongs to. It survives as long as one of its
        # sources does, so a discarded source need not be a loss.
        "proposition": render_text(row.get("proposition"), registry),
        "role": row.get("role"),
        "evidence_admissible": row.get("evidence_admissible"),
        "n_sources": row.get("n_sources"),
        "extraction": {
            "reasoning": row.get("extraction_reasoning"),
            "confidence": row.get("extraction_confidence"),
        },
        # Judged for the proposition, so it is shown as context of the item.
        "later_event": _later_event(row, item_blob),

        "source": {
            "name": row.get("source_name") or full.get("name"),
            "kind": row.get("source_kind") or full.get("kind"),
            "locator": locator,
            "domain": row.get("source_domain") or domain_of(locator),
            "proximity": row.get("source_proximity") or full.get("proximity"),
            "content": render_text(row.get("source_raw_content"), registry),
        },

        "available_since": row.get("available_since"),
        "accessed_at": row.get("accessed_at"),
        "accessible": row.get("accessible"),
        "faithfulness": _faithfulness(row, full),
        "temporal_validation": _temporal(row, full),

        "admissible": row.get("admissible"),
        "inadmissibility_reason": row.get("inadmissibility_reason"),
        "dismissed": row.get("dismissed"),
        "dismissed_reason": row.get("dismissed_reason"),
        # Set while the source waits out a rate limit; it is neither admissible nor
        # rejected until the window has passed.
        "deferred_until": row.get("deferred_until"),

        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),

        # The complete stored state, for the raw field inspector.
        "columns": columns,
        "full_source": full,
        "full_evidence": item_blob,
    }


def evidence_summary(row: dict, registry: MediaRegistry) -> dict:
    """One evidence source as the browser lists it.

    A lighter sibling of `evidence_source_payload`: the long texts and the JSONB
    blobs are left out - the card links to the claim detail for those - but the
    proposition is still rendered multimodally, so media show up in the list."""
    locator = row.get("source_locator")
    return {
        "id": row.get("id"),
        "claim_id": row.get("claim_id"),
        "review_id": row.get("review_id"),
        "article_id": row.get("article_id"),

        "proposition": render_text(row.get("proposition"), registry),

        "source": {
            "name": row.get("source_name"),
            "kind": row.get("source_kind"),
            "locator": locator,
            # Computed in SQL for the domain facet; recomputed here when absent.
            "domain": row.get("source_domain") or domain_of(locator),
            "proximity": row.get("source_proximity"),
        },

        "available_since": row.get("available_since"),
        "role": row.get("role"),
        # The evidence item this source belongs to. It survives as long as one of
        # its sources does, so a discarded source need not be a loss.
        "evidence_id": row.get("evidence_id"),
        "evidence_admissible": row.get("evidence_admissible"),
        "n_sources": row.get("n_sources"),
        "accessed_at": row.get("accessed_at"),

        "extraction": {"confidence": row.get("extraction_confidence")},
        "accessible": row.get("accessible"),
        "faithfulness": {
            "assessment": row.get("faithfulness_assessment"),
            "justification": row.get("faithfulness_justification"),
        } if row.get("faithfulness_assessment") is not None else None,
        "temporal_validation": {
            "before_claim": row.get("before_claim"),
            "before_fact_check": row.get("before_fact_check"),
        } if row.get("before_fact_check") is not None else None,
        "later_event": {
            "change_detected": row.get("later_event"),
            "justification": row.get("later_event_justification"),
        } if row.get("later_event") is not None else None,
        "zone": temporal_zone(row.get("before_claim"), row.get("before_fact_check")),

        "admissible": row.get("admissible"),
        "inadmissibility_reason": row.get("inadmissibility_reason"),
        "dismissed": row.get("dismissed"),
        "dismissed_reason": row.get("dismissed_reason"),
        "deferred_until": row.get("deferred_until"),

        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),

        # Enough claim context to make a decontextualized list entry readable.
        "claim": {
            "id": row.get("claim_id"),
            "data": row.get("claim_data"),
            "t_c": row.get("claim_date"),
            "language": row.get("claim_language"),
            "status": row.get("claim_status"),
        },
    }


def evidence_list_payload(page: dict, registry: MediaRegistry) -> dict:
    """A page of evidence rows, as returned by `queries.list_evidence`."""
    return {
        **{key: value for key, value in page.items() if key != "evidence"},
        "evidence": [evidence_summary(row, registry) for row in page.get("evidence") or []],
    }


#: How many media a claim preview shows before it says "+n more". A preview is a
#: thumbnail strip, not a gallery; the detail view renders all of them.
PREVIEW_MEDIA_LIMIT = 4


def claim_summary_payload(row: dict, registry: MediaRegistry) -> dict:
    """One claim as the browser lists it, with its own media resolved.

    Only the first few media are attached: a claim can reference many, and a
    list page should not have to load all of them."""
    claim = dict(row)
    content = render_text(claim.get("data"), registry)
    claim["content"] = {
        **content,
        "media": content["media"][:PREVIEW_MEDIA_LIMIT],
        "n_hidden_media": max(content["n_media"] - PREVIEW_MEDIA_LIMIT, 0),
    }
    return claim


def claim_list_payload(page: dict, registry: MediaRegistry) -> dict:
    """A page of claim rows, as returned by `queries.list_claims`."""
    return {
        **{key: value for key, value in page.items() if key != "claims"},
        "claims": [claim_summary_payload(row, registry) for row in page.get("claims") or []],
    }


def temporal_zone(before_claim, before_fact_check) -> str | None:
    """Where an item sits relative to the studied interval `t_c < t_e <= t_f`.

    Mirrors the classification `build_timeline` applies to dated items, but
    works from the stored flags alone, so undated items stay unclassified."""
    if before_fact_check is None:
        return None
    if before_claim:
        return "before_claim"
    return "in_window" if before_fact_check else "after_fact_check"


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
    """Where a source sits relative to the two cutoffs. Both are computed from
    `t_e`; the later-event judgement belongs to the item and is served with it."""
    blob = full.get("temporal_validation") or {}
    if not blob and row.get("before_fact_check") is None:
        return None
    return {
        "before_claim": _first(row.get("before_claim"), blob.get("before_claim")),
        "before_fact_check": _first(row.get("before_fact_check"),
                                    blob.get("before_fact_check")),
    }


def _later_event(row: dict, full: dict) -> dict | None:
    """The §3.3 (3) judgement of an evidence item: does the proposition rest on a
    change of the world that happened only after the claim was made?"""
    blob = full.get("later_event") or {}
    change = _first(row.get("later_event"), blob.get("change_detected"))
    if change is None and not blob:
        return None
    return {
        "change_detected": change,
        "reasoning": _first(row.get("later_event_reasoning"), blob.get("reasoning")),
        "justification": _first(row.get("later_event_justification"),
                                blob.get("justification")),
        "rater": _first(row.get("later_event_rater"), blob.get("rater")),
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

    evidence = [evidence_item_payload(row, registry) for row in detail["evidence"]]
    verdict = detail.get("verdict")

    return {
        "claim": claim,
        "verdict": _verdict_payload(verdict, registry) if verdict else None,
        "reviews": [dict(review) for review in detail.get("reviews") or []],
        "evidence": evidence,
        "rationales": [_rationale_payload(row, registry)
                       for row in detail.get("rationales") or []],
        "results": [dict(result) for result in detail.get("results") or []],
        "neighbours": detail.get("neighbours") or {},
        "timeline": build_timeline(claim, evidence),
    }


def _rationale_payload(row: dict, registry: MediaRegistry) -> dict:
    """A verdict rationale with its media references resolved."""
    payload = dict(row)
    payload["content"] = render_text(row.get("rationale"), registry)
    return payload


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
    """Positions of t_c, t_f and every dated *source* on one axis.

    Sources carry the dates, so they are what the timeline plots; each point names
    the evidence item it belongs to, which is what the UI scrolls to. The studied
    interval `t_c < t_e <= t_f` is classified once here rather than in three places
    in JS."""
    t_c, t_f = claim.get("t_c"), claim.get("t_f")
    points = []
    undated = 0
    for item in evidence:
        for source in item.get("sources") or []:
            available_since = source.get("available_since")
            if available_since is None:
                undated += 1
                continue
            temporal = source.get("temporal_validation") or {}
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
                "source_id": source.get("id"),
                "evidence_id": item.get("id"),
                "available_since": available_since,
                "zone": zone,
                "admissible": source.get("admissible"),
                # Whether losing this source cost the item its proposition.
                "evidence_admissible": item.get("admissible"),
                "role": item.get("role"),
            })

    points.sort(key=lambda point: point["available_since"])
    return {
        "t_c": t_c,
        "t_f": t_f,
        "points": points,
        "n_undated": undated,
    }
