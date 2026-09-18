"""Read-only SQL against the VeriTaS database.

Only the tables the Gold Evidence Reconstruction writes (`evidence`,
`gold_evidence_results`, `claims.gold_evidence_*`) plus the claim context needed
to make sense of them (`claims`, `reviews`, `articles`, `verdicts`) are touched.

`evidence.full_evidence` is the authoritative representation of an item - the
flat columns exist for querying - so detail views always return the JSONB blob
and the UI renders every field found in it.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime

from webui import database
from webui.stats import (
    INSTANCE_REASON_ORDER,
    REASON_ORDER,
    STATUS_ORDER,
    describe,
    evidence_funnel,
    recoverability,
    series,
    share,
    signed_log_histogram,
)

#: PostgreSQL regexes matching ezMM item references. Mirror `webui.parsing`.
MEDIA_REF_SQL = "<(image|video|audio):[0-9]+>"
IMAGE_REF_SQL = "<image:[0-9]+>"
VIDEO_REF_SQL = "<video:[0-9]+>"

#: The values the `media` filter accepts, and the pattern each one matches on.
#: `none` inverts `any`; every other value must be present at least once.
MEDIA_FILTERS = {
    "any": MEDIA_REF_SQL,
    "image": IMAGE_REF_SQL,
    "video": VIDEO_REF_SQL,
    "audio": "<audio:[0-9]+>",
    "none": MEDIA_REF_SQL,
}


def media_clause(column: str, media: str) -> str:
    """A regex test on `column` for one of the `MEDIA_FILTERS` values.

    The pattern comes from that fixed table, never from the request, so it is
    safe to interpolate - a regex cannot be a bind parameter in this position."""
    pattern = MEDIA_FILTERS.get(media)
    if pattern is None:
        raise ValueError(f"Unknown media filter: {media!r}. "
                         f"Expected one of {', '.join(MEDIA_FILTERS)}.")
    return f"{column} {'!~' if media == 'none' else '~'} '{pattern}'"


def domain_expr(alias: str = "s") -> str:
    """SQL extracting the bare host of `<alias>.locator`, without `www.`.

    Mirrors `webui.parsing.domain_of` so the facet counts and the rendered
    domains agree. `alias` is caller-supplied SQL, never user input."""
    return (r"lower(regexp_replace("
            rf"coalesce(substring({alias}.locator "
            r"from '^[a-zA-Z][a-zA-Z0-9+.-]*://([^/?#]+)'), "
            rf"{alias}.locator), '^www\.', ''))")


#: Claim columns the browser may sort by. Whitelisted, never interpolated blindly.
SORTABLE = {
    "id": "c.id",
    "date": "c.date",
    "updated": "c.gold_evidence_updated_at",
    "status": "c.gold_evidence_status",
}

#: Evidence columns the browser may sort by. Same whitelist discipline.
EVIDENCE_SORTABLE = {
    "id": "s.id",
    "claim": "s.claim_id",
    "available_since": "s.available_since",
    "confidence": "e.extraction_confidence",
    "faithfulness": "s.faithfulness_assessment",
    "updated": "s.updated_at",
}

#: Admissibility is a tri-state (yes / no / not yet filtered); the browser
#: filters on these names rather than on a raw boolean.
ADMISSIBILITY_SQL = {
    "admissible": "s.admissible",
    "inadmissible": "s.admissible IS FALSE",
    "unfiltered": "s.admissible IS NULL",
}

#: Where a source falls relative to the studied interval `t_c < t_e <= t_f`.
#: Derived from the flat columns, so no join to `claims`/`reviews` is needed.
ZONE_SQL = {
    "before_claim": "s.before_claim",
    "in_window": "(s.before_fact_check AND s.before_claim IS NOT TRUE)",
    "after_fact_check": "s.before_fact_check IS FALSE",
    "unvalidated": "s.before_fact_check IS NULL",
}

#: Reference times per claim, as defined in `gold_evidence.filtering`:
#: t_c is `claims.date`; t_f is the latest `published` among non-dismissed reviews.
REFERENCE_TIMES_SQL = """
    SELECT MAX(r.published) AS t_f
    FROM reviews r
    WHERE r.id = ANY (c.review_ids) AND r.dismissed = FALSE
"""

#: Counted per claim. The item-level numbers describe propositions, the
#: source-level ones where those propositions could be read and when - which is
#: why the window counts come from `evidence_sources`.
EVIDENCE_COUNTS_SQL = f"""
    SELECT COUNT(*)                                                   AS n_evidence,
           COUNT(*) FILTER (WHERE e.admissible)                       AS n_admissible,
           COUNT(*) FILTER (WHERE e.admissible IS FALSE)              AS n_inadmissible,
           COUNT(*) FILTER (WHERE e.admissible IS NULL)               AS n_unfiltered,
           COUNT(*) FILTER (WHERE e.proposition ~ '{MEDIA_REF_SQL}')  AS n_multimodal,
           COUNT(*) FILTER (WHERE e.proposition ~ '{IMAGE_REF_SQL}')  AS n_with_images,
           COUNT(*) FILTER (WHERE e.proposition ~ '{VIDEO_REF_SQL}')  AS n_with_videos,
           COALESCE(SUM(e.n_sources), 0)                              AS n_sources,
           (SELECT COUNT(*) FROM evidence_sources s
             WHERE s.claim_id = c.id AND s.admissible AND s.before_claim)
                                                                      AS n_before_claim,
           (SELECT COUNT(*) FROM evidence_sources s
             WHERE s.claim_id = c.id AND s.admissible AND s.before_fact_check
               AND s.before_claim IS FALSE)                           AS n_in_window
    FROM evidence e
    WHERE e.claim_id = c.id
"""


# ---------------------------------------------------------------------------
# Claim browser
# ---------------------------------------------------------------------------

async def list_claims(
        *,
        statuses: list[str] | None = None,
        reason: str | None = None,
        languages: list[str] | None = None,
        query: str | None = None,
        released: bool | None = None,
        media: str | None = None,
        date_from: date | datetime | None = None,
        date_to: date | datetime | None = None,
        sort: str = "updated",
        descending: bool = True,
        limit: int = 25,
        offset: int = 0,
) -> dict:
    """One page of claims that the reconstruction has touched, newest first.

    Claims without a `gold_evidence_status` were never processed and are not
    listed by default: the browser is about what the pipeline produced."""
    wants_unprocessed = bool(statuses) and "unprocessed" in statuses
    concrete = [status for status in (statuses or []) if status != "unprocessed"] or None

    where, args = build_claim_filters(
        statuses=concrete,
        wants_unprocessed=wants_unprocessed,
        reason=reason,
        languages=languages,
        query=query,
        released=released,
        media=media,
        date_from=date_from,
        date_to=date_to,
    )
    order_column = SORTABLE.get(sort, SORTABLE["updated"])
    direction = "DESC" if descending else "ASC"
    page_order = order_column.replace("c.", "p.")

    page_sql = f"""
        WITH page AS (
            SELECT c.id, c.data, c.date, c.language, c.review_ids,
                   c.gold_evidence_status, c.gold_evidence_reason, c.gold_evidence_updated_at,
                   c.released_quarter, c.released_longitudinal, c.is_rectified, c.variant_id
            FROM claims c
            WHERE {where}
            ORDER BY {order_column} {direction} NULLS LAST, c.id {direction}
            LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}
        )
        SELECT p.*, tf.t_f, ev.*
        FROM page p
        LEFT JOIN LATERAL (
            SELECT MAX(r.published) AS t_f
            FROM reviews r
            WHERE r.id = ANY (p.review_ids) AND r.dismissed = FALSE
        ) tf ON TRUE
        LEFT JOIN LATERAL (
            SELECT COUNT(*)                                                   AS n_evidence,
                   COUNT(*) FILTER (WHERE e.admissible)                       AS n_admissible,
                   COUNT(*) FILTER (WHERE e.admissible IS FALSE)              AS n_inadmissible,
                   COUNT(*) FILTER (WHERE e.admissible IS NULL)               AS n_unfiltered,
                   COUNT(*) FILTER (WHERE e.proposition ~ '{MEDIA_REF_SQL}')  AS n_multimodal,
                   COUNT(*) FILTER (WHERE e.proposition ~ '{IMAGE_REF_SQL}')  AS n_with_images,
                   COUNT(*) FILTER (WHERE e.proposition ~ '{VIDEO_REF_SQL}')  AS n_with_videos,
                   COALESCE(SUM(e.n_sources), 0)                              AS n_sources
            FROM evidence e
            WHERE e.claim_id = p.id
        ) ev ON TRUE
        ORDER BY {page_order} {direction} NULLS LAST, p.id {direction};
    """
    count_sql = f"SELECT COUNT(*) FROM claims c WHERE {where};"

    rows, total = await asyncio.gather(
        database.fetch(page_sql, *args, limit, offset),
        database.fetchval(count_sql, *args),
    )
    return {
        "total": int(total or 0),
        "limit": limit,
        "offset": offset,
        "claims": [claim_summary(row) for row in rows],
    }


def build_claim_filters(
        *,
        statuses: list[str] | None = None,
        wants_unprocessed: bool = False,
        reason: str | None = None,
        languages: list[str] | None = None,
        query: str | None = None,
        released: bool | None = None,
        media: str | None = None,
        date_from: date | datetime | None = None,
        date_to: date | datetime | None = None,
) -> tuple[str, list]:
    """Builds the WHERE clause shared by the page and the count query.

    Every user-supplied value becomes a positional parameter; only whitelisted
    fragments are ever interpolated into the SQL text."""
    clauses: list[str] = []
    args: list = []

    def add(template: str, value) -> None:
        args.append(value)
        clauses.append(template.format(n=len(args)))

    if statuses or wants_unprocessed:
        parts = []
        if statuses:
            args.append(statuses)
            parts.append(f"c.gold_evidence_status = ANY (${len(args)}::text[])")
        if wants_unprocessed:
            parts.append("c.gold_evidence_status IS NULL")
        clauses.append("(" + " OR ".join(parts) + ")")
    else:
        clauses.append("c.gold_evidence_status IS NOT NULL")

    if reason:
        add("c.gold_evidence_reason = ${n}", reason)
    if languages:
        add("c.language = ANY (${n}::text[])", languages)
    if query:
        add("c.data ILIKE '%' || ${n} || '%'", query)
    if released is not None:
        clauses.append("(c.released_quarter OR c.released_longitudinal)"
                       if released else
                       "NOT (c.released_quarter OR c.released_longitudinal)")
    if media:
        clauses.append(media_clause("c.data", media))
    if date_from is not None:
        add("c.date >= ${n}", as_datetime(date_from, end_of_day=False))
    if date_to is not None:
        add("c.date <= ${n}", as_datetime(date_to, end_of_day=True))

    return " AND ".join(clauses), args


def as_datetime(value: date | datetime, *, end_of_day: bool) -> datetime:
    """Widens a plain date to the start or the end of that day."""
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.max.time() if end_of_day else datetime.min.time())


def claim_summary(row) -> dict:
    """The claim fields every list entry and the detail header need."""
    row = dict(row)
    return {
        "id": row["id"],
        "data": row["data"],
        "t_c": row["date"],
        "t_f": row.get("t_f"),
        "language": row["language"],
        "status": row["gold_evidence_status"],
        "reason": row["gold_evidence_reason"],
        "updated_at": row["gold_evidence_updated_at"],
        "released": bool(row["released_quarter"] or row["released_longitudinal"]),
        "released_quarter": row["released_quarter"],
        "released_longitudinal": row["released_longitudinal"],
        "is_rectified": row["is_rectified"],
        "variant_id": row["variant_id"],
        "n_evidence": int(row.get("n_evidence") or 0),
        "n_admissible": int(row.get("n_admissible") or 0),
        "n_inadmissible": int(row.get("n_inadmissible") or 0),
        "n_unfiltered": int(row.get("n_unfiltered") or 0),
        "n_multimodal": int(row.get("n_multimodal") or 0),
        "n_with_images": int(row.get("n_with_images") or 0),
        "n_with_videos": int(row.get("n_with_videos") or 0),
    }


# ---------------------------------------------------------------------------
# Claim detail
# ---------------------------------------------------------------------------

async def get_claim(claim_id: int) -> dict | None:
    """Everything the detail view shows for one claim."""
    claim_sql = f"""
        SELECT c.id, c.data, c.date, c.language, c.review_ids, c.appearance_ids,
               c.gold_evidence_status, c.gold_evidence_reason, c.gold_evidence_updated_at,
               c.released_quarter, c.released_longitudinal, c.is_rectified, c.variant_id,
               c.dismissed, c.dismissed_reason, c.media_origin,
               tf.t_f, ev.*
        FROM claims c
        LEFT JOIN LATERAL ({REFERENCE_TIMES_SQL}) tf ON TRUE
        LEFT JOIN LATERAL ({EVIDENCE_COUNTS_SQL}) ev ON TRUE
        WHERE c.id = $1;
    """
    row = await database.fetchrow(claim_sql, claim_id)
    if row is None:
        return None

    evidence, rationales, reviews, verdict, results, neighbours = await asyncio.gather(
        get_evidence_for_claim(claim_id),
        get_verdict_rationales(claim_id),
        get_reviews(list(row["review_ids"] or [])),
        get_verdict(claim_id),
        get_results_for_claim(claim_id),
        get_neighbours(claim_id),
    )

    claim = claim_summary(row)
    claim.update({
        "appearance_ids": list(row["appearance_ids"] or []),
        "review_ids": list(row["review_ids"] or []),
        "dismissed": row["dismissed"],
        "dismissed_reason": row["dismissed_reason"],
        "media_origin": row["media_origin"],
        "n_before_claim": int(row["n_before_claim"] or 0),
        "n_in_window": int(row["n_in_window"] or 0),
    })
    return {
        "claim": claim,
        "evidence": evidence,
        "rationales": rationales,
        "reviews": reviews,
        "verdict": verdict,
        "results": results,
        "neighbours": neighbours,
    }


async def get_verdict_rationales(claim_id: int) -> list[dict]:
    """The reasoning that bridges evidence and verdict, one row per article."""
    rows = await database.fetch(
        """
        SELECT id, claim_id, review_id, article_id, rationale, extraction_reasoning,
               created_at, updated_at
        FROM verdict_rationales
        WHERE claim_id = $1
        ORDER BY id;
        """,
        claim_id,
    )
    return [dict(row) for row in rows]


async def get_reviews(review_ids: list[int]) -> list[dict]:
    """The professional fact-checks behind a claim, plus their scraped article."""
    if not review_ids:
        return []
    rows = await database.fetch(
        """
        SELECT r.id, r.url, r.published, r.modified, r.raw_rating, r.raw_claim,
               r.raw_publisher_name, r.raw_publisher_url, r.author_name, r.language,
               r.dismissed, r.dismissed_reason, r.stage,
               a.id AS article_id, a.title AS article_title, a.url AS article_url,
               length(a.extracted_article) AS article_length
        FROM reviews r
        LEFT JOIN articles a ON r.id = ANY (a.review_ids) AND a.dismissed = FALSE
        WHERE r.id = ANY ($1::int[])
        ORDER BY r.published DESC NULLS LAST, r.id;
        """,
        review_ids,
    )
    return [dict(row) for row in rows]


async def get_verdict(claim_id: int) -> dict | None:
    """The current gold verdict, which the reconstruction never modifies."""
    row = await database.fetchrow(
        """
        SELECT id, veracity, context_coverage, integrity, media, full_verdict, created_at
        FROM verdicts
        WHERE claim_id = $1 AND is_current
        LIMIT 1;
        """,
        claim_id,
    )
    return dict(row) if row else None


async def get_neighbours(claim_id: int) -> dict:
    """Previous/next processed claim by ID, so the detail view can be paged through.

    Written as an ordered walk along the primary key rather than as `MIN`/`MAX`
    over a predicate, so PostgreSQL stops at the first neighbour instead of
    scanning every processed claim on either side."""
    row = await database.fetchrow(
        """
        SELECT (SELECT id FROM claims
                 WHERE id < $1 AND gold_evidence_status IS NOT NULL
                 ORDER BY id DESC LIMIT 1) AS previous_id,
               (SELECT id FROM claims
                 WHERE id > $1 AND gold_evidence_status IS NOT NULL
                 ORDER BY id ASC LIMIT 1) AS next_id;
        """,
        claim_id,
    )
    return dict(row) if row else {"previous_id": None, "next_id": None}


async def get_results_for_claim(claim_id: int) -> list[dict]:
    """The stored sufficiency-validation results, one per condition and mode."""
    rows = await database.fetch(
        """
        SELECT id, claim_id, condition, ensemble_mode, n_evidence, predicted_verdict,
               member_responses, property_diffs, max_property_diff, is_close, threshold,
               model_specifiers, error, created_at, updated_at
        FROM gold_evidence_results
        WHERE claim_id = $1
        ORDER BY condition, ensemble_mode;
        """,
        claim_id,
    )
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

#: The evidence tables: one item per proposition, one row per source that reports
#: it. Everything Stage 2 decides belongs to the source, the proposition and its
#: role to the item, so the browser lists sources and names their item alongside.
EVIDENCE_FROM = "evidence_sources s JOIN evidence e ON e.id = s.evidence_id"

#: Aliased so that a row reads like one flat record: `source_*` for the source,
#: the item's own fields under their own names.
SOURCE_COLUMNS = """
    s.id, s.evidence_id, s.claim_id,
    s.name AS source_name, s.kind AS source_kind, s.locator AS source_locator,
    s.proximity AS source_proximity, s.raw_content AS source_raw_content,
    s.available_since, s.accessed_at, s.accessible,
    s.faithfulness_assessment, s.faithfulness_reasoning, s.faithfulness_justification,
    s.before_fact_check, s.before_claim,
    s.admissible, s.inadmissibility_reason, s.deferred_until, s.dismissed_reason,
    s.full_source, s.created_at, s.updated_at,
    e.review_id, e.article_id, e.proposition, e.role, e.extraction_reasoning,
    e.extraction_confidence, e.admissible AS evidence_admissible, e.n_sources,
    e.later_event, e.later_event_reasoning, e.later_event_justification,
    e.later_event_rater, e.dismissed, e.full_evidence
"""

#: The evidence items of one claim, without the sources (fetched separately).
EVIDENCE_ITEM_COLUMNS = """
    e.id, e.claim_id, e.review_id, e.article_id, e.proposition, e.role,
    e.extraction_reasoning, e.extraction_confidence, e.admissible,
    e.inadmissibility_reason, e.n_sources, e.n_admissible_sources,
    e.later_event, e.later_event_reasoning, e.later_event_justification,
    e.later_event_rater, e.available_since, e.dismissed, e.dismissed_reason,
    e.full_evidence, e.created_at, e.updated_at
"""

# --- the light column sets the claim view uses -------------------------------
#
# A claim can hold dozens of items with several sources each, and every source
# carries two scraped documents (`raw_content`), two reasoning traces and a JSONB
# blob - with the item's own blob and extraction reasoning repeated on every one
# of its source rows. That is megabytes per claim, all of it behind a collapsed
# disclosure. The claim view therefore asks only for what a collapsed card shows;
# `get_evidence_item` fetches the rest when one is expanded.

EVIDENCE_ITEM_SUMMARY_COLUMNS = """
    e.id, e.claim_id, e.review_id, e.article_id, e.proposition, e.role,
    e.extraction_confidence, e.admissible, e.inadmissibility_reason,
    e.n_sources, e.n_admissible_sources, e.available_since,
    e.later_event, e.later_event_justification,
    e.dismissed, e.dismissed_reason, e.created_at, e.updated_at
"""

SOURCE_SUMMARY_COLUMNS = """
    s.id, s.evidence_id, s.claim_id,
    s.name AS source_name, s.kind AS source_kind, s.locator AS source_locator,
    s.proximity AS source_proximity,
    s.available_since, s.accessed_at, s.accessible,
    s.faithfulness_assessment, s.faithfulness_justification,
    s.before_fact_check, s.before_claim,
    s.admissible, s.inadmissibility_reason, s.deferred_until, s.dismissed_reason,
    s.created_at, s.updated_at
"""


async def get_evidence_for_claim(claim_id: int) -> list[dict]:
    """The claim's evidence items with their sources, in the light column set."""
    items, sources = await asyncio.gather(
        database.fetch(
            f"SELECT {EVIDENCE_ITEM_SUMMARY_COLUMNS} FROM evidence e "
            f"WHERE e.claim_id = $1 ORDER BY e.id;",
            claim_id),
        database.fetch(
            f"SELECT {SOURCE_SUMMARY_COLUMNS} FROM evidence_sources s "
            f"WHERE s.claim_id = $1 ORDER BY s.evidence_id, s.id;",
            claim_id),
    )
    return _group_sources(items, sources)


async def get_evidence_item(evidence_id: int) -> dict | None:
    """One evidence item with its sources, every stored field included."""
    item, sources = await asyncio.gather(
        database.fetchrow(
            f"SELECT {EVIDENCE_ITEM_COLUMNS} FROM evidence e WHERE e.id = $1;", evidence_id),
        database.fetch(
            f"SELECT {SOURCE_COLUMNS} FROM {EVIDENCE_FROM} "
            f"WHERE s.evidence_id = $1 ORDER BY s.id;",
            evidence_id),
    )
    if item is None:
        return None
    return _group_sources([item], sources)[0]


def _group_sources(items, sources) -> list[dict]:
    """Attaches each source row to the item it reports."""
    by_evidence: dict[int, list[dict]] = {}
    for source in sources:
        by_evidence.setdefault(source["evidence_id"], []).append(dict(source))
    return [{**dict(item), "sources": by_evidence.get(item["id"], [])} for item in items]


async def get_evidence_source(source_id: int) -> dict | None:
    """One source, with the fields of the item it belongs to."""
    row = await database.fetchrow(
        f"SELECT {SOURCE_COLUMNS} FROM {EVIDENCE_FROM} WHERE s.id = $1;", source_id)
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Evidence browser
# ---------------------------------------------------------------------------

#: The list view never needs the long texts (reasoning traces, scraped content,
#: the JSONB blobs), so a page of sources stays small.
EVIDENCE_SUMMARY_COLUMNS = """
    s.id, s.evidence_id, s.claim_id, s.name AS source_name, s.kind AS source_kind,
    s.locator AS source_locator, s.proximity AS source_proximity,
    s.available_since, s.accessed_at, s.accessible, s.faithfulness_assessment,
    s.faithfulness_justification, s.before_fact_check, s.before_claim,
    s.admissible, s.inadmissibility_reason,
    s.deferred_until, s.dismissed_reason, s.created_at, s.updated_at,
    e.review_id, e.article_id, e.proposition, e.role, e.extraction_confidence,
    e.admissible AS evidence_admissible, e.n_sources, e.later_event,
    e.later_event_justification, e.dismissed
"""


async def list_evidence(
        *,
        claim_id: int | None = None,
        admissibility: list[str] | None = None,
        kinds: list[str] | None = None,
        proximities: list[str] | None = None,
        roles: list[str] | None = None,
        zones: list[str] | None = None,
        languages: list[str] | None = None,
        reason: str | None = None,
        domain: str | None = None,
        query: str | None = None,
        media: str | None = None,
        accessible: bool | None = None,
        later_event: bool | None = None,
        dismissed: bool | None = None,
        min_confidence: float | None = None,
        min_faithfulness: float | None = None,
        max_faithfulness: float | None = None,
        date_from: date | datetime | None = None,
        date_to: date | datetime | None = None,
        sort: str = "id",
        descending: bool = True,
        limit: int = 25,
        offset: int = 0,
) -> dict:
    """One page of evidence sources across all claims, with their claim context."""
    where, args = build_evidence_filters(
        claim_id=claim_id,
        admissibility=admissibility,
        kinds=kinds,
        proximities=proximities,
        roles=roles,
        zones=zones,
        languages=languages,
        reason=reason,
        domain=domain,
        query=query,
        media=media,
        accessible=accessible,
        later_event=later_event,
        dismissed=dismissed,
        min_confidence=min_confidence,
        min_faithfulness=min_faithfulness,
        max_faithfulness=max_faithfulness,
        date_from=date_from,
        date_to=date_to,
    )
    order_column = EVIDENCE_SORTABLE.get(sort, EVIDENCE_SORTABLE["id"])
    direction = "DESC" if descending else "ASC"

    page_sql = f"""
        SELECT {EVIDENCE_SUMMARY_COLUMNS},
               {domain_expr("s")}     AS source_domain,
               c.data                 AS claim_data,
               c.date                 AS claim_date,
               c.language             AS claim_language,
               c.gold_evidence_status AS claim_status
        FROM {EVIDENCE_FROM}
        JOIN claims c ON c.id = s.claim_id
        WHERE {where}
        ORDER BY {order_column} {direction} NULLS LAST, s.id {direction}
        LIMIT ${len(args) + 1} OFFSET ${len(args) + 2};
    """
    count_sql = (f"SELECT COUNT(*) FROM {EVIDENCE_FROM} "
                 f"JOIN claims c ON c.id = s.claim_id WHERE {where};")

    rows, total = await asyncio.gather(
        database.fetch(page_sql, *args, limit, offset),
        database.fetchval(count_sql, *args),
    )
    return {
        "total": int(total or 0),
        "limit": limit,
        "offset": offset,
        "evidence": [dict(row) for row in rows],
    }


def build_evidence_filters(
        *,
        claim_id: int | None = None,
        admissibility: list[str] | None = None,
        kinds: list[str] | None = None,
        proximities: list[str] | None = None,
        roles: list[str] | None = None,
        zones: list[str] | None = None,
        languages: list[str] | None = None,
        reason: str | None = None,
        domain: str | None = None,
        query: str | None = None,
        media: str | None = None,
        accessible: bool | None = None,
        later_event: bool | None = None,
        dismissed: bool | None = None,
        min_confidence: float | None = None,
        min_faithfulness: float | None = None,
        max_faithfulness: float | None = None,
        date_from: date | datetime | None = None,
        date_to: date | datetime | None = None,
) -> tuple[str, list]:
    """WHERE clause shared by the evidence page and its count query.

    As in `build_claim_filters`, every user-supplied value becomes a positional
    parameter; only whitelisted fragments are interpolated into the SQL text.
    The clause may reference `c.` - both queries join `claims` - as well as `s.`
    for the source and `e.` for the evidence item it belongs to."""
    clauses: list[str] = []
    args: list = []

    def add(template: str, value) -> None:
        args.append(value)
        clauses.append(template.format(n=len(args)))

    def any_of(vocabulary: dict[str, str], selected: list[str] | None) -> None:
        """OR over the whitelisted fragments the user selected."""
        parts = [vocabulary[name] for name in selected or [] if name in vocabulary]
        if parts:
            clauses.append("(" + " OR ".join(parts) + ")")

    if claim_id is not None:
        add("s.claim_id = ${n}", claim_id)
    any_of(ADMISSIBILITY_SQL, admissibility)
    any_of(ZONE_SQL, zones)
    if kinds:
        add("s.kind = ANY (${n}::text[])", kinds)
    if proximities:
        add("s.proximity = ANY (${n}::text[])", proximities)
    if roles:
        add("e.role = ANY (${n}::text[])", roles)
    if languages:
        add("c.language = ANY (${n}::text[])", languages)
    if reason:
        add("s.inadmissibility_reason = ${n}", reason)
    if domain:
        args.append(domain)
        clauses.append(f"{domain_expr('s')} = ${len(args)}")
    if query:
        add("e.proposition ILIKE '%' || ${n} || '%'", query)
    if media:
        clauses.append(media_clause("e.proposition", media))
    if accessible is not None:
        clauses.append("s.accessible" if accessible else "s.accessible IS FALSE")
    if later_event is not None:
        # §3.3 (3) is a property of the proposition, not of one place it is reported.
        clauses.append("e.later_event" if later_event else "e.later_event IS NOT TRUE")
    if dismissed is not None:
        clauses.append("e.dismissed" if dismissed else "e.dismissed IS NOT TRUE")
    if min_confidence is not None:
        add("e.extraction_confidence >= ${n}", min_confidence)
    if min_faithfulness is not None:
        add("s.faithfulness_assessment >= ${n}", min_faithfulness)
    if max_faithfulness is not None:
        add("s.faithfulness_assessment <= ${n}", max_faithfulness)
    if date_from is not None:
        add("s.available_since >= ${n}", as_datetime(date_from, end_of_day=False))
    if date_to is not None:
        add("s.available_since <= ${n}", as_datetime(date_to, end_of_day=True))

    return " AND ".join(clauses) if clauses else "TRUE", args


#: Facets offered by the evidence browser. Counted over *all* sources - unlike the
#: dashboard, which describes the admissible ones - because the browser must be
#: able to reach every stored source.
EVIDENCE_FACET_SQL = {
    "source_kinds": "SELECT kind AS label, COUNT(*) AS count FROM evidence_sources GROUP BY 1;",
    "proximities": "SELECT proximity AS label, COUNT(*) AS count FROM evidence_sources GROUP BY 1;",
    "roles":
        "SELECT e.role AS label, COUNT(*) AS count "
        f"FROM {EVIDENCE_FROM} GROUP BY 1;",
    "inadmissibility_reasons":
        "SELECT inadmissibility_reason AS label, COUNT(*) AS count "
        "FROM evidence_sources WHERE admissible IS FALSE GROUP BY 1;",
    "languages":
        "SELECT c.language AS label, COUNT(*) AS count "
        "FROM evidence_sources s JOIN claims c ON c.id = s.claim_id GROUP BY 1;",
}

EVIDENCE_ADMISSIBILITY_SQL = f"""
    SELECT COUNT(*) FILTER (WHERE {ADMISSIBILITY_SQL["admissible"]})   AS admissible,
           COUNT(*) FILTER (WHERE {ADMISSIBILITY_SQL["inadmissible"]}) AS inadmissible,
           COUNT(*) FILTER (WHERE {ADMISSIBILITY_SQL["unfiltered"]})   AS unfiltered
    FROM evidence_sources s;
"""

EVIDENCE_ZONE_SQL = f"""
    SELECT COUNT(*) FILTER (WHERE {ZONE_SQL["before_claim"]})     AS before_claim,
           COUNT(*) FILTER (WHERE {ZONE_SQL["in_window"]})        AS in_window,
           COUNT(*) FILTER (WHERE {ZONE_SQL["after_fact_check"]}) AS after_fact_check,
           COUNT(*) FILTER (WHERE {ZONE_SQL["unvalidated"]})      AS unvalidated
    FROM evidence_sources s;
"""

EVIDENCE_DOMAINS_SQL = f"""
    SELECT {domain_expr("s")} AS label, COUNT(*) AS count
    FROM evidence_sources s
    WHERE s.locator IS NOT NULL AND s.locator <> ''
    GROUP BY 1 ORDER BY 2 DESC LIMIT 40;
"""


async def get_evidence_filter_options() -> dict:
    """Distinct values the evidence browser offers as filters, with their counts."""
    names = list(EVIDENCE_FACET_SQL)
    *facets, domains, admissibility, zones = await asyncio.gather(
        *(database.fetch(EVIDENCE_FACET_SQL[name]) for name in names),
        database.fetch(EVIDENCE_DOMAINS_SQL),
        database.fetchrow(EVIDENCE_ADMISSIBILITY_SQL),
        database.fetchrow(EVIDENCE_ZONE_SQL),
    )
    rows = dict(zip(names, facets))

    return {
        "admissibility": counted(admissibility, ADMISSIBILITY_SQL),
        "zones": counted(zones, ZONE_SQL),
        "source_kinds": series(rows["source_kinds"]),
        "proximities": series(rows["proximities"], order=("primary", "secondary", "tertiary")),
        "roles": series(rows["roles"], order=("essential", "auxiliary", "background")),
        "inadmissibility_reasons": series(rows["inadmissibility_reasons"], order=REASON_ORDER),
        "languages": series(rows["languages"]),
        "domains": series(domains),
        "sorts": list(EVIDENCE_SORTABLE),
    }


def counted(row, vocabulary: dict[str, str]) -> list[dict]:
    """A one-row `COUNT(*) FILTER` result as a labelled series, in vocabulary order."""
    values = dict(row) if row else {}
    counts = {name: int(values.get(name) or 0) for name in vocabulary}
    total = sum(counts.values())
    return [
        {"label": name, "count": count, "share": share(count, total)}
        for name, count in counts.items()
    ]


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

CLAIM_STATUS_SQL = """
    SELECT gold_evidence_status AS label, COUNT(*) AS count
    FROM claims WHERE gold_evidence_status IS NOT NULL GROUP BY 1;
"""

CLAIM_REASON_SQL = """
    SELECT gold_evidence_reason AS label, COUNT(*) AS count
    FROM claims WHERE gold_evidence_reason IS NOT NULL GROUP BY 1;
"""

CLAIM_LANGUAGE_SQL = """
    SELECT language AS label, COUNT(*) AS count
    FROM claims WHERE gold_evidence_status IS NOT NULL GROUP BY 1;
"""

#: The dashboard's headline counts. Items and sources are counted separately:
#: an item is a proposition, a source is a place it can be read, and a lost source
#: costs the reconstruction nothing as long as its item kept another one.
EVIDENCE_TOTALS_SQL = f"""
    WITH items AS (
        SELECT COUNT(*)                                                AS n_evidence,
               COUNT(DISTINCT claim_id)                                AS n_claims,
               COUNT(*) FILTER (WHERE admissible)                      AS n_admissible,
               COUNT(*) FILTER (WHERE admissible IS FALSE)             AS n_inadmissible,
               COUNT(*) FILTER (WHERE admissible IS NULL)              AS n_unfiltered,
               COUNT(*) FILTER (WHERE proposition ~ '{MEDIA_REF_SQL}') AS n_multimodal,
               COUNT(*) FILTER (WHERE proposition ~ '{IMAGE_REF_SQL}') AS n_with_images,
               COUNT(*) FILTER (WHERE proposition ~ '{VIDEO_REF_SQL}') AS n_with_videos
        FROM evidence
    ),
         sources AS (
             SELECT COUNT(*)                                            AS n_sources,
                    COUNT(*) FILTER (WHERE admissible)                  AS n_sources_admissible,
                    COUNT(*) FILTER (WHERE admissible IS FALSE)         AS n_sources_rejected,
                    COUNT(*) FILTER (WHERE available_since IS NULL)     AS n_undated,
                    COUNT(*) FILTER (WHERE accessible IS FALSE)         AS n_inaccessible,
                    COUNT(*) FILTER (WHERE admissible AND before_claim) AS n_before_claim,
                    COUNT(*) FILTER (WHERE admissible AND before_fact_check
                                       AND before_claim IS FALSE)       AS n_in_window,
                    COUNT(*) FILTER (WHERE raw_content IS NOT NULL)     AS n_with_content,
                    COUNT(DISTINCT locator)                             AS n_distinct_locators
             FROM evidence_sources
         )
    SELECT * FROM items, sources;
"""

TOP_DOMAINS_SQL = f"""
    SELECT {domain_expr("s")} AS label, COUNT(*) AS count
    FROM evidence_sources s
    WHERE s.locator IS NOT NULL AND s.locator <> ''
    GROUP BY 1 ORDER BY 2 DESC LIMIT 15;
"""

DELTAS_SQL = """
    SELECT EXTRACT(EPOCH FROM (s.available_since - c.date)) / 86400.0   AS to_claim,
           EXTRACT(EPOCH FROM (s.available_since - ref.t_f)) / 86400.0  AS to_fact_check
    FROM evidence_sources s
    JOIN claims c ON c.id = s.claim_id
    LEFT JOIN LATERAL (
        SELECT MAX(r.published) AS t_f FROM reviews r
        WHERE r.id = ANY (c.review_ids) AND r.dismissed = FALSE
    ) ref ON TRUE
    WHERE s.available_since IS NOT NULL;
"""

DURATIONS_SQL = """
    SELECT EXTRACT(EPOCH FROM (ref.t_f - c.date)) / 86400.0 AS days
    FROM claims c
    LEFT JOIN LATERAL (
        SELECT MAX(r.published) AS t_f FROM reviews r
        WHERE r.id = ANY (c.review_ids) AND r.dismissed = FALSE
    ) ref ON TRUE
    WHERE c.gold_evidence_status IS NOT NULL
      AND c.date IS NOT NULL AND ref.t_f IS NOT NULL;
"""

CONTINGENCY_SQL = """
    SELECT gc.is_close AS claim_close, gf.is_close AS fact_check_close, COUNT(*) AS count
    FROM gold_evidence_results gc
    JOIN gold_evidence_results gf
         ON gf.claim_id = gc.claim_id
        AND gf.ensemble_mode = gc.ensemble_mode
        AND gf.condition = 'fact_check'
    WHERE gc.condition = 'claim'
    GROUP BY 1, 2;
"""

PROGRESS_SQL = """
    SELECT date_trunc('day', gold_evidence_updated_at)::date AS day,
           gold_evidence_status AS status,
           COUNT(*) AS count
    FROM claims
    WHERE gold_evidence_updated_at IS NOT NULL
    GROUP BY 1, 2 ORDER BY 1;
"""

RESULTS_META_SQL = """
    SELECT ensemble_mode, condition, COUNT(*) AS count,
           COUNT(*) FILTER (WHERE is_close) AS n_close,
           COUNT(*) FILTER (WHERE error IS NOT NULL) AS n_error,
           AVG(max_property_diff) AS mean_diff,
           MIN(threshold) AS threshold
    FROM gold_evidence_results GROUP BY 1, 2 ORDER BY 1, 2;
"""


def _admissible_distribution(column: str, table: str = "evidence_sources") -> str:
    """Distribution of a column over the admissible rows of one evidence table."""
    return (f"SELECT {column} AS label, COUNT(*) AS count "
            f"FROM {table} WHERE admissible GROUP BY 1;")


async def get_overview() -> dict:
    """Every number the statistics page shows, in one round of queries."""
    (claim_statuses, claim_reasons, evidence_totals, source_kinds, proximities,
     roles, inadmissibility, domains, faithfulness, confidence, deltas,
     durations, per_claim, contingency, progress, results_meta) = await asyncio.gather(
        database.fetch(CLAIM_STATUS_SQL),
        database.fetch(CLAIM_REASON_SQL),
        database.fetchrow(EVIDENCE_TOTALS_SQL),
        database.fetch(_admissible_distribution("kind")),
        database.fetch(_admissible_distribution("proximity")),
        database.fetch(_admissible_distribution("role", table="evidence")),
        database.fetch(
            "SELECT inadmissibility_reason AS label, COUNT(*) AS count "
            "FROM evidence_sources WHERE admissible IS FALSE GROUP BY 1;"),
        database.fetch(TOP_DOMAINS_SQL),
        database.fetch(
            "SELECT round(faithfulness_assessment::numeric, 3) AS value, COUNT(*) AS count "
            "FROM evidence_sources WHERE faithfulness_assessment IS NOT NULL GROUP BY 1 ORDER BY 1;"),
        database.fetch(
            "SELECT round(extraction_confidence::numeric, 2) AS value, COUNT(*) AS count "
            "FROM evidence WHERE extraction_confidence IS NOT NULL GROUP BY 1 ORDER BY 1;"),
        database.fetch(DELTAS_SQL),
        database.fetch(DURATIONS_SQL),
        database.fetch(
            "SELECT claim_id, COUNT(*) AS n, COUNT(*) FILTER (WHERE admissible) AS n_admissible "
            "FROM evidence GROUP BY claim_id;"),
        database.fetch(CONTINGENCY_SQL),
        database.fetch(PROGRESS_SQL),
        database.fetch(RESULTS_META_SQL),
    )

    totals = dict(evidence_totals) if evidence_totals else {}
    n_evidence = int(totals.get("n_evidence") or 0)
    n_admissible = int(totals.get("n_admissible") or 0)
    n_multimodal = int(totals.get("n_multimodal") or 0)
    n_in_window = int(totals.get("n_in_window") or 0)
    n_claims_processed = sum(int(row["count"]) for row in claim_statuses)

    statuses = series(claim_statuses, order=STATUS_ORDER)
    accepted = next((entry["count"] for entry in statuses if entry["label"] == "accepted"), 0)

    return {
        "claims": {
            "n_processed": n_claims_processed,
            "n_with_evidence": int(totals.get("n_claims") or 0),
            "n_accepted": accepted,
            "acceptance_rate": share(accepted, n_claims_processed),
            "statuses": statuses,
            "rejection_reasons": series(claim_reasons, order=INSTANCE_REASON_ORDER),
            "evidence_per_claim": describe([int(row["n"]) for row in per_claim]),
            "admissible_per_claim": describe([int(row["n_admissible"]) for row in per_claim]),
            "fact_check_duration_days": describe([row["days"] for row in durations]),
            "progress": [
                {"day": row["day"], "status": row["status"], "count": int(row["count"])}
                for row in progress
            ],
        },
        "evidence": {
            "n_total": n_evidence,
            "n_admissible": n_admissible,
            "n_inadmissible": int(totals.get("n_inadmissible") or 0),
            "n_unfiltered": int(totals.get("n_unfiltered") or 0),
            "n_multimodal": n_multimodal,
            "n_with_images": int(totals.get("n_with_images") or 0),
            "n_with_videos": int(totals.get("n_with_videos") or 0),
            "n_undated": int(totals.get("n_undated") or 0),
            "n_inaccessible": int(totals.get("n_inaccessible") or 0),
            "n_with_content": int(totals.get("n_with_content") or 0),
            "n_sources": int(totals.get("n_sources") or 0),
            "n_sources_admissible": int(totals.get("n_sources_admissible") or 0),
            "n_sources_rejected": int(totals.get("n_sources_rejected") or 0),
            "n_distinct_sources": int(totals.get("n_distinct_locators") or 0),
            "n_before_claim": int(totals.get("n_before_claim") or 0),
            "n_in_window": n_in_window,
            "admissible_rate": share(n_admissible, n_evidence),
            "multimodal_rate": share(n_multimodal, n_evidence),
            "in_window_rate": share(n_in_window, n_admissible),
            "source_kinds": series(source_kinds),
            "proximities": series(proximities, order=("primary", "secondary", "tertiary")),
            "roles": series(roles, order=("essential", "auxiliary", "background")),
            "inadmissibility_reasons": series(inadmissibility, order=REASON_ORDER),
            "top_domains": series(domains),
            "faithfulness": weighted_series(faithfulness),
            "extraction_confidence": weighted_series(confidence),
            "delta_to_claim": delta_summary([row["to_claim"] for row in deltas]),
            "delta_to_fact_check": delta_summary([row["to_fact_check"] for row in deltas]),
            "funnel": evidence_funnel(
                n_candidates=n_evidence,
                n_unfiltered=int(totals.get("n_unfiltered") or 0),
                reasons=[{"label": row["label"], "count": row["count"]}
                         for row in inadmissibility],
                n_admissible=n_admissible,
                n_before_claim=int(totals.get("n_before_claim") or 0),
            ),
        },
        "sufficiency": {
            "modes": [
                {
                    "ensemble_mode": row["ensemble_mode"],
                    "condition": row["condition"],
                    "count": int(row["count"]),
                    "n_close": int(row["n_close"] or 0),
                    "n_error": int(row["n_error"] or 0),
                    "close_rate": share(int(row["n_close"] or 0), int(row["count"])),
                    "mean_max_property_diff": as_float(row["mean_diff"]),
                    "threshold": as_float(row["threshold"]),
                }
                for row in results_meta
            ],
            "recoverability": recoverability(
                [{"claim_close": row["claim_close"],
                  "fact_check_close": row["fact_check_close"],
                  "count": row["count"]} for row in contingency]
            ),
        },
    }


def weighted_series(rows) -> list[dict]:
    """Rows of `(value, count)` into a series sorted by value, with shares."""
    entries = [{"value": as_float(row["value"]), "count": int(row["count"])} for row in rows]
    entries.sort(key=lambda entry: (entry["value"] is None, entry["value"]))
    total = sum(entry["count"] for entry in entries)
    for entry in entries:
        entry["share"] = share(entry["count"], total)
    return entries


def delta_summary(values) -> dict:
    """Summary statistics plus a symmetric-log histogram of day differences."""
    floats = [as_float(value) for value in values]
    return {"summary": describe(floats),
            "histogram": signed_log_histogram(floats, bins_per_side=24)}


def as_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def get_filter_options() -> dict:
    """Distinct values the browser offers as filters, with their counts."""
    statuses, reasons, languages = await asyncio.gather(
        database.fetch(CLAIM_STATUS_SQL),
        database.fetch(CLAIM_REASON_SQL),
        database.fetch(CLAIM_LANGUAGE_SQL),
    )
    return {
        "statuses": series(statuses, order=STATUS_ORDER),
        "rejection_reasons": series(reasons, order=INSTANCE_REASON_ORDER),
        "languages": series(languages),
        "sorts": list(SORTABLE),
    }
