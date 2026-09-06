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
    REASON_ORDER,
    STATUS_ORDER,
    describe,
    histogram,
    recoverability,
    series,
    share,
)

#: PostgreSQL regex matching an ezMM item reference. Mirrors `webui.parsing`.
MEDIA_REF_SQL = "<(image|video|audio):[0-9]+>"

#: Claim columns the browser may sort by. Whitelisted, never interpolated blindly.
SORTABLE = {
    "id": "c.id",
    "date": "c.date",
    "updated": "c.gold_evidence_updated_at",
    "status": "c.gold_evidence_status",
}

#: Reference times per claim, as defined in `gold_evidence.filtering`:
#: t_c is `claims.date`; t_f is the latest `published` among non-dismissed reviews.
REFERENCE_TIMES_SQL = """
    SELECT MAX(r.published) AS t_f
    FROM reviews r
    WHERE r.id = ANY (c.review_ids) AND r.dismissed = FALSE
"""

EVIDENCE_COUNTS_SQL = f"""
    SELECT COUNT(*)                                                   AS n_evidence,
           COUNT(*) FILTER (WHERE e.admissible)                       AS n_admissible,
           COUNT(*) FILTER (WHERE e.admissible IS FALSE)              AS n_inadmissible,
           COUNT(*) FILTER (WHERE e.admissible IS NULL)               AS n_unfiltered,
           COUNT(*) FILTER (WHERE e.proposition ~ '{MEDIA_REF_SQL}')  AS n_multimodal,
           COUNT(*) FILTER (WHERE e.admissible AND e.before_claim)    AS n_before_claim,
           COUNT(*) FILTER (WHERE e.admissible
                              AND e.before_fact_check
                              AND e.before_claim IS FALSE)            AS n_in_window
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
        query: str | None = None,
        released: bool | None = None,
        has_media: bool | None = None,
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
        query=query,
        released=released,
        has_media=has_media,
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
                   COUNT(*) FILTER (WHERE e.proposition ~ '{MEDIA_REF_SQL}')  AS n_multimodal
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
        query: str | None = None,
        released: bool | None = None,
        has_media: bool | None = None,
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
    if query:
        add("c.data ILIKE '%' || ${n} || '%'", query)
    if released is not None:
        clauses.append("(c.released_quarter OR c.released_longitudinal)"
                       if released else
                       "NOT (c.released_quarter OR c.released_longitudinal)")
    if has_media is not None:
        operator = "~" if has_media else "!~"
        clauses.append(f"c.data {operator} '{MEDIA_REF_SQL}'")
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

    evidence, reviews, verdict, results, neighbours = await asyncio.gather(
        get_evidence_for_claim(claim_id),
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
        "reviews": reviews,
        "verdict": verdict,
        "results": results,
        "neighbours": neighbours,
    }


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
    """Previous/next processed claim by ID, so the detail view can be paged through."""
    row = await database.fetchrow(
        """
        SELECT (SELECT MAX(id) FROM claims
                 WHERE id < $1 AND gold_evidence_status IS NOT NULL) AS previous_id,
               (SELECT MIN(id) FROM claims
                 WHERE id > $1 AND gold_evidence_status IS NOT NULL) AS next_id;
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

#: The flat columns are returned alongside `full_evidence` so the UI can show
#: what the database itself holds, next to the authoritative JSONB blob.
EVIDENCE_COLUMNS = """
    id, claim_id, review_id, article_id, proposition, source_name, source_kind,
    source_locator, source_proximity, source_raw_content, available_since, role,
    accessed_at, extraction_reasoning, extraction_confidence, accessible,
    faithfulness_assessment, faithfulness_reasoning, faithfulness_justification,
    before_fact_check, before_claim, later_event, temporal_reasoning,
    temporal_justification, admissible, inadmissibility_reason,
    dismissed, dismissed_reason, deferred_until, full_evidence,
    created_at, updated_at
"""


async def get_evidence_for_claim(claim_id: int) -> list[dict]:
    rows = await database.fetch(
        f"SELECT {EVIDENCE_COLUMNS} FROM evidence WHERE claim_id = $1 ORDER BY id;",
        claim_id,
    )
    return [dict(row) for row in rows]


async def get_evidence(evidence_id: int) -> dict | None:
    row = await database.fetchrow(
        f"SELECT {EVIDENCE_COLUMNS} FROM evidence WHERE id = $1;", evidence_id
    )
    return dict(row) if row else None


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

EVIDENCE_TOTALS_SQL = f"""
    SELECT COUNT(*)                                                  AS n_evidence,
           COUNT(DISTINCT claim_id)                                  AS n_claims,
           COUNT(*) FILTER (WHERE admissible)                        AS n_admissible,
           COUNT(*) FILTER (WHERE admissible IS FALSE)               AS n_inadmissible,
           COUNT(*) FILTER (WHERE admissible IS NULL)                AS n_unfiltered,
           COUNT(*) FILTER (WHERE proposition ~ '{MEDIA_REF_SQL}')   AS n_multimodal,
           COUNT(*) FILTER (WHERE available_since IS NULL)           AS n_undated,
           COUNT(*) FILTER (WHERE accessible IS FALSE)               AS n_inaccessible,
           COUNT(*) FILTER (WHERE admissible AND before_claim)       AS n_before_claim,
           COUNT(*) FILTER (WHERE admissible AND before_fact_check
                              AND before_claim IS FALSE)             AS n_in_window,
           COUNT(*) FILTER (WHERE source_raw_content IS NOT NULL)    AS n_with_content,
           COUNT(DISTINCT source_locator)                            AS n_sources
    FROM evidence;
"""

TOP_DOMAINS_SQL = r"""
    SELECT lower(regexp_replace(
               coalesce(substring(source_locator from '^[a-zA-Z][a-zA-Z0-9+.-]*://([^/?#]+)'),
                        source_locator),
               '^www\.', '')) AS label,
           COUNT(*) AS count
    FROM evidence
    WHERE source_locator IS NOT NULL AND source_locator <> ''
    GROUP BY 1 ORDER BY 2 DESC LIMIT 15;
"""

DELTAS_SQL = """
    SELECT EXTRACT(EPOCH FROM (e.available_since - c.date)) / 86400.0   AS to_claim,
           EXTRACT(EPOCH FROM (e.available_since - ref.t_f)) / 86400.0  AS to_fact_check
    FROM evidence e
    JOIN claims c ON c.id = e.claim_id
    LEFT JOIN LATERAL (
        SELECT MAX(r.published) AS t_f FROM reviews r
        WHERE r.id = ANY (c.review_ids) AND r.dismissed = FALSE
    ) ref ON TRUE
    WHERE e.available_since IS NOT NULL;
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


def _admissible_distribution(column: str) -> str:
    """Distribution of an evidence column over the admissible items only."""
    return (f"SELECT {column} AS label, COUNT(*) AS count "
            f"FROM evidence WHERE admissible GROUP BY 1;")


async def get_overview() -> dict:
    """Every number the statistics page shows, in one round of queries."""
    (claim_statuses, claim_reasons, evidence_totals, source_kinds, proximities,
     roles, inadmissibility, domains, faithfulness, confidence, deltas,
     durations, per_claim, contingency, progress, results_meta) = await asyncio.gather(
        database.fetch(CLAIM_STATUS_SQL),
        database.fetch(CLAIM_REASON_SQL),
        database.fetchrow(EVIDENCE_TOTALS_SQL),
        database.fetch(_admissible_distribution("source_kind")),
        database.fetch(_admissible_distribution("source_proximity")),
        database.fetch(_admissible_distribution("role")),
        database.fetch(
            "SELECT inadmissibility_reason AS label, COUNT(*) AS count "
            "FROM evidence WHERE admissible IS FALSE GROUP BY 1;"),
        database.fetch(TOP_DOMAINS_SQL),
        database.fetch(
            "SELECT round(faithfulness_assessment::numeric, 3) AS value, COUNT(*) AS count "
            "FROM evidence WHERE faithfulness_assessment IS NOT NULL GROUP BY 1 ORDER BY 1;"),
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
            "rejection_reasons": series(claim_reasons),
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
            "n_undated": int(totals.get("n_undated") or 0),
            "n_inaccessible": int(totals.get("n_inaccessible") or 0),
            "n_with_content": int(totals.get("n_with_content") or 0),
            "n_distinct_sources": int(totals.get("n_sources") or 0),
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
    """Summary statistics plus a histogram for a set of day differences."""
    floats = [as_float(value) for value in values]
    return {"summary": describe(floats), "histogram": histogram(floats, bins=32)}


def as_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def get_filter_options() -> dict:
    """Distinct values the browser offers as filters, with their counts."""
    statuses, reasons = await asyncio.gather(
        database.fetch(CLAIM_STATUS_SQL),
        database.fetch(CLAIM_REASON_SQL),
    )
    return {
        "statuses": series(statuses, order=STATUS_ORDER),
        "rejection_reasons": series(reasons),
        "sorts": list(SORTABLE),
    }
