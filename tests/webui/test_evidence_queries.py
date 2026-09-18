"""The evidence browser's filter builder: user input must always become a bound
parameter, and only whitelisted fragments may reach the SQL text."""

from datetime import date

import inspect

import pytest

from webui import queries
from webui.queries import (
    ADMISSIBILITY_SQL,
    EVIDENCE_SORTABLE,
    MEDIA_REF_SQL,
    ZONE_SQL,
    build_evidence_filters,
    counted,
    domain_expr,
)


def test_no_filters_matches_everything():
    where, args = build_evidence_filters()
    assert where == "TRUE"
    assert args == []


def test_enum_filters_are_parameterized():
    where, args = build_evidence_filters(
        kinds=["news_article", "fact_check"],
        proximities=["primary"],
        roles=["essential"],
    )
    assert "s.kind = ANY ($1::text[])" in where
    assert "s.proximity = ANY ($2::text[])" in where
    assert "e.role = ANY ($3::text[])" in where
    assert args == [["news_article", "fact_check"], ["primary"], ["essential"]]


def test_language_filters_on_the_joined_claim():
    where, args = build_evidence_filters(languages=["en", "de"])
    assert where == "c.language = ANY ($1::text[])"
    assert args == [["en", "de"]]


def test_search_term_never_reaches_the_sql_text():
    where, args = build_evidence_filters(query="'; DROP TABLE evidence; --")
    assert "DROP TABLE" not in where
    assert "e.proposition ILIKE '%' || $1 || '%'" in where
    assert args == ["'; DROP TABLE evidence; --"]


def test_domain_is_compared_against_the_same_expression_that_counts_facets():
    # The locator lives on the source, so both the filter and the facet count
    # must derive the domain from `evidence_sources`.
    where, args = build_evidence_filters(domain="bbc.co.uk")
    assert f"{domain_expr('s')} = $1" in where
    assert args == ["bbc.co.uk"]


@pytest.mark.parametrize("vocabulary,keyword", [
    (ADMISSIBILITY_SQL, "admissibility"),
    (ZONE_SQL, "zones"),
])
def test_vocabulary_filters_only_emit_known_fragments(vocabulary, keyword):
    # An unknown name is dropped rather than interpolated.
    where, args = build_evidence_filters(**{keyword: ["'; DROP TABLE evidence; --"]})
    assert where == "TRUE"
    assert args == []

    selected = list(vocabulary)[:2]
    where, args = build_evidence_filters(**{keyword: selected})
    assert where == "(" + " OR ".join(vocabulary[name] for name in selected) + ")"
    assert args == []


def test_several_vocabulary_values_are_ored_but_groups_are_anded():
    where, _ = build_evidence_filters(
        admissibility=["admissible"], zones=["in_window", "before_claim"])
    assert where == ("(s.admissible) AND "
                     "((s.before_fact_check AND s.before_claim IS NOT TRUE) OR s.before_claim)")


def test_boolean_filters_use_fixed_fragments():
    where, args = build_evidence_filters(
        media="any", accessible=False, later_event=False, dismissed=False)
    assert f"e.proposition ~ '{MEDIA_REF_SQL}'" in where
    assert "s.accessible IS FALSE" in where
    # `later_event`/`dismissed` are nullable, so "no" must also match NULL. The
    # later-event judgement belongs to the item, hence the `e.` alias.
    assert "e.later_event IS NOT TRUE" in where
    assert "e.dismissed IS NOT TRUE" in where
    assert args == []


def test_the_text_only_filter_uses_the_negated_operator():
    where, _ = build_evidence_filters(media="none")
    assert f"e.proposition !~ '{MEDIA_REF_SQL}'" in where


def test_evidence_can_be_filtered_by_media_kind():
    images, _ = build_evidence_filters(media="image")
    videos, _ = build_evidence_filters(media="video")
    assert "<image:[0-9]+>" in images and "<video:" not in images
    assert "<video:[0-9]+>" in videos and "<image:" not in videos


def test_dates_are_widened_to_cover_the_whole_day():
    _, args = build_evidence_filters(
        date_from=date(2024, 5, 1), date_to=date(2024, 5, 31))
    assert args[0].hour == 0
    assert args[1].hour == 23


def test_placeholders_are_numbered_consecutively():
    where, args = build_evidence_filters(
        claim_id=12,
        kinds=["news_article"],
        proximities=["primary"],
        roles=["essential"],
        languages=["en"],
        reason="unfaithful",
        domain="bbc.co.uk",
        query="flood",
        min_confidence=0.5,
        min_faithfulness=-0.2,
        max_faithfulness=0.9,
        date_from=date(2024, 1, 1),
        date_to=date(2024, 12, 31),
    )
    assert len(args) == 13
    for index in range(1, len(args) + 1):
        assert f"${index}" in where


def test_sortable_columns_are_whitelisted():
    for column in EVIDENCE_SORTABLE.values():
        # Source columns are addressed through `s`, the item's own through `e`.
        assert column.startswith(("s.", "e."))


def test_counted_turns_one_filter_row_into_a_series():
    row = {"admissible": 6, "inadmissible": 2, "unfiltered": 0}
    result = counted(row, ADMISSIBILITY_SQL)
    assert [entry["label"] for entry in result] == list(ADMISSIBILITY_SQL)
    assert [entry["count"] for entry in result] == [6, 2, 0]
    assert result[0]["share"] == pytest.approx(0.75)


def test_counted_tolerates_a_missing_row():
    result = counted(None, ZONE_SQL)
    assert [entry["count"] for entry in result] == [0, 0, 0, 0]
    assert all(entry["share"] is None for entry in result)


@pytest.mark.parametrize("alias", ["s", "src"])
def test_domain_expr_is_built_around_the_given_alias(alias):
    expression = domain_expr(alias)
    assert expression.count(f"{alias}.locator") == 2
    assert expression.startswith("lower(regexp_replace(")


# ------------------------------------------- the claim view's light column set

HEAVY_COLUMNS = ("raw_content", "full_source", "full_evidence",
                 "faithfulness_reasoning", "later_event_reasoning", "extraction_reasoning")


def test_the_claim_views_column_sets_omit_every_heavy_field():
    """Opening a claim must not transfer the scraped documents and JSONB blobs of
    every source; they sit behind a collapsed card and are fetched on demand."""
    light = queries.EVIDENCE_ITEM_SUMMARY_COLUMNS + queries.SOURCE_SUMMARY_COLUMNS
    for column in HEAVY_COLUMNS:
        assert column not in light, f"{column} would be shipped with every claim"


def test_the_full_column_sets_still_carry_the_heavy_fields():
    full = queries.EVIDENCE_ITEM_COLUMNS + queries.SOURCE_COLUMNS
    for column in HEAVY_COLUMNS:
        assert column in full, f"{column} is unreachable for the detail view"


def _columns(block: str) -> set[str]:
    """The column expressions of one of the `*_COLUMNS` string constants."""
    return {part.strip() for part in block.split(",") if part.strip()}


@pytest.mark.parametrize("light,full", [
    ("EVIDENCE_ITEM_SUMMARY_COLUMNS", "EVIDENCE_ITEM_COLUMNS"),
    ("SOURCE_SUMMARY_COLUMNS", "SOURCE_COLUMNS"),
])
def test_every_light_column_exists_in_the_full_set(light, full):
    """The light sets are hand-maintained subsets, so a column renamed in the
    schema can leave them asking PostgreSQL for something that is gone - and the
    claim view is the one place that would then fail."""
    missing = _columns(getattr(queries, light)) - _columns(getattr(queries, full))
    assert not missing, f"{light} names columns {full} does not: {sorted(missing)}"


def test_the_light_source_columns_keep_what_a_collapsed_card_shows():
    for column in ("s.name", "s.kind", "s.locator", "s.proximity", "s.admissible",
                   "s.available_since", "s.faithfulness_assessment", "s.before_claim",
                   "s.before_fact_check", "s.deferred_until"):
        assert column in queries.SOURCE_SUMMARY_COLUMNS
    # §3.3 (3) is judged for the item, so it is not a source column.
    assert "s.later_event" not in queries.SOURCE_SUMMARY_COLUMNS
    assert "e.later_event" in queries.EVIDENCE_ITEM_SUMMARY_COLUMNS


def test_the_light_item_columns_keep_the_proposition_and_its_verdict():
    for column in ("e.proposition", "e.role", "e.admissible", "e.inadmissibility_reason",
                   "e.n_sources", "e.n_admissible_sources", "e.available_since",
                   "e.later_event"):
        assert column in queries.EVIDENCE_ITEM_SUMMARY_COLUMNS


def test_neighbours_walk_the_primary_key_instead_of_aggregating():
    """`MIN`/`MAX` over `gold_evidence_status IS NOT NULL` scans every processed
    claim; an ordered walk stops at the first neighbour."""
    source = inspect.getsource(queries.get_neighbours)
    assert "ORDER BY id DESC LIMIT 1" in source
    assert "ORDER BY id ASC LIMIT 1" in source
    assert "MAX(id)" not in source and "MIN(id)" not in source
