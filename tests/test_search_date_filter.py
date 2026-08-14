"""Temporal-leakage guards for the baseline search date filters.

The benchmark is only valid if evidence retrieved for a claim predates the claim.
Upstream search APIs filter at day granularity with inclusive end dates, so these
tests pin the boundary: the claim's own day must never appear in a filter value.
"""

from datetime import date, datetime

import pytest

from eval.baselines.common.search import SearchService, strict_cutoff_date
from eval.baselines.providers.perplexity import PerplexityFactChecker


CLAIM_DAY = date(2024, 2, 5)


@pytest.mark.parametrize(
    "claim_date",
    [
        "2024-02-05T12:22:02",
        "2024-02-05T00:00:00",
        "2024-02-05T23:59:59",
        "2024-02-05T12:22:02Z",
        "2024-02-05",
        datetime(2024, 2, 5, 12, 22, 2),
        date(2024, 2, 5),
    ],
    ids=["midday", "midnight", "last-second", "zulu", "date-only", "datetime", "date"],
)
def test_cutoff_excludes_the_claim_day(claim_date):
    """Every accepted input form resolves to the day *before* the claim."""
    cutoff = strict_cutoff_date(claim_date)

    assert cutoff < CLAIM_DAY, "claim day must not be retrievable"
    assert cutoff == date(2024, 2, 4)


def test_cutoff_crosses_month_boundary():
    assert strict_cutoff_date("2024-03-01T08:00:00") == date(2024, 2, 29)


def test_cutoff_crosses_year_boundary():
    assert strict_cutoff_date("2020-01-01T00:00:00") == date(2019, 12, 31)


def test_serper_tbs_uses_the_strict_cutoff():
    service = SearchService.__new__(SearchService)  # no API key / network needed

    assert service._format_date_for_serper("2024-02-05T12:22:02") == "cdr:1,cd_max:2/4/2024"


def test_perplexity_filter_uses_the_strict_cutoff():
    checker = PerplexityFactChecker.__new__(PerplexityFactChecker)

    assert checker._format_date_filter("2024-02-05T12:22:02") == "2/4/2024"


def test_perplexity_filter_returns_none_on_unparseable_date():
    """A malformed date must not silently degrade into an unfiltered search."""
    checker = PerplexityFactChecker.__new__(PerplexityFactChecker)

    assert checker._format_date_filter("not-a-date") is None
