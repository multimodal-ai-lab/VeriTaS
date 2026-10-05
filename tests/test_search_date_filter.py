"""Temporal-leakage guards for the baseline's web_search date filter.

The benchmark is only valid if evidence retrieved for a claim predates the claim.
Upstream search APIs filter at day granularity with inclusive end dates, so these
tests pin the boundary: the claim's own day must never appear in a filter value.
The Serper request body is built on the client (scrapeMM's SerperQuery), so this
needs no network.
"""

from datetime import date, datetime

import pytest

from eval.baselines.common.tools import ToolSession, build_search_query, parse_claim_day


CLAIM_DAY = date(2024, 2, 5)


def _cd_max(claim_date) -> str:
    """The inclusive end date Google receives for a search on behalf of a claim."""
    tbs = build_search_query("q", claim_date).request_body()["tbs"]
    return tbs.split("cd_max:")[1]


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
    assert parse_claim_day(claim_date) == CLAIM_DAY
    assert _cd_max(claim_date) == "2/4/2024", "claim day must not be retrievable"


def test_cutoff_crosses_month_boundary():
    assert _cd_max("2024-03-01T08:00:00") == "2/29/2024"


def test_cutoff_crosses_year_boundary():
    assert _cd_max("2020-01-01T00:00:00") == "12/31/2019"


def test_unparseable_date_raises():
    """A malformed date must not silently degrade into an unfiltered search."""
    with pytest.raises(ValueError):
        build_search_query("q", "not-a-date")
    with pytest.raises(ValueError):
        ToolSession(claim_date="not-a-date")


def test_missing_date_is_unfiltered():
    assert "tbs" not in build_search_query("q", None).request_body()
