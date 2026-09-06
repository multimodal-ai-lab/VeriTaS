"""Round-tripping of Evidence through the DB representation, and the additive
`gold_evidence_*` columns on Claim."""

from datetime import datetime

import pytest

from tests.gold_evidence.conftest import make_evidence
from veritas.common import Claim
from veritas.db.veritas_db import _evidence_columns, row_to_evidence
from veritas.gold_evidence.models import (
    Evidence,
    EvidenceRole,
    ProximityLevel,
    SourceKind,
    to_naive,
)


def roundtrip(evidence: Evidence, evidence_id: int = 42) -> Evidence:
    """Flattens to columns and reconstructs, exactly as the DB layer does."""
    columns, values = _evidence_columns(evidence)
    row = dict(zip(columns, values))
    row["id"] = evidence_id
    return row_to_evidence(row)


def test_evidence_survives_the_roundtrip():
    original = make_evidence(available_since=datetime(2024, 4, 15, 10, 30))
    restored = roundtrip(original)

    assert restored.id == 42
    assert restored.claim_id == original.claim_id
    assert restored.proposition == original.proposition
    assert restored.source.locator == original.source.locator
    assert restored.source.kind is original.source.kind
    assert restored.source.proximity is original.source.proximity
    assert restored.role is original.role
    assert restored.available_since == original.available_since
    assert restored.faithfulness.assessment == original.faithfulness.assessment
    assert restored.temporal_validation.before_claim is True


def test_media_references_in_the_proposition_survive():
    original = make_evidence(
        proposition="The clip <video:12> shows the square at dusk.")
    restored = roundtrip(original)
    assert restored.proposition == "The clip <video:12> shows the square at dusk."


def test_flat_columns_mirror_the_nested_fields():
    """The flat columns exist for querying; they must not drift from the blob."""
    evidence = make_evidence(available_since=datetime(2024, 4, 15),
                             faithfulness=2 / 3, later_event=True)
    columns, values = _evidence_columns(evidence)
    row = dict(zip(columns, values))

    assert row["faithfulness_assessment"] == pytest.approx(2 / 3)
    assert row["before_claim"] is True
    assert row["later_event"] is True
    assert row["source_kind"] == SourceKind.NEWS_ARTICLE.value
    assert row["source_proximity"] == ProximityLevel.SECONDARY.value
    assert row["role"] == EvidenceRole.ESSENTIAL.value
    assert row["full_evidence"]["proposition"] == evidence.proposition


def test_unfiltered_evidence_serializes_with_nulls():
    evidence = make_evidence(filtered=False)
    columns, values = _evidence_columns(evidence)
    row = dict(zip(columns, values))
    assert row["accessible"] is None
    assert row["faithfulness_assessment"] is None
    assert row["before_claim"] is None
    assert row["admissible"] is None

    restored = roundtrip(evidence)
    assert restored.faithfulness is None
    assert restored.temporal_validation is None


def test_hashes_are_stable_and_distinguish_items():
    a = make_evidence()
    b = make_evidence(proposition="A different proposition.")
    columns, values_a = _evidence_columns(a)
    _, values_b = _evidence_columns(b)
    row_a, row_b = dict(zip(columns, values_a)), dict(zip(columns, values_b))

    assert row_a["source_locator_hash"] == row_b["source_locator_hash"]
    assert row_a["proposition_hash"] != row_b["proposition_hash"]
    # Deterministic across calls
    assert _evidence_columns(a)[1] == values_a


def test_a_missing_locator_stays_null_in_the_columns():
    """Both locator columns are nullable, so a source that is not a publication is
    stored as what it is rather than as an empty string."""
    from veritas.gold_evidence.models import SourceKind as Kind

    evidence = make_evidence(locator=None, kind=Kind.OFFLINE, available_since=None)
    columns, values = _evidence_columns(evidence)
    row = dict(zip(columns, values))

    assert row["source_locator"] is None
    assert row["source_locator_hash"] is None
    assert roundtrip(evidence).source.locator is None


def test_a_present_locator_is_still_hashed():
    evidence = make_evidence(locator="https://example.org/record/1")
    columns, values = _evidence_columns(evidence)
    row = dict(zip(columns, values))

    assert row["source_locator"] == "https://example.org/record/1"
    assert isinstance(row["source_locator_hash"], int)


def test_the_deferral_window_is_flattened_and_restored():
    from datetime import timedelta

    evidence = make_evidence()
    until = datetime.now() + timedelta(hours=24)
    evidence.deferred_until = until
    columns, values = _evidence_columns(evidence)
    assert dict(zip(columns, values))["deferred_until"] == until
    assert roundtrip(evidence).deferred_until == until


# --- Claim columns ---------------------------------------------------------

def claim_kwargs(**overrides) -> dict:
    base = dict(data="A claim", date=datetime(2024, 5, 1),
                appearance_ids=set(), review_ids={1})
    base.update(overrides)
    return base


def test_gold_evidence_columns_default_to_none():
    claim = Claim(**claim_kwargs())
    assert claim.gold_evidence_status is None
    assert claim.gold_evidence_reason is None
    assert claim.gold_evidence_updated_at is None
    assert claim.gold_evidence_rejected is False


def test_gold_evidence_columns_roundtrip_through_model_validate():
    """`get_claim_by_id` does SELECT * -> model_validate, so the new columns must
    survive a load/save cycle rather than being reset to NULL."""
    row = claim_kwargs(id=5, gold_evidence_status="rejected",
                       gold_evidence_reason="insufficient_evidence",
                       gold_evidence_updated_at=datetime(2026, 9, 1, 12, 0))
    claim = Claim.model_validate(row)
    assert claim.gold_evidence_status == "rejected"
    assert claim.gold_evidence_rejected is True

    dumped = claim.model_dump()
    assert dumped["gold_evidence_reason"] == "insufficient_evidence"
    assert Claim.model_validate(dumped).gold_evidence_updated_at == datetime(2026, 9, 1, 12, 0)


def test_rejection_does_not_touch_the_dismissed_flag():
    """A rejected instance stays a valid VeriTaS claim."""
    claim = Claim.model_validate(claim_kwargs(id=5, gold_evidence_status="rejected",
                                              gold_evidence_reason="insufficient_evidence"))
    assert claim.dismissed is False
    assert claim.dismissed_reason is None


# --- Time normalization ----------------------------------------------------

def test_to_naive_strips_timezones_via_utc():
    from datetime import date, timezone, timedelta

    aware = datetime(2024, 5, 1, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    assert to_naive(aware) == datetime(2024, 5, 1, 10, 0)
    assert to_naive(datetime(2024, 5, 1, 12, 0)) == datetime(2024, 5, 1, 12, 0)
    assert to_naive(date(2024, 5, 1)) == datetime(2024, 5, 1, 0, 0)
    assert to_naive(None) is None


def test_to_naive_rejects_other_types():
    with pytest.raises(TypeError):
        to_naive("2024-05-01")


def test_time_deltas_are_in_days():
    evidence = make_evidence(available_since=datetime(2024, 5, 11))
    assert evidence.time_to_claim(datetime(2024, 5, 1)) == pytest.approx(10.0)
    assert evidence.time_to_fact_check(datetime(2024, 5, 21)) == pytest.approx(-10.0)
    assert evidence.time_to_claim(None) is None
