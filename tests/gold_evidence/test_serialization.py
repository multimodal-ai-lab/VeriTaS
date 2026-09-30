"""Round-tripping of Evidence, its citations and the sources they cite through the
DB representation, and the additive `gold_evidence_*` columns on Claim."""

from datetime import datetime

import pytest

from tests.gold_evidence.conftest import make_citation, make_evidence, make_source
from veritas.common import Claim
from veritas.db.veritas_db import (
    _citation_columns,
    _evidence_columns,
    _rationale_columns,
    _source_columns,
    row_to_citation,
    row_to_evidence,
    row_to_source,
    row_to_verdict_rationale,
)
from veritas.gold_evidence.models import (
    Evidence,
    EvidenceRole,
    ProximityLevel,
    SourceKind,
    to_naive,
)


def source_roundtrip(source, source_id: int = 7):
    columns, values = _source_columns(source)
    row = dict(zip(columns, values))
    row["id"] = source_id
    return row_to_source(row)


def citation_roundtrip(citation, evidence_id: int = 42, citation_id: int = 9):
    columns, values = _citation_columns(citation, evidence_id, claim_id=1)
    row = dict(zip(columns, values))
    row["id"] = citation_id
    return row_to_citation(row)


def roundtrip(evidence: Evidence, evidence_id: int = 42) -> Evidence:
    """Flattens to columns and reconstructs, exactly as the DB layer does: the item,
    its citations and the sources they cite live in separate tables and are
    reassembled on read, citations of one source sharing one instance."""
    columns, values = _evidence_columns(evidence)
    row = dict(zip(columns, values))
    row["id"] = evidence_id

    restored = row_to_evidence(row)
    sources = {}
    for index, citation in enumerate(evidence.citations):
        copy = citation_roundtrip(citation, evidence_id, citation_id=index + 1)
        if citation.source is not None:
            key = id(citation.source)
            if key not in sources:
                sources[key] = source_roundtrip(citation.source, source_id=len(sources) + 1)
            copy.source = sources[key]
        restored.citations.append(copy)
    return restored


def test_evidence_survives_the_roundtrip():
    original = make_evidence(available_since=datetime(2024, 4, 15, 10, 30))
    restored = roundtrip(original)

    assert restored.id == 42
    assert restored.claim_id == original.claim_id
    assert restored.proposition == original.proposition
    assert restored.role is original.role
    assert restored.extraction_confidence == original.extraction_confidence

    citation, original_citation = restored.citations[0], original.citations[0]
    assert citation.locator == original_citation.locator
    assert citation.kind is original_citation.kind
    assert citation.proximity is original_citation.proximity
    assert citation.available_since == original_citation.available_since
    assert citation.faithfulness.assessment == original_citation.faithfulness.assessment
    assert citation.temporal_validation.before_claim is True
    assert citation.admissible is True
    assert citation.judged_at == original_citation.judged_at


def test_media_references_in_the_proposition_survive():
    original = make_evidence(
        proposition="The clip <video:12> shows the square at dusk.")
    restored = roundtrip(original)
    assert restored.proposition == "The clip <video:12> shows the square at dusk."


def test_every_citation_of_an_item_survives_the_roundtrip():
    original = make_evidence(citations=[make_citation(locator="https://a/1", name="Reuters"),
                                        make_citation(locator="https://a/2", name="Register")])
    restored = roundtrip(original)

    assert [c.name for c in restored.citations] == ["Reuters", "Register"]


def test_the_source_survives_the_roundtrip_with_its_content():
    source = make_source(content="The page text.", available_since=datetime(2024, 4, 15))
    source.dating_method = "meta"
    restored = source_roundtrip(source)

    assert restored.id == 7
    assert restored.locator == source.locator
    assert restored.raw_content == "The page text."
    assert restored.accessible is True
    assert restored.available_since == datetime(2024, 4, 15)
    assert restored.dating_method == "meta"
    assert restored.is_fact_check is False


def test_the_content_is_stored_once_per_source():
    """It is the bulk of the row: a column of its own, not a second copy in the blob."""
    columns, values = _source_columns(make_source(content="The page text."))
    row = dict(zip(columns, values))
    assert row["raw_content"] == "The page text."
    assert "raw_content" not in row["full_source"]


def test_a_source_is_stored_under_its_normalized_locator():
    a, b = make_source(locator="https://Example.org/a/"), make_source(locator="https://example.org/a")
    key_a = dict(zip(*_source_columns(a)))["locator_key"]
    key_b = dict(zip(*_source_columns(b)))["locator_key"]
    assert key_a == key_b
    assert len(key_a) == 40  # SHA-1: collision-free across the global table


def test_the_citation_columns_mirror_the_nested_fields():
    """The flat columns exist for querying; they must not drift from the blob."""
    evidence = make_evidence(available_since=datetime(2024, 4, 15), faithfulness=2 / 3)
    citation = evidence.citations[0]
    citation.source.id = 5
    columns, values = _citation_columns(citation, evidence_id=42, claim_id=1)
    row = dict(zip(columns, values))

    assert row["source_id"] == 5
    assert row["faithfulness_assessment"] == pytest.approx(2 / 3)
    assert row["before_claim"] is True
    assert row["kind"] == SourceKind.NEWS_ARTICLE.value
    assert row["proximity"] == ProximityLevel.SECONDARY.value
    assert row["citation_key"] == citation.source.key
    assert row["full_citation"]["name"] == citation.name
    # The source is stored in its own table, not a second time in the citation.
    assert "source" not in row["full_citation"]


def test_the_later_event_judgement_is_stored_with_the_item():
    """It is about the proposition, so it belongs to the item, not to a citation."""
    evidence = make_evidence(later_event=True)
    columns, values = _evidence_columns(evidence)
    row = dict(zip(columns, values))

    assert row["later_event"] is True
    assert row["admissible"] is False
    assert row["inadmissibility_reason"] == "later_event"
    assert roundtrip(evidence).later_event.change_detected is True


def test_the_item_columns_carry_what_the_citations_decided():
    """The aggregation layer stores the outcome so that queries need not join."""
    from veritas.gold_evidence.admissibility import apply_admissibility_to_item

    evidence = make_evidence(citations=[make_citation(locator="https://a/1", accessible=False),
                                        make_citation(locator="https://a/2")])
    apply_admissibility_to_item(evidence)
    columns, values = _evidence_columns(evidence)
    row = dict(zip(columns, values))

    assert row["role"] == EvidenceRole.KEY.value
    assert row["admissible"] is True          # one citation survived
    assert row["n_sources"] == 2
    assert row["n_admissible_sources"] == 1
    assert row["full_evidence"]["proposition"] == evidence.proposition
    # The citations live in their own table, not a second time in the blob.
    assert "citations" not in row["full_evidence"]
    assert "sources" not in row["full_evidence"]


def test_unfiltered_evidence_serializes_with_nulls():
    evidence = make_evidence(decided=False)
    citation = evidence.citations[0]
    columns, values = _citation_columns(citation, evidence_id=42, claim_id=1)
    row = dict(zip(columns, values))
    assert row["faithfulness_assessment"] is None
    assert row["before_claim"] is None
    assert row["admissible"] is None
    assert row["judged_at"] is None
    assert dict(zip(*_source_columns(citation.source)))["accessible"] is None

    restored = citation_roundtrip(citation)
    assert restored.faithfulness is None
    assert restored.temporal_validation is None


def test_hashes_are_stable_and_distinguish_items():
    a = make_evidence()
    b = make_evidence(proposition="A different proposition.")
    columns, values_a = _evidence_columns(a)
    _, values_b = _evidence_columns(b)
    row_a, row_b = dict(zip(columns, values_a)), dict(zip(columns, values_b))

    assert row_a["proposition_hash"] != row_b["proposition_hash"]
    # Deterministic across calls
    assert _evidence_columns(a)[1] == values_a


def test_citations_are_keyed_by_source_and_fall_back_to_kind_and_name():
    a = make_citation(locator="https://example.org/a")
    b = make_citation(locator="https://example.org/a", name="Another label")
    assert a.key == b.key

    # A citation without a source is identified by kind and name instead, so that
    # one interview is not stored twice within an item.
    interview = make_citation(locator=None, kind=SourceKind.OFFLINE, name="Prof. Meier")
    same = make_citation(locator=None, kind=SourceKind.OFFLINE, name="prof.  meier")
    assert interview.key == same.key
    assert interview.key != a.key


def test_a_citation_without_a_source_stays_null_in_the_columns():
    """A source that is not a publication is stored as what it is rather than as
    an empty string."""
    evidence = make_evidence(locator=None, kind=SourceKind.OFFLINE, available_since=None)
    columns, values = _citation_columns(evidence.citations[0], evidence_id=42, claim_id=1)
    row = dict(zip(columns, values))

    assert row["source_id"] is None
    restored = roundtrip(evidence)
    assert restored.citations[0].source is None
    assert restored.citations[0].locator is None


def test_citations_of_one_source_share_it_after_the_roundtrip():
    shared = make_source()
    evidence = make_evidence(citations=[make_citation(source=shared, name="A"),
                                        make_citation(locator=None, kind=SourceKind.TOOL,
                                                      name="B")])
    restored = roundtrip(evidence)
    assert restored.citations[0].source is not None
    assert restored.sources == [restored.citations[0].source]


def test_the_deferral_window_is_flattened_and_restored():
    from datetime import timedelta

    source = make_source()
    until = datetime.now() + timedelta(hours=24)
    source.deferred_until = until
    columns, values = _source_columns(source)

    assert dict(zip(columns, values))["deferred_until"] == until
    assert source_roundtrip(source).deferred_until == until


# --- Verdict rationales ----------------------------------------------------

def test_a_rationale_survives_the_roundtrip():
    from tests.gold_evidence.conftest import make_rationale

    original = make_rationale("The clip <video:12> shows daylight.")
    columns, values = _rationale_columns(original)
    row = dict(zip(columns, values))
    row["id"] = 11

    assert row["rationale"] == "The clip <video:12> shows daylight."
    assert row["claim_id"] == 1
    assert row["review_id"] == 7

    restored = row_to_verdict_rationale(row)
    assert restored.id == 11
    assert restored.rationale == original.rationale
    assert restored.article_id == original.article_id


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
