"""Rows written before `essential` was renamed to `key` stay in the database as they
are; they must read as the current names wherever they are loaded."""

import pytest

from tests.gold_evidence.conftest import make_evidence
from veritas.gold_evidence import CONDITION_CLAIM
from veritas.db.veritas_db import _evidence_columns, row_to_evidence
from veritas.gold_evidence.admissibility import key_evidence, lost_key, select
from veritas.gold_evidence.analysis import build_claim_record
from veritas.gold_evidence.extraction import _parse_enum
from veritas.gold_evidence.models import (
    Evidence,
    EvidenceRole,
    LEGACY_REASON_ALIASES,
    LEGACY_ROLE_ALIASES,
    normalize_reason,
)
from veritas.gold_evidence.pipeline import REJECT_KEY_EVIDENCE_LOST


def test_the_role_values_are_the_ones_the_prompt_asks_for():
    assert [role.value for role in EvidenceRole] == ["key", "auxiliary", "background"]


@pytest.mark.parametrize("value", ["essential", "ESSENTIAL", " Essential "])
def test_the_enum_reads_the_legacy_role_as_key(value):
    assert EvidenceRole(value.strip().lower()) is EvidenceRole.KEY


def test_the_enum_still_rejects_unknown_roles():
    with pytest.raises(ValueError):
        EvidenceRole("crucial")


@pytest.mark.parametrize("value", ["essential", "Essential", " essential "])
def test_the_model_reads_the_legacy_role_as_key(value):
    item = Evidence.model_validate({"claim_id": 1, "proposition": "p", "role": value})
    assert item.role is EvidenceRole.KEY
    assert item.is_key


def test_a_stored_legacy_row_loads_as_key():
    """The JSONB blob of an old row carries `essential`; loading it must neither fail
    nor demote the item to the default role."""
    columns, values = _evidence_columns(make_evidence(role=EvidenceRole.KEY))
    row = dict(zip(columns, values))
    row["id"] = 3
    row["role"] = "essential"
    row["full_evidence"] = {**row["full_evidence"], "role": "essential"}

    restored = row_to_evidence(row)

    assert restored.role is EvidenceRole.KEY


def test_new_rows_are_written_under_the_new_name():
    columns, values = _evidence_columns(make_evidence(role=EvidenceRole.KEY))
    row = dict(zip(columns, values))
    assert row["role"] == "key"
    assert row["full_evidence"]["role"] == "key"


def test_legacy_items_count_as_key_everywhere():
    legacy = make_evidence(proposition="p", role=EvidenceRole("essential"))
    auxiliary = make_evidence(proposition="q", role=EvidenceRole.AUXILIARY)

    assert key_evidence([legacy, auxiliary]) == [legacy]
    assert lost_key([legacy, auxiliary]) == []
    assert select([auxiliary, legacy], CONDITION_CLAIM)[0] is legacy


@pytest.mark.parametrize("value", ["key", "KEY", "Key ", "essential", "Essential"])
def test_extraction_accepts_the_new_and_the_old_role_name(value):
    """A model that still answers with the old name must not lose the item's weight."""
    assert _parse_enum(value, EvidenceRole, EvidenceRole.AUXILIARY) is EvidenceRole.KEY


def test_extraction_falls_back_to_auxiliary_for_an_unknown_role():
    assert _parse_enum("crucial", EvidenceRole, EvidenceRole.AUXILIARY) is EvidenceRole.AUXILIARY


def test_the_legacy_rejection_reason_reads_as_the_current_one():
    assert normalize_reason("essential_evidence_lost") == REJECT_KEY_EVIDENCE_LOST
    assert normalize_reason(REJECT_KEY_EVIDENCE_LOST) == REJECT_KEY_EVIDENCE_LOST
    assert normalize_reason("insufficient_evidence") == "insufficient_evidence"
    assert normalize_reason(None) is None


def test_the_aliases_point_at_current_names():
    assert set(LEGACY_ROLE_ALIASES.values()) <= {role.value for role in EvidenceRole}
    assert set(LEGACY_REASON_ALIASES.values()) == {REJECT_KEY_EVIDENCE_LOST}


def test_the_webui_mirrors_the_aliases():
    """`webui` does not import `veritas`, so it keeps its own copy of the tables."""
    from webui import queries

    assert queries.LEGACY_ROLES == LEGACY_ROLE_ALIASES
    assert queries.LEGACY_REASONS == LEGACY_REASON_ALIASES


def test_the_analysis_reports_legacy_rejections_under_the_current_reason():
    class _Claim:
        id = 1
        gold_evidence_status = "rejected"
        gold_evidence_reason = "essential_evidence_lost"
        released = True
        is_rectified = False
        language = "en"

    record = build_claim_record(claim=_Claim(), gold=None, evidence=[], t_c=None, t_f=None)
    assert record.reason == REJECT_KEY_EVIDENCE_LOST
