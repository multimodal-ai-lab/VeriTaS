"""Tests for the SQL building blocks of the media index.

The index itself runs inside PostgreSQL, but the reference pattern it is driven
with decides which media are considered "in use" - and therefore which files
survive the cleanup. It is worth pinning down precisely.
"""

import inspect
import re

import pytest

from veritas.db.veritas_db import (
    MEDIA_IS_USED,
    MEDIA_REF_PATTERN,
    VeritasDB,
    _usage_agg,
)

PATTERN = re.compile(MEDIA_REF_PATTERN)


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_every_ezmm_kind_is_matched(kind):
    assert PATTERN.findall(f"see <{kind}:42> here") == [(kind, "42")]


def test_all_references_in_a_text_are_found():
    text = "Intro <image:1> middle <video:22> and <image:1> again <audio:333>."

    assert PATTERN.findall(text) == [
        ("image", "1"), ("video", "22"), ("image", "1"), ("audio", "333"),
    ]


@pytest.mark.parametrize("text", [
    "<pdf:1>",           # not an ezMM kind
    "<image:>",          # no ID
    "<image:abc>",       # non-numeric ID
    "<image 1>",         # no colon
    "image:1",           # no brackets
    "<Image:1>",         # ezMM references are lower-case
])
def test_non_references_are_not_matched(text):
    assert PATTERN.findall(text) == []


def test_ids_stay_within_postgres_integer_range():
    """The pattern casts to `INTEGER` in SQL, so an ID that cannot fit must not
    match at all - a match would abort the whole index rebuild."""
    assert PATTERN.findall("<image:999999999>") == [("image", "999999999")]
    assert PATTERN.findall("<image:1234567890>") == []
    assert int("9" * 9) < 2 ** 31 - 1


def test_a_reference_inside_surrounding_text_is_still_found():
    assert PATTERN.findall("![alt](x)<image:7>**bold**") == [("image", "7")]


def test_usage_aggregation_filters_by_source():
    sql = _usage_agg("claim")

    assert "ARRAY_AGG(DISTINCT source_id)" in sql
    assert "source = 'claim'" in sql
    assert "'{}'::INTEGER[]" in sql  # no NULL arrays; the columns are NOT NULL


def test_the_used_predicate_covers_every_usage_column():
    for column in ("claim_ids", "appearance_ids", "article_ids", "evidence_ids"):
        assert f"CARDINALITY({column}) > 0" in MEDIA_IS_USED
    assert MEDIA_IS_USED.startswith("(") and MEDIA_IS_USED.endswith(")")


def test_rebuilding_the_index_never_touches_an_embedding():
    """Embeddings are expensive to recompute and the index rebuild has no reason
    to write them. Neither the reset nor the upsert may name the column: the
    upsert must leave it out of both its column list and its `DO UPDATE SET`."""
    source = inspect.getsource(VeritasDB.rebuild_media_index)

    assert "embedding" not in source


def test_the_migration_only_ever_fills_in_a_missing_embedding():
    """The one place the schema setup writes `embedding` is the rescue path for a
    stranded `media_embeddings` table, and it must not overwrite what is there."""
    source = inspect.getsource(VeritasDB._create_tables)
    writes = [line.strip() for line in source.splitlines()
              if "embedding" in line and ("SET " in line or "INSERT INTO media" in line)]

    assert writes == [
        "INSERT INTO media (kind, media_id, embedding)",
        "SET embedding = COALESCE(media.embedding, EXCLUDED.embedding);",
    ]
    assert "DROP TABLE" not in source
