"""The gold-evidence tables can be emptied and dropped without fighting their own
references: every foreign key they hold cascades.

Checked on the schema source, like the media index tests, so no database is needed.
"""

import inspect
import re

from veritas.db.veritas_db import VeritasDB

GOLD_EVIDENCE_TABLES = ("evidence", "evidence_sources", "sources", "citations",
                        "verdict_rationales", "gold_evidence_results")


def table_definitions() -> dict[str, str]:
    """`CREATE TABLE` body per table name."""
    source = inspect.getsource(VeritasDB._create_tables)
    return {name: body for name, body in re.findall(
        r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\n\s*\);", source, flags=re.DOTALL)}


def test_every_gold_evidence_table_is_defined():
    assert set(GOLD_EVIDENCE_TABLES) <= set(table_definitions())


def test_every_foreign_key_of_the_gold_evidence_tables_cascades():
    definitions = table_definitions()
    for table in GOLD_EVIDENCE_TABLES:
        for line in definitions[table].splitlines():
            if "REFERENCES" in line:
                assert "ON DELETE CASCADE" in line, f"{table}: {line.strip()}"


def test_existing_databases_get_the_cascading_source_reference():
    """`CREATE TABLE IF NOT EXISTS` does not touch a table created earlier."""
    source = inspect.getsource(VeritasDB._create_tables)
    assert "con.confdeltype <> 'c'" in source
    assert "REFERENCES sources (id) ON DELETE CASCADE;" in source
