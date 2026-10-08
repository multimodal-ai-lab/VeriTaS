"""Fixtures for DB integration tests that write data.

Unlike the parent `db` fixture, which connects to the configured database, the
fixture below creates a fresh, uniquely named database on the configured server
for every test and drops it afterward. Hence these tests never touch existing
data. The global `veritas.db.db` object is pointed at the temporary database
for the duration of the test, so that pipeline code using it works as usual.
"""

import uuid

import asyncpg
import pytest_asyncio

from veritas.db import db as veritas_db


@pytest_asyncio.fixture(scope="function", autouse=True)
async def db():
    """Overrides the DB fixture from the parent conftest with a temporary database."""
    original_name, original_dsn = veritas_db._database, veritas_db.dsn
    name = f"veritas_unittest_{uuid.uuid4().hex[:12]}"
    veritas_db._database = name
    veritas_db.dsn = (f"postgresql://{veritas_db._user}:{veritas_db._password}"
                      f"@{veritas_db._host}:{veritas_db._port}/{name}")
    try:
        await veritas_db.connect_maybe_initialize(max_connections=5)
        yield veritas_db
    finally:
        await veritas_db.close()
        veritas_db.pool = None
        veritas_db._database, veritas_db.dsn = original_name, original_dsn
        conn = await asyncpg.connect(
            user=veritas_db._user, password=veritas_db._password,
            host=veritas_db._host, port=veritas_db._port, database="postgres",
        )
        try:
            await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        finally:
            await conn.close()
