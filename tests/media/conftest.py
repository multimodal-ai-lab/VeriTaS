"""Fixtures for the media index and cleanup tests.

These tests are pure: no database, no filesystem outside `tmp_path`, no network.
The `db` fixture below deliberately overrides the autouse fixture in
`tests/conftest.py` so that no connection pool is opened for them.
"""

import pytest_asyncio


@pytest_asyncio.fixture(scope="function", autouse=True)
async def db():
    """Overrides the DB fixture from the parent conftest: these tests need no DB."""
    yield None
