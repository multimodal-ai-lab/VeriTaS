"""Fixtures for the web UI tests.

Like the Gold Evidence tests, these are pure: no database, no network, no LLM
calls. The `db` fixture below deliberately overrides the autouse fixture in
`tests/conftest.py` so that no connection pool is opened for them.
"""

import pytest_asyncio


@pytest_asyncio.fixture(scope="function", autouse=True)
async def db():
    """Overrides the DB fixture from the parent conftest: these tests need no DB."""
    yield None
