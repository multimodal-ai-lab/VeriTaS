"""These tests are pure: no database, no network, no API calls. The `db` fixture
below overrides the autouse fixture in `tests/conftest.py` so that no connection
pool is opened for them."""

import pytest_asyncio


@pytest_asyncio.fixture(scope="function", autouse=True)
async def db():
    yield None
