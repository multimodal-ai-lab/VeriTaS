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


class FakeRegistry:
    """Stands in for the ezMM registry: `missing` references resolve to nothing."""

    def __init__(self, missing=()):
        self.missing = set(missing)

    def describe(self, references):
        return [
            {
                "reference": reference,
                "kind": reference.strip("<>").split(":")[0],
                "id": int(reference.strip("<>").split(":")[1]),
                "exists": reference not in self.missing,
                "url": f"/api/media/{reference.strip('<>').replace(':', '/')}",
            }
            for reference in references
        ]

    def get_by_reference(self, reference):
        return None
