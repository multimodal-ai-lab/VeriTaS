import logging

import pytest_asyncio

from veritas.db import db as veritas_db

# Set up logging
logger = logging.getLogger("VeriTaS")
logger.setLevel("DEBUG")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


@pytest_asyncio.fixture(scope="function", autouse=True)
async def db():
    # Create a fresh connection pool per test so it's bound to the same event loop
    # that the test runs in, avoiding cross-loop asyncpg issues.
    await veritas_db.connect_maybe_initialize(max_connections=1)
    try:
        yield veritas_db
    finally:
        # Ensure the pool is closed after each test to prevent cross-test interference
        await veritas_db.close()
