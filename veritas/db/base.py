import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

logger = logging.getLogger("VeriTaS")


class Database:
    """Base class for wrapping a PostgreSQL database asynchronously."""

    def __init__(self, database: str, user: str, password: str, host: str, port: int):
        """
        Initialize the Database class with a Data Source Name (DSN).
        Example DSN: 'postgresql://user:password@localhost:5432/veritas_db'
        """
        self._database = database
        self._user = user
        self._password = password
        self._host = host
        self._port = port
        self.dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self.pool: asyncpg.Pool | None = None

    def is_initialized(self) -> bool:
        """Check if the connection pool is initialized."""
        return self.pool is not None

    async def connect(self, max_connections: int = 1):
        """Initialize the connection pool."""
        self.pool = await asyncpg.create_pool(
            dsn=self.dsn,
            init=_setup_connection,
            min_size=1,  # Number of connection the pool will be initialized with.
            max_size=max_connections
        )
        logger.debug("Database connection pool created.")

    async def _ensure_db(self) -> bool:
        """Logs in into the default `postgres` DB and initializes a fresh
        database if it does not exist yet. Returns True if a new DB was created."""
        default_dsn = f"postgresql://{self._user}:{self._password}@{self._host}:{self._port}/postgres"

        # Connect to 'postgres'
        conn = await asyncpg.connect(dsn=default_dsn)

        # Check if database exists (optional, otherwise CREATE will fail)
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", self._database)

        if not exists:
            # Create database
            await conn.execute(f'CREATE DATABASE "{self._database}"')
            logger.info(f"Database '{self._database}' created.")
            await conn.close()
            return True
        else:
            logger.debug(f"Found existing database '{self._database}'.")
            await conn.close()
            return False

    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logger.debug("Database connection pool closed.")

    async def _fetch(self, query: str, *args) -> list[asyncpg.Record]:
        """Fetch multiple rows."""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def _fetchrow(self, query: str, *args) -> asyncpg.Record | None:
        """Fetch a single row."""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def _execute(self, query: str, *args) -> str:
        """Execute a query (INSERT, UPDATE, DELETE). Returns status string."""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def _fetchval(self, query: str, *args) -> Any:
        """Fetch a single value."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    @asynccontextmanager
    async def _transaction(self):
        """Yields a pooled connection wrapped in a transaction, so that a group of
        statements either all take effect or none of them do."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def _copy_records_to_table(self, *args, **kwargs):
        """Saves the provided data (`records`) to the specified table."""
        async with self.pool.acquire() as conn:
            await conn.copy_records_to_table(*args, **kwargs)


async def _setup_connection(conn):
    """Setup function called for every new connection in the pool.
    It adds a custom JSONB codec to the connection."""

    def encoder(obj):
        return b"\x01" + json.dumps(obj).encode("utf-8")

    def decoder(obj):
        return json.loads(obj.decode("utf-8")[1:])

    await conn.set_type_codec("jsonb", encoder=encoder, decoder=decoder, schema="pg_catalog", format="binary")
