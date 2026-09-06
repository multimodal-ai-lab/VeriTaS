"""Connection pool for the VeriTaS database.

Every connection is put into `default_transaction_read_only` mode, so this
service cannot write to the pipeline's data even if a query were wrong.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from webui.config import get_settings

logger = logging.getLogger("veritas-webui")

_pool: asyncpg.Pool | None = None


async def _init_connection(connection: asyncpg.Connection) -> None:
    await connection.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    await connection.set_type_codec(
        "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    # Belt and braces: this UI is a viewer, never a writer.
    await connection.execute("SET default_transaction_read_only = on;")


async def connect(max_connections: int = 8) -> asyncpg.Pool:
    global _pool
    if _pool is None:
        settings = get_settings()
        logger.info("Connecting to %s:%s/%s as %s", settings.db_host,
                    settings.db_port, settings.db_name, settings.db_user)
        _pool = await asyncpg.create_pool(
            dsn=settings.dsn, init=_init_connection, min_size=1, max_size=max_connections
        )
    return _pool


async def disconnect() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialized.")
    return _pool


async def fetch(query: str, *args) -> list[asyncpg.Record]:
    async with get_pool().acquire() as connection:
        return await connection.fetch(query, *args)


async def fetchrow(query: str, *args) -> asyncpg.Record | None:
    async with get_pool().acquire() as connection:
        return await connection.fetchrow(query, *args)


async def fetchval(query: str, *args) -> Any:
    async with get_pool().acquire() as connection:
        return await connection.fetchval(query, *args)


async def healthy() -> bool:
    try:
        return await fetchval("SELECT 1;") == 1
    except Exception:
        return False
