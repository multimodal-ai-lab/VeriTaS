"""Connection pool for the VeriTaS database.

Every connection is put into `default_transaction_read_only` mode, so this
service cannot write to the pipeline's data even if a query were wrong.
"""

from __future__ import annotations

import asyncio
import json
import logging

from pathlib import Path
from typing import Any

import asyncpg

from webui.config import get_settings

logger = logging.getLogger("veritas-webui")

#: Seconds a single connection attempt may take. A host that refuses the
#: connection fails in milliseconds; one that drops the packets - a firewall, or
#: an address that does not route - would otherwise hang for the OS TCP timeout,
#: and with it every request that needs the database, including the health check.
CONNECT_TIMEOUT = 5.0

#: Seconds a single query may take. The dashboard aggregates over every evidence
#: row, so this is generous; it exists to stop one pathological query from
#: holding a pool connection forever.
COMMAND_TIMEOUT = 60.0

_pool: asyncpg.Pool | None = None

#: Serializes the connection attempts of concurrent requests, so a burst of
#: them opens one pool instead of one pool each.
_lock = asyncio.Lock()


async def _init_connection(connection: asyncpg.Connection) -> None:
    await connection.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    await connection.set_type_codec(
        "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    # Belt and braces: this UI is a viewer, never a writer.
    await connection.execute("SET default_transaction_read_only = on;")


#: Marker files every common container runtime drops into the root filesystem.
_CONTAINER_MARKERS = ("/run/.containerenv", "/.dockerenv")


def in_container() -> bool:
    return any(Path(marker).exists() for marker in _CONTAINER_MARKERS)


def connection_hint() -> str | None:
    """A hint for the most common way this deployment is misconfigured.

    Inside a container on its own network, `localhost` is the *container*, not
    the machine running PostgreSQL - so a database on the host's loopback is
    unreachable no matter how it is spelled. The message names the two ways out
    rather than leaving a bare "connection refused" in the log."""
    settings = get_settings()
    if settings.db_host not in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return None
    if not in_container():
        return None
    return (f"VERITAS_DB_HOST is {settings.db_host!r}, which inside a container "
            f"means the container itself. Either share the host's network "
            f"(docker-compose.podman.yaml does this with network_mode: host), "
            f"or point VERITAS_DB_HOST at the host - host.docker.internal on "
            f"Docker, or the host's LAN address.")


async def connect(max_connections: int = 8) -> asyncpg.Pool:
    global _pool
    async with _lock:
        if _pool is None:
            settings = get_settings()
            logger.info("Connecting to %s:%s/%s as %s", settings.db_host,
                        settings.db_port, settings.db_name, settings.db_user)
            _pool = await asyncpg.create_pool(
                dsn=settings.dsn, init=_init_connection,
                min_size=1, max_size=max_connections,
                timeout=CONNECT_TIMEOUT, command_timeout=COMMAND_TIMEOUT,
            )
    return _pool


async def disconnect() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def get_pool() -> asyncpg.Pool:
    """The pool, connecting on first use. The database being unreachable while
    the service starts is not fatal - and not permanent either: every request
    retries, so the UI recovers on its own once the database is back."""
    if _pool is not None:
        return _pool
    try:
        return await connect()
    except Exception as error:
        raise RuntimeError(_unreachable(error)) from error


def _unreachable(error: Exception) -> str:
    """Names the endpoint that could not be reached - without the password."""
    settings = get_settings()
    return (f"Cannot reach the database '{settings.db_name}' at "
            f"{settings.db_host}:{settings.db_port} as user "
            f"'{settings.db_user}': {type(error).__name__}: {error}")


async def fetch(query: str, *args) -> list[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as connection:
        return await connection.fetch(query, *args)


async def fetchrow(query: str, *args) -> asyncpg.Record | None:
    pool = await get_pool()
    async with pool.acquire() as connection:
        return await connection.fetchrow(query, *args)


async def fetchval(query: str, *args) -> Any:
    pool = await get_pool()
    async with pool.acquire() as connection:
        return await connection.fetchval(query, *args)


async def check() -> str | None:
    """`None` if the database answers, otherwise why it does not."""
    try:
        if await fetchval("SELECT 1;") == 1:
            return None
        return "The database did not answer 'SELECT 1'."
    except RuntimeError as error:  # Already carries the endpoint description.
        logger.warning("Health check failed: %s", error)
        return str(error)
    except Exception as error:
        message = _unreachable(error)
        logger.warning("Health check failed: %s", message)
        return message


async def healthy() -> bool:
    return await check() is None
