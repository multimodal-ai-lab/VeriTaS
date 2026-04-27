from __future__ import annotations

import asyncio
import logging
from abc import ABC

from veritas.db import db
from veritas.models import QuotaExceededError

logger = logging.getLogger("VeriTaS")


class Stage(ABC):
    """Base abstract class for all pipeline stages."""

    id: int
    name: str

    db_max_connections: int = 1
    default_interval_seconds: float = 10

    done: bool = False

    async def setup(self, **kwargs) -> None:
        """Called once before the loop starts. Override to allocate state."""
        await db.connect_maybe_initialize(max_connections=self.db_max_connections)

    async def get_queued_items(self, **kwargs) -> list:
        """Fetch the next batch of items to process from the DB."""
        raise NotImplementedError

    async def step(self, **kwargs) -> None:
        """Perform one iteration. Set `self.done = True` when fully done."""
        raise NotImplementedError

    async def teardown(self) -> None:
        """Called once after the loop exits (success and error)."""
        pass

    async def run(self, *, interval_seconds: float | None = None, **kwargs) -> None:
        """Run the stage loop until ``self.done`` is set or quota is hit."""
        if interval_seconds is None:
            interval_seconds = self.default_interval_seconds
        await self.setup(**kwargs)
        try:
            while not self.done:
                try:
                    await self.step(**kwargs)
                except QuotaExceededError as e:
                    logger.error(f"❌ Quota exceeded: {e}")
                    return
                except Exception as e:
                    logger.error(f"❌ Stage {self.id} ({self.name}) error: {e}", exc_info=True)
                    await _safe_sleep(interval_seconds)
                    continue

                if self.done:
                    break

                await _safe_sleep(interval_seconds)
            logger.info(f"✅ Stage {self.id} ({self.name}) processing completed")
        finally:
            await self.teardown()


async def _safe_sleep(seconds: float) -> None:
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        pass
