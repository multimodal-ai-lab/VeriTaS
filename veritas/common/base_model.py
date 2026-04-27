from __future__ import annotations

import logging
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("VeriTaS")


class VeritasBaseModel(BaseModel):
    id: int | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def save_to_db(self):
        """Inserts the instance into the database and saves the ID
        assigned by the DB."""
        from veritas.db import db

        if self.id is None:
            self.id = await db.insert(self)
        else:
            await db.update(self)

    @classmethod
    async def get(cls, id: int) -> VeritasBaseModel | None:
        """Returns an existing object from the database by its ID."""
        # TODO: Return existing class instance if it exists using an object cache
        from veritas.db import db

        return await db.get(cls, id)

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.id is not None
                and other.id is not None
                and self.id == other.id
        )

    def __hash__(self):
        return hash((self.id, str(type(self).__name__)))


class Dismissable(VeritasBaseModel):
    """VeriTaS model that can be dismissed (removed from the pipeline) for quality control."""

    dismissed: bool = False  # Whether this object is excluded from VeriTaS
    dismissed_reason: str | None = None  # Reason for exclusion (incl. any error messages if applicable)

    async def dismiss(self, reason: str) -> None:
        """Excludes this object from VeriTaS."""
        if not self.dismissed:
            logger.debug(f"Dismissing {type(self).__name__} {self.id}, reason: {reason}")
            self.dismissed = True
            self.dismissed_reason = reason
            await self.save_to_db()

    async def take_back_dismissal(self) -> None:
        """Undoes the exclusion of this object."""
        if self.dismissed:
            logger.debug(f"Taking back dismissal of {type(self).__name__} {self.id}")
            self.dismissed = False
            self.dismissed_reason = None
            await self.save_to_db()


class Deferrable(VeritasBaseModel):
    """VeriTaS model that can be deferred (postponed) for processing, e.g.,
    to wait for rate limit cooldown."""

    # When set, processing this object should be postponed until the given datetime
    deferred_until: datetime | None = None

    async def defer(self, until: datetime | None = None, *, hours: int | None = None) -> None:
        """Postpones processing this object until the specified datetime, or by a
        number of hours. If both are provided, 'until' takes precedence."""
        # Compute the actual datetime to defer until
        if until is None and hours is not None:
            until = datetime.now() + timedelta(hours=hours)
        if until is None:
            # Default to 24 hours if no value provided
            until = datetime.now() + timedelta(hours=24)
        assert isinstance(until, datetime)

        # Only update if this extends the defer window or wasn't set
        if self.deferred_until is None or until > self.deferred_until:
            logger.debug(f"Deferring {type(self).__name__} {self.id} until {until.isoformat()}.")
            self.deferred_until = until
            await self.save_to_db()

    @property
    def deferred(self) -> bool:
        """Returns True if the deferred_until datetime is in the future."""
        return self.deferred_until is not None and self.deferred_until > datetime.now()
