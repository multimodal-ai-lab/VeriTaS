from __future__ import annotations

import logging
from datetime import date, datetime
from enum import Enum
from typing import Any

from veritas.common.base_model import VeritasBaseModel

logger = logging.getLogger("VeriTaS")


class SignatoryStatus(str, Enum):
    NOT_A_SIGNATORY = "not_a_signatory"
    ACTIVE = "active"
    EXPIRED = "expired"
    IN_RENEWAL = "in_renewal"


class Publisher(VeritasBaseModel):
    name: str
    domains: set[str]  # All domains (not HTTP URLs or subdomains) that belong to this publisher

    # IFCN signatory status
    ifcn_status: SignatoryStatus | None = None
    ifcn_expires: date | None = None

    # EFCSN member status
    efcsn_status: SignatoryStatus | None = None
    efcsn_expires: date | None = None

    country: str | None = None
    language: str | None = None  # ISO code, e.g., 'en', 'de'

    def model_post_init(self, __context: Any) -> None:
        # Validate signatory status by checking expiration
        if self.ifcn_status == SignatoryStatus.ACTIVE:
            if self.ifcn_expires and self.ifcn_expires <= datetime.now().date():
                logger.info(
                    f"The IFCN signatory status of publisher {self.name} is set to "
                    f"active, but has expired on {self.ifcn_expires.strftime('%Y-%m-%d')}."
                )
        if self.efcsn_status == SignatoryStatus.ACTIVE:
            if self.efcsn_expires and self.efcsn_expires <= datetime.now().date():
                logger.info(
                    f"The EFCSN membership status of publisher {self.name} is set to "
                    f"active, but has expired on {self.efcsn_expires.strftime('%Y-%m-%d')}."
                )
