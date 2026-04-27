"""Utilities for working with claims data."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Default path to claims file
DEFAULT_CLAIMS_PATH = Path(__file__).parent.parent / "data" / "veritas_release" / "veritas_longitudinal_2020_q1_2025_q4" / "claims.json"

# Module-level cache for claims data
_claims_cache: Optional[dict[int, str]] = None


def _load_claims_to_quarter_map(claims_path: Path = DEFAULT_CLAIMS_PATH) -> dict[int, str]:
    """Load claims and build a mapping from claim ID to quarter string."""
    with open(claims_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_quarter = {}
    for claim in data["claims"]:
        claim_id = claim["id"]
        date_str = claim["date"]
        # Parse ISO date format (e.g., "2020-01-07T00:00:00")
        dt = datetime.fromisoformat(date_str)
        quarter = (dt.month - 1) // 3 + 1
        quarter_str = f"Q{quarter} {dt.year}"
        id_to_quarter[claim_id] = quarter_str

    return id_to_quarter


def get_claim_quarter(claim_id: int, claims_path: Path = DEFAULT_CLAIMS_PATH) -> str:
    """
    Get the quarter for a given claim ID.

    Args:
        claim_id: The claim ID to look up.
        claims_path: Path to the claims.json file. Uses default if not specified.

    Returns:
        A string representing the quarter, e.g., "Q4 2024".

    Raises:
        KeyError: If the claim ID is not found.
    """
    global _claims_cache

    if _claims_cache is None:
        _claims_cache = _load_claims_to_quarter_map(claims_path)

    if claim_id not in _claims_cache:
        raise KeyError(f"Claim ID {claim_id} not found in claims data")

    return _claims_cache[claim_id]
