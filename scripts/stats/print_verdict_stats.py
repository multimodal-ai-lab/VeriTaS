import argparse
import asyncio
from collections import Counter

from veritas.db import db
from veritas.common.annotation.rating import Category3Bin

async def print_verdict_stats(released_only: bool = False):
    """Prints statistics about verdicts in the database."""
    await db.connect_maybe_initialize()
    verdicts = await db.get_verdicts()
    
    if released_only:
        verdicts = [v for v in verdicts if (await v.claim).released]

    print(f"Total verdicts: {len(verdicts)}")
    verdicts_non_dismissed_claims = [verdict for verdict in verdicts if not (await verdict.claim).dismissed]
    print(f"Verdicts of non-dismissed claims: {len(verdicts_non_dismissed_claims)}")
    verdicts_sufficient_agreement = sum(verdict.sufficient_agreement for verdict in verdicts_non_dismissed_claims)
    print(f"Verdicts of non-dismissed claims with sufficient agreement: {verdicts_sufficient_agreement}")

    # Count ratings for the 5 properties
    properties = ["authenticity", "contextualization", "veracity", "context_coverage", "integrity"]
    stats = {prop: Counter() for prop in properties}

    for verdict in verdicts_non_dismissed_claims:
        for prop in properties:
            rating = getattr(verdict, prop)
            if rating:
                stats[prop][rating.as_3_bin()] += 1

    print("\nRating counts for all properties (non-dismissed claims):")
    header = f"{'Property':<20} | {'Negative':>10} | {'Neutral':>10} | {'Positive':>10}"
    print(header)
    print("-" * len(header))
    for prop in properties:
        counts = stats[prop]
        print(f"{prop:<20} | {counts[Category3Bin.NEGATIVE]:>10} | {counts[Category3Bin.NEUTRAL]:>10} | {counts[Category3Bin.POSITIVE]:>10}")


if __name__ == "__main__":
    asyncio.run(print_verdict_stats(released_only=True))
