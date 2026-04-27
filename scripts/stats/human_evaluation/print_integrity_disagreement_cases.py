#!/usr/bin/env python3
"""Print the top 10 integrity disagreement cases between aggregated human and automatic ratings.

Similar to print_annotator_disagreement_cases.py, but focuses on the integrity property
across the whole human evaluation using aggregated human ratings.

Run this script manually (requires DB access configured in config/globals.yaml):
    python -m scripts.stats.human_evaluation.print_integrity_disagreement_cases
"""
import asyncio

from tabulate import tabulate

from scripts.stats.human_evaluation.plot_human_vs_automatic import fetch_verdicts, filter_verdicts
from veritas.db import db
from veritas.db.annotation_db import annotation_db


async def main() -> None:
    await db.connect_maybe_initialize()
    await annotation_db.connect_maybe_initialize()

    print("Fetching data...")
    verdicts = await fetch_verdicts()
    verdicts = filter_verdicts(verdicts, exclude_single_human_ratings=True, exclude_human_disagreement=True)

    # Fetch annotation IDs per claim, excluding dismissed claims
    ann_rows = await annotation_db._fetch(
        "SELECT a.claim_id, ARRAY_AGG(a.id ORDER BY a.id) as annotation_ids "
        "FROM annotations a "
        "JOIN claims c ON a.claim_id = c.id "
        "WHERE a.status = 'completed' AND NOT a.excluded AND NOT c.dismissed "
        "GROUP BY a.claim_id"
    )
    annotation_ids_by_claim: dict[int, list[int]] = {r['claim_id']: r['annotation_ids'] for r in ann_rows}

    # Collect integrity disagreements: (|diff|, claim_id, human_score, auto_score, human_decisive, auto_decisive, ann_ids)
    disagreements: list[tuple[float, int, float, float, str, str, list[int]]] = []

    for cid, hv, av in verdicts:
        h_integrity = hv.integrity
        a_integrity = av.integrity
        if h_integrity is None or a_integrity is None:
            continue

        diff = abs(h_integrity.score - a_integrity.score)
        disagreements.append((
            diff,
            cid,
            h_integrity.score,
            a_integrity.score,
            hv.compromising_property_name,
            av.compromising_property_name,
            annotation_ids_by_claim.get(cid, []),
        ))

    disagreements.sort(key=lambda x: x[0], reverse=True)

    print(f"\nTotal integrity comparisons: {len(disagreements)}")
    print("\nTop 10 Most Disagreeing Integrity Cases (Human vs. Automatic):")

    if disagreements:
        rows = []
        for diff, cid, h_score, a_score, h_prop, a_prop, ann_ids in disagreements[:10]:
            rows.append([
                cid,
                f"{h_score:.4f}",
                f"{a_score:.4f}",
                f"{diff:.4f}",
                h_prop,
                a_prop,
                ", ".join(str(a) for a in ann_ids),
            ])
        headers = ["Claim ID", "Human Score", "Auto Score", "|Diff|",
                    "Human Decisive Prop.", "Auto Decisive Prop.", "Annotation IDs"]
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print("  No integrity comparisons available.")


if __name__ == "__main__":
    asyncio.run(main())
