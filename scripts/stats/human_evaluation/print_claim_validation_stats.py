#!/usr/bin/env python3
"""Print a comprehensive table summarizing claim quality statistics from the human evaluation.

Reports:
- Total number of annotated claims
- Number of dismissed claims
- Detailed tabulation of dismissed reasons and their counts

Run this script manually (requires DB access configured in config/globals.yaml):
    python -m scripts.stats.human_evaluation.print_claim_validation_stats
"""
import asyncio
import textwrap

from tabulate import tabulate

from veritas.db.annotation_db import annotation_db


async def main() -> None:
    await annotation_db.connect_maybe_initialize()

    # Total annotated claims (distinct claim IDs with at least one completed or dismissed, non-excluded annotation)
    # We also exclude claims that are dismissed in the claims table
    total_claims = (await annotation_db._fetch(
        "SELECT COUNT(DISTINCT a.claim_id) as count FROM annotations a "
        "JOIN claims c ON a.claim_id = c.id "
        "WHERE a.status IN ('completed', 'dismissed') AND NOT a.excluded AND NOT c.dismissed"
    ))[0]['count']

    # Dismissed claims (distinct claim IDs where all non-excluded annotations are dismissed)
    dismissed_claims = (await annotation_db._fetch(
        "SELECT COUNT(DISTINCT a.claim_id) as count FROM annotations a "
        "JOIN claims c ON a.claim_id = c.id "
        "WHERE a.status = 'dismissed' AND NOT a.excluded AND NOT c.dismissed"
    ))[0]['count']

    completed_claims = (await annotation_db._fetch(
        "SELECT COUNT(DISTINCT a.claim_id) as count FROM annotations a "
        "JOIN claims c ON a.claim_id = c.id "
        "WHERE a.status = 'completed' AND NOT a.excluded AND NOT c.dismissed"
    ))[0]['count']

    # Summary table
    summary = [
        ["Total annotated claims", total_claims],
        ["Claims with completed annotations", completed_claims],
        ["Claims with dismissed annotations", dismissed_claims],
    ]
    print("\nClaim Validation Summary:")
    print(tabulate(summary, headers=["Metric", "Count"], tablefmt="grid"))

    # Dismissed reasons breakdown
    reason_rows = await annotation_db._fetch(
        "SELECT a.dismiss_reason, COUNT(*) as count FROM annotations a "
        "JOIN claims c ON a.claim_id = c.id "
        "WHERE a.status = 'dismissed' AND NOT a.excluded AND NOT c.dismissed "
        "GROUP BY a.dismiss_reason ORDER BY count DESC"
    )

    if reason_rows:
        # Aggregate "Failed manual validation checks" into one row
        manual_check_prefix = "Failed manual validation checks"
        manual_check_total = 0
        check_counts: dict[str, int] = {}
        reasons_table = []

        for r in reason_rows:
            reason = r['dismiss_reason'] or "(no reason)"
            count = r['count']
            if reason.startswith(manual_check_prefix):
                manual_check_total += count
                # Extract individual checks after the colon
                if ":" in reason:
                    checks_part = reason.split(":", 1)[1]
                    for check in checks_part.split(","):
                        check = check.strip()
                        if check:
                            check_counts[check] = check_counts.get(check, 0) + count
            else:
                reasons_table.append([textwrap.fill(reason, width=80), count])

        if manual_check_total > 0:
            reasons_table.insert(0, [manual_check_prefix, manual_check_total])

        print("\nDismissed Reasons:")
        print(tabulate(reasons_table, headers=["Reason", "Count"], tablefmt="grid"))

        # Detailed breakdown of failed manual validation checks
        if check_counts:
            checks_table = [[check, cnt, f"{cnt / total_claims:.1%}"] for check, cnt in sorted(check_counts.items(), key=lambda x: x[1], reverse=True)]
            checks_table.append(["**Total**", manual_check_total, f"{manual_check_total / total_claims:.1%}"])
            print("\nFailed Manual Validation Checks (detailed):")
            print(tabulate(checks_table, headers=["Check", "Count", "% of Annotated Claims"], tablefmt="grid"))
    else:
        print("\nNo dismissed annotations found.")


if __name__ == "__main__":
    asyncio.run(main())
