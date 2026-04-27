import asyncio
from collections import Counter
from tabulate import tabulate
from veritas.db import db
from veritas.common import Claim

async def print_rectified_stats():
    await db.connect()
    
    # Fetch all rectified claims
    query = "SELECT * FROM claims WHERE is_rectified = TRUE"
    rows = await db._fetch(query)
    rectified_claims = [Claim.model_validate(dict(row)) for row in rows]
    
    total = len(rectified_claims)
    dismissed = [c for c in rectified_claims if c.dismissed]
    non_dismissed = [c for c in rectified_claims if not c.dismissed]
    
    validated_active = [c for c in non_dismissed if c.check_completed]
    not_validated = [c for c in non_dismissed if not c.check_completed]
    
    print(f"Summary of Rectified Claims")
    print(f"===========================")
    print(f"Total rectified claims:        {total}")
    print(f"Dismissed:                     {len(dismissed)}")
    print(f"Validated & Active:            {len(validated_active)}")
    print(f"Not yet validated:             {len(not_validated)}")
    print()
    
    if dismissed:
        print(f"Reasons for Dismissal (of {len(dismissed)} claims)")
        print(f"-------------------------------------------")
        
        inconsistent = [c for c in dismissed if c.is_inconsistent]
        unshareable = [c for c in dismissed if c.is_unshareable]
        missing_media = [c for c in dismissed if c.missing_referenced_media]
        too_long = [c for c in dismissed if c.dismissed_reason and "too long" in c.dismissed_reason.lower()]
        other = [c for c in dismissed if not (c.is_inconsistent or c.is_unshareable or c.missing_referenced_media or (c.dismissed_reason and "too long" in c.dismissed_reason.lower()))]
        
        rows_data = [
            ["Inconsistent", len(inconsistent), f"{len(inconsistent)/len(dismissed):.1%}", f"{len(inconsistent)/total:.1%}"],
            ["Unshareable", len(unshareable), f"{len(unshareable)/len(dismissed):.1%}", f"{len(unshareable)/total:.1%}"],
            ["Missing referenced media", len(missing_media), f"{len(missing_media)/len(dismissed):.1%}", f"{len(missing_media)/total:.1%}"],
            ["Too long", len(too_long), f"{len(too_long)/len(dismissed):.1%}", f"{len(too_long)/total:.1%}"],
        ]
        
        if other:
            # Count individual "other" reasons
            other_reasons = Counter(c.dismissed_reason or "Unknown" for c in other)
            for reason, count in other_reasons.most_common():
                reason_display = reason.split("\n")[0]
                rows_data.append([reason_display, count, f"{count/len(dismissed):.1%}", f"{count/total:.1%}"])
        
        print(tabulate(rows_data, headers=["Reason", "Count", "% of Dismissed", "% of Total"], tablefmt="simple"))

    await db.close()

if __name__ == "__main__":
    asyncio.run(print_rectified_stats())
