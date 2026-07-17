import asyncio
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from veritas.db import db
from scripts.stats.common import COLORS, _ensure_plots_dir
import os

async def main():
    # Connect to the database
    await db.connect_maybe_initialize()

    # Query to get claim date and the corresponding review's date (published)
    # We join claims and reviews. Since a claim can have multiple reviews, 
    # and the prompt says "for all claims, computes the difference between claim date 
    # and the corresponding review's date", it might mean we should consider all claim-review pairs
    # or just one review per claim. Usually, "the corresponding review" in this context
    # might refer to the reviews linked to the claim.
    
    query = """
        SELECT c.id as claim_id, c.data as claim_statement, c.dismissed as claim_dismissed, c.date as claim_date, r.published as review_date
        FROM claims c
        JOIN reviews r ON r.id = ANY(c.review_ids)
        WHERE c.date IS NOT NULL AND r.published IS NOT NULL
    """
    
    rows = await db._fetch(query)
    
    if not rows:
        print("No claims with valid dates found.")
        return

    diff_data = []
    skipped_zero = 0
    skipped_negative = 0
    for row in rows:
        c_date = row['claim_date']
        r_date = row['review_date']
        
        # Ensure both are datetime objects
        if isinstance(c_date, datetime) and isinstance(r_date, datetime):
            diff = r_date - c_date
            # Difference in days (as float)
            diff_days = diff.total_seconds() / (24 * 3600)
            
            # 0 often means raw_claim_date was missing and defaulted to published.
            if diff_days == 0:
                skipped_zero += 1
                continue
            if diff_days < 0:
                skipped_negative += 1
                continue
                
            diff_data.append({
                'id': row['claim_id'],
                'statement': row['claim_statement'],
                'dismissed': row['claim_dismissed'],
                'diff_days': diff_days
            })

    if not diff_data:
        print("No valid positive date differences computed.")
        return

    diffs = np.array([d['diff_days'] for d in diff_data])
    avg_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    median_diff = np.median(diffs)

    # Convert average and median to days and hours
    avg_days = int(abs(avg_diff))
    avg_hours = int((abs(avg_diff) - avg_days) * 24)
    avg_sign = "-" if avg_diff < 0 else ""
    
    med_days = int(abs(median_diff))
    med_hours = int((abs(median_diff) - med_days) * 24)
    med_sign = "-" if median_diff < 0 else ""

    # Calculate statistics for differences > 30 days
    diffs_gt_30 = diffs[diffs > 30]
    count_gt_30 = len(diffs_gt_30)
    percent_gt_30 = (count_gt_30 / len(diffs)) * 100

    print(f"Number of claim-review pairs analyzed: {len(diffs)}")
    print(f"Skipped 0-day differences (likely missing raw_claim_date): {skipped_zero}")
    print(f"Skipped negative differences: {skipped_negative}")
    print(f"Differences > 30 days: {count_gt_30} ({percent_gt_30:.2f}%)")
    print(f"Average difference: {avg_sign}{avg_days} days and {avg_hours} hours")
    print(f"Median difference: {med_sign}{med_days} days and {med_hours} hours")
    print(f"Standard deviation: {std_diff:.2f} days")

    # Print top 20 claims with highest difference
    print("\nTop 20 claims with highest date difference:")
    print(f"{'ID':<10} | {'Diff (Days)':<12} | {'Dismissed':<10} | {'Statement'}")
    print("-" * 100)
    
    # Sort by diff_days descending
    top_20 = sorted(diff_data, key=lambda x: x['diff_days'], reverse=True)[:20]
    for item in top_20:
        statement_short = (item['statement'][:80] + '...') if len(item['statement']) > 80 else item['statement']
        # Clean up statement from newlines for better console printing
        statement_short = statement_short.replace('\n', ' ').replace('\r', '')
        print(f"{item['id']:<10} | {item['diff_days']:<12.2f} | {str(item['dismissed']):<10} | {statement_short}")
    print()

    # Filter data for histogram (0-30 days)
    diffs_filtered = diffs[diffs <= 30]

    # Plotting
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Calculate weights to show percentage on y-axis
    # Note: Using len(diffs) for total count to represent % of overall dataset (including those > 30)
    weights = np.ones_like(diffs_filtered) / len(diffs) * 100
    
    # Use 30 bins for 0-30 days range (one bar per day)
    plt.hist(diffs_filtered, bins=30, weights=weights, color=COLORS['blue'], edgecolor='black', alpha=0.7)
    plt.axvline(avg_diff, color=COLORS['orange'], linestyle='dashed', linewidth=2, label=f'Mean: {avg_sign}{avg_days}d {avg_hours}h')
    plt.axvline(median_diff, color='green', linestyle='dotted', linewidth=2, label=f'Median: {med_sign}{med_days}d {med_hours}h')
    
    plt.title("Distribution of Differences (0-30 days)", fontsize=14)
    plt.xlabel("Difference (Days)", fontsize=12)
    plt.ylabel("Percentage of Total Dataset (%)", fontsize=12)
    plt.xlim(0, 30)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    out_dir = _ensure_plots_dir()
    out_path = os.path.join(out_dir, "claim_review_date_diff.png")
    plt.savefig(out_path)
    print(f"Histogram saved to: {out_path}")
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())
