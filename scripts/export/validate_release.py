import asyncio
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from scripts.stats.common import COLORS, _ensure_plots_dir
from scripts.stats.quarters.common import quarter_key, quarter_label
from veritas.common import Claim
from veritas.common.annotation.rating import Category3Bin
from veritas.db import db


async def validate_release():
    await db.connect_maybe_initialize()

    # Fetch all released claims
    query = "SELECT * FROM claims WHERE (released_quarter = TRUE OR released_longitudinal = TRUE)"
    rows = await db._fetch(query)
    claims = [Claim.model_validate(dict(row)) for row in rows]

    if not claims:
        print("No released claims found.")
        return

    # Group by quarter
    claims_by_q = defaultdict(list)
    for c in claims:
        if c.date:
            qk = quarter_key(c.date)
            claims_by_q[qk].append(c)
        else:
            print(f"Warning: Claim {c.id} has no date.")

    q_keys = sorted(claims_by_q.keys())

    plots_dir = _ensure_plots_dir()
    quarter_stats_dir = os.path.join(plots_dir, "quarter_stats")
    os.makedirs(quarter_stats_dir, exist_ok=True)

    for qk in q_keys:
        label = quarter_label(qk)
        q_claims = claims_by_q[qk]
        n_claims = len(q_claims)

        print(f"\nValidating Quarter {label} ({n_claims} claims):")

        # 1. Check count
        if n_claims == 1000:
            print("  ✅ Exactly 1000 claims")
        else:
            print(f"  ❌ {n_claims} claims (expected 1000)")

        # Gather data for balance and media share
        intact_count = 0
        compromised_count = 0
        nei_count = 0

        has_media_count = 0

        # For bias plot
        intact_with_media = 0
        compromised_with_media = 0

        for c in q_claims:
            v = await c.current_verdict
            cat = v.integrity.as_3_bin()

            mm = c.as_multimodal_sequence()
            has_media = mm.has_images() or mm.has_videos()

            if has_media:
                has_media_count += 1

            if cat == Category3Bin.POSITIVE:
                intact_count += 1
                if has_media:
                    intact_with_media += 1
            elif cat == Category3Bin.NEGATIVE:
                compromised_count += 1
                if has_media:
                    compromised_with_media += 1
            elif cat == Category3Bin.NEUTRAL:
                nei_count += 1

        # 2. Check balance
        if intact_count == compromised_count:
            print(f"  ✅ Balanced: {intact_count} intact, {compromised_count} compromised")
        else:
            print(f"  ❌ Not balanced: {intact_count} intact vs {compromised_count} compromised")
        print(f"      (NEI: {nei_count})")

        # 3. Check media share
        media_share = has_media_count / n_claims
        if media_share <= 0.8:
            print(f"  ✅ Media share: {media_share:.1%} (<= 80%)")
        else:
            print(f"  ❌ Media share: {media_share:.1%} (> 80%)")

        # 4. Plot bias chart
        # We compare the share of media within each group
        intact_media_share = intact_with_media / intact_count if intact_count > 0 else 0
        compromised_media_share = compromised_with_media / compromised_count if compromised_count > 0 else 0

        fig, ax = plt.subplots(figsize=(6, 5))
        groups = ["Intact", "Compromised"]
        shares = [intact_media_share * 100, compromised_media_share * 100]

        bars = ax.bar(groups, shares, color=[COLORS["positive"], COLORS["negative"]])
        ax.set_ylabel("Media Share (%)")
        ax.set_title(f"Media Bias Verification - {label}")
        ax.set_ylim(0, 100)

        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
        plt.close()
        print(f"  ✅ Bias plot generated.")


if __name__ == "__main__":
    asyncio.run(validate_release())
