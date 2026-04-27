import asyncio
from typing import Sequence

import matplotlib.pyplot as plt

from scripts.stats.common import COLORS
from veritas.db import db


async def fetch_cr_origins() -> list[tuple[bool, bool, bool]]:
    """Fetch which sources each review has."""
    await db.connect_maybe_initialize()
    query = """
            SELECT (google_claim_review IS NOT NULL)      as has_google,
                   (datacommons_claim_review IS NOT NULL) as has_datacommons,
                   (direct_claim_review IS NOT NULL)      as has_direct
            FROM reviews;
            """
    records: Sequence = await db.fetch(query)  # type: ignore[attr-defined]
    return [
        (bool(r["has_google"]), bool(r["has_datacommons"]), bool(r["has_direct"]))
        for r in records
    ]


async def async_main(save: str | None = None, show: bool = True):
    data = await fetch_cr_origins()

    if not data:
        print("No reviews found in the database.")
        return

    # Count sets
    google_only = 0
    datacommons_only = 0
    direct_only = 0
    google_datacommons = 0
    google_direct = 0
    datacommons_direct = 0
    all_three = 0

    for g, dc, d in data:
        if g and dc and d:
            all_three += 1
        elif g and dc:
            google_datacommons += 1
        elif g and d:
            google_direct += 1
        elif dc and d:
            datacommons_direct += 1
        elif g:
            google_only += 1
        elif dc:
            datacommons_only += 1
        elif d:
            direct_only += 1

    # Total counts for each source
    total_google = google_only + google_datacommons + google_direct + all_three
    total_dc = datacommons_only + google_datacommons + datacommons_direct + all_three
    total_direct = direct_only + google_direct + datacommons_direct + all_three

    print(f"Total Reviews: {len(data):,}")
    print(f"Google: {total_google:,}")
    print(f"DataCommons: {total_dc:,}")
    print(f"Direct: {total_direct:,}")
    print(f"Intersection All: {all_three:,}")

    # Plotting using matplotlib-venn if available, otherwise fallback
    try:
        from matplotlib_venn import venn3

        plt.figure(figsize=(10, 8), dpi=300)
        v = venn3(subsets=(
            google_only,
            datacommons_only,
            google_datacommons,
            direct_only,
            google_direct,
            datacommons_direct,
            all_three
        ), set_labels=('Google Fact-Check Explorer', 'DataCommons', 'Direct Download'),
            subset_label_formatter=lambda x: f"{x:,}")

        # Set colors from common.py
        # Google: blue, DataCommons: orange, Direct: positive (green)
        # Set alpha to 1.0 to avoid washed-out colors
        for patch_id in ['100', '010', '001', '110', '101', '011', '111']:
            patch = v.get_patch_by_id(patch_id)
            if patch:
                patch.set_alpha(1.0)

        if v.get_patch_by_id('100'): v.get_patch_by_id('100').set_color(COLORS["blue"])
        if v.get_patch_by_id('010'): v.get_patch_by_id('010').set_color(COLORS["orange"])
        if v.get_patch_by_id('001'): v.get_patch_by_id('001').set_color(COLORS["neutral"])

        # Intersections
        if v.get_patch_by_id('110'): v.get_patch_by_id('110').set_color(COLORS["light_orange"])
        if v.get_patch_by_id('101'): v.get_patch_by_id('101').set_color(COLORS["soft_blue"])
        if v.get_patch_by_id('011'): v.get_patch_by_id('011').set_color(COLORS["soft_orange"])
        if v.get_patch_by_id('111'): v.get_patch_by_id('111').set_color(COLORS["soft_light_orange"])

        plt.title("Origin of ClaimReviews")

        if save:
            plt.savefig(save)
            print(f"Saved Venn diagram to {save}")
    except ImportError:
        print("matplotlib-venn not installed. Please install it using 'pip install matplotlib-venn'.")
        # Simple bar chart as fallback
        labels = ['Google Explorer', 'DataCommons', 'Direct Download']
        counts = [total_google, total_dc, total_direct]
        colors = [COLORS["blue"], COLORS["orange"], COLORS["neutral"]]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, counts, color=colors)
        plt.title("Number of ClaimReviews per Source")
        plt.ylabel("Count")

        # Add commas to bar labels
        plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height):,}', ha='center', va='bottom')

        plt.tight_layout()

        if save:
            plt.savefig(save)
            print(f"Saved fallback bar chart to {save}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    asyncio.run(async_main(save="plots/cr_origin.pdf", show=True))


if __name__ == "__main__":
    main()
