from __future__ import annotations
from collections import Counter
from typing import Any

from scripts.stats.common import COLORS
from scripts.stats.quarters.common import aggregate_by_quarter, select_top_categories, remap_to_top, build_single_chart, Row

def plot_publishers(rows: list[Row], q_keys: list[tuple[int, int]], top_publishers: int) -> list[tuple[str, Any]]:
    q_keys_pub, q_counts_pub = aggregate_by_quarter(rows, lambda r: (r.publisher_name or "(unknown)"))
    # align
    q_counts_pub = {k: q_counts_pub.get(k, Counter()) for k in q_keys}
    pub_top = select_top_categories(q_counts_pub, top_publishers, other_label="Other")
    q_counts_pub_top = remap_to_top(q_counts_pub, pub_top, other_label="Other")

    return [
        build_single_chart(
            "publishers",
            q_keys,
            q_counts_pub_top,
            pub_top,
            "Reviews per quarter by publisher",
            "# Reviews",
            category_colors={
                cat: color
                for cat, color in zip(
                    [c for c in pub_top if c != "Other"],
                    [
                        COLORS["orange"],
                        COLORS["light_orange"],
                        COLORS["darkblue"],
                        COLORS["blue"],
                    ],
                )
            },
        )
    ]
