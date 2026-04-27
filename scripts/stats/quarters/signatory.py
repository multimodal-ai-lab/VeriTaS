from __future__ import annotations
from collections import Counter
from typing import Any

from scripts.stats.common import COLORS
from scripts.stats.quarters.common import aggregate_by_quarter, build_single_chart, Row

def _signatory_label(r: Row) -> str:
    if r.publisher_name:
        if r.ifcn_status is None and r.efcsn_status is None:
            return "Neither"
        else:
            # Determine signatory combinations explicitly
            has_ifcn = r.ifcn_status in ["active", "in_renewal"]
            has_efcsn = r.efcsn_status == "active"
            if has_ifcn and has_efcsn:
                return "Both"
            if has_ifcn:
                return "IFCN signatory"
            if has_efcsn:
                return "EFCSN member"
            return "Neither"
    else:
        # No publisher assigned, so unknown yet
        return "Unknown"

def plot_signatory(rows: list[Row], q_keys: list[tuple[int, int]]) -> list[tuple[str, Any]]:
    _q_keys_sig, q_counts_sig = aggregate_by_quarter(rows, _signatory_label)
    # align
    q_counts_sig = {k: q_counts_sig.get(k, Counter()) for k in q_keys}
    signatory_categories = ["IFCN signatory", "EFCSN member", "Both", "Neither", "Unknown"]

    return [
        build_single_chart(
            "signatory",
            q_keys,
            q_counts_sig,
            signatory_categories,
            "Reviews per quarter by signatory status",
            "# Reviews",
            category_colors={
                "IFCN signatory": COLORS["orange"],
                "EFCSN member": COLORS["darkblue"],
                "Both": COLORS["light_orange"]
            },
        )
    ]
