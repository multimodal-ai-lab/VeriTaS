from __future__ import annotations
from collections import Counter
from typing import Any

from scripts.stats.common import COLORS
from scripts.stats.quarters.common import aggregate_by_quarter, build_single_chart, Row

def plot_stages(rows: list[Row], q_keys: list[tuple[int, int]]) -> list[tuple[str, Any]]:
    q_keys_stage, q_counts_stage = aggregate_by_quarter(
        rows,
        lambda r: ("Dismissed" if r.dismissed else f"S{r.stage or 0}"),
    )
    # Build category list: all stage labels sorted numerically + optional Dismissed at the end
    all_stage_labels = set(k for c in q_counts_stage.values() for k in c.keys())
    stage_numeric = sorted([s for s in all_stage_labels if s.startswith("S")], key=lambda s: int(s[1:]))
    stage_categories = stage_numeric + (["Dismissed"] if "Dismissed" in all_stage_labels else [])
    
    # align
    q_counts_stage = {k: q_counts_stage.get(k, Counter()) for k in q_keys}

    figs = []
    figs.append(
        build_single_chart(
            "stages",
            q_keys,
            q_counts_stage,
            stage_categories,
            "Reviews per quarter by current processing stage",
            "# Reviews",
            category_colors={
                "S1": COLORS["orange"],
                "S2": COLORS["light_orange"],
                "S3": COLORS["darkblue"],
                "S4": COLORS["blue"],
                "Dismissed": COLORS["neutral"],
            },
        )
    )

    # Extra zoomed stages plot
    stage_only = [s for s in stage_categories if s.startswith("S")]
    stage_desc = sorted(stage_only, key=lambda s: int(s[1:]), reverse=True)
    stage_zoom_categories = stage_desc + (["Dismissed"] if "Dismissed" in stage_categories else [])
    figs.append(
        build_single_chart(
            "stages_zoomed",
            q_keys,
            q_counts_stage,
            stage_zoom_categories,
            "Stage progress (0–1000)",
            "# Reviews",
            category_colors={
                "S1": COLORS["darkblue"],
                "S2": COLORS["blue"],
                "S3": COLORS["soft_blue"],
                "S4": COLORS["soft_light_orange"],
                "S5": COLORS["light_orange"],
                "S6": COLORS["soft_orange"],
                "S7": COLORS["orange"],
                "Dismissed": COLORS["neutral"],
            },
            y_max=1500,
            hline=1000,
        )
    )
    return figs
