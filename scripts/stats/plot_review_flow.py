"""Plot a Sankey diagram of review flows from Stage 1 to Stage 6.

The flows are grouped by dismissal reason at each stage and the dismissed
flows are labeled with wrapped reason text for readability.

This script reuses DB access patterns from print_review_stats.py.
"""

from __future__ import annotations

import asyncio
import textwrap
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sankeyflow import Sankey

from veritas.db import db

# --- Optional static configuration for date filtering ---
# Set START and END to strings (YYYY-MM-DD) or date objects to filter reviews by
# their published timestamp in the half-open interval [START, END).
# Leave as None to include all available reviews.
START: str | date | None = "2016-01-01"
END: str | date | None = "2025-12-31"


def _to_date(d: str | date | None) -> date | None:
    if d is None:
        return None
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


CMAP = LinearSegmentedColormap.from_list("mycmap", ["#EC6500", "#F5A300"])


async def _fetch_rows(start: date | None = None, end: date | None = None) -> list[dict[str, Any]]:
    """Fetch minimal fields for building stage flows.

    Returns a list of dicts with keys: id, stage, dismissed, dismissed_reason.
    """
    await db.connect_maybe_initialize()
    where = []
    params: list[object] = []
    if start is not None:
        where.append("published >= $%d" % (len(params) + 1))
        params.append(datetime.combine(start, datetime.min.time()))
    if end is not None:
        where.append("published < $%d" % (len(params) + 1))
        params.append(datetime.combine(end, datetime.min.time()))

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    query = f"""
        SELECT id,
               COALESCE(stage, 0) AS stage,
               COALESCE(dismissed, FALSE) AS dismissed,
               dismissed_reason
        FROM reviews{where_sql}
    """
    rows = await db._fetch(query, *params)
    return [
        {
            "id": int(r["id"]),
            "stage": int(r["stage"]) if r["stage"] is not None else 0,
            "dismissed": bool(r["dismissed"]),
            "dismissed_reason": r["dismissed_reason"],
        }
        for r in rows
    ]


def _trim_reason(reason: str | None, max_len: int = 60) -> str:
    if not reason:
        return "(Unspecified)"
    # Keep only the first line and trim excessive length, then wrap to narrow width
    text = reason.splitlines()[0]
    if len(text) > max_len:
        text = text[: max_len] + "…"
    # Wrap for better label width (about 18 chars per line, up to 3 lines)
    wrapped = textwrap.fill(
        text,
        width=18,
        max_lines=3,
        placeholder="…",
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped


def _build_sankey(rows: list[dict[str, Any]]) -> tuple[list[list[tuple[str, int]]], list[tuple]]:
    """Build nodes and flows for sankeyflow from Stage 1 to 6.

    - Columns 1..6: Stage k plus dismissal reason nodes for dismissals at stage k
    - Flows:
        Stage s -> Stage s+1 (kept), for s in 1..5
        Stage s -> Dismissed reason at stage s+1, for s in 1..5 (thus dismissal nodes for k in 2..6)
    """
    if not rows:
        return [], []

    # Pre-compute counts
    # count reaching each stage k (>= k)
    reach = {k: 0 for k in range(0, 7)}
    for k in range(0, 7):
        reach[k] = sum(1 for r in rows if (r.get("stage") or 0) >= k and not r.get("dismissed", False))

    # Dismissals grouped by stage k (1..6) and reason
    dismissals: dict[int, Counter[str]] = defaultdict(Counter)
    for r in rows:
        st = int(r.get("stage") or 0)
        if r.get("dismissed", False):
            # Dismissed at stage (st + 1) per print_review_stats convention
            k = st + 1
            if 1 <= k <= 6:
                dismissals[k][_trim_reason(r.get("dismissed_reason"))] += 1

    # Define color map for primary nodes (non-dismissed reviews)
    cm = LinearSegmentedColormap.from_list("mai_cmap", ["#EC6500", "#F5A300"])
    colors = cm(np.linspace(0, 1, 7))

    # Build nodes per column
    nodes: list[list[tuple[str, int, dict]]] = []

    # Columns 1..6
    for k in range(1, 7):
        col_nodes: list[tuple[str, int, dict]] = []
        # Stage k node value = number that reached k (i.e., kept from k-1)
        kept_k = sum(1 for r in rows if (r.get("stage") or 0) >= k)
        col_nodes.append((f"Stage {k}", kept_k, dict(color=colors[k])))
        # Dismissal nodes for stage k (only for k >= 2, since Stage 0 column is removed)
        if k >= 2:
            dismissals_stage = dismissals.get(k)
            if dismissals_stage:
                entrants_k = kept_k  # number that entered stage k
                threshold = 0.01 * entrants_k
                major: list[tuple[str, int]] = []
                other_count = 0
                for reason, cnt in dismissals_stage.most_common():
                    if cnt < threshold:
                        other_count += cnt
                    else:
                        major.append((reason, cnt))
                for reason, cnt in major:
                    col_nodes.append((f"S{k} {reason}", cnt, dict(color="#DDDDDD")))
                if other_count > 0:
                    col_nodes.append((f"S{k} Other", other_count, dict(color="#DDDDDD")))
        nodes.append(col_nodes)

    # Build flows
    flows = []
    # From Stage s to Stage s+1 (kept), s in 1..5
    for s in range(1, 6):
        kept_next = sum(1 for r in rows if (r.get("stage") or 0) >= (s + 1))
        if kept_next:
            flows.append((f"Stage {s}", f"Stage {s + 1}", kept_next))
        # Dismissals at stage s+1 (k), only for k >= 2
        k = s + 1
        dismissals_stage = dismissals.get(k, {})
        if dismissals_stage:
            entrants_k = sum(1 for r in rows if (r.get("stage") or 0) >= k)
            threshold = 0.01 * entrants_k
            other_count = 0
            for reason, cnt in dismissals_stage.items():
                if cnt < threshold:
                    other_count += cnt
                else:
                    flows.append((f"Stage {s}", f"S{k} {reason}", cnt))
            if other_count > 0:
                flows.append((f"Stage {s}", f"S{k} Other", other_count))

    return nodes, flows


async def main() -> None:
    start = _to_date(START)
    end = _to_date(END)
    rows = await _fetch_rows(start, end)
    nodes, flows = _build_sankey(rows)

    if not nodes or not flows:
        print("No data available to plot review flow.")
        return

    plt.figure(figsize=(20, 10), dpi=144)
    s = Sankey(flows=flows, nodes=nodes, node_width=0.1, flow_color_mode="dest")
    s.draw()
    plt.title("Review processing flow (Stage 1 → 6) with dismissal reasons", fontsize=18)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
