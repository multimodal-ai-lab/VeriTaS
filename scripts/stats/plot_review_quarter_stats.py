from __future__ import annotations

import asyncio
from datetime import datetime, date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from scripts.stats.quarters.common import parse_date, fetch_review_rows, generate_quarter_keys
from scripts.stats.quarters.languages import plot_languages
from scripts.stats.quarters.publishers import plot_publishers
from scripts.stats.quarters.signatory import plot_signatory
from scripts.stats.quarters.stages import plot_stages


async def main_async(
        start: str | None,
        end: str | None,
        top_languages: int,
        top_publishers: int,
        save: str | None,
) -> None:
    start_d = parse_date(start) or date(2016, 1, 1)
    end_d = parse_date(end) or date.today()
    rows = await fetch_review_rows(start_d, end_d)

    # Drop rows that lie in the future to avoid plotting future quarters
    if rows:
        now_ref = datetime.now(rows[0].published.tzinfo) if rows[0].published.tzinfo else datetime.now()
        rows = [r for r in rows if r.published <= now_ref]
    if not rows:
        print("No rows found for the given filters.")
        return

    # Initial quarter keys from reviews
    q_keys = generate_quarter_keys(start_d, end_d)

    # Collect figures from individual plotters
    figs: list[tuple[str, Any]] = []

    # 1) Review Stages
    figs.extend(plot_stages(rows, q_keys))

    # 2) Review Languages
    figs.extend(plot_languages(rows, q_keys, top_languages))

    # 3) Review Publishers
    figs.extend(plot_publishers(rows, q_keys, top_publishers))

    # 4) Review Signatory
    figs.extend(plot_signatory(rows, q_keys))

    if save:
        base = Path(save) / f"quarter_stats/reviews"
        base.mkdir(exist_ok=True, parents=True)
        for suffix, f in figs:
            out_path = base / f"{suffix}.pdf"
            f.savefig(out_path, dpi=300)
            print(f"Saved figure to {out_path}")
    # Show all figures
    plt.show()
    plt.close('all')


if __name__ == "__main__":
    asyncio.run(
        main_async(
            start="2016-01-01",
            end="2026-06-30",
            top_languages=8,
            top_publishers=15,
            save="plots/",
        )
    )
