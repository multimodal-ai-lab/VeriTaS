from __future__ import annotations

import asyncio
from datetime import datetime, date
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt

from scripts.stats.quarters.authenticity import plot_authenticity
from scripts.stats.quarters.claims import plot_claim_integrity, plot_claim_languages, plot_claim_publishers
from scripts.stats.quarters.common import parse_date, generate_quarter_keys
from scripts.stats.quarters.media import plot_media
from scripts.stats.quarters.platforms import plot_platforms


async def main_async(
        start: str | None,
        end: str | None,
        top_languages: int,
        top_publishers: int,
        save: str | None,
        mode: Literal["all", "release", "natural"] = "all"
) -> None:
    """

    :param start: Beginning of time frame to plot
    :param end: Inclusive end of time frame to plot
    :param top_languages: Number of languages to color in the plot
    :param top_publishers: Number of publishers to color in the plot
    :param save:
    :param mode:
    :return:
    """
    start_d = parse_date(start) or date(2016, 1, 1)
    end_d = parse_date(end) or date.today()

    q_keys = generate_quarter_keys(start_d, end_d)

    if not q_keys:
        print("No quarters found for the given range.")
        return

    # Collect figures from individual plotters
    figs: list[tuple[str, Any]] = []

    # 1) Platforms
    figs.extend(await plot_platforms(start_d, end_d, q_keys, mode=mode))

    # 2) Media
    figs.extend(await plot_media(start_d, end_d, q_keys, mode=mode))

    # 3) Authenticity
    auth_figs = await plot_authenticity(start_d, end_d, q_keys, mode=mode)
    figs.extend(auth_figs)

    # 4) Claims Integrity
    claim_figs = await plot_claim_integrity(start_d, end_d, q_keys, mode=mode)
    figs.extend(claim_figs)

    # 5) Claim Languages
    lang_figs = await plot_claim_languages(start_d, end_d, q_keys, top_languages, mode=mode)
    figs.extend(lang_figs)

    # 6) Claim Publishers
    pub_figs = await plot_claim_publishers(start_d, end_d, q_keys, top_publishers, mode=mode)
    figs.extend(pub_figs)

    if save:
        base = Path(save) / f"quarter_stats/claims/{mode}"
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
            start="2020-01-01",
            end="2026-03-31",
            top_languages=9,
            top_publishers=9,
            save="plots/",
            mode="release",
        )
    )
