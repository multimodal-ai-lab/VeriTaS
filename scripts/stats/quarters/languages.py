from __future__ import annotations
from collections import Counter
from typing import Any

from scripts.stats.quarters.common import aggregate_by_quarter, select_top_categories, remap_to_top, build_single_chart, Row

def plot_languages(rows: list[Row], q_keys: list[tuple[int, int]], top_languages: int) -> list[tuple[str, Any]]:
    q_keys_lang, q_counts_lang = aggregate_by_quarter(rows, lambda r: (r.language or "(unknown)").lower())
    # align
    q_counts_lang = {k: q_counts_lang.get(k, Counter()) for k in q_keys}

    lang_top = select_top_categories(q_counts_lang, top_languages, other_label="Other")
    q_counts_lang_top = remap_to_top(q_counts_lang, lang_top, other_label="Other")

    return [
        build_single_chart(
            "languages",
            q_keys,
            q_counts_lang_top,
            lang_top,
            "Reviews per quarter by language (top categories)",
            "# Reviews",
        )
    ]
