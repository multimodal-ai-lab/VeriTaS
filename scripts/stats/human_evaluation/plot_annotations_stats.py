"""Reads all annotations from the database, computes, and plots statistics.

What is plotted:
1) The distribution of tendencies for each single property in a 2x2 grid of subplots
   (six properties: clarity, veracity, context_coverage, intent, authenticity, contextualization).
2) Per property, the distribution of the maximum tendency difference per claim (disagreement).

Run this script manually (requires DB access configured in config/globals.yaml):
    python -m scripts.stats.plot_annotations_stats
"""

import asyncio
import os
from collections import defaultdict, Counter
from statistics import stdev

import numpy as np

from veritas.db.annotation_db import annotation_db
from scripts.stats.plot_verdict_stats import _plot_properties_hist_grid, _ensure_plots_dir, _per_claim_stats
from scripts.stats.common import COLORS, plot_ring_chart
from scripts.stats.human_evaluation.common import value_to_sign, to_tendency
from veritas.db import db


# Properties to consider
CLAIM_LEVEL_PROPERTIES = ["clarity", "veracity", "context_coverage", "intent"]
MEDIA_PROPERTIES = ["media_authenticity", "media_contextualization"]


async def _extract_tendencies() -> tuple[dict[str, list[list[float]]], dict, dict]:
    """Collect raw tendencies per property across all annotations, grouped by claim/media."""
    per_prop: dict[str, list[list[float]]] = defaultdict(list)

    # 1. Fetch claim-level annotations
    query = """
        SELECT a.id, a.claim_id, a.clarity, a.clarity_confidence, a.veracity, a.veracity_confidence, 
               a.context_coverage, a.context_coverage_confidence, a.intent, a.intent_confidence,
               a.intent_tags
        FROM annotations a
        JOIN claims c ON a.claim_id = c.id
        WHERE a.status = 'completed' AND NOT a.excluded AND NOT c.dismissed
    """
    rows = await annotation_db._fetch(query)

    # Group claim-level annotations by claim_id
    ann_by_claim = defaultdict(list)
    for r in rows:
        ann_by_claim[r['claim_id']].append(r)

    for claim_id, anns in ann_by_claim.items():
        for prop in CLAIM_LEVEL_PROPERTIES:
            tendencies = []
            for ann in anns:
                t = to_tendency(prop, ann[prop], ann[f"{prop}_confidence"])
                if t is not None:
                    tendencies.append(t)
            if tendencies:
                per_prop[prop].append(tendencies)

    # 2. Fetch media evaluations
    query = """
        SELECT me.media_id, me.media_authenticity, me.media_authenticity_confidence, 
               me.media_contextualization, me.media_contextualization_confidence,
               a.claim_id
        FROM media_evaluations me
        JOIN annotations a ON me.annotation_id = a.id
        JOIN claims c ON a.claim_id = c.id
        WHERE a.status = 'completed' AND NOT a.excluded AND NOT c.dismissed
    """
    media_rows = await annotation_db._fetch(query)

    # Group media evaluations by (claim_id, media_id)
    media_by_item = defaultdict(list)
    for r in media_rows:
        media_by_item[(r['claim_id'], r['media_id'])].append(r)

    for item_key, rows in media_by_item.items():
        for prop in MEDIA_PROPERTIES:
            tendencies = []
            prop_short = prop.replace("media_", "")
            for r in rows:
                t = to_tendency(prop_short, r[prop], r[f"{prop}_confidence"])
                if t is not None:
                    tendencies.append(t)
            if tendencies:
                per_prop[prop_short].append(tendencies)

    return per_prop, ann_by_claim, media_by_item


def plot_score_dist(per_prop_values: dict[str, list[float]], title: str, filename: str):
    _plot_properties_hist_grid(
        per_prop_values,
        title,
        filename,
        xticks=np.arange(-1, 1.3, step=1/3),
        xlim=(-1.2, 1.2),
        xlabel="Score",
        ylabel="Count",
        gradient_colors=(COLORS["negative"], COLORS["neutral"], COLORS["positive"]),
        neutral_value=0.0,
    )


def plot_disagreement_dist(per_prop_values: dict[str, list[float]], title: str, filename: str):
    _plot_properties_hist_grid(
        per_prop_values,
        title,
        filename,
        xticks=np.arange(0, 2.3, step=1/3),
        xlim=(-0.2, 2.2),
        xlabel="Maximum tendency difference among annotators",
        ylabel="Count",
        gradient_colors=(COLORS["neutral"], COLORS["neutral"], COLORS["orange"]),
        neutral_value=0.0,
    )


async def run():
    await annotation_db.connect_maybe_initialize()
    await db.connect_maybe_initialize()

    # Print summary of annotation counts
    total = (await annotation_db._fetch(
        "SELECT COUNT(a.*) as count FROM annotations a "
        "JOIN claims c ON a.claim_id = c.id "
        "WHERE NOT c.dismissed"
    ))[0]['count']
    excluded = (await annotation_db._fetch(
        "SELECT COUNT(a.*) as count FROM annotations a "
        "JOIN claims c ON a.claim_id = c.id "
        "WHERE a.excluded AND NOT c.dismissed"
    ))[0]['count']
    completed = (await annotation_db._fetch(
        "SELECT COUNT(a.*) as count FROM annotations a "
        "JOIN claims c ON a.claim_id = c.id "
        "WHERE a.status = 'completed' AND NOT a.excluded AND NOT c.dismissed"
    ))[0]['count']
    dismissed = (await annotation_db._fetch(
        "SELECT COUNT(a.*) as count FROM annotations a "
        "JOIN claims c ON a.claim_id = c.id "
        "WHERE a.status = 'dismissed' AND NOT a.excluded AND NOT c.dismissed"
    ))[0]['count']

    print(f"Total annotations: {total}")
    print(f"Excluded annotations: {excluded}")
    print(f"Completed (non-excluded) annotations: {completed}")
    print(f"Dismissed (non-excluded) annotations: {dismissed}")

    per_prop_and_claim_tendencies, ann_by_claim, media_by_item = await _extract_tendencies()

    # 1) Per-property distributions
    def _flatten(nested: list[list]) -> list:
        return [item for sublist in nested for item in sublist]

    per_prop_tendencies = {prop: _flatten(t) for prop, t in per_prop_and_claim_tendencies.items()}
    plot_score_dist(
        per_prop_tendencies,
        title="Distribution of annotation scores",
        filename="annotation_score_distributions.pdf",
    )

    # 2) Disagreement
    per_prop_variances, per_prop_maxdiffs = _per_claim_stats(per_prop_and_claim_tendencies)
    plot_disagreement_dist(
        per_prop_maxdiffs,
        title="Distribution of per-claim annotation disagreement",
        filename="annotation_disagreement_distributions.pdf",
    )

    # 3) Language distribution
    await plot_language_distribution()

    # 4) Integrity distribution
    plot_integrity_distribution(ann_by_claim, media_by_item)

    # 5) Dismissal distribution
    await plot_dismissal_distribution()

    # 6) Annotations per claim
    await plot_annotations_per_claim()


def plot_integrity_distribution(ann_by_claim, media_by_item):
    """Ring chart showing the integrity of the annotated claims."""
    from veritas.common.annotation.rating import Rating, Category3Bin

    integrity_counts = Counter()

    for claim_id, anns in ann_by_claim.items():
        # Calculate aggregated scores for each property
        # Following the logic in veritas/common/verdict.py _determine_integrity
        
        # 1. Check for humor
        # We check if any annotator marked it as humor and the average intent is positive
        intents = []
        is_humor_count = 0
        for ann in anns:
            t = to_tendency("intent", ann["intent"], ann["intent_confidence"])
            if t is not None:
                intents.append(t)
            if ann["intent_tags"] and "Humor" in ann["intent_tags"]:
                is_humor_count += 1
        
        avg_intent = sum(intents) / len(intents) if intents else 0
        is_humor = avg_intent > 0 and is_humor_count >= len(anns) / 2

        if is_humor:
            decisive_score = avg_intent
        else:
            # 2. Check context coverage and veracity
            v_scores = []
            for ann in anns:
                t = to_tendency("veracity", ann["veracity"], ann["veracity_confidence"])
                if t is not None:
                    v_scores.append(t)
            avg_veracity = sum(v_scores) / len(v_scores) if v_scores else 0

            cc_scores = []
            for ann in anns:
                t = to_tendency("context_coverage", ann["context_coverage"], ann["context_coverage_confidence"])
                if t is not None:
                    cc_scores.append(t)
            
            if cc_scores:
                avg_cc = sum(cc_scores) / len(cc_scores)
                # In _determine_integrity: if cc.score < veracity.score: return cc
                if avg_cc < avg_veracity:
                    decisive_score = avg_cc
                else:
                    decisive_score = avg_veracity
            elif v_scores:
                decisive_score = avg_veracity
            else:
                # 3. Fallback to contextualization
                c_scores = []
                for (c_id, m_id), m_anns in media_by_item.items():
                    if c_id == claim_id:
                        for m_ann in m_anns:
                            t = to_tendency("contextualization", m_ann["media_contextualization"], m_ann["media_contextualization_confidence"])
                            if t is not None:
                                c_scores.append(t)
                decisive_score = min(c_scores) if c_scores else 0

        # Map decisive_score to category
        if decisive_score < -1/3:
            integrity_counts["compromised"] += 1
        elif decisive_score > 1/3:
            integrity_counts["intact"] += 1
        else:
            integrity_counts["unknown"] += 1

    plot_ring_chart(
        integrity_counts, 
        title="Integrity of Annotated Claims", 
        filename="annotation_integrity_distribution.pdf",
        colors=[COLORS["negative"], COLORS["neutral"], COLORS["positive"]]
    )


async def plot_dismissal_distribution():
    """Ring chart showing dismissed vs actually annotated claims."""
    # Get total count of completed and dismissed annotations
    query = """
        SELECT a.status, COUNT(*) as count 
        FROM annotations a
        JOIN claims c ON a.claim_id = c.id
        WHERE a.status IN ('completed', 'dismissed')
            AND NOT a.excluded
            AND NOT c.dismissed
        GROUP BY a.status
    """
    rows = await annotation_db._fetch(query)
    counts = {row['status']: row['count'] for row in rows}
    
    plot_ring_chart(
        counts, 
        title="Annotation Dismissal Rate", 
        filename="annotation_dismissal_distribution.pdf",
        colors=[COLORS["negative"], COLORS["positive"]]
    )


async def plot_annotations_per_claim():
    """Stacked bar chart showing the number of annotations per claim, colored by language."""
    # Query annotation counts per claim with language
    query = """
        SELECT annotation_count, language, COUNT(*) as claim_count
        FROM (
            SELECT a.claim_id, COUNT(*) as annotation_count, c.language
            FROM annotations a
            JOIN claims c ON a.claim_id = c.id
            WHERE a.status = 'completed' AND NOT a.excluded AND NOT c.dismissed
            GROUP BY a.claim_id, c.language
        ) subq
        GROUP BY annotation_count, language
        ORDER BY annotation_count, language
    """
    rows = await annotation_db._fetch(query)

    if not rows:
        print("No annotation distribution data available.")
        return

    # Build a mapping: annotation_count -> {language: claim_count}
    count_lang: dict[int, dict[str, int]] = {}
    all_languages: set[str] = set()
    for r in rows:
        ac = r['annotation_count']
        lang = r['language'] or "unknown"
        all_languages.add(lang)
        if ac not in count_lang:
            count_lang[ac] = {}
        count_lang[ac][lang] = r['claim_count']

    counts = sorted(count_lang.keys())
    # Sort languages by total count (descending) for consistent legend order
    lang_totals = {}
    for lang in all_languages:
        lang_totals[lang] = sum(count_lang.get(c, {}).get(lang, 0) for c in counts)
    languages = sorted(all_languages, key=lambda l: lang_totals[l], reverse=True)

    # Assign colors to languages, cycling through available palette
    lang_color_palette = [
        COLORS["orange"], COLORS["blue"], COLORS["light_orange"],
        COLORS["darkblue"], COLORS["positive"], COLORS["negative"],
        COLORS["soft_orange"], COLORS["soft_blue"], COLORS["neutral"],
        COLORS["soft_light_orange"],
    ]
    lang_colors = {lang: lang_color_palette[i % len(lang_color_palette)] for i, lang in enumerate(languages)}

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x_labels = [str(c) for c in counts]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    bottoms = [0] * len(counts)
    for lang in languages:
        values = [count_lang.get(c, {}).get(lang, 0) for c in counts]
        ax.bar(x_labels, values, bottom=bottoms, color=lang_colors[lang], edgecolor="white", label=lang)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    # Add total counts on top of bars
    for i, total in enumerate(bottoms):
        ax.text(i, total + 0.1, f'{int(total)}', ha='center', va='bottom')

    ax.set_title("Number of Annotations per Claim", fontsize=16)
    ax.set_xlabel("Number of Annotators", fontsize=12)
    ax.set_ylabel("Number of Claims", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title="Language")

    out_dir = _ensure_plots_dir()
    out_path = os.path.join(out_dir, "annotation_count_per_claim.pdf")
    plt.savefig(out_path)
    plt.show()
    plt.close()
    print(f"Saved annotations per claim plot: {out_path}")


async def plot_language_distribution():
    """Plot a bar chart of the languages for all annotations."""
    # Fetch all claim languages for completed annotations
    query = """
        SELECT c.language, COUNT(a.id) as count
        FROM annotations a
        JOIN claims c ON a.claim_id = c.id
        WHERE a.status = 'completed' AND NOT a.excluded AND NOT c.dismissed
        GROUP BY c.language
        ORDER BY count DESC
    """
    rows = await db._fetch(query)

    if not rows:
        print("No language data available for annotations.")
        return

    languages = [row['language'] or "unknown" for row in rows]
    counts = [row['count'] for row in rows]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required to generate plots.")
        return

    plt.figure(figsize=(10, 6), dpi=300)
    bars = plt.bar(languages, counts, color=COLORS["orange"], edgecolor="white")
    
    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')

    plt.title("Language Distribution for Annotations", fontsize=16)
    plt.xlabel("Language", fontsize=12)
    plt.ylabel("Number of Annotations", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    out_dir = _ensure_plots_dir()
    out_path = os.path.join(out_dir, "annotation_language_distribution.pdf")
    plt.savefig(out_path)
    plt.show()
    plt.close()
    print(f"Saved language distribution plot: {out_path}")


if __name__ == "__main__":
    asyncio.run(run())
