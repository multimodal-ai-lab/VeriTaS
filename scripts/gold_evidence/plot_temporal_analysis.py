"""Plots for the Gold Evidence temporal analysis.

Consumes the `aggregates.json` written by `run_temporal_analysis.py`, so plotting
never touches the DB and can be re-run offline on an exported result directory.

Example:
    python -m scripts.gold_evidence.plot_temporal_analysis exports/gold_evidence/2026-09-01_10-00-00
"""

import argparse
import json
import os

try:  # Reuse the project's TUDa palette when importable
    from scripts.stats.common import COLORS
except ImportError:  # pragma: no cover - allows plotting from an exported directory
    COLORS = dict(orange="#EC6500", soft_orange="#FFC599", light_orange="#F5A300",
                  darkblue="#004E73", blue="#0083CC", negative="#E03440",
                  neutral="#a1a1a1", positive="#09C479")

FIGURES = ("time_differences", "composition", "recoverability", "rejections")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", help="Directory containing aggregates.json")
    parser.add_argument("--out", default=None,
                        help="Where to write the figures (default: <results_dir>/plots)")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    return parser.parse_args()


def load(results_dir: str) -> dict:
    with open(os.path.join(results_dir, "aggregates.json"), encoding="utf-8") as f:
        return json.load(f)


def plot_time_differences(aggregates: dict, out_dir: str, fmt: str) -> None:
    """Histograms of (t_e - t_c), (t_e - t_f) and (t_f - t_c)."""
    import matplotlib.pyplot as plt

    panels = [
        ("$t_e - t_c$ (days)", "distribution_t_e_minus_t_c", COLORS["orange"]),
        ("$t_e - t_f$ (days)", "distribution_t_e_minus_t_f", COLORS["blue"]),
        ("$t_f - t_c$ (days)", "distribution_t_f_minus_t_c", COLORS["darkblue"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
    for ax, (label, key, color) in zip(axes, panels):
        values = aggregates.get(key, {}).get("values") or []
        if not values:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(values, bins=40, color=color, edgecolor="white", linewidth=0.4)
            ax.axvline(0, color=COLORS["neutral"], linestyle="--", linewidth=1)
            median = aggregates[key]["median"]
            ax.axvline(median, color=COLORS["negative"], linestyle="-", linewidth=1,
                       label=f"median {median:.0f} d")
            ax.legend(frameon=False, fontsize=9)
        ax.set_xlabel(label)
        ax.set_ylabel("evidence items" if key != "distribution_t_f_minus_t_c" else "claims")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Temporal structure of reconstructed gold evidence")
    _save(fig, out_dir, f"time_differences.{fmt}")


def plot_composition(aggregates: dict, out_dir: str, fmt: str) -> None:
    """Source type, proximity, role and modality of the admissible evidence."""
    import matplotlib.pyplot as plt

    panels = [
        ("Source type", aggregates.get("source_kind_distribution", {}), COLORS["orange"]),
        ("Proximity", aggregates.get("source_proximity_distribution", {}), COLORS["light_orange"]),
        ("Role", aggregates.get("role_distribution", {}), COLORS["blue"]),
        ("Modality", {k: v for k, v in aggregates.get("modality_composition", {}).items()
                      if isinstance(v, int)}, COLORS["darkblue"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=300)
    for ax, (title, counts, color) in zip(axes, panels):
        if not counts:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        items = sorted(counts.items(), key=lambda kv: kv[1])
        labels = [k.replace("_", " ") for k, _ in items]
        values = [v for _, v in items]
        ax.barh(labels, values, color=color)
        ax.set_title(title)
        ax.set_xlabel("admissible items")
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir, f"composition.{fmt}")


def plot_recoverability(aggregates: dict, out_dir: str, fmt: str) -> None:
    """The central comparison: recoverability from E_claim vs E_factcheck."""
    import matplotlib.pyplot as plt

    recoverability = aggregates.get("recoverability", {})
    contingency = recoverability.get("contingency", {})
    if not contingency:
        return

    fig, (ax_bar, ax_matrix) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    rates = [recoverability.get("rate_E_claim") or 0.0,
             recoverability.get("rate_E_factcheck") or 0.0]
    ax_bar.bar(["$E_{claim}$", "$E_{factcheck}$"], rates,
               color=[COLORS["blue"], COLORS["orange"]], width=0.55)
    for i, rate in enumerate(rates):
        ax_bar.text(i, rate, f"{rate:.1%}", ha="center", va="bottom")
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("gold verdict recoverable")
    ax_bar.set_title("Recoverability by evidence cutoff")
    ax_bar.spines[["top", "right"]].set_visible(False)

    matrix = [
        [contingency.get("both", 0), contingency.get("only_E_claim", 0)],
        [contingency.get("only_E_factcheck", 0), contingency.get("neither", 0)],
    ]
    image = ax_matrix.imshow(matrix, cmap="Oranges")
    ax_matrix.set_xticks([0, 1], ["recoverable", "not recoverable"])
    ax_matrix.set_yticks([0, 1], ["recoverable", "not recoverable"])
    ax_matrix.set_xlabel("$E_{factcheck}$")
    ax_matrix.set_ylabel("$E_{claim}$")
    for i in range(2):
        for j in range(2):
            ax_matrix.text(j, i, matrix[i][j], ha="center", va="center", fontsize=13)
    p_value = recoverability.get("mcnemar_exact_p")
    title = "Paired outcomes"
    if isinstance(p_value, (int, float)):
        title += f"  (McNemar exact p = {p_value:.3g})"
    ax_matrix.set_title(title)
    fig.colorbar(image, ax=ax_matrix, fraction=0.046)
    _save(fig, out_dir, f"recoverability.{fmt}")


def plot_rejections(aggregates: dict, out_dir: str, fmt: str) -> None:
    """Why evidence candidates and whole instances were rejected."""
    import matplotlib.pyplot as plt

    panels = [
        ("Evidence rejection reasons", aggregates.get("rejection_reasons", {}), COLORS["negative"]),
        ("Instance rejection reasons", aggregates.get("instance_rejection_reasons", {}),
         COLORS["soft_orange"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)
    for ax, (title, counts, color) in zip(axes, panels):
        if not counts:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        items = sorted(counts.items(), key=lambda kv: kv[1])
        ax.barh([k.replace("_", " ") for k, _ in items], [v for _, v in items], color=color)
        ax.set_title(title)
        ax.set_xlabel("count")
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir, f"rejections.{fmt}")


def _save(fig, out_dir: str, filename: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    aggregates = load(args.results_dir)
    out_dir = args.out or os.path.join(args.results_dir, "plots")

    plotters = {
        "time_differences": plot_time_differences,
        "composition": plot_composition,
        "recoverability": plot_recoverability,
        "rejections": plot_rejections,
    }
    for name in args.figures:
        plotters[name](aggregates, out_dir, args.format)


if __name__ == "__main__":
    main()
