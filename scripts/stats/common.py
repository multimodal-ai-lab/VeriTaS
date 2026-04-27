from __future__ import annotations
import os

COLORS = dict(
    orange="#EC6500",  # Primary, TUDa
    soft_orange="#FFC599",
    light_orange="#F5A300",  # Secondary, TUDa
    soft_light_orange="#FFCE6B",
    darkblue="#004E73",  # Tertiary, TUDa
    blue="#0083CC",  # Quaternary, TUDa
    soft_blue="#83D3FF",
    negative="#E03440",
    neutral="#a1a1a1",
    positive="#09C479",
)

TITLE_APPEND = {
    "all": "",
    "release": " (VeriTaS release)",
    "natural": " (natural)"
}


def _ensure_plots_dir() -> str:
    out_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_ring_chart(counts_dict, title, filename, colors=None):
    """Helper to plot a ring (donut) chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = list(counts_dict.keys())
    values = list(counts_dict.values())

    if not values or sum(values) == 0:
        print(f"No data for {title}")
        return

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # Sort labels and values for consistent display if colors are provided
    if colors and labels == ["intact", "compromised", "unknown"]:
        # Match colors to fixed order
        order = ["intact", "compromised", "unknown"]
        values = [counts_dict.get(l, 0) for l in order]
        labels = order
    elif colors and labels == ["completed", "dismissed"]:
        order = ["completed", "dismissed"]
        values = [counts_dict.get(l, 0) for l in order]
        labels = order

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.85
    )

    # Draw a white circle at the center to make it a donut
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    ax.axis('equal')
    plt.title(title, fontsize=18)
    plt.tight_layout()

    out_dir = _ensure_plots_dir()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path)
    plt.show()
    plt.close()
    print(f"Saved ring chart: {out_path}")
