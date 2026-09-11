"""
Visualise PRISM2 temperature-sampling uncertainty scores.

Produces a two-panel figure:
  Left  — horizontal bar chart of mean entropy per concept (model uncertainty)
  Right — horizontal bar chart of mean P(Yes|T=2) per concept (prevalence)

Concepts are sorted descending by entropy so the most uncertain sit at the top.

Output: <out_dir>/temperature_scores_summary.pdf  (+ .png at 150 dpi)

Usage:
  python 18_plot_temperature_scores.py
  python 18_plot_temperature_scores.py --csv <path> --out_dir <dir>
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_PATH = Path(
    "/home/jovyan/kgbk271-ibd-volume/results/"
    "prism2_manual_knn/uncertainty_estimation/temperature_sampling/"
    "prism2_temperature_scores.csv"
)
OUT_DIR = CSV_PATH.parent

# ---------------------------------------------------------------------------
# Palette (reference palette from dataviz skill)
# ---------------------------------------------------------------------------
SURFACE   = "#fcfcfb"
INK_PRI   = "#0b0b0b"
INK_SEC   = "#52514e"
INK_MUTED = "#898781"
GRID      = "#e1e0d9"
BLUE      = "#2a78d6"   # sequential slot 1 — entropy
ORANGE    = "#eb6834"   # sequential slot 2 — P(Yes)

UAMP_TERMS = [
    "Inflammation involvement",
    "Crypt architectural distortion",
    "Neutrophil granulocytic infiltration",
    "Crypt abscesses",
    "Lymphoid aggregates",
    "Histiocytic granulomas",
    "Mucin depletion",
    "Pyloric gland metaplasia",
    "Paneth cell metaplasia",
    "Neuronal hyperplasia",
    "Muscular hypertrophy",
]

def col(term, suffix):
    return term.lower().replace(" ", "_") + "_" + suffix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default=str(CSV_PATH))
    p.add_argument("--out_dir", default=str(OUT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-concept means
    means = {}
    for term in UAMP_TERMS:
        means[term] = {
            "entropy": df[col(term, "entropy")].mean(),
            "p_yes":   df[col(term, "p_yes_T")].mean(),
        }

    # Sort by entropy descending
    sorted_terms = sorted(UAMP_TERMS, key=lambda t: means[t]["entropy"], reverse=True)
    entropies = [means[t]["entropy"] for t in sorted_terms]
    pyes      = [means[t]["p_yes"]   for t in sorted_terms]

    # Short labels for y-axis
    labels = sorted_terms  # full names fit in a horizontal bar

    n = len(sorted_terms)
    y = np.arange(n)

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig, (ax_ent, ax_py) = plt.subplots(
        1, 2,
        figsize=(12, 5.2),
        facecolor=SURFACE,
        gridspec_kw={"wspace": 0.55},
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.08)

    bar_h = 0.62

    def style_ax(ax):
        ax.set_facecolor(SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5)
        ax.xaxis.label.set_color(INK_SEC)
        ax.xaxis.label.set_size(9)
        ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    # --- Left: entropy ---
    bars_e = ax_ent.barh(
        y, entropies, height=bar_h,
        color=BLUE, alpha=0.88,
        linewidth=0, zorder=3,
        clip_on=False,
    )
    ax_ent.set_yticks(y)
    ax_ent.set_yticklabels(labels, fontsize=8.5, color=INK_PRI)
    ax_ent.set_xlim(0, 1.08)
    ax_ent.set_xlabel("Mean entropy (bits, max = 1.0)")
    ax_ent.axvline(1.0, color=GRID, linewidth=0.8, linestyle="--", zorder=1)
    # Direct labels
    for bar, val in zip(bars_e, entropies):
        ax_ent.text(
            val + 0.015, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left",
            fontsize=7.5, color=INK_SEC,
        )
    ax_ent.set_title("Model uncertainty", fontsize=10, color=INK_PRI,
                     fontweight="semibold", pad=8, loc="left")
    style_ax(ax_ent)
    ax_ent.invert_yaxis()

    # --- Right: P(Yes) ---
    bars_p = ax_py.barh(
        y, pyes, height=bar_h,
        color=ORANGE, alpha=0.88,
        linewidth=0, zorder=3,
        clip_on=False,
    )
    ax_py.set_yticks(y)
    ax_py.set_yticklabels(labels, fontsize=8.5, color=INK_PRI)
    ax_py.set_xlim(0, 0.85)
    ax_py.set_xlabel("Mean P(Yes | T = 2)")
    ax_py.axvline(0.5, color=GRID, linewidth=0.8, linestyle="--", zorder=1)
    # Direct labels
    for bar, val in zip(bars_p, pyes):
        ax_py.text(
            val + 0.010, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left",
            fontsize=7.5, color=INK_SEC,
        )
    ax_py.set_title("Prevalence score", fontsize=10, color=INK_PRI,
                    fontweight="semibold", pad=8, loc="left")
    style_ax(ax_py)
    ax_py.invert_yaxis()

    # --- Suptitle ---
    fig.suptitle(
        "PRISM2 temperature-scaled histological scores  ·  manual_knn  ·  T = 2.0  ·  n = 3 318 slides",
        fontsize=9.5, color=INK_SEC, y=0.97,
    )

    pdf_path = out_dir / "temperature_scores_summary.pdf"
    png_path = out_dir / "temperature_scores_summary.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved → {pdf_path}")
    print(f"Saved → {png_path}")


if __name__ == "__main__":
    main()
