"""
Visualise PRISM2 multi-feature 10-way scoring results.

Two panels:
  Left  — mean P per option across all slides (sorted descending)
  Right — number of slides where each option has the highest probability (winner count)

Output: uncertainty_estimation/multi_feature/multi_feature_summary.png  + .pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = Path(
    "/home/jovyan/kgbk271-ibd-volume/results/prism2_manual_knn/"
    "uncertainty_estimation/multi_feature/prism2_multi_feature_scores.csv"
)
OUT_DIR = CSV_PATH.parent

# Palette
SURFACE   = "#fcfcfb"
INK_PRI   = "#0b0b0b"
INK_SEC   = "#52514e"
INK_MUTED = "#898781"
GRID      = "#e1e0d9"
BLUE      = "#2a78d6"
ORANGE    = "#eb6834"

OPTIONS = [
    ("A", "Inflammation involvement"),
    ("B", "Crypt architectural distortion"),
    ("C", "Neutrophil granulocytic infiltration"),
    ("D", "Crypt abscesses"),
    ("E", "Lymphoid aggregates"),
    ("F", "Histiocytic granulomas"),
    ("G", "Mucin depletion"),
    ("H", "Pyloric gland metaplasia"),
    ("I", "Paneth cell metaplasia"),
    ("J", "Neuronal hyperplasia"),
]

COLS = [f"p_{label.lower().replace(' ', '_')}" for _, label in OPTIONS]
LABELS = [label for _, label in OPTIONS]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default=str(CSV_PATH))
    p.add_argument("--out_dir", default=str(OUT_DIR))
    return p.parse_args()


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


def main():
    args = parse_args()
    df   = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    means   = df[COLS].mean().values
    winners = df[COLS].idxmax(axis=1).map(dict(zip(COLS, range(len(COLS)))))
    winner_counts = np.bincount(winners, minlength=len(COLS))

    # Sort by mean P descending
    order   = np.argsort(means)[::-1]
    s_means  = means[order]
    s_counts = winner_counts[order]
    s_labels = [LABELS[i] for i in order]

    n = len(s_labels)
    y = np.arange(n)
    bar_h = 0.62

    fig, (ax_mean, ax_win) = plt.subplots(
        1, 2, figsize=(13, 5.2), facecolor=SURFACE,
        gridspec_kw={"wspace": 0.55},
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.08)

    # --- Left: mean P ---
    bars = ax_mean.barh(y, s_means, height=bar_h, color=BLUE, alpha=0.88,
                        linewidth=0, zorder=3, clip_on=False)
    ax_mean.set_yticks(y)
    ax_mean.set_yticklabels(s_labels, fontsize=8.5, color=INK_PRI)
    ax_mean.set_xlim(0, max(s_means) * 1.22)
    ax_mean.set_xlabel("Mean P (option selected)")
    for bar, val in zip(bars, s_means):
        ax_mean.text(val + max(s_means) * 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", ha="left",
                     fontsize=7.5, color=INK_SEC)
    ax_mean.set_title("Mean probability per option", fontsize=10,
                      color=INK_PRI, fontweight="semibold", pad=8, loc="left")
    style_ax(ax_mean)
    ax_mean.invert_yaxis()

    # --- Right: winner count ---
    bars2 = ax_win.barh(y, s_counts, height=bar_h, color=ORANGE, alpha=0.88,
                        linewidth=0, zorder=3, clip_on=False)
    ax_win.set_yticks(y)
    ax_win.set_yticklabels(s_labels, fontsize=8.5, color=INK_PRI)
    ax_win.set_xlim(0, max(s_counts) * 1.22)
    ax_win.set_xlabel("Number of slides where option wins")
    for bar, val in zip(bars2, s_counts):
        if val > 0:
            ax_win.text(val + max(s_counts) * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:,}", va="center", ha="left",
                        fontsize=7.5, color=INK_SEC)
    ax_win.set_title("Winner count (highest P per slide)", fontsize=10,
                     color=INK_PRI, fontweight="semibold", pad=8, loc="left")
    style_ax(ax_win)
    ax_win.invert_yaxis()

    fig.suptitle(
        "PRISM2 multi-feature 10-way prompt  ·  manual_knn  ·  n = 3 318 slides",
        fontsize=9.5, color=INK_SEC, y=0.97,
    )

    for ext in ("pdf", "png"):
        p = out_dir / f"multi_feature_summary.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor=SURFACE,
                    **({} if ext == "pdf" else {"dpi": 150}))
        print(f"Saved → {p}")


if __name__ == "__main__":
    main()
