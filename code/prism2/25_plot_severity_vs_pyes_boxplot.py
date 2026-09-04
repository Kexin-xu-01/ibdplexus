"""
Boxplot of P(Yes|T=1) against severity quartile groups, one panel per concept.

Severity score (0–3 continuous) is binned into 4 equal-frequency quartiles per concept.
P(Yes|T=1) is recovered from P(Yes|T=2) via:  logodds_T1 = 2 * logit(p_T2)

Output: uncertainty_estimation/severity_vs_pyes_boxplot.png + .pdf
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

BASE = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2_manual_knn/uncertainty_estimation")
OUT  = BASE

SURFACE   = "#fcfcfb"
INK_PRI   = "#0b0b0b"
INK_SEC   = "#52514e"
INK_MUTED = "#898781"
GRID      = "#e1e0d9"
SEQ       = ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab"]   # Q1→Q4 lightblue→darkblue

TERMS = [
    ("inflammation_involvement",               "Inflammation involvement"),
    ("crypt_architectural_distortion",         "Crypt architectural distortion"),
    ("neutrophil_granulocytic_infiltration",   "Neutrophil granulocytic infiltration"),
    ("crypt_abscesses",                        "Crypt abscesses"),
    ("lymphoid_aggregates",                    "Lymphoid aggregates"),
    ("histiocytic_granulomas",                 "Histiocytic granulomas"),
    ("mucin_depletion",                        "Mucin depletion"),
    ("pyloric_gland_metaplasia",               "Pyloric gland metaplasia"),
    ("paneth_cell_metaplasia",                 "Paneth cell metaplasia"),
    ("neuronal_hyperplasia",                   "Neuronal hyperplasia"),
    ("muscular_hypertrophy",                   "Muscular hypertrophy"),
]

Q_LABELS = ["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"]


def p_T2_to_T1(p_T2):
    p_T2 = np.clip(p_T2, 1e-7, 1 - 1e-7)
    logodds = 2 * np.log(p_T2 / (1 - p_T2))
    return 1 / (1 + np.exp(-logodds))


def load_data():
    ts = pd.read_csv(BASE / "temperature_sampling/prism2_temperature_scores.csv")
    sv = pd.read_csv(BASE / "severity/prism2_severity_scores.csv")
    return ts.merge(sv, on="slide")


def make_boxplot_props(color):
    return dict(
        boxprops=dict(color=color, linewidth=1.2),
        medianprops=dict(color=INK_PRI, linewidth=1.8),
        whiskerprops=dict(color=color, linewidth=1.0),
        capprops=dict(color=color, linewidth=1.0),
        flierprops=dict(marker="o", markersize=1.5, markerfacecolor=color,
                        markeredgewidth=0, alpha=0.3),
    )


def main():
    df = load_data()

    ncols = 3
    nrows = 4   # 11 concepts → 12 cells, last empty
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 14), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.05,
                        hspace=0.60, wspace=0.35)

    for idx, (col, label) in enumerate(TERMS):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor(SURFACE)

        p_T1 = p_T2_to_T1(df[f"{col}_p_yes_T"].values)
        sev  = df[f"{col}_severity"].values

        # Quartile bins per concept
        quartiles = pd.qcut(pd.Series(sev), q=4, labels=False, duplicates="drop").values
        n_bins = len(np.unique(quartiles[~np.isnan(quartiles.astype(float))]))

        groups = [p_T1[quartiles == q] for q in range(n_bins)]
        sev_means = [sev[quartiles == q].mean() for q in range(n_bins)]
        counts    = [len(g) for g in groups]

        positions = np.arange(n_bins)
        for q, (grp, color) in enumerate(zip(groups, SEQ[:n_bins])):
            bp = ax.boxplot(
                grp,
                positions=[q],
                widths=0.55,
                patch_artist=True,
                **make_boxplot_props(color),
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.75)

        # Annotate with n and mean severity
        for q, (cnt, sm) in enumerate(zip(counts, sev_means)):
            ax.text(q, -0.06, f"n={cnt:,}\nsev={sm:.2f}",
                    ha="center", va="top", fontsize=6, color=INK_MUTED,
                    transform=ax.get_xaxis_transform())

        ql = Q_LABELS[:n_bins]
        ax.set_xticks(positions)
        ax.set_xticklabels(ql, fontsize=7.5, color=INK_SEC)
        ax.set_ylabel("P(Yes | T=1)", fontsize=7.5, color=INK_SEC)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(label, fontsize=8.5, color=INK_PRI, fontweight="semibold", pad=5)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=7.5)

    # Hide unused panel
    axes[-1][-1].set_visible(False)

    # Legend
    leg_ax = axes[-1][-1]
    leg_ax.set_visible(True)
    leg_ax.set_facecolor(SURFACE)
    for sp in leg_ax.spines.values():
        sp.set_visible(False)
    leg_ax.set_xticks([])
    leg_ax.set_yticks([])
    patches = [mpatches.Patch(facecolor=SEQ[i], alpha=0.75, label=Q_LABELS[i].replace("\n", " "))
               for i in range(4)]
    leg_ax.legend(handles=patches, title="Severity quartile\n(concept-specific)",
                  title_fontsize=8, fontsize=8, loc="center",
                  frameon=False, labelcolor=INK_SEC)
    leg_ax.text(0.5, 0.15,
                "Quartiles split the per-slide severity\nscore (0–3) into 4 equal-frequency\nbins within each concept.",
                ha="center", va="bottom", fontsize=7, color=INK_MUTED,
                transform=leg_ax.transAxes)

    fig.suptitle(
        "P(Yes | T=1) vs severity quartile  ·  PRISM2 manual_knn  ·  n = 3 318 slides",
        fontsize=10, color=INK_SEC, y=0.965,
    )

    for ext in ("pdf", "png"):
        p = OUT / f"severity_vs_pyes_boxplot.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor=SURFACE,
                    **({} if ext == "pdf" else {"dpi": 150}))
        print(f"Saved → {p}")


if __name__ == "__main__":
    main()
