"""
4-panel comparison figure across all uncertainty estimation experiments.

Panel A (top-left)   — Prevalence: P(Yes) from 3 methods side by side
Panel B (top-right)  — Severity: stacked bars P(mild/moderate/severe)
Panel C (bottom-left)— Uncertainty heatmap: 3 measures × 11 concepts (normalised within measure)
Panel D (bottom-right)— Robustness scatter: mean P(Yes) vs SD across phrasings

Output: uncertainty_estimation/uncertainty_comparison.png + .pdf
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

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
SURFACE   = "#fcfcfb"
INK_PRI   = "#0b0b0b"
INK_SEC   = "#52514e"
INK_MUTED = "#898781"
GRID      = "#e1e0d9"
# Categorical slots
BLUE   = "#2a78d6"
ORANGE = "#eb6834"
AQUA   = "#1baf7a"
# Sequential blue ramp (for severity/heatmap)
SEQ = ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]  # steps 100,300,450,550,700

TERMS = [
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

def c(t): return t.lower().replace(" ", "_")


def load_data():
    ts = pd.read_csv(BASE / "temperature_sampling/prism2_temperature_scores.csv")
    mc = pd.read_csv(BASE / "multiple_choice/prism2_mc_scores.csv")
    sv = pd.read_csv(BASE / "severity/prism2_severity_scores.csv")
    rb = pd.read_csv(BASE / "question_robustness/prism2_robustness_scores.csv")
    rows = []
    for t in TERMS:
        col = c(t)
        rows.append({
            "term":       t,
            "ts_pyes":    ts[f"{col}_p_yes_T"].mean(),
            "mc_pterm":   mc[f"{col}_p_term"].mean(),
            "rb_mean":    rb[f"{col}_mean"].mean(),
            "rb_sd":      rb[f"{col}_sd"].mean(),
            "entropy":    ts[f"{col}_entropy"].mean(),
            "p_not_sure": mc[f"{col}_p_not_sure"].mean(),
            "p_mild":     sv[f"{col}_p_mild"].mean(),
            "p_moderate": sv[f"{col}_p_moderate"].mean(),
            "p_severe":   sv[f"{col}_p_severe"].mean(),
            "severity":   sv[f"{col}_severity"].mean(),
        })
    return pd.DataFrame(rows)


def style_ax(ax, x_label=None):
    ax.set_facecolor(SURFACE)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    if x_label:
        ax.set_xlabel(x_label, color=INK_SEC, fontsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_a(ax, df):
    """Grouped horizontal bars — prevalence from 3 methods."""
    order = df.sort_values("ts_pyes", ascending=False)
    labels = order["term"].tolist()
    n  = len(labels)
    y  = np.arange(n)
    h  = 0.24
    gap = 0.28

    ax.barh(y + gap,  order["ts_pyes"],   height=h, color=BLUE,   alpha=0.88, linewidth=0, zorder=3, label="Temp. sampling P(Yes|T=2)")
    ax.barh(y,        order["mc_pterm"],  height=h, color=ORANGE, alpha=0.88, linewidth=0, zorder=3, label="Multiple choice P(term)")
    ax.barh(y - gap,  order["rb_mean"],   height=h, color=AQUA,   alpha=0.88, linewidth=0, zorder=3, label="Robustness mean P(Yes)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK_PRI)
    ax.set_xlim(0, 1.15)
    ax.axvline(0.5, color=GRID, linewidth=0.8, linestyle="--", zorder=1)
    ax.legend(fontsize=7, loc="lower right", frameon=False, labelcolor=INK_SEC)
    ax.set_title("A   Prevalence across prompting methods", fontsize=9.5,
                 color=INK_PRI, fontweight="semibold", pad=7, loc="left")
    style_ax(ax, "Mean P(Yes / term present)")
    ax.invert_yaxis()


def panel_b(ax, df):
    """Stacked horizontal bars — severity breakdown."""
    order = df.sort_values("severity", ascending=False)
    labels = order["term"].tolist()
    n = len(labels)
    y = np.arange(n)
    h = 0.55

    ax.barh(y, order["p_mild"],     height=h, color=SEQ[1], linewidth=0, zorder=3, label="Mild")
    ax.barh(y, order["p_moderate"], height=h, color=SEQ[2], linewidth=0, zorder=3,
            left=order["p_mild"], label="Moderate")
    ax.barh(y, order["p_severe"],   height=h, color=SEQ[3], linewidth=0, zorder=3,
            left=order["p_mild"] + order["p_moderate"], label="Severe")

    # Severity score as text
    for i, (_, row) in enumerate(order.iterrows()):
        ax.text(row["p_mild"] + row["p_moderate"] + row["p_severe"] + 0.01,
                i, f"{row['severity']:.2f}", va="center", ha="left",
                fontsize=7, color=INK_SEC)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK_PRI)
    ax.set_xlim(0, 1.18)
    ax.legend(fontsize=7, loc="lower right", frameon=False, labelcolor=INK_SEC)
    ax.set_title("B   Severity breakdown  (score 0–3 annotated)", fontsize=9.5,
                 color=INK_PRI, fontweight="semibold", pad=7, loc="left")
    style_ax(ax, "P(mild) + P(moderate) + P(severe)")
    ax.invert_yaxis()


def panel_c(ax, df):
    """Heatmap — 3 uncertainty measures × 11 concepts (min-max normalised within row)."""
    measures = {
        "Entropy\n(temp. sampling)": "entropy",
        "P(not sure)\n(mult. choice)": "p_not_sure",
        "SD across\nphrasings":       "rb_sd",
    }
    # Sort concepts by entropy descending
    order = df.sort_values("entropy", ascending=True)  # imshow y goes bottom-up
    labels = order["term"].tolist()

    mat = np.zeros((len(measures), len(labels)))
    for i, (_, key) in enumerate(measures.items()):
        vals = order[key].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        mat[i] = (vals - vmin) / (vmax - vmin + 1e-9)

    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5, color=INK_PRI)
    ax.set_yticks(range(len(measures)))
    ax.set_yticklabels(list(measures.keys()), fontsize=8, color=INK_PRI)

    # Value annotations
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt_col = "white" if mat[i, j] > 0.55 else INK_SEC
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color=txt_col)

    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK_MUTED, length=0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalised score\n(within measure)", fontsize=7, color=INK_SEC)
    cbar.ax.tick_params(labelsize=6.5, colors=INK_MUTED)

    ax.set_title("C   Uncertainty heatmap  (normalised within each measure)",
                 fontsize=9.5, color=INK_PRI, fontweight="semibold", pad=7, loc="left")


def panel_d(ax, df):
    """Scatter — mean P(Yes) vs SD across phrasings."""
    ax.scatter(df["rb_mean"], df["rb_sd"], color=BLUE, s=60, alpha=0.85,
               edgecolors="white", linewidth=0.8, zorder=3)

    for _, row in df.iterrows():
        # Short label
        words = row["term"].split()
        short = " ".join(words[:2]) if len(words) > 2 else row["term"]
        ax.annotate(short,
                    (row["rb_mean"], row["rb_sd"]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=7, color=INK_SEC)

    ax.set_xlabel("Mean P(Yes) across phrasings", color=INK_SEC, fontsize=8.5)
    ax.set_ylabel("SD across phrasings", color=INK_SEC, fontsize=8.5)
    ax.set_xlim(-0.02, 0.70)
    ax.set_ylim(-0.01, 0.28)
    ax.set_facecolor(SURFACE)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("D   Robustness: prevalence vs phrasing sensitivity",
                 fontsize=9.5, color=INK_PRI, fontweight="semibold", pad=7, loc="left")


def main():
    df = load_data()

    fig = plt.figure(figsize=(16, 12), facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.10,
                        hspace=0.55, wspace=0.38)

    ax_a = fig.add_subplot(2, 2, 1)
    ax_b = fig.add_subplot(2, 2, 2)
    ax_c = fig.add_subplot(2, 2, 3)
    ax_d = fig.add_subplot(2, 2, 4)

    panel_a(ax_a, df)
    panel_b(ax_b, df)
    panel_c(ax_c, df)
    panel_d(ax_d, df)

    fig.suptitle(
        "PRISM2 uncertainty estimation — comparison across 5 experiments  ·  manual_knn  ·  n = 3 318 slides",
        fontsize=10, color=INK_SEC, y=0.965,
    )

    for ext in ("pdf", "png"):
        p = OUT / f"uncertainty_comparison.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor=SURFACE,
                    **({} if ext == "pdf" else {"dpi": 150}))
        print(f"Saved → {p}")


if __name__ == "__main__":
    main()
