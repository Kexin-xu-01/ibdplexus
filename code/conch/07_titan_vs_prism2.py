"""
Compare Titan zero-shot scores vs PRISM2 histological scores across the IBD cohort.

Outputs (in results/titan_zero_shot/):
    titan_vs_prism2_scatter.png    — 11 scatter panels, one per UAMP label
    titan_vs_prism2_correlations.png — Pearson r bar chart + rank-correlation
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

TITAN_CSV  = Path("/home/jovyan/kgbk271-ibd-volume/results/titan_zero_shot/titan_zero_shot_scores.csv")
PRISM2_CSV = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2_manual_knn/prism2_histological_score.csv")
OUT_DIR    = Path("/home/jovyan/kgbk271-ibd-volume/results/titan_zero_shot")

# 11 shared labels (PRISM2 has no normal_mucosa)
LABELS = [
    "inflammation_involvement",
    "crypt_architectural_distortion",
    "neutrophil_granulocytic_infiltration",
    "crypt_abscesses",
    "lymphoid_aggregates",
    "histiocytic_granulomas",
    "mucin_depletion",
    "pyloric_gland_metaplasia",
    "paneth_cell_metaplasia",
    "neuronal_hyperplasia",
    "muscular_hypertrophy",
]
LABEL_DISPLAY = {
    "inflammation_involvement":             "Inflammation",
    "crypt_architectural_distortion":       "Crypt distortion",
    "neutrophil_granulocytic_infiltration": "Neutrophilic infiltration",
    "crypt_abscesses":                      "Crypt abscesses",
    "lymphoid_aggregates":                  "Lymphoid aggregates",
    "histiocytic_granulomas":               "Histiocytic granulomas",
    "mucin_depletion":                      "Mucin depletion",
    "pyloric_gland_metaplasia":             "Pyloric gland metaplasia",
    "paneth_cell_metaplasia":               "Paneth cell metaplasia",
    "neuronal_hyperplasia":                 "Neuronal hyperplasia",
    "muscular_hypertrophy":                 "Muscular hypertrophy",
}

# Load and merge on slide id
titan  = pd.read_csv(TITAN_CSV)
prism2 = pd.read_csv(PRISM2_CSV)
print(f"Titan  : {len(titan)} slides")
print(f"PRISM2 : {len(prism2)} slides")

merged = titan.merge(prism2, on="slide", suffixes=("_titan", "_prism2"))
print(f"Merged : {len(merged)} slides in common\n")

# Compute correlations
results = []
for lbl in LABELS:
    t = merged[f"{lbl}_titan"].to_numpy()
    p = merged[f"{lbl}_prism2"].to_numpy()
    r_p, p_p = pearsonr(t, p)
    r_s, p_s = spearmanr(t, p)
    results.append({"label": lbl, "pearson": r_p, "spearman": r_s})
    print(f"  {lbl:<42} Pearson={r_p:+.3f}  Spearman={r_s:+.3f}")

results_df = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Plot 1: 11 scatter panels (Titan vs PRISM2 per label)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 6, figsize=(19, 6.8))
axes = axes.flatten()

for i, lbl in enumerate(LABELS):
    ax = axes[i]
    t = merged[f"{lbl}_titan"].to_numpy()
    p = merged[f"{lbl}_prism2"].to_numpy()
    r_p = results[i]["pearson"]
    r_s = results[i]["spearman"]

    # Colour by density (2D hist)
    from matplotlib.colors import LogNorm
    ax.hist2d(t, p, bins=40, cmap="viridis", cmin=1, norm=LogNorm())

    # Reference line at each variable's median
    ax.axhline(np.median(p), color='white', linewidth=0.5, alpha=0.6, linestyle=':')
    ax.axvline(np.median(t), color='white', linewidth=0.5, alpha=0.6, linestyle=':')

    ax.set_title(LABEL_DISPLAY[lbl], fontsize=10, fontweight="bold")
    ax.text(0.03, 0.97,
            f"r={r_p:+.2f}\nρ={r_s:+.2f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="white",
            bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=3))
    ax.set_xlabel("Titan (cosine)", fontsize=9)
    ax.set_ylabel("PRISM2", fontsize=9)
    ax.tick_params(labelsize=8)

# hide 12th axis (2x6 grid, 11 labels)
axes[11].axis("off")

fig.suptitle(f"Titan zero-shot vs PRISM2 histological scores  ({len(merged)} slides)",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.99])

out1 = OUT_DIR / "titan_vs_prism2_scatter.png"
fig.savefig(out1, dpi=200, bbox_inches="tight")
print(f"\nSaved → {out1}")

# ---------------------------------------------------------------------------
# Plot 2: correlation bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))
order = results_df.sort_values("spearman", ascending=False).reset_index(drop=True)
x = np.arange(len(order))
bw = 0.4

def _c(r):
    return "#2b8cbe" if r >= 0 else "#e34a33"

ax.bar(x - bw/2, order["pearson"], bw,
       color=[_c(v) for v in order["pearson"]],
       label="Pearson r", edgecolor="white", linewidth=0.5)
ax.bar(x + bw/2, order["spearman"], bw,
       color=[_c(v) for v in order["spearman"]], alpha=0.55, hatch="//",
       label="Spearman ρ", edgecolor="white", linewidth=0.5)

for i, row in order.iterrows():
    ax.text(i - bw/2, row["pearson"] + 0.01, f"{row['pearson']:+.2f}",
            ha="center", va="bottom", fontsize=8)
    ax.text(i + bw/2, row["spearman"] + 0.01, f"{row['spearman']:+.2f}",
            ha="center", va="bottom", fontsize=8)

ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels([LABEL_DISPLAY[l] for l in order["label"]],
                   rotation=35, ha="right", fontsize=10)
ax.set_ylabel("Correlation (Titan vs PRISM2)", fontsize=11)
ax.set_title(f"Titan vs PRISM2 per-label correlation  ({len(merged)} slides)",
             fontsize=12, fontweight="bold", pad=8)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
ax.spines[["top","right"]].set_visible(False)
ax.set_ylim(min(order["spearman"].min(), order["pearson"].min()) - 0.06, 1.0)

plt.tight_layout()
out2 = OUT_DIR / "titan_vs_prism2_correlations.png"
fig.savefig(out2, dpi=200, bbox_inches="tight")
print(f"Saved → {out2}")

results_df.to_csv(OUT_DIR / "titan_vs_prism2_correlations.csv", index=False)
print(f"Saved → {OUT_DIR / 'titan_vs_prism2_correlations.csv'}")

# ---------------------------------------------------------------------------
# Plot 3: boxplot of Titan scores per PRISM2 bin, per concept
# ---------------------------------------------------------------------------
# PRISM2 scores are calibrated probabilities (0-1). Bin at 0.25 / 0.5 / 0.75.
BIN_EDGES = [0.0, 0.25, 0.5, 0.75, 1.001]
BIN_LABELS = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]
BIN_COLORS = ["#4c78a8", "#8fbcd4", "#f28e2b", "#e15759"]

fig, axes = plt.subplots(2, 6, figsize=(19, 7))
axes = axes.flatten()

for i, lbl in enumerate(LABELS):
    ax = axes[i]
    t = merged[f"{lbl}_titan"].to_numpy()
    p = merged[f"{lbl}_prism2"].to_numpy()
    bin_idx = np.digitize(p, BIN_EDGES) - 1  # 0..3

    data_per_bin, counts, positions, colours = [], [], [], []
    for b in range(len(BIN_LABELS)):
        vals = t[bin_idx == b]
        if len(vals) >= 5:
            data_per_bin.append(vals)
            counts.append(len(vals))
            positions.append(b)
            colours.append(BIN_COLORS[b])

    bp = ax.boxplot(data_per_bin, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.4),
                    whiskerprops=dict(color="black", linewidth=0.8),
                    capprops=dict(color="black", linewidth=0.8),
                    boxprops=dict(edgecolor="black", linewidth=0.6))
    for patch, c in zip(bp['boxes'], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    # X-tick labels combine bin + n
    ax.set_xticks(range(len(BIN_LABELS)))
    xtick_labels = []
    counts_by_bin = {p: n for p, n in zip(positions, counts)}
    for b in range(len(BIN_LABELS)):
        n = counts_by_bin.get(b, 0)
        xtick_labels.append(f"{BIN_LABELS[b]}\nn={n}")
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.set_title(LABEL_DISPLAY[lbl], fontsize=10, fontweight="bold")
    ax.set_ylabel("Titan (cosine)", fontsize=9)
    ax.set_xlabel("PRISM2 score bin", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # spearman for reference
    r_s = results[i]["spearman"]
    ax.text(0.03, 0.97, f"ρ={r_s:+.2f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="black",
            bbox=dict(facecolor="white", edgecolor="lightgray", pad=2, alpha=0.9))

axes[11].axis("off")

fig.suptitle(f"Titan zero-shot score vs PRISM2 bin  ({len(merged)} slides)",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.98])

out3 = OUT_DIR / "titan_vs_prism2_boxplot.png"
fig.savefig(out3, dpi=200, bbox_inches="tight")
print(f"Saved → {out3}")

plt.close("all")
