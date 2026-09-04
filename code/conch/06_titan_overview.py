"""
Cohort-level Titan zero-shot overview plots.

Reads: /home/jovyan/kgbk271-ibd-volume/results/titan_zero_shot/titan_zero_shot_scores.csv
Writes:
    titan_score_distributions.png   — violin per label
    titan_label_correlations.png    — pairwise correlation matrix
    titan_top_slides.png            — top-scoring slide per label (score bars)
    titan_overview.png              — combined single figure for slides
"""

from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

CSV = Path("/home/jovyan/kgbk271-ibd-volume/results/titan_zero_shot/titan_zero_shot_scores.csv")
OUT_DIR = Path("/home/jovyan/kgbk271-ibd-volume/results/titan_zero_shot")

LABEL_ORDER = [
    "normal_mucosa",
    "inflammation_involvement",
    "neutrophil_granulocytic_infiltration",
    "crypt_abscesses",
    "crypt_architectural_distortion",
    "mucin_depletion",
    "lymphoid_aggregates",
    "histiocytic_granulomas",
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
    "normal_mucosa":                        "Normal mucosa",
}

df = pd.read_csv(CSV)
print(f"Loaded {len(df)} slides")

X = df[LABEL_ORDER].to_numpy()
display_labels = [LABEL_DISPLAY[l] for l in LABEL_ORDER]

# -----------------------------------------------------------------------------
# Plot 1: violin distributions per label (ordered by median, colored by category)
# -----------------------------------------------------------------------------
fig1, ax = plt.subplots(figsize=(10, 6.5))

# Order by median score for readability
medians = np.median(X, axis=0)
order = np.argsort(-medians)
data_sorted = [X[:, i] for i in order]
labels_sorted = [display_labels[i] for i in order]
label_keys_sorted = [LABEL_ORDER[i] for i in order]

# Group colours
inflammation_terms = {
    "inflammation_involvement", "neutrophil_granulocytic_infiltration",
    "crypt_abscesses", "histiocytic_granulomas", "lymphoid_aggregates"
}
architecture_terms = {"crypt_architectural_distortion", "mucin_depletion"}
metaplasia_terms   = {"pyloric_gland_metaplasia", "paneth_cell_metaplasia"}
neuromuscular      = {"neuronal_hyperplasia", "muscular_hypertrophy"}

def _colour(key):
    if key == "normal_mucosa":      return "#4C8BF5"
    if key in inflammation_terms:   return "#E15759"
    if key in architecture_terms:   return "#F28E2B"
    if key in metaplasia_terms:     return "#59A14F"
    if key in neuromuscular:        return "#B07AA1"
    return "#888888"

parts = ax.violinplot(data_sorted, showmeans=False, showmedians=True, widths=0.85)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(_colour(label_keys_sorted[i]))
    pc.set_alpha(0.55)
    pc.set_edgecolor('black')
    pc.set_linewidth(0.6)
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(1.5)
parts['cbars'].set_color('black')
parts['cmaxes'].set_color('black')
parts['cmins'].set_color('black')

# Annotate median
for i, d in enumerate(data_sorted):
    ax.text(i + 1, np.median(d), f"{np.median(d):.2f}",
            ha='center', va='center', fontsize=8, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.85))

ax.set_xticks(range(1, len(labels_sorted) + 1))
ax.set_xticklabels(labels_sorted, rotation=35, ha='right', fontsize=10)
ax.set_ylabel("Titan cosine similarity", fontsize=11)
ax.set_title(f"Titan zero-shot score distribution across {len(df)} IBD biopsies",
             fontsize=12, fontweight='bold', pad=8)
ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.grid(axis='y', alpha=0.25)
ax.spines[['top','right']].set_visible(False)

# Category legend
from matplotlib.patches import Patch
handles = [
    Patch(facecolor="#4C8BF5", alpha=0.55, edgecolor='black', label='Normal'),
    Patch(facecolor="#E15759", alpha=0.55, edgecolor='black', label='Inflammation'),
    Patch(facecolor="#F28E2B", alpha=0.55, edgecolor='black', label='Architecture/mucin'),
    Patch(facecolor="#59A14F", alpha=0.55, edgecolor='black', label='Metaplasia'),
    Patch(facecolor="#B07AA1", alpha=0.55, edgecolor='black', label='Neuromuscular'),
]
ax.legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.9, ncol=1)

plt.tight_layout()
out1 = OUT_DIR / "titan_score_distributions.png"
fig1.savefig(out1, dpi=200, bbox_inches='tight')
print(f"Saved → {out1}")


# -----------------------------------------------------------------------------
# Plot 2: pairwise correlation matrix
# -----------------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(9, 8))
corr = np.corrcoef(X.T)
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

for i in range(len(LABEL_ORDER)):
    for j in range(len(LABEL_ORDER)):
        val = corr[i, j]
        text_col = 'white' if abs(val) > 0.55 else 'black'
        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                fontsize=8.5, color=text_col,
                fontweight='bold' if i == j else 'normal')

ax.set_xticks(range(len(LABEL_ORDER)))
ax.set_yticks(range(len(LABEL_ORDER)))
ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(display_labels, fontsize=10)
ax.set_title(f"Titan zero-shot score correlations across {len(df)} slides",
             fontsize=12, fontweight='bold', pad=8)
cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label("Pearson r", fontsize=10)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
out2 = OUT_DIR / "titan_label_correlations.png"
fig2.savefig(out2, dpi=200, bbox_inches='tight')
print(f"Saved → {out2}")


# -----------------------------------------------------------------------------
# Plot 3: cohort-wide slide × label heatmap (rows sorted by dominant finding)
# -----------------------------------------------------------------------------
fig3, ax = plt.subplots(figsize=(12, 7))

# Row-standardise each label to make co-occurrence patterns readable
Z = (X - X.mean(axis=0)) / X.std(axis=0)

# Sort slides by hierarchical clustering (fast alternative: sort by argmax of Z)
top_label = np.argmax(Z, axis=1)
sort_idx = np.lexsort((-Z.max(axis=1), top_label))
Z_sorted = Z[sort_idx]

im = ax.imshow(Z_sorted.T, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5,
               interpolation='nearest')
ax.set_yticks(range(len(LABEL_ORDER)))
ax.set_yticklabels(display_labels, fontsize=10)
ax.set_xlabel(f"Slides ({len(df)} biopsies, sorted by top-scoring label)", fontsize=11)
ax.set_title("Titan zero-shot: standardised score per slide × label",
             fontsize=12, fontweight='bold', pad=8)
ax.tick_params(axis='both', length=0)
cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cbar.set_label("Z-score", fontsize=10)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
out3 = OUT_DIR / "titan_cohort_heatmap.png"
fig3.savefig(out3, dpi=200, bbox_inches='tight')
print(f"Saved → {out3}")


# -----------------------------------------------------------------------------
# Plot 4: fraction of slides above threshold per label
# -----------------------------------------------------------------------------
fig4, ax = plt.subplots(figsize=(10, 5.5))

thresholds = [0.4, 0.5, 0.6, 0.7]
n = len(df)
bar_w = 0.20
x = np.arange(len(LABEL_ORDER))

colors = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']
for k, thr in enumerate(thresholds):
    frac = np.mean(X >= thr, axis=0)  # per label
    ax.bar(x + (k - 1.5) * bar_w, frac * 100, bar_w,
           label=f"≥ {thr}", color=colors[k], edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(display_labels, rotation=35, ha='right', fontsize=10)
ax.set_ylabel("% of slides", fontsize=11)
ax.set_title(f"Titan zero-shot: fraction of {n} slides scoring above threshold",
             fontsize=12, fontweight='bold', pad=8)
ax.grid(axis='y', alpha=0.25)
ax.spines[['top','right']].set_visible(False)
ax.legend(title='Threshold', fontsize=9, title_fontsize=9, loc='upper right')

plt.tight_layout()
out4 = OUT_DIR / "titan_threshold_fractions.png"
fig4.savefig(out4, dpi=200, bbox_inches='tight')
print(f"Saved → {out4}")

plt.close('all')
print("\nAll Titan overview plots saved.")
