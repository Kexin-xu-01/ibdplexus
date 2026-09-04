"""
Compare CONCH slide-level scores (attention-weighted across top-K patches)
against PRISM2 histological scores, and side-by-side against Titan.

CONCH slide-level score = sum_j attn_j * conch_score(patch_j, label) / sum_j attn_j
where attn_j is the PRISM2 attention weight for patch j.

Outputs (in results/conch_zero_shot/):
    conch_slide_scores.csv              — attention-weighted per (slide, label)
    conch_vs_prism2_scatter.png         — 11 scatter panels
    conch_vs_prism2_boxplot.png         — Titan-style boxplots
    conch_titan_vs_prism2_bars.png      — side-by-side Pearson/Spearman for both models
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr

CONCH_DIR   = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/"
                   "tissue_threshold_15_filtered_no_darkspot_manual_knn/"
                   "20x_224px_0px_overlap/prism2_conch_zero_shot")
PRISM2_CSV  = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2_manual_knn/"
                   "prism2_histological_score.csv")
TITAN_CSV   = Path("/home/jovyan/kgbk271-ibd-volume/results/titan_zero_shot/"
                   "titan_zero_shot_scores.csv")
OUT_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/results/conch_zero_shot")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# ---------------------------------------------------------------------------
# 1. Aggregate CONCH patch scores → slide-level attention-weighted score
# ---------------------------------------------------------------------------
def aggregate_slide(json_path: Path) -> dict:
    """Return {label: attention-weighted score} for one slide."""
    with open(json_path) as f:
        patches = json.load(f)
    if not patches:
        return {}
    attn = np.array([p["attn_score"] for p in patches], dtype=float)
    if attn.sum() == 0:
        weights = np.ones_like(attn) / len(attn)
    else:
        weights = attn / attn.sum()
    result = {}
    for lbl in LABELS + ["normal_mucosa"]:
        vals = np.array([
            next((x["score"] for x in p["labels"] if x["label"] == lbl), np.nan)
            for p in patches
        ], dtype=float)
        result[lbl] = float(np.nansum(weights * vals))
    return result

json_paths = sorted(CONCH_DIR.glob("*.json"))
print(f"Aggregating {len(json_paths)} CONCH JSON files ...")
rows = []
for jp in json_paths:
    row = {"slide": jp.stem, **aggregate_slide(jp)}
    rows.append(row)
conch_slide = pd.DataFrame(rows)
conch_slide.to_csv(OUT_DIR / "conch_slide_scores.csv", index=False)
print(f"Saved → {OUT_DIR / 'conch_slide_scores.csv'}   shape={conch_slide.shape}")

# ---------------------------------------------------------------------------
# 2. Merge with PRISM2 + Titan
# ---------------------------------------------------------------------------
prism2 = pd.read_csv(PRISM2_CSV)
titan  = pd.read_csv(TITAN_CSV)

conch_p2 = conch_slide.merge(prism2, on="slide", suffixes=("_conch", "_prism2"))
print(f"\nCONCH ∩ PRISM2 : {len(conch_p2)} slides")

conch_p2_t = conch_p2.merge(titan, on="slide")
# Rename Titan columns
rename_map = {c: f"{c}_titan" for c in LABELS if c in titan.columns}
conch_p2_t = conch_p2_t.rename(columns=rename_map)
print(f"CONCH ∩ PRISM2 ∩ Titan : {len(conch_p2_t)} slides\n")

# ---------------------------------------------------------------------------
# 3. Correlations: CONCH vs PRISM2 and Titan vs PRISM2
# ---------------------------------------------------------------------------
records = []
for lbl in LABELS:
    c = conch_p2_t[f"{lbl}_conch"].to_numpy()
    p = conch_p2_t[f"{lbl}_prism2"].to_numpy()
    t = conch_p2_t[f"{lbl}_titan"].to_numpy()
    r_c_p, _ = pearsonr(c, p)
    r_c_s, _ = spearmanr(c, p)
    r_t_p, _ = pearsonr(t, p)
    r_t_s, _ = spearmanr(t, p)
    records.append({
        "label": lbl,
        "conch_pearson": r_c_p,   "conch_spearman": r_c_s,
        "titan_pearson": r_t_p,   "titan_spearman": r_t_s,
    })
    print(f"  {lbl:<42}  CONCH ρ={r_c_s:+.3f}  |  Titan ρ={r_t_s:+.3f}")

corr_df = pd.DataFrame(records)
corr_df.to_csv(OUT_DIR / "conch_vs_prism2_correlations.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Scatter panels (CONCH weighted vs PRISM2), 2×6 layout
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 6, figsize=(19, 6.8))
axes = axes.flatten()

for i, lbl in enumerate(LABELS):
    ax = axes[i]
    c = conch_p2_t[f"{lbl}_conch"].to_numpy()
    p = conch_p2_t[f"{lbl}_prism2"].to_numpy()
    ax.hist2d(c, p, bins=40, cmap="magma", cmin=1, norm=LogNorm())
    ax.axhline(np.median(p), color='white', linewidth=0.5, alpha=0.6, linestyle=':')
    ax.axvline(np.median(c), color='white', linewidth=0.5, alpha=0.6, linestyle=':')
    r_p = corr_df.loc[i, "conch_pearson"]
    r_s = corr_df.loc[i, "conch_spearman"]
    ax.set_title(LABEL_DISPLAY[lbl], fontsize=10, fontweight="bold")
    ax.text(0.03, 0.97, f"r={r_p:+.2f}\nρ={r_s:+.2f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="white",
            bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=3))
    ax.set_xlabel("CONCH (attn-wtd)", fontsize=9)
    ax.set_ylabel("PRISM2", fontsize=9)
    ax.tick_params(labelsize=8)
axes[11].axis("off")

fig.suptitle(f"CONCH attention-weighted slide score vs PRISM2  ({len(conch_p2_t)} slides)",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.99])
out = OUT_DIR / "conch_vs_prism2_scatter.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved → {out}")

# ---------------------------------------------------------------------------
# 5. Boxplot CONCH vs PRISM2 bin
# ---------------------------------------------------------------------------
BIN_EDGES = [0.0, 0.25, 0.5, 0.75, 1.001]
BIN_LABELS = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]
BIN_COLORS = ["#4c78a8", "#8fbcd4", "#f28e2b", "#e15759"]

fig, axes = plt.subplots(2, 6, figsize=(19, 7))
axes = axes.flatten()

for i, lbl in enumerate(LABELS):
    ax = axes[i]
    c = conch_p2_t[f"{lbl}_conch"].to_numpy()
    p = conch_p2_t[f"{lbl}_prism2"].to_numpy()
    bin_idx = np.digitize(p, BIN_EDGES) - 1

    data_per_bin, counts, positions, colours = [], [], [], []
    for b in range(len(BIN_LABELS)):
        vals = c[bin_idx == b]
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
    for patch, col in zip(bp['boxes'], colours):
        patch.set_facecolor(col); patch.set_alpha(0.75)

    ax.set_xticks(range(len(BIN_LABELS)))
    counts_by_bin = {pos: n for pos, n in zip(positions, counts)}
    xlab = [f"{BIN_LABELS[b]}\nn={counts_by_bin.get(b, 0)}" for b in range(len(BIN_LABELS))]
    ax.set_xticklabels(xlab, fontsize=8)

    r_s = corr_df.loc[i, "conch_spearman"]
    ax.text(0.03, 0.97, f"ρ={r_s:+.2f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="black",
            bbox=dict(facecolor="white", edgecolor="lightgray", pad=2, alpha=0.9))
    ax.set_title(LABEL_DISPLAY[lbl], fontsize=10, fontweight="bold")
    ax.set_ylabel("CONCH (attn-wtd)", fontsize=9)
    ax.set_xlabel("PRISM2 score bin", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

axes[11].axis("off")
fig.suptitle(f"CONCH attention-weighted score vs PRISM2 bin  ({len(conch_p2_t)} slides)",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.98])
out2 = OUT_DIR / "conch_vs_prism2_boxplot.png"
fig.savefig(out2, dpi=200, bbox_inches="tight")
print(f"Saved → {out2}")

# ---------------------------------------------------------------------------
# 6. Side-by-side bars: CONCH vs Titan (both vs PRISM2)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.8))
order = corr_df.sort_values("conch_spearman", ascending=False).reset_index(drop=True)
x = np.arange(len(order))
bw = 0.36

ax.bar(x - bw/2, order["conch_spearman"], bw,
       color="#4C8BF5", edgecolor="white", linewidth=0.5,
       label="CONCH (attn-wtd) ρ")
ax.bar(x + bw/2, order["titan_spearman"], bw,
       color="#F28E2B", edgecolor="white", linewidth=0.5,
       label="Titan ρ")

for i, row in order.iterrows():
    ax.text(i - bw/2, row["conch_spearman"] + 0.01, f"{row['conch_spearman']:+.2f}",
            ha="center", va="bottom", fontsize=8)
    ax.text(i + bw/2, row["titan_spearman"] + 0.01, f"{row['titan_spearman']:+.2f}",
            ha="center", va="bottom", fontsize=8)

ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels([LABEL_DISPLAY[l] for l in order["label"]],
                   rotation=35, ha="right", fontsize=10)
ax.set_ylabel("Spearman ρ  vs  PRISM2", fontsize=11)
ax.set_title(f"CONCH (attention-weighted) vs Titan  —  correlation with PRISM2  "
             f"({len(conch_p2_t)} slides)",
             fontsize=12, fontweight="bold", pad=8)
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", alpha=0.25)
ax.spines[["top","right"]].set_visible(False)

y_min = min(order[["conch_spearman", "titan_spearman"]].to_numpy().min() - 0.06, -0.05)
ax.set_ylim(y_min, 1.0)

plt.tight_layout()
out3 = OUT_DIR / "conch_titan_vs_prism2_bars.png"
fig.savefig(out3, dpi=200, bbox_inches="tight")
print(f"Saved → {out3}")

plt.close("all")
