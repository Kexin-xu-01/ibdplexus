"""
Compare CD vs UC classification performance and histological scores
across different patch filtering strategies.

Filtering conditions:
  1. original        — trident_processed (no extra filter)
  2. tissue_t15      — tissue_threshold_15 (Laplacian t100 only)
  3. lap_intensity   — tissue_threshold_15_filtered (Lap t100 + intensity p98)
  4. no_darkspot     — tissue_threshold_15_filtered_no_darkspot (+ GrandQC dark spots)

Outputs:
  <OUT_DIR>/filter_rf_comparison.png        RF AUC/AP bar chart
  <OUT_DIR>/filter_histoscore_comparison.png histological score distributions
  <OUT_DIR>/filter_comparison.csv           tabular summary
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

BASE   = "/home/jovyan/kgbk271-ibd-volume"
OUT_DIR = os.path.join(BASE, "training/cd_vs_uc/filter_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Result directories per condition ─────────────────────────────────────────
CONDITIONS = {
    "Original\n(trident)":          os.path.join(BASE, "training/cd_vs_uc/02_04_imaging_allsites/results"),
    "Laplacian\nt100":              os.path.join(BASE, "training/cd_vs_uc/tissue_threshold_15/results"),
    "Lap +\nintensity":             os.path.join(BASE, "training/cd_vs_uc/tissue_threshold_15_filtered/results"),
    "Lap + intensity\n+ no-darkspot": os.path.join(BASE, "training/cd_vs_uc/no_darkspot/results"),
}

HISTOSCORE_CSVS = {
    "Original\n(trident)":          os.path.join(BASE, "results/prism2/prism2_histological_score.csv"),
    "Laplacian\nt100":              None,
    "Lap +\nintensity":             os.path.join(BASE, "results/prism2_tissue_threshold_15_filtered/prism2_histological_score.csv"),
    "Lap + intensity\n+ no-darkspot": os.path.join(BASE, "results/prism2_no_darkspot/prism2_histological_score.csv"),
}

HISTO_COLS = [
    "inflammation_involvement", "crypt_architectural_distortion",
    "neutrophil_granulocytic_infiltration", "crypt_abscesses",
    "lymphoid_aggregates", "histiocytic_granulomas", "mucin_depletion",
    "pyloric_gland_metaplasia", "paneth_cell_metaplasia",
    "neuronal_hyperplasia", "muscular_hypertrophy",
]

MODELS = ["prism2_base", "prism2_diagnostic"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── Load RF summaries ─────────────────────────────────────────────────────────
def load_rf_summary(results_dir, model):
    path = os.path.join(results_dir, f"{model}_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

rows = []
for cond, rdir in CONDITIONS.items():
    for model in MODELS:
        s = load_rf_summary(rdir, model)
        if s is None:
            continue
        rows.append({
            "condition":  cond,
            "model":      model,
            "mean_auc":   s["mean_auc"],
            "std_auc":    s["std_auc"],
            "mean_ap":    s["mean_ap"],
            "std_ap":     s["std_ap"],
            "mean_acc":   s["mean_acc"],
            "std_acc":    s["std_acc"],
        })
rf_df = pd.DataFrame(rows)

if rf_df.empty:
    print("No RF results found yet.")
else:
    print("RF results loaded:")
    print(rf_df[["condition","model","mean_auc","mean_ap","mean_acc"]].to_string(index=False))

    # ── RF bar chart ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CD vs UC classification — effect of patch quality filtering\n(Random Forest, 5-fold CV)",
                 fontsize=13, fontweight="bold")

    for ax, metric, metric_std, ylabel in [
        (axes[0], "mean_auc", "std_auc",  "AUC (mean ± std)"),
        (axes[1], "mean_ap",  "std_ap",   "Average Precision (mean ± std)"),
    ]:
        cond_list = list(CONDITIONS.keys())
        x = np.arange(len(cond_list))
        width = 0.35

        for mi, (model, col) in enumerate(zip(MODELS, [COLORS[0], COLORS[1]])):
            sub = rf_df[rf_df["model"] == model]
            vals = [sub[sub["condition"] == c][metric].values[0]
                    if len(sub[sub["condition"] == c]) else np.nan
                    for c in cond_list]
            errs = [sub[sub["condition"] == c][metric_std].values[0]
                    if len(sub[sub["condition"] == c]) else 0
                    for c in cond_list]
            ax.bar(x + mi * width - width / 2, vals, width,
                   yerr=errs, capsize=4, color=col, alpha=0.85,
                   label=model.replace("_", " "))

        ax.set_xticks(x)
        ax.set_xticklabels(cond_list, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.65, 0.85)
        ax.axhline(0.79, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_rf = os.path.join(OUT_DIR, "filter_rf_comparison.png")
    fig.savefig(out_rf, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_rf}")

# ── Histological scores ───────────────────────────────────────────────────────
histo_dfs = {}
for cond, csv_path in HISTOSCORE_CSVS.items():
    if csv_path is None or not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"slide": "slide_id"}) if "slide" in df.columns else df
    histo_dfs[cond] = df
    print(f"Histoscores loaded [{cond}]: {len(df)} slides")

if histo_dfs:
    n_cols = len(HISTO_COLS)
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle("Histological score distributions across patch filtering strategies\n(PRISM2 Yes/No probabilities)",
                 fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()

    cond_colors = {c: col for c, col in zip(histo_dfs.keys(), COLORS)}

    for i, col in enumerate(HISTO_COLS):
        ax = axes_flat[i]
        for cond, df in histo_dfs.items():
            if col not in df.columns:
                continue
            ax.hist(df[col].dropna(), bins=30, alpha=0.55, density=True,
                    color=cond_colors.get(cond, "gray"),
                    label=cond.replace("\n", " "))
        ax.set_title(col.replace("_", " "), fontsize=9)
        ax.set_xlabel("P(Yes)", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for j in range(n_cols, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared legend
    patches = [mpatches.Patch(color=c, label=l.replace("\n", " "))
               for l, c in cond_colors.items()]
    fig.legend(handles=patches, loc="lower right", fontsize=9,
               bbox_to_anchor=(0.98, 0.02))

    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    out_hist = os.path.join(OUT_DIR, "filter_histoscore_comparison.png")
    fig.savefig(out_hist, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_hist}")

    # Mean score per condition table
    mean_rows = []
    for cond, df in histo_dfs.items():
        row = {"condition": cond.replace("\n", " ")}
        for col in HISTO_COLS:
            if col in df.columns:
                row[col] = df[col].mean()
        mean_rows.append(row)
    mean_df = pd.DataFrame(mean_rows).set_index("condition")
    print("\nMean histological scores per condition:")
    print(mean_df.round(3).T.to_string())

# ── Save tabular summary ──────────────────────────────────────────────────────
if not rf_df.empty:
    rf_df["condition"] = rf_df["condition"].str.replace("\n", " ")
    rf_df.to_csv(os.path.join(OUT_DIR, "filter_comparison.csv"), index=False)
    print(f"\nTabular summary: {OUT_DIR}/filter_comparison.csv")

print("\nDone.")
