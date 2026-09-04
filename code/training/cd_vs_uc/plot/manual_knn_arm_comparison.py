"""
Side-by-side AUC comparison: baseline (no_darkspot patches) vs manual KNN patches.

Three panels, one per arm group:
  1. All-sites imaging: prism2_base, prism2_diagnostic
  2. At-20-cm visit-level: img_base_visit, rna_visit, concat_raw_visit
  3. Histoscore / concept-learning: img_histoscore_visit, rna_visit, concat_histoscore_visit

Each panel shows old (no_darkspot) and new (manual KNN) bars in paired colours.

Output
------
    <OUT_DIR>/manual_knn_arm_comparison.png
    <OUT_DIR>/manual_knn_arm_comparison.pdf
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = "/home/jovyan/kgbk271-ibd-volume"
OUT_DIR = os.path.join(BASE, "training/cd_vs_uc/plots/manual_knn")
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette (Nature style) ─────────────────────────────────────────────
SURF  = "#fcfcfb"
INK   = "#0b0b0b"
INK2  = "#52514e"
MUTED = "#898781"
GRID  = "#e1e0d9"
BASE_COLOR = "#c3c2b7"

OLD_COLOR  = "#6baed6"   # muted blue — baseline (no_darkspot)
NEW_COLOR  = "#2171b5"   # darker blue — manual KNN
OLD_ALPHA  = 0.75
NEW_ALPHA  = 0.95

# ── data loaders ──────────────────────────────────────────────────────────────

def load_json_summary(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def stats_from_fold_csv(csv_path, strategy):
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    sub = df[df["strategy"] == strategy]["auc"].values
    if len(sub) == 0:
        return None, None
    return float(sub.mean()), float(sub.std(ddof=1))

def stats_from_json_summary(summary, model_or_strategy):
    if summary is None:
        return None, None
    if isinstance(summary, list):
        for s in summary:
            if s.get("strategy") == model_or_strategy:
                return s["mean_auc"], s["std_auc"]
    elif isinstance(summary, dict):
        if summary.get("model") == model_or_strategy:
            return summary["mean_auc"], summary["std_auc"]
    return None, None

# ── panel definitions ──────────────────────────────────────────────────────────

ALLSITES_OLD_BASE = os.path.join(BASE, "training/cd_vs_uc/02_imaging_no_darkspot/results")
ALLSITES_NEW_BASE = os.path.join(BASE, "training/cd_vs_uc/02_imaging_manual_knn/results")

AT20_OLD_CSV   = os.path.join(BASE, "training/cd_vs_uc/08_09_at20cm_site_controlled/results/at20cm_visit_fold_metrics.csv")
AT20_NEW_CSV   = os.path.join(BASE, "training/cd_vs_uc/08_09_at20cm_manual_knn/results/at20cm_visit_fold_metrics.csv")

HISTO_OLD_CSV  = os.path.join(BASE, "training/cd_vs_uc/08_concept_learning/results/at20cm_histoscore_fold_metrics.csv")
HISTO_NEW_CSV  = os.path.join(BASE, "training/cd_vs_uc/08_concept_learning_manual_knn/results/at20cm_histoscore_fold_metrics.csv")

# arms: (label, key, old_csv_or_dir, new_csv_or_dir, kind)
#   kind = "allsites_json" | "visit_csv"
PANELS = [
    {
        "title": "All-sites imaging\n(slide-level RF, all colon biopsies)",
        "arms": [
            ("prism2_base\n(2,560-d)",       "prism2_base",       "allsites_json"),
            ("prism2_diagnostic\n(3,072-d)",  "prism2_diagnostic", "allsites_json"),
        ],
    },
    {
        "title": "At-20-cm · visit-level\n(945 visits, 817 patients)",
        "arms": [
            ("Imaging\n(prism2 base)",          "img_base_visit",   "visit_csv"),
            ("RNA-seq\n(VST, 17,963 genes)",    "rna_visit",        "visit_csv"),
            ("RNA + Imaging\n(concat raw)",     "concat_raw_visit", "visit_csv"),
        ],
    },
    {
        "title": "Histoscore / concept-learning\n(945 visits, 817 patients)",
        "arms": [
            ("Imaging\n(11 histo scores)",       "img_histoscore_visit",   "visit_csv"),
            ("RNA-seq\n(VST, 17,963 genes)",     "rna_visit",              "visit_csv"),
            ("RNA + Histo scores\n(multimodal)", "concat_histoscore_visit","visit_csv"),
        ],
    },
]


def get_stats(key, kind, old_src, new_src):
    if kind == "allsites_json":
        old_j = load_json_summary(os.path.join(old_src, f"{key}_summary.json"))
        new_j = load_json_summary(os.path.join(new_src, f"{key}_summary.json"))
        om, os_ = stats_from_json_summary(old_j, key)
        nm, ns_ = stats_from_json_summary(new_j, key)
    else:  # visit_csv
        om, os_ = stats_from_fold_csv(old_src, key)
        nm, ns_ = stats_from_fold_csv(new_src, key)
    return om, os_, nm, ns_


# ── plotting ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
fig.patch.set_facecolor(SURF)
fig.patch.set_alpha(1.0)

for ax, panel in zip(axes, PANELS):
    ax.set_facecolor(SURF)

    # Pick the right csv / dir sources per panel
    if panel["title"].startswith("All-sites"):
        old_src, new_src = ALLSITES_OLD_BASE, ALLSITES_NEW_BASE
        kind = "allsites_json"
    elif panel["title"].startswith("At-20-cm"):
        old_src, new_src = AT20_OLD_CSV, AT20_NEW_CSV
        kind = "visit_csv"
    else:
        old_src, new_src = HISTO_OLD_CSV, HISTO_NEW_CSV
        kind = "visit_csv"

    n_arms = len(panel["arms"])
    y_base = np.arange(n_arms)
    bar_h = 0.32
    gap   = 0.06

    for i, (label, key, _) in enumerate(panel["arms"]):
        om, os_, nm, ns_ = get_stats(key, kind, old_src, new_src)
        y = y_base[i]

        # old bar (top)
        if om is not None:
            ax.barh(y + gap / 2 + bar_h / 2, om - 0.5, height=bar_h,
                    left=0.5, color=OLD_COLOR, alpha=OLD_ALPHA, linewidth=0, zorder=2)
            ax.errorbar(om, y + gap / 2 + bar_h / 2,
                        xerr=os_, fmt="none", color=INK,
                        capsize=2.0, capthick=0.7, elinewidth=0.7, zorder=4)
            ax.text(0.502, y + gap / 2 + bar_h / 2,
                    f"{om:.3f}±{os_:.3f}", va="center", ha="left",
                    fontsize=5.5, color=INK2, fontfamily="sans-serif")

        # new bar (bottom)
        if nm is not None:
            ax.barh(y - gap / 2 - bar_h / 2, nm - 0.5, height=bar_h,
                    left=0.5, color=NEW_COLOR, alpha=NEW_ALPHA, linewidth=0, zorder=2)
            ax.errorbar(nm, y - gap / 2 - bar_h / 2,
                        xerr=ns_, fmt="none", color=INK,
                        capsize=2.0, capthick=0.7, elinewidth=0.7, zorder=4)
            ax.text(0.502, y - gap / 2 - bar_h / 2,
                    f"{nm:.3f}±{ns_:.3f}", va="center", ha="left",
                    fontsize=5.5, color=INK2, fontfamily="sans-serif")

    ax.set_yticks(y_base)
    ax.set_yticklabels([a[0] for a in panel["arms"]],
                       fontsize=6.5, color=INK, fontfamily="sans-serif")
    ax.set_xlabel("AUC (5-fold CV, patient-level split)",
                  fontsize=7, color=INK, fontfamily="sans-serif")
    ax.set_xlim(0.50, 0.99)
    ax.set_ylim(-0.7, n_arms - 0.3)

    ax.axvline(0.5, color=MUTED, lw=0.6, ls="--", zorder=1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color(BASE_COLOR)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.set_tick_params(width=0.6, length=3, labelsize=6.5, colors=INK)
    ax.tick_params(axis="y", length=0)

    ax.set_title(panel["title"], fontsize=7.5, fontweight="bold",
                 color=INK, pad=6, fontfamily="sans-serif", loc="left")

# shared legend
legend_handles = [
    mpatches.Patch(color=OLD_COLOR, alpha=OLD_ALPHA, label="Baseline (no-darkspot)"),
    mpatches.Patch(color=NEW_COLOR, alpha=NEW_ALPHA, label="+ Manual KNN exclusion"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01),
           handleheight=0.8, handlelength=1.4)

fig.suptitle("CD vs UC classification — effect of manual KNN patch exclusion",
             fontsize=9, fontweight="bold", color=INK, y=1.01,
             fontfamily="sans-serif")

plt.tight_layout(pad=0.6, rect=[0, 0.04, 1, 1])

for ext in ("png", "pdf"):
    out = os.path.join(OUT_DIR, f"manual_knn_arm_comparison.{ext}")
    plt.savefig(out, dpi=200 if ext == "png" else None,
                bbox_inches="tight", facecolor=SURF)
    print(f"Saved: {out}")

plt.close()
