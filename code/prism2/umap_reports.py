"""
Extract categorical variables from PRISM2 free-text reports and plot
them on the prism2_diagnostic UMAP (reusing cached coordinates).
Also adds UAMP yes/no P(Yes) scores (continuous, sequential colour scale).

Report-extracted variables:
  report_tissue, report_inflammation, report_finding,
  report_active_inflam, report_granuloma, report_dysplasia,
  report_crypt_injury, report_h_pylori

UAMP yes/no scores (P(Yes), 0–1 continuous):
  inflammation_involvement, crypt_architectural_distortion,
  neutrophil_granulocytic_infiltration, crypt_abscesses, lymphoid_aggregates,
  histiocytic_granulomas, mucin_depletion, pyloric_gland_metaplasia,
  paneth_cell_metaplasia, neuronal_hyperplasia, muscular_hypertrophy

Ground-truth variables added to HTML dropdown:
  diagnosis, normal_lesional, macroscopic_appearance, disease_activity_60, tissue_site

Outputs:
  results/prism2/umap/umap_prism2_diagnostic_reports.html
  results/prism2/umap/umap_prism2_diagnostic_report_<var>.{png,pdf}
  results/prism2/umap/umap_prism2_diagnostic_compare_<pair>.{png,pdf}
  results/metadata/slide_report_features.csv
"""

import json
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

JSONL_PATH  = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2/prism2_reports.jsonl")
COORDS_NPZ  = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2/umap/umap_prism2_diagnostic_coords.npz")
META_CSV    = Path("/home/jovyan/kgbk271-ibd-volume/results/metadata/slide_metadata.csv")
UAMP_CSV    = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2/prism2_reports.csv")
OUT_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2/umap")
COMPARE_DIR = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2/umap/compare")
META_OUT    = Path("/home/jovyan/kgbk271-ibd-volume/results/metadata/slide_report_features.csv")
PROMPT      = "write a report"

# UAMP P(Yes) columns — continuous 0–1, shown with sequential colour scale
UAMP_COLS = [
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

UAMP_TITLES = {c: c.replace("_", " ").title() + " (UAMP P(Yes))" for c in UAMP_COLS}

# ── palettes ───────────────────────────────────────────────────────────────
# Report-extracted
REPORT_PALETTES = {
    "report_tissue": {
        "Colonic":          "#4C72B0",
        "Small intestinal": "#DD8452",
        "Gastric":          "#55A868",
        "Duodenal":         "#9467BD",
        "Unknown":          "#BBBBBB",
    },
    "report_inflammation": {
        "None":             "#3A9E64",
        "Mild chronic":     "#F0C040",
        "Chronic inactive": "#E07B2A",
        "Active":           "#C0392B",
        "Active chronic":   "#8B0000",
        "Unknown":          "#BBBBBB",
    },
    "report_finding": {
        "Normal":              "#3A9E64",
        "Hyperplastic polyp":  "#4C72B0",
        "Lymphoid aggregate":  "#9467BD",
        "Chronic colitis":     "#F0C040",
        "Active colitis":      "#E07B2A",
        "Lymphocytic colitis": "#DD8452",
        "Gastritis":           "#8C564B",
        "Dysplasia/Carcinoma": "#C0392B",
        "Lymphoma":            "#E377C2",
        "Other":               "#7F7F7F",
        "Unknown":             "#BBBBBB",
    },
    "report_active_inflam": {"Yes": "#C0392B", "No": "#3A9E64", "Unknown": "#BBBBBB"},
    "report_granuloma":     {"Yes": "#C0392B", "No": "#3A9E64", "Unknown": "#BBBBBB"},
    "report_dysplasia":     {"Yes": "#C0392B", "No": "#3A9E64", "Unknown": "#BBBBBB"},
    "report_crypt_injury":  {"Yes": "#C0392B", "No": "#3A9E64", "Unknown": "#BBBBBB"},
    "report_h_pylori":      {"Yes": "#C0392B", "No": "#3A9E64", "Unknown": "#BBBBBB"},
}

# Ground-truth
GT_PALETTES = {
    "diagnosis": {
        "Crohn's Disease":    "#4C72B0",
        "Ulcerative Colitis": "#DD8452",
        "IBD Unclassified":   "#55A868",
        "Unknown":            "#BBBBBB",
    },
    "normal_lesional": {
        "Normal":   "#3A9E64",
        "Lesional": "#C0392B",
        "Unknown":  "#BBBBBB",
    },
    "macroscopic_appearance": {
        "Normal":                "#3A9E64",
        "Possible inflammation": "#F0C040",
        "Erosions or Ulcers":    "#C0392B",
        "Unknown":               "#BBBBBB",
    },
    "disease_activity_60": {
        "Remission": "#3A9E64",
        "Mild":      "#F0C040",
        "Moderate":  "#E07B2A",
        "Severe":    "#C0392B",
        "Unknown":   "#BBBBBB",
    },
    "tissue_site": {
        "Ileum":            "#4C72B0",
        "Rectum":           "#DD8452",
        "Sigmoid Colon":    "#55A868",
        "Cecum":            "#9467BD",
        "Ascending Colon":  "#8C564B",
        "Descending Colon": "#E377C2",
        "Other":            "#7F7F7F",
        "Unknown":          "#BBBBBB",
    },
}

ALL_PALETTES = {**REPORT_PALETTES, **GT_PALETTES}

LABEL_TITLES = {
    # report
    "report_tissue":        "Tissue Type (from report)",
    "report_inflammation":  "Inflammation Grade (from report)",
    "report_finding":       "Primary Finding (from report)",
    "report_active_inflam": "Active Inflammation (from report)",
    "report_granuloma":     "Granulomas (from report)",
    "report_dysplasia":     "Dysplasia/Carcinoma (from report)",
    "report_crypt_injury":  "Crypt Injury (from report)",
    "report_h_pylori":      "H. pylori (from report)",
    # ground truth
    "diagnosis":              "Diagnosis [GT]",
    "normal_lesional":        "Normal vs Lesional [GT]",
    "macroscopic_appearance": "Macroscopic Appearance [GT]",
    "disease_activity_60":    "Disease Activity [GT]",
    "tissue_site":            "Tissue Site [GT]",
}

# Pairs for side-by-side comparison: (report_col, gt_col, shared_palette)
# Shared palette must work for both — map each gt value to the same colour
# scheme as its report counterpart where possible.
COMPARE_PAIRS = [
    (
        "report_active_inflam", "Active Inflammation\n(from report)",
        "normal_lesional",      "Normal vs Lesional\n[ground truth]",
        # shared 2-colour palette: Yes/Lesional = red, No/Normal = green
        {"Yes": "#C0392B", "No": "#3A9E64",
         "Lesional": "#C0392B", "Normal": "#3A9E64", "Unknown": "#BBBBBB"},
        "compare_active_inflam_vs_normal_lesional",
    ),
    (
        "report_tissue",    "Tissue Type\n(from report)",
        "tissue_site",      "Tissue Site\n[ground truth]",
        # map tissue_site values to same hues as report_tissue
        {"Colonic": "#4C72B0", "Small intestinal": "#DD8452",
         "Gastric": "#55A868", "Duodenal": "#9467BD",
         "Ileum": "#DD8452",          # small intestinal hue
         "Rectum": "#4C72B0",         # colonic hue
         "Sigmoid Colon": "#4C72B0",
         "Cecum": "#4C72B0",
         "Ascending Colon": "#4C72B0",
         "Descending Colon": "#4C72B0",
         "Other": "#7F7F7F",
         "Unknown": "#BBBBBB"},
        "compare_tissue",
    ),
    (
        "report_finding",       "Primary Finding\n(from report)",
        "macroscopic_appearance", "Macroscopic Appearance\n[ground truth]",
        # map macro to closest finding hue
        {"Normal": "#3A9E64", "Possible inflammation": "#F0C040",
         "Erosions or Ulcers": "#E07B2A",
         "Hyperplastic polyp": "#4C72B0", "Lymphoid aggregate": "#9467BD",
         "Chronic colitis": "#F0C040", "Active colitis": "#E07B2A",
         "Lymphocytic colitis": "#DD8452", "Gastritis": "#8C564B",
         "Dysplasia/Carcinoma": "#C0392B", "Lymphoma": "#E377C2",
         "Other": "#7F7F7F", "Unknown": "#BBBBBB"},
        "compare_finding_vs_macroscopic",
    ),
]


# ── extraction rules ───────────────────────────────────────────────────────

def extract_tissue(text: str) -> str:
    t = text.lower()
    if re.search(r"\bgastric\b|\bstomach\b|\bgastritis\b", t):
        return "Gastric"
    if re.search(r"\bduodenal\b|\bduodenum\b", t):
        return "Duodenal"
    if re.search(r"\bsmall intestin|\bsmall bowel\b|\bileal\b|\bileum\b", t):
        return "Small intestinal"
    if re.search(r"\bcolon|\brect|\bcecum|\bsigmoid\b", t):
        return "Colonic"
    return "Unknown"


def extract_inflammation(text: str) -> str:
    t = text.lower()
    if re.search(r"active chronic", t):
        return "Active chronic"
    if re.search(r"active (colitis|inflammation|inflam)", t):
        return "Active"
    if re.search(r"chronic inactive", t):
        return "Chronic inactive"
    if re.search(r"mild chronic", t):
        return "Mild chronic"
    if re.search(r"no significant patholog|no evidence of|without.*abnormal|no patholog|unremarkable", t):
        return "None"
    return "Unknown"


def extract_finding(text: str) -> str:
    t = text.lower()
    if re.search(r"dysplasia|carcinoma|adenocarcinoma|malignant", t):
        return "Dysplasia/Carcinoma"
    if re.search(r"lymphoma|malt|b-cell lymph|lymphoproliferative", t):
        return "Lymphoma"
    if re.search(r"lymphocytic colitis", t):
        return "Lymphocytic colitis"
    if re.search(r"active (chronic )?colitis|active colitis", t):
        return "Active colitis"
    if re.search(r"chronic (inactive )?colitis|chronic colitis|chronic inflam", t):
        return "Chronic colitis"
    if re.search(r"gastritis", t):
        return "Gastritis"
    if re.search(r"hyperplastic polyp", t):
        return "Hyperplastic polyp"
    if re.search(r"lymphoid aggregate|lymphoid follicle", t):
        return "Lymphoid aggregate"
    if re.search(r"no significant patholog|no evidence of|without.*abnormal|no patholog|unremarkable|no diagnostic abnormal", t):
        return "Normal"
    return "Other"


def flag(text: str, pattern: str) -> str:
    return "Yes" if re.search(pattern, text.lower()) else "No"


def extract_features(report: str) -> dict:
    return {
        "report_tissue":        extract_tissue(report),
        "report_inflammation":  extract_inflammation(report),
        "report_finding":       extract_finding(report),
        "report_active_inflam": flag(report, r"neutrophil|crypt abscess|active (colitis|inflam)"),
        "report_granuloma":     flag(report, r"granuloma"),
        "report_dysplasia":     flag(report, r"dysplasia|carcinoma|adenocarcinoma"),
        "report_crypt_injury":  flag(report, r"crypt (abscess|injur|dropout|destruct|distort|irregul)"),
        "report_h_pylori":      flag(report, r"helicobacter|h\.?\s*pylori"),
    }


# ── data loading ───────────────────────────────────────────────────────────

def load_report_features() -> pd.DataFrame:
    rows = []
    with open(JSONL_PATH) as f:
        for line in f:
            r = json.loads(line.strip())
            if r["prompt"] != PROMPT:
                continue
            feats = extract_features(r["report"])
            feats["slide_id"] = r["slide"]
            feats["report"]   = r["report"]
            rows.append(feats)
    df = pd.DataFrame(rows).drop_duplicates("slide_id").set_index("slide_id")
    print(f"  Extracted report features for {len(df)} slides")
    return df


# ── HTML (interactive, dropdown) ──────────────────────────────────────────

def _build_continuous_trace(df: pd.DataFrame, col: str, title: str) -> list:
    """Single Scattergl trace with a continuous Viridis colour scale for P(Yes) scores."""
    mask = df[col].notna()
    sub = df[mask]
    if sub.empty:
        return []
    trace = go.Scattergl(
        x=sub["umap_x"], y=sub["umap_y"],
        mode="markers",
        name=title,
        marker=dict(
            color=sub[col],
            colorscale="Viridis",
            cmin=0, cmax=1,
            size=5, opacity=0.85,
            line=dict(width=0.3, color="rgba(255,255,255,0.2)"),
            colorbar=dict(
                title=dict(text="P(Yes)", side="right"),
                thickness=14, len=0.7,
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0", "0.25", "0.5", "0.75", "1"],
            ),
            showscale=True,
        ),
        text=sub.apply(lambda r: (
            f"<b>{r.name}</b><br>"
            f"{title}: <b>{r[col]:.3f}</b><br>"
            f"Diagnosis [GT]: {r.get('diagnosis','?')}<br>"
            f"Normal/Lesional [GT]: {r.get('normal_lesional','?')}<br>"
            f"Tissue site [GT]: {r.get('tissue_site','?')}<br>"
            f"Disease activity [GT]: {r.get('disease_activity_60','?')}"
        ), axis=1),
        hovertemplate="%{text}<extra></extra>",
        visible=False,
    )
    return [trace]


def _build_traces(df: pd.DataFrame, col: str, palette: dict) -> list:
    traces = []
    order = [k for k in palette if k != "Unknown"] + ["Unknown"]
    for cat in order:
        mask = df[col] == cat
        if not mask.any():
            continue
        sub = df[mask]
        traces.append(go.Scattergl(
            x=sub["umap_x"], y=sub["umap_y"],
            mode="markers", name=cat,
            marker=dict(color=palette[cat], size=5, opacity=0.75,
                        line=dict(width=0.5, color="white")),
            text=sub.apply(lambda r: (
                f"<b>{r.name}</b><br>"
                f"Diagnosis [GT]: {r.get('diagnosis','?')}<br>"
                f"Normal/Lesional [GT]: {r.get('normal_lesional','?')}  "
                f"Macro [GT]: {r.get('macroscopic_appearance','?')}<br>"
                f"Finding (report): {r.get('report_finding','?')}<br>"
                f"Tissue (report): {r.get('report_tissue','?')}  "
                f"Inflam (report): {r.get('report_inflammation','?')}<br>"
                f"Active: {r.get('report_active_inflam','?')}  "
                f"Granuloma: {r.get('report_granuloma','?')}  "
                f"Dysplasia: {r.get('report_dysplasia','?')}<br>"
                f"<i>{r.get('report','')[:120]}…</i>"
            ), axis=1),
            hovertemplate="%{text}<extra></extra>",
            legendgroup=col, visible=True,
        ))
    return traces


def make_html(df: pd.DataFrame, out_path: Path):
    # Order: report vars, GT vars, then UAMP continuous
    color_vars = list(REPORT_PALETTES.keys()) + list(GT_PALETTES.keys())
    all_groups = [(col, _build_traces(df, col, ALL_PALETTES[col])) for col in color_vars]

    # Add UAMP continuous groups (only for slides that have scores)
    uamp_present = [c for c in UAMP_COLS if c in df.columns and df[c].notna().any()]
    uamp_groups = [(col, _build_continuous_trace(df, col, UAMP_TITLES[col]))
                   for col in uamp_present]

    all_groups_combined = all_groups + uamp_groups

    fig = go.Figure()
    for _, traces in all_groups_combined:
        for t in traces:
            t.visible = False
            fig.add_trace(t)
    for t in all_groups_combined[0][1]:
        t.visible = True

    flat = [t for _, trs in all_groups_combined for t in trs]
    buttons, cum = [], 0
    for col, traces in all_groups_combined:
        if col in LABEL_TITLES:
            label = LABEL_TITLES[col]
        else:
            label = UAMP_TITLES.get(col, col)
        vis = [False] * len(flat)
        for j in range(cum, cum + len(traces)):
            vis[j] = True
        buttons.append(dict(
            label=label, method="update",
            args=[{"visible": vis},
                  {"title": f"PRISM2 Diagnostic UMAP — {label}"}],
        ))
        cum += len(traces)

    # Separators between sections
    sep_gt = len(REPORT_PALETTES)
    sep_uamp = len(REPORT_PALETTES) + len(GT_PALETTES)
    buttons.insert(sep_gt, dict(label="── Ground Truth ──", method="skip", args=[]))
    if uamp_groups:
        buttons.insert(sep_uamp + 1, dict(label="── UAMP Scores ──", method="skip", args=[]))

    fig.update_layout(
        title=dict(text=f"PRISM2 Diagnostic UMAP — {LABEL_TITLES[color_vars[0]]}",
                   font=dict(size=16)),
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.01, xanchor="left", y=1.13, yanchor="top",
            showactive=True, bgcolor="#F0F0F0", bordercolor="#CCCCCC",
        )],
        annotations=[dict(text="Colour by:", x=0.01, xref="paper",
                          y=1.17, yref="paper", showarrow=False, font=dict(size=12))],
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(itemsizing="constant", font=dict(size=11)),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        width=1000, height=700, margin=dict(t=110, r=20, b=40, l=60),
    )
    fig.add_annotation(text=f"{len(df)} slides · top section = report-extracted · bottom section = ground truth",
                       x=1, xref="paper", y=-0.05, yref="paper",
                       showarrow=False, font=dict(size=10, color="#888888"), xanchor="right")
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  HTML → {out_path}")


# ── single UMAP static plots ───────────────────────────────────────────────

def _scatter_ax(ax, df: pd.DataFrame, col: str, palette: dict, title: str, n_label: str):
    ax.set_facecolor("#FAFAFA")
    order = [k for k in palette if k != "Unknown"] + ["Unknown"]
    for cat in order:
        mask = df[col] == cat
        if not mask.any():
            continue
        ax.scatter(df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
                   c=palette[cat], s=6, alpha=0.7, linewidths=0,
                   rasterized=True, label=cat)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [mpatches.Patch(color=palette[c], label=c)
               for c in order if (df[col] == c).any()]
    ax.legend(handles=handles, fontsize=7, title_fontsize=8,
              loc="lower right", framealpha=0.85,
              markerscale=1.4, handlelength=1.0)
    ax.text(1, -0.06, n_label, transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="#888888")


def save_single_static(df: pd.DataFrame, out_dir: Path, dpi: int = 200):
    """One PNG+PDF per report variable (solo plots)."""
    for col, palette in REPORT_PALETTES.items():
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("white")
        n_known = (df[col] != "Unknown").sum()
        _scatter_ax(ax, df, col, palette, LABEL_TITLES[col],
                    f"n={len(df)} · {n_known} labelled")
        stem = f"umap_prism2_diagnostic_report_{col.replace('report_', '')}"
        fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"    {stem}.png/.pdf")


# ── side-by-side comparison plots ─────────────────────────────────────────

def save_comparison_plots(df: pd.DataFrame, out_dir: Path, dpi: int = 200):
    """Side-by-side: report-extracted (left) vs ground truth (right)."""
    for (r_col, r_title, gt_col, gt_title, shared_pal, stem) in COMPARE_PAIRS:
        # Only use slides where both columns are known
        both = df[(df[r_col] != "Unknown") & (df[gt_col].notna()) & (df[gt_col] != "Unknown")]
        n = len(both)
        if n == 0:
            print(f"    [SKIP] {stem}: no overlapping data")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        fig.patch.set_facecolor("white")

        # Compute a simple per-category accuracy note for active_inflam comparison
        accuracy_note = ""
        if r_col == "report_active_inflam" and gt_col == "normal_lesional":
            # Yes→Lesional, No→Normal: compute agreement
            agree = ((both[r_col] == "Yes") & (both[gt_col] == "Lesional")) | \
                    ((both[r_col] == "No")  & (both[gt_col] == "Normal"))
            pct = 100 * agree.sum() / n
            accuracy_note = f"Agreement: {agree.sum()}/{n} ({pct:.1f}%)"

        # Left: report
        r_pal = {k: v for k, v in shared_pal.items()
                 if k in df[r_col].unique() or k == "Unknown"}
        _scatter_ax(axes[0], both, r_col, r_pal,
                    f"Report-extracted\n{r_title}", f"n={n}")

        # Right: ground truth (restrict palette to values present)
        gt_pal = {k: v for k, v in shared_pal.items()
                  if k in df[gt_col].unique() or k == "Unknown"}
        _scatter_ax(axes[1], both, gt_col, gt_pal,
                    f"Ground truth\n{gt_title}", f"n={n}")

        # Super title
        suptitle = f"PRISM2 Report vs Ground Truth — {r_col.replace('report_','').replace('_',' ').title()}"
        if accuracy_note:
            suptitle += f"\n{accuracy_note}"
        fig.suptitle(suptitle, fontsize=13, y=1.01)
        fig.tight_layout()

        fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"    {stem}.png/.pdf")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting report features ...")
    feat_df = load_report_features()
    feat_df.to_csv(META_OUT)
    print(f"  Saved features → {META_OUT}")

    print("Loading UMAP coordinates ...")
    d = np.load(COORDS_NPZ, allow_pickle=True)
    coord_df = pd.DataFrame({"slide_id": list(d["slides"]),
                             "umap_x": d["xy"][:, 0],
                             "umap_y": d["xy"][:, 1]}).set_index("slide_id")

    print("Loading ground-truth metadata ...")
    meta = pd.read_csv(META_CSV, index_col="slide_id")
    gt_cols = list(GT_PALETTES.keys())
    for c in gt_cols:
        if c in meta.columns:
            meta[c] = meta[c].fillna("Unknown")
        else:
            meta[c] = "Unknown"

    # Combine: coords + report features + GT metadata
    df = coord_df.join(feat_df, how="inner").join(meta[gt_cols], how="left")
    for col in ALL_PALETTES:
        df[col] = df[col].fillna("Unknown").astype(str).replace("nan", "Unknown")

    # Join UAMP P(Yes) scores (left join — NaN for slides not yet scored)
    if UAMP_CSV.exists():
        uamp = pd.read_csv(UAMP_CSV, index_col="slide")
        uamp.index.name = "slide_id"
        available = [c for c in UAMP_COLS if c in uamp.columns]
        df = df.join(uamp[available], how="left")
        n_uamp = df[UAMP_COLS[0]].notna().sum()
        print(f"  Joined UAMP scores for {n_uamp}/{len(df)} slides")
    else:
        print(f"  UAMP CSV not found at {UAMP_CSV}, skipping")

    print(f"  {len(df)} slides total")

    print("Building interactive HTML ...")
    make_html(df, OUT_DIR / "umap_prism2_diagnostic_reports.html")

    print("Saving single-variable static plots ...")
    save_single_static(df, OUT_DIR)

    print("Saving side-by-side comparison plots ...")
    save_comparison_plots(df, COMPARE_DIR)

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
