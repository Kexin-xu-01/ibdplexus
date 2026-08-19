"""
Head-to-head comparison: tissue_threshold_15 vs tissue_threshold_15_filtered

Compares on the same patient cohort and CV splits:
  1. Random Forest CD vs UC performance (AUC, AP, Accuracy per fold)
  2. PRISM2 UAMP histological score distributions (11 features)
  3. UMAP structure: silhouette score, CD/UC separation, cluster stats

Outputs
-------
  <OUT_DIR>/comparison_rf.csv
  <OUT_DIR>/comparison_histoscore.csv
  <OUT_DIR>/comparison_umap.csv
  <OUT_DIR>/comparison_report.html
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = Path("/home/jovyan/kgbk271-ibd-volume")

BEFORE = dict(
    label="no_filter",
    rf_dir=BASE / "training/cd_vs_uc/02_04_imaging_allsites/results",
    histoscore_csv=BASE / "results/prism2/prism2_histological_score.csv",
    umap_base_npz=BASE / "results/prism2/umap/no_tissue_filter/umap_prism2_base_coords.npz",
    umap_diag_npz=BASE / "results/prism2/umap/no_tissue_filter/umap_prism2_diagnostic_coords.npz",
)

AFTER = dict(
    label="tissue_threshold_15_filtered",
    rf_dir=BASE / "training/cd_vs_uc/tissue_threshold_15_filtered/results",
    histoscore_csv=BASE / "results/prism2_tissue_threshold_15_filtered/prism2_histological_score.csv",
    umap_base_npz=BASE / "results/prism2/umap/tissue_threshold_15_filtered/umap_prism2_base_coords.npz",
    umap_diag_npz=BASE / "results/prism2/umap/tissue_threshold_15_filtered/umap_prism2_diagnostic_coords.npz",
)

SLIDE_META = BASE / "results/metadata/slide_metadata.csv"
OUT_DIR    = BASE / "results/comparison_no_filter_vs_filtered"

HISTO_COLS = [
    "inflammation_involvement", "crypt_architectural_distortion",
    "neutrophil_granulocytic_infiltration", "crypt_abscesses",
    "lymphoid_aggregates", "histiocytic_granulomas", "mucin_depletion",
    "pyloric_gland_metaplasia", "paneth_cell_metaplasia",
    "neuronal_hyperplasia", "muscular_hypertrophy",
]

DIAG_MAP = {
    "Crohn's Disease": "CD", "Crohn's disease": "CD",
    "Ulcerative Colitis": "UC", "Ulcerative colitis": "UC",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_rf_summary(rf_dir: Path, model: str) -> dict | None:
    p = rf_dir / f"{model}_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_rf_preds(rf_dir: Path, model: str) -> pd.DataFrame | None:
    p = rf_dir / f"{model}_slide_predictions.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_umap_coords(npz_path: Path) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    if not npz_path.exists():
        return None, None
    d = np.load(npz_path, allow_pickle=True)
    coords = d["xy"]
    slide_ids = list(d["slides"])
    return coords, slide_ids


def silhouette_for_diagnosis(coords, slide_ids, meta_df):
    sil_df = pd.DataFrame({"slide_id": slide_ids})
    sil_df = sil_df.merge(
        meta_df[["slide_id", "diagnosis_short"]].dropna(), on="slide_id", how="inner"
    )
    sil_df = sil_df[sil_df["diagnosis_short"].isin(["CD", "UC"])]
    if len(sil_df) < 10:
        return None, None, 0, 0

    id2idx = {s: i for i, s in enumerate(slide_ids)}
    idx = [id2idx[s] for s in sil_df["slide_id"] if s in id2idx]
    sil_df = sil_df[sil_df["slide_id"].isin(id2idx)]
    xy  = coords[idx]
    labels = LabelEncoder().fit_transform(sil_df["diagnosis_short"])
    sil = silhouette_score(xy, labels, sample_size=min(2000, len(labels)), random_state=42)
    n_cd = (sil_df["diagnosis_short"] == "CD").sum()
    n_uc = (sil_df["diagnosis_short"] == "UC").sum()
    return sil, sil_df, n_cd, n_uc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_df = pd.read_csv(SLIDE_META)
    meta_df["diagnosis_short"] = meta_df["diagnosis"].map(DIAG_MAP)

    conditions = [BEFORE, AFTER]

    # ── 1. RF metrics ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RANDOM FOREST  CD vs UC  (5-fold patient-level CV)")
    print("="*70)

    rf_rows = []
    for cond in conditions:
        for model in ["prism2_base", "prism2_diagnostic"]:
            s = load_rf_summary(cond["rf_dir"], model)
            if s is None:
                print(f"  [{cond['label']} / {model}]  NOT FOUND — skipping")
                continue
            row = dict(
                condition=cond["label"],
                model=model,
                mean_auc=s["mean_auc"], std_auc=s["std_auc"],
                mean_ap=s["mean_ap"],  std_ap=s["std_ap"],
                mean_acc=s["mean_acc"], std_acc=s["std_acc"],
                n_slides=sum(f["n_train"] + f["n_val"] for f in s["folds"]) // len(s["folds"]),
            )
            rf_rows.append(row)
            print(f"  {cond['label']:40s}  {model:22s}  "
                  f"AUC={s['mean_auc']:.4f}±{s['std_auc']:.4f}  "
                  f"AP={s['mean_ap']:.4f}±{s['std_ap']:.4f}  "
                  f"Acc={s['mean_acc']:.4f}±{s['std_acc']:.4f}")

    rf_df = pd.DataFrame(rf_rows)
    rf_df.to_csv(OUT_DIR / "comparison_rf.csv", index=False)

    # ── 2. PRISM2 histological scores ─────────────────────────────────────────
    print("\n" + "="*70)
    print("PRISM2 UAMP  HISTOLOGICAL SCORES  (mean P(Yes) per feature)")
    print("="*70)

    histo_rows = []
    histo_dfs = {}
    for cond in conditions:
        p = cond["histoscore_csv"]
        if not p.exists():
            print(f"  [{cond['label']}]  histoscore CSV NOT FOUND")
            continue
        df = pd.read_csv(p, index_col="slide")
        df.index.name = "slide_id"
        histo_dfs[cond["label"]] = df
        # merge with diagnosis
        df_meta = df.reset_index().merge(meta_df[["slide_id","diagnosis_short"]],
                                         on="slide_id", how="left")
        available = [c for c in HISTO_COLS if c in df.columns]
        means = df[available].mean()
        row = {"condition": cond["label"], "n_slides": len(df)}
        for col in available:
            row[col] = round(means[col], 4)
        histo_rows.append(row)
        print(f"\n  {cond['label']}  (n={len(df)} slides)")
        for col in available:
            print(f"    {col:45s} {means[col]:.4f}")

    if len(histo_dfs) == 2:
        labels = list(histo_dfs.keys())
        df_b = histo_dfs[labels[0]]
        df_a = histo_dfs[labels[1]]
        common = df_b.index.intersection(df_a.index)
        available = [c for c in HISTO_COLS if c in df_b.columns and c in df_a.columns]
        print(f"\n  Delta (filtered − unfiltered) on {len(common)} common slides:")
        for col in available:
            delta = df_a.loc[common, col].mean() - df_b.loc[common, col].mean()
            print(f"    {col:45s} {delta:+.4f}")

    histo_df = pd.DataFrame(histo_rows)
    histo_df.to_csv(OUT_DIR / "comparison_histoscore.csv", index=False)

    # ── 3. UMAP structure ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("UMAP  CD/UC SEPARATION  (silhouette score, higher=better separated)")
    print("="*70)

    umap_rows = []
    umap_data = {}
    for cond in conditions:
        for emb_key, npz_path in [("prism2_base", cond["umap_base_npz"]),
                                   ("prism2_diagnostic", cond["umap_diag_npz"])]:
            coords, slide_ids = load_umap_coords(npz_path)
            if coords is None:
                print(f"  [{cond['label']} / {emb_key}]  coords NOT FOUND")
                continue
            sil, sil_df, n_cd, n_uc = silhouette_for_diagnosis(coords, slide_ids, meta_df)
            if sil is None:
                print(f"  [{cond['label']} / {emb_key}]  no diagnosable slides")
                continue
            row = dict(condition=cond["label"], embedding=emb_key,
                       silhouette_CD_UC=round(sil, 4),
                       n_cd=int(n_cd), n_uc=int(n_uc),
                       n_total=len(slide_ids))
            umap_rows.append(row)
            umap_data[f"{cond['label']}_{emb_key}"] = (coords, slide_ids, sil_df)
            print(f"  {cond['label']:40s}  {emb_key:22s}  "
                  f"silhouette={sil:.4f}  (CD={n_cd}, UC={n_uc})")

    umap_df = pd.DataFrame(umap_rows)
    umap_df.to_csv(OUT_DIR / "comparison_umap.csv", index=False)

    # ── 4. HTML report ────────────────────────────────────────────────────────
    _write_html(rf_df, histo_df, umap_df, umap_data, meta_df, OUT_DIR)
    print(f"\nOutputs written to {OUT_DIR}/")


def _write_html(rf_df, histo_df, umap_df, umap_data, meta_df, out_dir):
    sections = []

    # RF table
    if not rf_df.empty:
        rf_pivot = rf_df.pivot_table(
            index="model", columns="condition",
            values=["mean_auc", "mean_ap", "mean_acc"]
        ).round(4)
        sections.append("<h2>Random Forest CD vs UC (5-fold CV)</h2>")
        sections.append(rf_pivot.to_html(classes="table"))

    # Histoscore table
    if not histo_df.empty:
        sections.append("<h2>PRISM2 Histological Scores (mean P(Yes))</h2>")
        available = [c for c in HISTO_COLS if c in histo_df.columns]
        sections.append(histo_df[["condition", "n_slides"] + available].to_html(
            index=False, classes="table"))

        # bar chart
        fig = go.Figure()
        cmap = {"tissue_threshold_15": "#2196F3", "tissue_threshold_15_filtered": "#FF9800"}
        for _, row in histo_df.iterrows():
            fig.add_trace(go.Bar(
                name=row["condition"],
                x=[c.replace("_", " ") for c in available],
                y=[row[c] for c in available],
                marker_color=cmap.get(row["condition"], None),
            ))
        fig.update_layout(barmode="group", title="Histological Score Comparison",
                          xaxis_tickangle=-45, height=500, template="plotly_white")
        sections.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    # UMAP scatter plots
    diag_colors = {"CD": "#E53935", "UC": "#1E88E5", "other": "#BDBDBD"}
    for key, (coords, slide_ids, sil_df) in umap_data.items():
        merged = pd.DataFrame({"slide_id": slide_ids})
        merged = merged.merge(meta_df[["slide_id", "diagnosis_short"]].fillna("other"),
                              on="slide_id", how="left")
        merged["diagnosis_short"] = merged["diagnosis_short"].fillna("other")
        x, y = coords[:, 0], coords[:, 1]

        fig = go.Figure()
        for diag in ["CD", "UC", "other"]:
            mask = merged["diagnosis_short"] == diag
            fig.add_trace(go.Scattergl(
                x=x[mask], y=y[mask],
                mode="markers",
                marker=dict(size=3, color=diag_colors[diag], opacity=0.6),
                name=diag,
                text=merged.loc[mask, "slide_id"],
            ))

        sil_val = umap_df.loc[umap_df.apply(
            lambda r: f"{r['condition']}_{r['embedding']}" == key, axis=1
        ), "silhouette_CD_UC"]
        sil_str = f"  silhouette={sil_val.values[0]:.4f}" if len(sil_val) else ""
        title = key.replace("_", " ") + sil_str
        fig.update_layout(title=title, template="plotly_white",
                          height=600, xaxis_title="UMAP-1", yaxis_title="UMAP-2")
        sections.append(f"<h2>UMAP: {title}</h2>")
        sections.append(fig.to_html(full_html=False, include_plotlyjs=False))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Tissue Filter Comparison</title>
<style>
  body {{ font-family: sans-serif; margin: 40px; }}
  h1 {{ color: #333; }}
  h2 {{ color: #555; margin-top: 40px; }}
  .table {{ border-collapse: collapse; margin-bottom: 20px; }}
  .table th, .table td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: right; }}
  .table th {{ background: #f5f5f5; }}
</style>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head><body>
<h1>tissue_threshold_15 vs tissue_threshold_15_filtered</h1>
<p>Same patient cohort, same 5-fold patient-level CV splits.</p>
{''.join(sections)}
</body></html>"""

    (out_dir / "comparison_report.html").write_text(html)


if __name__ == "__main__":
    main()
