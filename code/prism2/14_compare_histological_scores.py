"""
Compare PRISM2 UAMP histological scores across dataset versions.

For each of the 11 UAMP terms, shows how P(Yes) distributions shift as
successive QC filters are applied:
  trident_processed → tissue_threshold_15 → tissue_threshold_15_filtered
  → no_darkspot → manual_knn

Outputs (all in <out_dir>/):
  score_distributions.html   — interactive violin + mean-line plot per term
  score_means.html           — heatmap of mean P(Yes) per term × dataset
  per_slide_delta.html       — per-slide P(Yes) change: manual_knn vs no_darkspot
  comparison_summary.csv     — mean ± SD per term per dataset

Usage:
  python 17_compare_histological_scores.py [--out_dir PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_ROOT = Path("/home/jovyan/kgbk271-ibd-volume/results")

DATASETS = [
    ("trident_processed",    RESULTS_ROOT / "prism2"                          / "prism2_histological_score.csv"),
    ("tissue_threshold_15",  RESULTS_ROOT / "prism2_tissue_threshold_15"      / "prism2_histological_score.csv"),
    ("filtered",             RESULTS_ROOT / "prism2_tissue_threshold_15_filtered" / "prism2_histological_score.csv"),
    ("no_darkspot",          RESULTS_ROOT / "prism2_no_darkspot"              / "prism2_histological_score.csv"),
    ("manual_knn",           RESULTS_ROOT / "prism2_manual_knn"               / "prism2_histological_score.csv"),
]

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

TERM_LABELS = [c.replace("_", " ").title() for c in UAMP_COLS]

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def load_all() -> dict[str, pd.DataFrame]:
    dfs = {}
    for name, path in DATASETS:
        if not path.exists():
            print(f"  [SKIP] {name}: {path} not found")
            continue
        df = pd.read_csv(path).set_index("slide")
        # keep only UAMP columns present
        df = df[[c for c in UAMP_COLS if c in df.columns]]
        dfs[name] = df
        print(f"  {name}: {len(df)} slides")
    return dfs


def violin_html(dfs: dict, out_path: Path):
    n_terms = len(UAMP_COLS)
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=TERM_LABELS,
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    names = list(dfs.keys())
    showlegend_set = set()

    for idx, col in enumerate(UAMP_COLS):
        row, col_idx = divmod(idx, 4)
        row += 1; col_idx += 1

        for di, (name, df) in enumerate(dfs.items()):
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            show = name not in showlegend_set
            if show:
                showlegend_set.add(name)
            fig.add_trace(
                go.Violin(
                    y=vals,
                    name=name,
                    legendgroup=name,
                    showlegend=show,
                    line_color=PALETTE[di],
                    fillcolor=PALETTE[di],
                    opacity=0.55,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    width=0.9,
                ),
                row=row, col=col_idx,
            )

    fig.update_layout(
        title=dict(text="UAMP P(Yes) distributions by QC filter stage", font=dict(size=16)),
        violinmode="group",
        height=820, width=1300,
        legend=dict(
            title="Dataset",
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(size=11),
        ),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        margin=dict(t=100, b=40, l=50, r=20),
    )
    fig.update_yaxes(range=[-0.05, 1.05], tickvals=[0, 0.25, 0.5, 0.75, 1.0])

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  → {out_path}")


def heatmap_html(dfs: dict, out_path: Path):
    names = list(dfs.keys())
    means = np.array([
        [dfs[n][c].mean() if n in dfs and c in dfs[n].columns else np.nan
         for c in UAMP_COLS]
        for n in names
    ])

    fig = go.Figure(go.Heatmap(
        z=means,
        x=TERM_LABELS,
        y=names,
        colorscale="Viridis",
        zmin=0, zmax=1,
        text=np.round(means, 3),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="Mean P(Yes)", thickness=14),
    ))
    fig.update_layout(
        title=dict(text="Mean UAMP P(Yes) per term per dataset", font=dict(size=15)),
        xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        height=320 + 50 * len(names),
        width=1100,
        margin=dict(t=80, b=160, l=180, r=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  → {out_path}")


def delta_html(dfs: dict, out_path: Path):
    if "no_darkspot" not in dfs or "manual_knn" not in dfs:
        print("  [SKIP] delta plot: need no_darkspot and manual_knn")
        return

    base = dfs["no_darkspot"]
    new  = dfs["manual_knn"]
    common = base.index.intersection(new.index)
    delta = new.loc[common] - base.loc[common]   # positive = higher in manual_knn

    n_terms = len(UAMP_COLS)
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=TERM_LABELS,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for idx, col in enumerate(UAMP_COLS):
        row, col_idx = divmod(idx, 4)
        row += 1; col_idx += 1

        if col not in delta.columns:
            continue

        d = delta[col].dropna()
        mean_d = d.mean()
        fig.add_trace(
            go.Histogram(
                x=d.values,
                nbinsx=60,
                name=col,
                showlegend=False,
                marker_color="#4C72B0",
                opacity=0.75,
            ),
            row=row, col=col_idx,
        )
        # Vertical line at zero and at mean
        for xval, color, dash in [(0, "#888888", "dash"), (mean_d, "#C44E52", "solid")]:
            fig.add_vline(
                x=xval, line_dash=dash, line_color=color, line_width=1.5,
                row=row, col=col_idx,
            )

        # Annotation: mean delta + % slides that increased
        pct_up = 100 * (d > 0.01).mean()
        pct_dn = 100 * (d < -0.01).mean()
        fig.add_annotation(
            x=0.97, y=0.97, xref="x domain", yref="y domain",
            text=f"Δ̄={mean_d:+.3f}<br>↑{pct_up:.0f}% ↓{pct_dn:.0f}%",
            showarrow=False, align="right",
            font=dict(size=9, color="#333333"),
            bgcolor="rgba(255,255,255,0.8)",
            row=row, col=col_idx,
        )

    fig.update_layout(
        title=dict(
            text="Per-slide Δ P(Yes): manual_knn − no_darkspot<br>"
                 "<sup>Grey dashed = 0, red = mean Δ | ↑/↓ = fraction with |Δ| > 0.01</sup>",
            font=dict(size=15),
        ),
        height=840, width=1300,
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        margin=dict(t=110, b=40, l=50, r=20),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  → {out_path}")


def summary_csv(dfs: dict, out_path: Path):
    rows = []
    for name, df in dfs.items():
        for col in UAMP_COLS:
            if col not in df.columns:
                continue
            rows.append({
                "dataset": name,
                "term": col,
                "n_slides": df[col].notna().sum(),
                "mean": df[col].mean(),
                "sd": df[col].std(),
                "median": df[col].median(),
                "p25": df[col].quantile(0.25),
                "p75": df[col].quantile(0.75),
            })
    pd.DataFrame(rows).round(4).to_csv(out_path, index=False)
    print(f"  → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(RESULTS_ROOT / "prism2" / "compare_histological_scores"))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSVs ...")
    dfs = load_all()
    print(f"  {len(dfs)} datasets loaded\n")

    print("Score distributions (violin) ...")
    violin_html(dfs, out_dir / "score_distributions.html")

    print("Mean score heatmap ...")
    heatmap_html(dfs, out_dir / "score_means.html")

    print("Per-slide delta: manual_knn vs no_darkspot ...")
    delta_html(dfs, out_dir / "per_slide_delta.html")

    print("Summary CSV ...")
    summary_csv(dfs, out_dir / "comparison_summary.csv")

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
