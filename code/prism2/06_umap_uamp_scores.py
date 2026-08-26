"""
Visualise PRISM2 UAMP histological scores on UMAP.

Loads per-slide P(Yes) scores from prism2_histological_score.csv and overlays
them on pre-computed UMAP coordinates (from umap_embeddings.py).  Each of the
11 UAMP terms is selectable via a dropdown; colour encodes P(Yes) on a
continuous Viridis scale.  Clinical ground-truth variables are also included
as categorical dropdown options.

Inputs (required):
  --umap_csv     prism2_histological_score.csv produced by run_prism2_umap.py
  --coords_npz   one or more <embed>_coords.npz files from umap_embeddings.py
                 (e.g. umap_prism2_diagnostic_coords.npz)
  --out_dir      directory for output HTML files

Inputs (optional):
  --meta_csv     slide_metadata.csv with clinical columns (diagnosis, etc.)

Outputs (one per coords_npz):
  <out_dir>/umap_scores_<embed_name>.html

Usage:
  python umap_scores.py \\
    --umap_csv  .../results/prism2_manual_knn/prism2_histological_score.csv \\
    --coords_npz .../results/prism2/umap/manual_knn/umap_prism2_base_coords.npz \\
                 .../results/prism2/umap/manual_knn/umap_prism2_diagnostic_coords.npz \\
    --out_dir   .../results/prism2/umap/manual_knn
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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

UAMP_TITLES = {c: c.replace("_", " ").title() for c in UAMP_COLS}

GT_PALETTES = {
    "diagnosis": {
        "Crohn's Disease":    "#4C72B0",
        "Ulcerative Colitis": "#DD8452",
        "IBD Unclassified":   "#55A868",
        "Unknown":            "#BBBBBB",
    },
    "disease_activity_60": {
        "Remission": "#3A9E64",
        "Mild":      "#F0C040",
        "Moderate":  "#E07B2A",
        "Severe":    "#C0392B",
        "Unknown":   "#BBBBBB",
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

GT_LABEL_TITLES = {
    "diagnosis":              "Diagnosis [GT]",
    "disease_activity_60":    "Disease Activity [GT]",
    "normal_lesional":        "Normal vs Lesional [GT]",
    "macroscopic_appearance": "Macroscopic Appearance [GT]",
    "tissue_site":            "Tissue Site [GT]",
}


def _continuous_trace(df: pd.DataFrame, col: str, title: str) -> go.Scattergl:
    mask = df[col].notna()
    sub = df[mask]
    return go.Scattergl(
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
            + "".join(
                f"{k}: {r.get(k, '?')}<br>"
                for k in ["diagnosis", "normal_lesional", "tissue_site", "disease_activity_60"]
                if r.get(k, "?") != "?"
            )
        ), axis=1),
        hovertemplate="%{text}<extra></extra>",
        visible=False,
    )


def _categorical_traces(df: pd.DataFrame, col: str, palette: dict) -> list:
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
                + "".join(
                    f"{k}: {r.get(k, '?')}<br>"
                    for k in ["diagnosis", "normal_lesional", "tissue_site",
                               "disease_activity_60", "macroscopic_appearance"]
                )
            ), axis=1),
            hovertemplate="%{text}<extra></extra>",
            legendgroup=col, visible=False,
        ))
    return traces


def make_html(df: pd.DataFrame, embed_name: str, out_path: Path):
    groups = []

    # UAMP continuous groups
    available_uamp = [c for c in UAMP_COLS if c in df.columns and df[c].notna().any()]
    for col in available_uamp:
        groups.append((col, UAMP_TITLES[col], [_continuous_trace(df, col, UAMP_TITLES[col])]))

    # GT categorical groups
    for col, palette in GT_PALETTES.items():
        if col in df.columns:
            traces = _categorical_traces(df, col, palette)
            if traces:
                groups.append((col, GT_LABEL_TITLES[col], traces))

    if not groups:
        print(f"  [SKIP] no data to plot for {embed_name}")
        return

    fig = go.Figure()
    for _, _, traces in groups:
        for t in traces:
            fig.add_trace(t)

    # Show first group by default
    for t in groups[0][2]:
        t.visible = True

    flat = [t for _, _, trs in groups for t in trs]
    buttons, cum = [], 0

    # Separator label before GT section
    gt_start = len(available_uamp)

    for i, (col, label, traces) in enumerate(groups):
        if i == gt_start and gt_start > 0:
            buttons.append(dict(label="── Ground Truth ──", method="skip", args=[]))
        vis = [False] * len(flat)
        for j in range(cum, cum + len(traces)):
            vis[j] = True
        buttons.append(dict(
            label=label, method="update",
            args=[{"visible": vis},
                  {"title": f"{embed_name} UMAP — {label}"}],
        ))
        cum += len(traces)

    fig.update_layout(
        title=dict(text=f"{embed_name} UMAP — {UAMP_TITLES[available_uamp[0]] if available_uamp else ''}",
                   font=dict(size=16)),
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.01, xanchor="left", y=1.13, yanchor="top",
            showactive=True, bgcolor="#F0F0F0", bordercolor="#CCCCCC",
        )],
        annotations=[dict(
            text="Colour by:", x=0.01, xref="paper",
            y=1.17, yref="paper", showarrow=False, font=dict(size=12),
        )],
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(itemsizing="constant", font=dict(size=11)),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        width=1000, height=700,
        margin=dict(t=110, r=20, b=40, l=60),
    )
    n_uamp = df[available_uamp[0]].notna().sum() if available_uamp else 0
    fig.add_annotation(
        text=f"{len(df)} slides · {n_uamp} with UAMP scores",
        x=1, xref="paper", y=-0.05, yref="paper",
        showarrow=False, font=dict(size=10, color="#888888"), xanchor="right",
    )

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--umap_csv", required=True,
                   help="prism2_histological_score.csv from run_prism2_umap.py")
    p.add_argument("--coords_npz", nargs="+", required=True,
                   help="umap_*_coords.npz files from umap_embeddings.py")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for HTML files")
    p.add_argument("--meta_csv", default=None,
                   help="Optional slide_metadata.csv with clinical columns")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading UAMP scores from {args.umap_csv} ...")
    scores = pd.read_csv(args.umap_csv)
    scores = scores.rename(columns={"slide": "slide_id"})
    scores = scores.set_index("slide_id")
    print(f"  {len(scores)} slides, columns: {list(scores.columns)}")

    meta = None
    if args.meta_csv:
        print(f"Loading metadata from {args.meta_csv} ...")
        meta = pd.read_csv(args.meta_csv, index_col="slide_id")
        for col in GT_PALETTES:
            if col in meta.columns:
                meta[col] = meta[col].fillna("Unknown").astype(str).replace("nan", "Unknown")
            else:
                meta[col] = "Unknown"
        print(f"  {len(meta)} slides with metadata")

    for npz_path in args.coords_npz:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            print(f"[SKIP] coords not found: {npz_path}")
            continue

        d = np.load(npz_path, allow_pickle=True)
        slides = list(d["slides"])
        xy = d["xy"]

        embed_name = npz_path.stem.replace("umap_", "").replace("_coords", "")
        print(f"\n── {embed_name} ({len(slides)} slides) ──")

        df = pd.DataFrame({"slide_id": slides, "umap_x": xy[:, 0], "umap_y": xy[:, 1]})
        df = df.set_index("slide_id")

        df = df.join(scores[[c for c in UAMP_COLS if c in uamp.columns]], how="left")

        if meta is not None:
            gt_cols = [c for c in GT_PALETTES if c in meta.columns]
            df = df.join(meta[gt_cols], how="left")
            for col in GT_PALETTES:
                if col in df.columns:
                    df[col] = df[col].fillna("Unknown")
                else:
                    df[col] = "Unknown"

        n_scored = df[UAMP_COLS[0]].notna().sum() if UAMP_COLS[0] in df.columns else 0
        print(f"  {n_scored}/{len(df)} slides have UAMP scores")

        out_path = out_dir / f"umap_scores_{embed_name}.html"
        make_html(df, embed_name, out_path)

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
