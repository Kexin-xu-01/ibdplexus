"""
UMAP of slide-level embeddings coloured by IBD-relevant metadata.

Loads three embedding types:
  - prism2_base       (2560-dim)
  - prism2_diagnostic (3072-dim)
  - titan             (768-dim)

Links slides to patient/clinical metadata from omics_samples.csv.
Outputs one interactive HTML per embedding type, each with a colour
dropdown across: diagnosis, disease_activity, disease_location,
macroscopic_appearance, crohn_phenotype, gender.

Outputs: /home/jovyan/kgbk271-ibd-volume/results/prism2/umap/
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────
EMBED_DIRS = {
    "prism2_base": Path("/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/20x_224px_0px_overlap/prism2_base"),
    "prism2_diagnostic": Path("/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/20x_224px_0px_overlap/prism2_diagnostic"),
    "titan": Path("/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/20x_512px_0px_overlap/slide_features_titan"),
}
OMICS_CSV   = Path("/home/jovyan/shared-data/ibd_plexus_sparc_processed/omics_samples.csv")
OUT_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2/umap")

# ── categorical colour palettes (CVD-safe, fixed order) ────────────────────
# Validated against WCAG / OKLab ΔE ≥ 8 between adjacent pairs
PALETTES = {
    "diagnosis": {
        "Crohn's Disease":    "#4C72B0",   # blue
        "Ulcerative Colitis": "#DD8452",   # orange
        "IBD Unclassified":   "#55A868",   # green
        "Unknown":            "#BBBBBB",
    },
    "disease_activity_60": {
        "Remission": "#3A9E64",   # green
        "Mild":      "#F0C040",   # yellow
        "Moderate":  "#E07B2A",   # orange
        "Severe":    "#C0392B",   # red
        "Unknown":   "#BBBBBB",
    },
    "disease_location": {
        "Ileocolonic": "#4C72B0",   # blue
        "Ileal":       "#DD8452",   # orange
        "Colonic":     "#55A868",   # green
        "Unknown":     "#BBBBBB",
    },
    "macroscopic_appearance": {
        "Normal":                 "#3A9E64",
        "Possible inflammation":  "#F0C040",
        "Erosions or Ulcers":     "#C0392B",
        "Unknown":                "#BBBBBB",
    },
    "crohn_phenotype": {
        "Inflammatory non-penetrating, non-stricturing (B1)": "#4C72B0",
        "Stricturing (B2)":                                   "#DD8452",
        "Penetrating (B3)":                                   "#55A868",
        "Both stricturing and penetrating (B2B3)":            "#9467BD",
        "Unknown":                                            "#BBBBBB",
    },
    "gender": {
        "Male":    "#4C72B0",
        "Female":  "#DD8452",
        "Unknown": "#BBBBBB",
    },
}

LABEL_TITLES = {
    "diagnosis":              "Diagnosis",
    "disease_activity_60":    "Disease Activity",
    "disease_location":       "Disease Location",
    "macroscopic_appearance": "Macroscopic Appearance",
    "crohn_phenotype":        "Crohn's Phenotype",
    "gender":                 "Gender",
}


def load_embeddings(embed_dir: Path) -> tuple[list[str], np.ndarray]:
    slides, vecs = [], []
    for h5_path in sorted(embed_dir.glob("*.h5")):
        with h5py.File(h5_path) as f:
            vecs.append(f["features"][:].astype(np.float32))
        slides.append(h5_path.stem)
    return slides, np.stack(vecs)


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(OMICS_CSV)
    df["slide_id"] = df["image_vsi_path"].str.replace(".vsi", "", regex=False).str.strip()
    df = df[df["slide_id"].notna() & (df["slide_id"] != "")]

    # Crohn's phenotype: shorten for display
    df["crohn_phenotype"] = df["crohn_s_disease_phenotype"].fillna("Unknown")

    # One row per slide: prefer histology rows, else first
    df = df.sort_values("omics_type").drop_duplicates("slide_id", keep="first")

    keep = ["slide_id", "diagnosis", "disease_activity_60", "disease_location",
            "macroscopic_appearance", "crohn_phenotype", "gender",
            "deidentified_master_patient_id", "age_at_diagnosis"]
    return df[keep].set_index("slide_id")


def map_colour(series: pd.Series, palette: dict) -> list[str]:
    return [palette.get(v, palette["Unknown"]) for v in series]


def build_traces(df: pd.DataFrame, col: str, palette: dict) -> list[go.Scattergl]:
    """One trace per category value — enables legend toggle."""
    traces = []
    order = [k for k in palette if k != "Unknown"] + ["Unknown"]
    for cat in order:
        mask = df[col] == cat
        if not mask.any():
            continue
        sub = df[mask]
        traces.append(go.Scattergl(
            x=sub["umap_x"], y=sub["umap_y"],
            mode="markers",
            name=cat,
            marker=dict(color=palette[cat], size=5, opacity=0.75,
                        line=dict(width=0.5, color="white")),
            text=sub.apply(
                lambda r: (
                    f"<b>{r.name}</b><br>"
                    f"Diagnosis: {r.get('diagnosis','?')}<br>"
                    f"Activity: {r.get('disease_activity_60','?')}<br>"
                    f"Location: {r.get('disease_location','?')}<br>"
                    f"Macro: {r.get('macroscopic_appearance','?')}<br>"
                    f"Gender: {r.get('gender','?')}<br>"
                    f"Age at Dx: {r.get('age_at_diagnosis','?')}"
                ), axis=1
            ),
            hovertemplate="%{text}<extra></extra>",
            legendgroup=col,
            visible=True,
        ))
    return traces


def make_html(slides: list[str], xy: np.ndarray, meta: pd.DataFrame,
              embed_name: str, out_path: Path):
    df = pd.DataFrame({"slide_id": slides, "umap_x": xy[:, 0], "umap_y": xy[:, 1]})
    df = df.join(meta, on="slide_id")

    # Fill missing with "Unknown"
    for col in PALETTES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = "Unknown"

    color_vars = list(PALETTES.keys())
    n_matched = df["diagnosis"].ne("Unknown").sum()

    # Build all trace sets; show first by default
    all_traces = []
    visibility_sets = []   # list of bool lists, one per colour var

    for i, col in enumerate(color_vars):
        traces = build_traces(df, col, PALETTES[col])
        all_traces.extend(traces)
        n = len(traces)
        visibility_sets.append((i, n, traces))

    fig = go.Figure()
    for _, _, traces in visibility_sets:
        for t in traces:
            t.visible = False
        for t in traces:
            fig.add_trace(t)

    # Show first colour var by default
    first_col, first_n, first_traces = visibility_sets[0]
    for t in first_traces:
        t.visible = True

    # Build dropdown buttons
    buttons = []
    trace_list = [t for _, _, trs in visibility_sets for t in trs]
    cum = 0
    for i, (_, n, _) in enumerate(visibility_sets):
        vis = [False] * len(trace_list)
        for j in range(cum, cum + n):
            vis[j] = True
        buttons.append(dict(
            label=LABEL_TITLES[color_vars[i]],
            method="update",
            args=[{"visible": vis},
                  {"title": f"{embed_name} UMAP — {LABEL_TITLES[color_vars[i]]}"}],
        ))
        cum += n

    fig.update_layout(
        title=dict(text=f"{embed_name} UMAP — {LABEL_TITLES[color_vars[0]]}",
                   font=dict(size=16)),
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=0.01, xanchor="left",
            y=1.12, yanchor="top",
            showactive=True,
            bgcolor="#F0F0F0",
            bordercolor="#CCCCCC",
        )],
        annotations=[dict(
            text="Colour by:", x=0.01, xref="paper",
            y=1.16, yref="paper", showarrow=False, font=dict(size=12),
        )],
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False,
                   showticklabels=False),
        legend=dict(itemsizing="constant", tracegroupgap=2,
                    font=dict(size=11)),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        width=1000, height=700,
        margin=dict(t=100, r=20, b=40, l=60),
    )

    # Annotation: slide count
    fig.add_annotation(
        text=f"{len(slides)} slides · {n_matched} with clinical metadata",
        x=1, xref="paper", y=-0.04, yref="paper",
        showarrow=False, font=dict(size=10, color="#888888"),
        xanchor="right",
    )

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  → {out_path}")


def run_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.25,
             seed: int = 42) -> np.ndarray:
    print(f"  Running UMAP on {X.shape} ...", flush=True)
    X_scaled = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, metric="cosine",
                        random_state=seed, verbose=False)
    return reducer.fit_transform(X_scaled)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", nargs="+",
                   default=["prism2_base", "prism2_diagnostic", "titan"],
                   choices=list(EMBED_DIRS.keys()),
                   help="Which embedding types to visualise")
    p.add_argument("--n_neighbors", type=int, default=30)
    p.add_argument("--min_dist",    type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading metadata ...")
    meta = load_metadata()
    print(f"  {len(meta)} slides with clinical metadata")

    for name in args.embeddings:
        embed_dir = EMBED_DIRS[name]
        if not embed_dir.exists():
            print(f"[SKIP] {name}: directory not found")
            continue

        print(f"\n── {name} ──")
        slides, X = load_embeddings(embed_dir)
        print(f"  Loaded {len(slides)} slides, dim={X.shape[1]}")

        xy = run_umap(X, n_neighbors=args.n_neighbors, min_dist=args.min_dist)

        out_path = OUT_DIR / f"umap_{name}.html"
        make_html(slides, xy, meta, name, out_path)

    print(f"\nDone. HTML files in {OUT_DIR}")


if __name__ == "__main__":
    main()
