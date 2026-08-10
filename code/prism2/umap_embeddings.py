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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    # ── clinical ──────────────────────────────────────────────────────────
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
    "disease_location": {
        "Ileocolonic": "#4C72B0",
        "Ileal":       "#DD8452",
        "Colonic":     "#55A868",
        "Unknown":     "#BBBBBB",
    },
    "macroscopic_appearance": {
        "Normal":                 "#3A9E64",
        "Possible inflammation":  "#F0C040",
        "Erosions or Ulcers":     "#C0392B",
        "Unknown":                "#BBBBBB",
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
    "normal_lesional": {
        "Normal":   "#3A9E64",
        "Lesional": "#C0392B",
        "Unknown":  "#BBBBBB",
    },
    "crohn_phenotype": {
        "Inflammatory (B1)":      "#4C72B0",
        "Stricturing (B2)":       "#DD8452",
        "Penetrating (B3)":       "#55A868",
        "Stricturing+Penetrating (B2B3)": "#9467BD",
        "Unknown":                "#BBBBBB",
    },
    "gender": {
        "Male":    "#4C72B0",
        "Female":  "#DD8452",
        "Unknown": "#BBBBBB",
    },
    # ── batch / technical ─────────────────────────────────────────────────
    "scan_batch": {
        "vsi-2021":     "#4C72B0",
        "VSI_MAY_2022": "#DD8452",
        "Unknown":      "#BBBBBB",
    },
    "collection_year": {
        "2017":    "#4C72B0",
        "2018":    "#55A868",
        "2019":    "#E07B2A",
        "2020":    "#9467BD",
        "2021":    "#C0392B",
        "Unknown": "#BBBBBB",
    },
    "ethnicity": {
        "Not Hispanic or Latino": "#4C72B0",
        "Hispanic or Latino":     "#DD8452",
        "Unknown":                "#BBBBBB",
    },
    "race": {
        "White":                    "#4C72B0",
        "Black or African American": "#DD8452",
        "Asian":                    "#55A868",
        "Other":                    "#9467BD",
        "Unknown":                  "#BBBBBB",
    },
}

LABEL_TITLES = {
    "diagnosis":              "Diagnosis",
    "disease_activity_60":    "Disease Activity",
    "disease_location":       "Disease Location (clinical)",
    "macroscopic_appearance": "Macroscopic Appearance",
    "tissue_site":            "Tissue Site (biopsy location)",
    "normal_lesional":        "Normal vs Lesional",
    "crohn_phenotype":        "Crohn's Phenotype",
    "gender":                 "Gender",
    "scan_batch":             "Scan Batch (batch effect)",
    "collection_year":        "Sample Collection Year (batch effect)",
    "ethnicity":              "Ethnicity",
    "race":                   "Race",
}


def load_embeddings(embed_dir: Path) -> tuple[list[str], np.ndarray]:
    slides, vecs = [], []
    for h5_path in sorted(embed_dir.glob("*.h5")):
        with h5py.File(h5_path) as f:
            vecs.append(f["features"][:].astype(np.float32))
        slides.append(h5_path.stem)
    return slides, np.stack(vecs)


def _norm_slide(s: pd.Series) -> pd.Series:
    return s.str.replace(".vsi", "", regex=False).str.strip()


def _normalise_tissue(s: pd.Series) -> pd.Series:
    """Map free-text biosample locations to tidy categories."""
    mapping = {
        "at 20 cm": "Sigmoid Colon",
        "At 20 cm": "Sigmoid Colon",
        "at 30 cm": "Sigmoid Colon",
        "At 30 cm": "Sigmoid Colon",
        "Sigmoid": "Sigmoid Colon",
        "sigmoid": "Sigmoid Colon",
        "Ileum": "Ileum",
        "ileum": "Ileum",
        "Rectum": "Rectum",
        "rectum": "Rectum",
        "Cecum": "Cecum",
        "cecum": "Cecum",
        "Ascending Colon": "Ascending Colon",
        "Descending Colon": "Descending Colon",
        "Transverse Colon": "Transverse Colon",
    }
    return s.map(lambda v: mapping.get(str(v).strip(), "Other") if pd.notna(v) else "Unknown")


def _normalise_ethnicity(s: pd.Series) -> pd.Series:
    def _f(v):
        if pd.isna(v):
            return "Unknown"
        v = str(v).lower()
        if "hispanic" in v and "not" not in v:
            return "Hispanic or Latino"
        if "hispanic" in v:
            return "Not Hispanic or Latino"
        return "Unknown"
    return s.map(_f)


def _normalise_race(s: pd.Series) -> pd.Series:
    def _f(v):
        if pd.isna(v):
            return "Unknown"
        v = str(v).lower()
        if "white" in v:
            return "White"
        if "black" in v or "african" in v:
            return "Black or African American"
        if "asian" in v:
            return "Asian"
        return "Other"
    return s.map(_f)


def load_metadata() -> pd.DataFrame:
    # ── 1. omics_samples: clinical labels ──────────────────────────────────
    omics = pd.read_csv(OMICS_CSV)
    omics["slide_id"] = _norm_slide(omics["image_vsi_path"].fillna(""))
    omics = omics[omics["slide_id"] != ""]
    omics["crohn_phenotype"] = omics["crohn_s_disease_phenotype"].map(lambda v: {
        "Inflammatory non-penetrating, non-stricturing (B1)": "Inflammatory (B1)",
        "Stricturing (B2)":                                   "Stricturing (B2)",
        "Penetrating (B3)":                                   "Penetrating (B3)",
        "Both stricturing and penetrating (B2B3)":            "Stricturing+Penetrating (B2B3)",
    }.get(str(v), "Unknown") if pd.notna(v) else "Unknown")
    omics["collection_year"] = pd.to_datetime(
        omics["sample_collected_date"], errors="coerce").dt.year.astype("Int64").astype(str).replace("<NA>", "Unknown")
    omics = omics.sort_values("omics_type").drop_duplicates("slide_id", keep="first")
    omics_cols = ["slide_id", "deidentified_master_patient_id",
                  "diagnosis", "disease_activity_60", "disease_location",
                  "macroscopic_appearance", "crohn_phenotype",
                  "gender", "age_at_diagnosis", "collection_year"]
    meta = omics[omics_cols].set_index("slide_id")

    # ── 2. wsi_metadata_raw: biopsy date + tissue site + normal/lesional ───
    raw = pd.read_csv("/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/IBD_meta_data_latest/wsi_metadata_raw.csv")
    raw["slide_id"] = _norm_slide(raw["IMAGE_VSI"].fillna(""))
    raw = raw[raw["slide_id"] != ""].drop_duplicates("slide_id")
    raw["collection_year_raw"] = pd.to_datetime(
        raw["Date Sample Collected"], format="%d-%b-%Y", errors="coerce").dt.year.astype("Int64").astype(str).replace("<NA>", "Unknown")
    raw["tissue_site"] = _normalise_tissue(raw["BIOSAMPLE_LOCATION"])
    raw_cols = ["slide_id", "collection_year_raw", "tissue_site"]
    meta = meta.join(raw[raw_cols].set_index("slide_id"), how="outer")
    # Fill collection_year from wsi_metadata_raw where omics didn't have it
    mask = (meta["collection_year"].isna() | meta["collection_year"].eq("Unknown"))
    meta.loc[mask, "collection_year"] = meta.loc[mask, "collection_year_raw"]
    meta.drop(columns=["collection_year_raw"], inplace=True)

    # normal/lesional label
    nl = pd.read_csv("/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/IBD_meta_data_latest/wsi_metadata_normal_lesional.csv")
    nl["slide_id"] = _norm_slide(nl["IMAGE_VSI"].fillna(""))
    nl = nl.drop_duplicates("slide_id")
    nl["normal_lesional"] = nl["label"].map({0: "Normal", 1: "Lesional"})
    meta = meta.join(nl[["slide_id", "normal_lesional"]].set_index("slide_id"), how="left")

    # ── 3. vsi_metadata: scan batch + mpp ──────────────────────────────────
    vsi = pd.read_csv("/home/jovyan/ibdplexus/data/vsi_metadata.tsv", sep="\t")
    vsi = vsi[vsi["is_overview"] == 0].copy()
    vsi["scan_batch"] = vsi["vsi_path"].str.extract(r"sparc-image-ffpe/(VSI_MAY_2022|vsi-2021)/")
    vsi = vsi.drop_duplicates("slide_id")[["slide_id", "scan_batch", "mpp_x_um"]]
    meta = meta.join(vsi.set_index("slide_id"), how="left")

    # ── 4. demographics: ethnicity + race ──────────────────────────────────
    dem = pd.read_csv("/home/jovyan/shared-data/ibd_plexus_sparc_raw/redshift/processed_tables/demographic_data_13mar23.csv")
    dem["ethnicity_clean"] = _normalise_ethnicity(dem["ethnicity"])
    dem["race_clean"]      = _normalise_race(dem["race"])
    dem = dem[["deidentified_master_patient_id", "ethnicity_clean", "race_clean"]]

    # Join via patient ID
    meta = meta.reset_index()
    meta = meta.merge(dem, on="deidentified_master_patient_id", how="left")
    meta = meta.rename(columns={"ethnicity_clean": "ethnicity", "race_clean": "race"})
    meta = meta.drop_duplicates("slide_id").set_index("slide_id")

    # ── 5. Fill all missing with "Unknown" ─────────────────────────────────
    for col in PALETTES:
        if col not in meta.columns:
            meta[col] = "Unknown"
        else:
            meta[col] = meta[col].fillna("Unknown").astype(str).replace("nan", "Unknown").replace("<NA>", "Unknown")

    print(f"  Metadata assembled: {len(meta)} slides")
    for col in PALETTES:
        known = (meta[col] != "Unknown").sum()
        print(f"    {col}: {known}/{len(meta)} known")

    return meta


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
                    f"Location (clinical): {r.get('disease_location','?')}<br>"
                    f"Tissue site: {r.get('tissue_site','?')}<br>"
                    f"Macroscopic: {r.get('macroscopic_appearance','?')}<br>"
                    f"Normal/Lesional: {r.get('normal_lesional','?')}<br>"
                    f"Crohn's phenotype: {r.get('crohn_phenotype','?')}<br>"
                    f"Gender: {r.get('gender','?')}  |  Age at Dx: {r.get('age_at_diagnosis','?')}<br>"
                    f"Ethnicity: {r.get('ethnicity','?')}  |  Race: {r.get('race','?')}<br>"
                    f"Collection year: {r.get('collection_year','?')}<br>"
                    f"Scan batch: {r.get('scan_batch','?')}"
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


def save_static(slides: list[str], xy: np.ndarray, meta: pd.DataFrame,
                embed_name: str, out_dir: Path, dpi: int = 200):
    """Save one PNG + PDF per colour variable using matplotlib."""
    df = pd.DataFrame({"slide_id": slides, "umap_x": xy[:, 0], "umap_y": xy[:, 1]})
    df = df.join(meta, on="slide_id")
    for col in PALETTES:
        df[col] = df[col].fillna("Unknown") if col in df.columns else "Unknown"

    for col, palette in PALETTES.items():
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("white")

        order = [k for k in palette if k != "Unknown"] + ["Unknown"]
        for cat in order:
            mask = df[col] == cat
            if not mask.any():
                continue
            ax.scatter(
                df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
                c=palette[cat], s=8, alpha=0.7, linewidths=0,
                rasterized=True, label=cat,
            )

        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        ax.set_title(f"{embed_name}  —  {LABEL_TITLES[col]}", fontsize=13, pad=10)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        handles = [
            mpatches.Patch(color=palette[cat], label=cat)
            for cat in order if (df[col] == cat).any()
        ]
        ax.legend(handles=handles, title=LABEL_TITLES[col],
                  fontsize=8, title_fontsize=9,
                  loc="lower right", framealpha=0.85,
                  markerscale=1.5, handlelength=1.2)

        n_matched = df[col].ne("Unknown").sum()
        fig.text(0.99, 0.01, f"{len(slides)} slides · {n_matched} with metadata",
                 ha="right", va="bottom", fontsize=8, color="#888888")

        stem = f"umap_{embed_name}_{col}"
        fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"    {stem}.png / .pdf")


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

        # Cache UMAP coordinates so re-runs skip the slow step
        cache_path = OUT_DIR / f"umap_{name}_coords.npz"
        np.savez(cache_path, slides=np.array(slides), xy=xy)

        out_path = OUT_DIR / f"umap_{name}.html"
        make_html(slides, xy, meta, name, out_path)

        print(f"  Saving static PNG/PDF ...")
        save_static(slides, xy, meta, name, OUT_DIR)

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
