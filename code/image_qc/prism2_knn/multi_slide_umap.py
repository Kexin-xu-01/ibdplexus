#!/usr/bin/env python3
"""
Build a combined interactive HTML UMAP viewer for patch embeddings across all slides.

Loads Virchow2 features from features_virchow2/*.h5, subsamples, runs a joint
PCA → UMAP, joins with patient metadata for multi-variable coloring, extracts
patch thumbnails from pyramidal TIFFs, and writes one self-contained HTML with
Plotly scatter + hover panel.

Usage:
  python multi_slide_umap.py
  python multi_slide_umap.py --max_patches 20000 --thumb_px 128
  python multi_slide_umap.py --n_slides 200
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from umap import UMAP

FEAT_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15/20x_224px_0px_overlap/features_virchow2")
TIFF_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
META_CSV = Path("/home/jovyan/kgbk271-ibd-volume/metadata/omics_samples_metadata.csv")
OUT_DIR  = Path("/home/jovyan/kgbk271-ibd-volume/results/umap_patch_viewer/tissue_threshold_15")

PATCH_PX = 672  # patch footprint at full res
PYRLEVEL = 1    # pyramid level (2× downsampled)
PYRDIV   = 2    # divisor matching PYRLEVEL

UNKNOWN_COLOR = "#888888"
TABLEAU10 = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
             "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"]

PALETTES = {
    "diagnosis": {
        "Crohn's Disease":    "#4e79a7",
        "Ulcerative Colitis": "#e15759",
        "IBD Unclassified":   "#f28e2b",
    },
    "disease_activity_60": {
        "Remission": "#59a14f",
        "Mild":      "#edc948",
        "Moderate":  "#f28e2b",
        "Severe":    "#e15759",
    },
    "macroscopic_appearance": {
        "Normal":               "#59a14f",
        "Possible inflammation":"#f28e2b",
        "Erosions or Ulcers":   "#e15759",
    },
    "disease_location": {
        "Ileal":       "#4e79a7",
        "Colonic":     "#e15759",
        "Ileocolonic": "#76b7b2",
        "Unknown":     "#888",
    },
    "gender": {
        "Male":   "#4e79a7",
        "Female": "#e15759",
    },
}

META_VARIABLES = [
    ("diagnosis",                  "Diagnosis"),
    ("disease_activity_60",        "Disease Activity"),
    ("macroscopic_appearance",     "Macroscopic Appearance"),
    ("disease_location",           "Disease Location"),
    ("characteristics_bio_material", "Tissue Site"),
    ("gender",                     "Gender"),
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir",       default=str(FEAT_DIR))
    p.add_argument("--tiff_dir",       default=str(TIFF_DIR))
    p.add_argument("--meta_csv",       default=str(META_CSV))
    p.add_argument("--out_dir",        default=str(OUT_DIR))
    p.add_argument("--n_slides",       type=int,   default=None,
                   help="Limit to first N slides (default: all)")
    p.add_argument("--max_patches",    type=int,   default=10000,
                   help="Max total patches sampled across all slides")
    p.add_argument("--pca_components", type=int,   default=50)
    p.add_argument("--umap_neighbors", type=int,   default=15)
    p.add_argument("--umap_min_dist",  type=float, default=0.1)
    p.add_argument("--thumb_px",       type=int,   default=160)
    p.add_argument("--jpeg_quality",   type=int,   default=72,
                   help="JPEG quality for thumbnails (lower = smaller file)")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--out_name",       default="all_slides_umap.html",
                   help="Output HTML filename (default: all_slides_umap.html)")
    p.add_argument("--exclusion_csv",  default=None,
                   help="CSV with slide_id,x,y columns to exclude (e.g. from qc_find_nn.py)")
    return p.parse_args()


# ── Metadata ──────────────────────────────────────────────────────────────────

def load_metadata(meta_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    df["slide_id"] = df["image_vsi_path"].str.replace(".vsi", "", regex=False)
    df = df.dropna(subset=["slide_id"])
    df = df.drop_duplicates(subset="slide_id", keep="first")
    return df.set_index("slide_id")


# ── Feature sampling ──────────────────────────────────────────────────────────

def load_exclusion_set(exclusion_csv: Path) -> set:
    df = pd.read_csv(exclusion_csv)
    return set(zip(df["slide_id"].astype(str), df["x"].astype(int), df["y"].astype(int)))


def sample_patches(slides, feat_dir: Path, max_patches: int, seed: int,
                   exclusion: set | None = None):
    rng = np.random.default_rng(seed)

    # Pass 1: count patches per slide
    counts = {}
    for slide in slides:
        h5 = feat_dir / f"{slide}.h5"
        if h5.exists():
            with h5py.File(h5) as f:
                counts[slide] = f["features"].shape[0]

    valid = list(counts.keys())
    total = sum(counts.values())
    n_sample = min(max_patches, total)
    print(f"  {len(valid)} slides, {total:,} patches → sampling {n_sample:,}", flush=True)

    # Proportional allocation (at least 1 per slide)
    alloc = {s: max(1, round(counts[s] / total * n_sample)) for s in valid}

    # Pass 2: load and subsample
    feats_list, coords_list, slide_ids = [], [], []
    slide_patch_map: dict[str, list[int]] = {}
    g = 0  # global patch index

    for slide in valid:
        h5 = feat_dir / f"{slide}.h5"
        with h5py.File(h5) as f:
            feats  = f["features"][:]
            coords = f["coords"][:]

        # Filter excluded patches before sampling
        if exclusion:
            keep = np.array([
                (slide, int(cx), int(cy)) not in exclusion
                for cx, cy in coords
            ])
            feats  = feats[keep]
            coords = coords[keep]

        n = len(feats)
        if n == 0:
            continue
        k = min(alloc[slide], n)
        if k < n:
            idx = rng.choice(n, k, replace=False)
            idx.sort()
            feats  = feats[idx]
            coords = coords[idx]

        slide_patch_map[slide] = list(range(g, g + len(feats)))
        g += len(feats)

        feats_list.append(feats)
        coords_list.append(coords)
        slide_ids.extend([slide] * len(feats))

    feats_all  = np.concatenate(feats_list,  axis=0)
    coords_all = np.concatenate(coords_list, axis=0)
    return feats_all, coords_all, slide_ids, slide_patch_map, total


# ── UMAP ──────────────────────────────────────────────────────────────────────

def run_umap(feats: np.ndarray, pca_d: int, n_neighbors: int, min_dist: float, seed: int):
    n = feats.shape[0]
    n_pca = min(pca_d, n - 1, feats.shape[1])
    print(f"  PCA {feats.shape[1]}→{n_pca} ...", end=" ", flush=True)
    reduced = PCA(n_components=n_pca, random_state=seed).fit_transform(feats)
    print("done")

    nn = min(n_neighbors, n - 1)
    print(f"  UMAP {n_pca}→2 (n={n:,}) ...", end=" ", flush=True)
    xy = UMAP(n_components=2, n_neighbors=nn, min_dist=min_dist, random_state=seed).fit_transform(reduced)
    print("done")
    return xy, n_pca


# ── Thumbnail extraction ───────────────────────────────────────────────────────

def _thumb_b64(sl: openslide.OpenSlide, x: int, y: int, thumb_px: int, jpeg_quality: int = 72) -> str:
    sz = PATCH_PX // PYRDIV  # patch size at PYRLEVEL (level 1 = 2× down)
    region = sl.read_region((x, y), PYRLEVEL, (sz, sz))
    img = region.convert("RGB").resize((thumb_px, thumb_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode()


def _placeholder(thumb_px: int) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.full((thumb_px, thumb_px, 3), 45, dtype=np.uint8)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_all_thumbs(slide_patch_map: dict, coords_all: np.ndarray,
                       tiff_dir: Path, thumb_px: int, jpeg_quality: int = 72) -> list[str]:
    n_total = sum(len(v) for v in slide_patch_map.values())
    thumbs = [None] * n_total
    ph = _placeholder(thumb_px)
    n_slides = len(slide_patch_map)

    for si, (slide, patch_indices) in enumerate(slide_patch_map.items()):
        tiff = tiff_dir / f"{slide}.tiff"
        print(f"  [{si+1}/{n_slides}] {slide} ({len(patch_indices)} patches) ...",
              end=" ", flush=True)
        if not tiff.exists():
            for gi in patch_indices:
                thumbs[gi] = ph
            print("no TIFF")
            continue
        try:
            with openslide.OpenSlide(tiff) as sl:
                for gi in patch_indices:
                    cx, cy = coords_all[gi]
                    thumbs[gi] = _thumb_b64(sl, int(cx), int(cy), thumb_px, jpeg_quality)
            print("done")
        except Exception as e:
            for gi in patch_indices:
                thumbs[gi] = ph
            print(f"ERROR: {e}")

    return thumbs


# ── Metadata colours ──────────────────────────────────────────────────────────

def _cat_colors(values: list[str], palette: dict | None):
    unique = sorted({v for v in values if v != "Unknown"}, key=str)
    if palette:
        cmap = {**{v: palette.get(v, UNKNOWN_COLOR) for v in unique},
                "Unknown": UNKNOWN_COLOR}
    else:
        cmap = {v: TABLEAU10[i % len(TABLEAU10)] for i, v in enumerate(unique)}
        cmap["Unknown"] = UNKNOWN_COLOR
    colors = [cmap.get(v, UNKNOWN_COLOR) for v in values]
    present = [v for v in list(cmap) if v in set(values)]
    legend = {v: cmap[v] for v in present}
    return colors, legend


def build_meta_colors(slide_ids: list[str], meta_df: pd.DataFrame) -> dict:
    result = {}
    for col, label in META_VARIABLES:
        vals = []
        for s in slide_ids:
            if s in meta_df.index:
                v = meta_df.loc[s, col]
                vals.append(str(v) if pd.notna(v) else "Unknown")
            else:
                vals.append("Unknown")
        colors, legend = _cat_colors(vals, PALETTES.get(col))
        result[col] = {"label": label, "values": vals, "colors": colors, "legend": legend}
    return result


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UMAP &mdash; All Slides</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #111; color: #e8e8e8;
          display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
  header {{ padding: 8px 16px; background: #1a1a1a; border-bottom: 1px solid #333;
             display: flex; align-items: center; gap: 14px; flex-shrink: 0; flex-wrap: wrap; }}
  header h1 {{ font-size: 15px; font-weight: 600; letter-spacing: 0.03em; white-space: nowrap; }}
  .stats {{ font-size: 12px; color: #666; white-space: nowrap; }}
  .ctrl {{ display: flex; align-items: center; gap: 7px; margin-left: auto; }}
  .ctrl label {{ font-size: 12px; color: #888; }}
  select {{ background: #222; color: #d0d0d0; border: 1px solid #444; border-radius: 5px;
             padding: 4px 28px 4px 8px; font-size: 12px; cursor: pointer;
             appearance: none; background-image:
               url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23888'/%3E%3C/svg%3E");
             background-repeat: no-repeat; background-position: right 8px center; }}
  select:focus {{ outline: none; border-color: #666; }}
  /* QC toolbar */
  #qc-bar {{ display: flex; align-items: center; gap: 8px; padding: 5px 16px;
              background: #141414; border-bottom: 1px solid #252525;
              flex-shrink: 0; flex-wrap: wrap; }}
  #qc-bar .sep {{ width: 1px; height: 18px; background: #2a2a2a; margin: 0 2px; }}
  .qc-lbl {{ font-size: 10px; color: #444; text-transform: uppercase;
              letter-spacing: 0.08em; white-space: nowrap; }}
  button {{ background: #222; color: #bbb; border: 1px solid #333; border-radius: 4px;
             padding: 3px 9px; font-size: 12px; cursor: pointer; white-space: nowrap; }}
  button:hover {{ background: #2c2c2c; border-color: #484848; }}
  button.btn-active {{ background: #2a2200; border-color: #886600; color: #ffcc44; }}
  button.btn-danger {{ background: #1e1010; border-color: #663333; color: #ee6666; }}
  button.btn-danger:hover {{ background: #2a1515; }}
  button.btn-export {{ background: #0f1e0f; border-color: #336633; color: #66cc66; }}
  button.btn-export:hover {{ background: #152615; }}
  .nn-wrap {{ display: flex; align-items: center; gap: 4px; }}
  .nn-wrap input[type=number] {{ width: 46px; background: #1e1e1e; color: #bbb;
    border: 1px solid #333; border-radius: 4px; padding: 3px 5px;
    font-size: 12px; text-align: center; }}
  #bad-count {{ font-size: 12px; color: #ee6666; margin-left: 4px; font-weight: 600; }}
  /* layout */
  #main {{ display: flex; flex: 1; overflow: hidden; }}
  #plot {{ flex: 1; min-width: 0; }}
  #panel {{ width: 290px; flex-shrink: 0; background: #181818;
             border-left: 1px solid #282828; display: flex; flex-direction: column;
             align-items: center; padding: 14px 14px 10px; gap: 10px; overflow-y: auto; }}
  .hint {{ font-size: 12px; color: #484848; text-align: center; margin-top: 28px;
            line-height: 1.6; }}
  #patch-img {{ width: 230px; height: 230px; border-radius: 6px;
                 border: 1px solid #2e2e2e; object-fit: cover; display: none;
                 image-rendering: auto; }}
  #patch-meta {{ font-size: 11px; color: #999; line-height: 1.85; display: none;
                  width: 100%; background: #202020; border-radius: 6px;
                  padding: 10px 12px; }}
  #patch-meta b {{ color: #d8d8d8; }}
  #patch-meta .bad-badge {{ display: inline-block; background: #3a1010;
    color: #ee5555; border-radius: 3px; padding: 1px 5px; font-size: 10px; }}
  #legend-box {{ width: 100%; margin-top: auto; padding-top: 10px;
                  border-top: 1px solid #242424; }}
  .leg-title {{ font-size: 10px; color: #4a4a4a; text-transform: uppercase;
                 letter-spacing: 0.08em; margin-bottom: 7px; }}
  .leg-item {{ display: flex; align-items: center; gap: 7px;
                font-size: 11px; color: #888; padding: 2px 0; }}
  .leg-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}
</style>
</head>
<body>
<header>
  <h1>UMAP &mdash; All Slides</h1>
  <span class="stats">{n_sampled:,} patches &thinsp;&middot;&thinsp; {n_slides:,} slides
    &thinsp;&middot;&thinsp; {n_total:,} total &thinsp;&middot;&thinsp;
    Virchow2 2560-d &rarr; PCA {pca_d}-d &rarr; UMAP 2-d</span>
  <div class="ctrl">
    <label for="csel">Color by</label>
    <select id="csel">{color_opts_html}</select>
  </div>
</header>

<!-- QC / bad-patch toolbar -->
<div id="qc-bar">
  <span class="qc-lbl">QC</span>
  <button id="btn-lasso" title="Toggle lasso select mode (or use Plotly toolbar)" onclick="toggleLasso()">&#9670; Lasso</button>
  <button onclick="markSelectedBad()" title="Add lasso-selected points to the bad set">&#9679; Mark Bad</button>
  <div class="sep"></div>
  <div class="nn-wrap">
    <button onclick="findNN()" title="Expand bad set by adding nearest neighbours (UMAP 2-D) of each bad patch">&#8857; Find NN</button>
    <span style="font-size:12px;color:#555">K=</span>
    <input id="nn-k" type="number" value="10" min="1" max="500" title="Neighbours per bad patch">
  </div>
  <div class="sep"></div>
  <button onclick="clearPending()" title="Clear current lasso selection without marking bad">Clear sel.</button>
  <button class="btn-danger" onclick="resetAll()" title="Remove all bad-patch markings">Reset all</button>
  <div class="sep"></div>
  <button class="btn-export" onclick="exportCSV()" title="Download bad patches as CSV for qc_find_nn.py">&#8595; Export CSV</button>
  <span id="bad-count"></span>
</div>

<div id="main">
  <div id="plot"></div>
  <div id="panel">
    <div class="hint" id="hint">Hover over a point<br>to see the patch</div>
    <img id="patch-img" alt="patch">
    <div id="patch-meta"></div>
    <div id="legend-box">
      <div class="leg-title" id="leg-title"></div>
      <div id="leg-items"></div>
    </div>
  </div>
</div>
<script>
const DATA = {data_json};

let currentKey = DATA.color_opts[0].key;

// ── Bad-patch state ──────────────────────────────────────────────────────────
const badSet     = new Set();   // indices confirmed as bad
let   pendingSet = new Set();   // indices in the current lasso selection
let   lassoActive = false;

// ── Plotly trace ─────────────────────────────────────────────────────────────
const trace = {{
  type: "scattergl",
  mode: "markers",
  x: DATA.x,
  y: DATA.y,
  marker: {{
    size: 4,
    color: DATA.colors[currentKey],
    opacity: 0.75,
    line: {{ width: 0 }},
  }},
  text: DATA.labels,
  hovertemplate: "%{{text}}<extra></extra>",
  customdata: DATA.x.map(function(_, i) {{ return i; }}),
  selected:   {{ marker: {{ color: "#ffdd00", size: 7, opacity: 1 }} }},
  unselected: {{ marker: {{ opacity: 0.35 }} }},
}};

const layout = {{
  paper_bgcolor: "#111",
  plot_bgcolor: "#111",
  margin: {{ l: 44, r: 10, t: 12, b: 44 }},
  xaxis: {{
    title: {{ text: "UMAP 1", font: {{ size: 11, color: "#4a4a4a" }} }},
    gridcolor: "#1c1c1c", zerolinecolor: "#282828",
    tickfont: {{ color: "#484848", size: 10 }},
  }},
  yaxis: {{
    title: {{ text: "UMAP 2", font: {{ size: 11, color: "#4a4a4a" }} }},
    gridcolor: "#1c1c1c", zerolinecolor: "#282828",
    tickfont: {{ color: "#484848", size: 10 }},
  }},
  hovermode: "closest",
  dragmode: "pan",
  selectdirection: "any",
}};

const config = {{
  scrollZoom: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["autoScale2d"],
  responsive: true,
}};

const gd = document.getElementById("plot");
Plotly.newPlot(gd, [trace], layout, config);

// ── Lasso toggle (custom button mirrors Plotly toolbar lasso) ────────────────
function toggleLasso() {{
  lassoActive = !lassoActive;
  Plotly.relayout(gd, {{ dragmode: lassoActive ? "lasso" : "pan" }});
  document.getElementById("btn-lasso").classList.toggle("btn-active", lassoActive);
}}

// Keep custom button in sync if user clicks Plotly's own mode buttons
gd.on("plotly_relayout", function(ev) {{
  if (ev.dragmode !== undefined) {{
    lassoActive = (ev.dragmode === "lasso" || ev.dragmode === "select");
    document.getElementById("btn-lasso").classList.toggle("btn-active", lassoActive);
  }}
}});

// ── Plotly selection events ──────────────────────────────────────────────────
gd.on("plotly_selected", function(ev) {{
  if (!ev || !ev.points) {{ pendingSet.clear(); return; }}
  pendingSet = new Set(ev.points.map(function(p) {{ return p.customdata; }}));
}});

gd.on("plotly_deselect", function() {{
  pendingSet.clear();
}});

// ── Mark pending selection as bad ────────────────────────────────────────────
function markSelectedBad() {{
  if (pendingSet.size === 0) {{
    alert("No patches selected.\\nUse the Lasso or Box-select tool first, then click Mark Bad.");
    return;
  }}
  pendingSet.forEach(function(i) {{ badSet.add(i); }});
  pendingSet.clear();
  Plotly.restyle(gd, {{ selectedpoints: [null] }});
  applyMarkers();
  updateBadCount();
}}

// ── Find K nearest neighbours (UMAP 2-D) of every bad patch ─────────────────
// Nearest-neighbour search uses UMAP coordinates embedded in the page.
// For full-feature-space NN across all slides, export the CSV and run qc_find_nn.py.
function findNN() {{
  if (badSet.size === 0) {{
    alert("No bad patches yet. Mark some patches as bad first.");
    return;
  }}
  const k = Math.max(1, parseInt(document.getElementById("nn-k").value, 10) || 10);
  const badArr = Array.from(badSet);
  let added = 0;

  for (let bi = 0; bi < badArr.length; bi++) {{
    const bx = DATA.x[badArr[bi]];
    const by = DATA.y[badArr[bi]];
    // collect distances to all non-bad points
    const dists = [];
    for (let j = 0; j < DATA.x.length; j++) {{
      if (badSet.has(j)) continue;
      const dx = DATA.x[j] - bx;
      const dy = DATA.y[j] - by;
      dists.push({{ idx: j, d2: dx * dx + dy * dy }});
    }}
    dists.sort(function(a, b) {{ return a.d2 - b.d2; }});
    const take = Math.min(k, dists.length);
    for (let di = 0; di < take; di++) {{
      if (!badSet.has(dists[di].idx)) {{
        badSet.add(dists[di].idx);
        added++;
      }}
    }}
  }}

  applyMarkers();
  updateBadCount();
  console.log("findNN: added " + added + " neighbours → " + badSet.size + " bad total");
}}

// ── Apply marker colours (bad = red, rest = colour scheme) ──────────────────
function applyMarkers() {{
  const base   = DATA.colors[currentKey];
  const colors  = base.slice();
  const sizes   = new Array(colors.length).fill(4);
  const opacity = new Array(colors.length).fill(0.75);
  badSet.forEach(function(i) {{
    colors[i]  = "#ff3333";
    sizes[i]   = 7;
    opacity[i] = 1.0;
  }});
  Plotly.restyle(gd, {{
    "marker.color":   [colors],
    "marker.size":    [sizes],
    "marker.opacity": [opacity],
    selectedpoints:   [null],
  }});
}}

// ── Clear pending selection (yellow highlights) without marking bad ───────────
function clearPending() {{
  pendingSet.clear();
  Plotly.restyle(gd, {{ selectedpoints: [null] }});
}}

// ── Reset all bad patches ────────────────────────────────────────────────────
function resetAll() {{
  if (!confirm("Clear all " + badSet.size + " bad-patch markings?")) return;
  badSet.clear();
  pendingSet.clear();
  Plotly.restyle(gd, {{
    "marker.color":   [DATA.colors[currentKey]],
    "marker.size":    [4],
    "marker.opacity": [0.75],
    selectedpoints:   [null],
  }});
  updateBadCount();
}}

// ── Update counter display ───────────────────────────────────────────────────
function updateBadCount() {{
  const el = document.getElementById("bad-count");
  el.textContent = badSet.size > 0 ? badSet.size + " bad" : "";
}}

// ── Export bad patches as CSV ────────────────────────────────────────────────
// Feed this file to qc_find_nn.py for full-feature-space NN search.
function exportCSV() {{
  if (badSet.size === 0) {{ alert("No bad patches to export."); return; }}
  const lines = ["global_index,slide_id,x,y,umap_x,umap_y"];
  const sorted = Array.from(badSet).sort(function(a, b) {{ return a - b; }});
  sorted.forEach(function(i) {{
    lines.push([
      i,
      DATA.slide_ids[i],
      DATA.slide_x[i],
      DATA.slide_y[i],
      DATA.x[i].toFixed(4),
      DATA.y[i].toFixed(4),
    ].join(","));
  }});
  const blob = new Blob([lines.join("\\n")], {{ type: "text/csv" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "bad_patches.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}}

// ── Color selector ───────────────────────────────────────────────────────────
document.getElementById("csel").addEventListener("change", function() {{
  currentKey = this.value;
  applyMarkers();
  renderLegend(currentKey);
}});

// ── Hover ────────────────────────────────────────────────────────────────────
let lastIdx = -1;
gd.on("plotly_hover", function(ev) {{
  const pt  = ev.points[0];
  const idx = pt.customdata;
  if (idx === lastIdx) return;
  lastIdx = idx;

  document.getElementById("hint").style.display = "none";

  const img = document.getElementById("patch-img");
  img.src = "data:image/jpeg;base64," + DATA.thumbs[idx];
  img.style.display = "block";

  const isBad = badSet.has(idx);
  const rows = [
    ["Slide",  DATA.slide_ids[idx]],
    ["Coord",  "(" + DATA.slide_x[idx] + ", " + DATA.slide_y[idx] + ") px"],
    ["UMAP",   "(" + pt.x.toFixed(3) + ", " + pt.y.toFixed(3) + ")"],
    ["Status", isBad ? "<span class=\\"bad-badge\\">BAD</span>" : "OK"],
  ];
  DATA.color_opts.forEach(function(opt) {{
    const v = DATA.meta[opt.key][idx];
    if (v && v !== "Unknown") rows.push([opt.label, v]);
  }});

  document.getElementById("patch-meta").innerHTML =
    rows.map(function(r) {{ return "<b>" + r[0] + "</b>: " + r[1]; }}).join("<br>");
  document.getElementById("patch-meta").style.display = "block";
}});

// ── Legend ────────────────────────────────────────────────────────────────────
function renderLegend(key) {{
  const legend = DATA.legends[key] || {{}};
  const opt    = DATA.color_opts.find(function(o) {{ return o.key === key; }}) || {{}};
  document.getElementById("leg-title").textContent = opt.label || "";
  document.getElementById("leg-items").innerHTML =
    Object.entries(legend)
      .map(function(e) {{
        return "<div class=\\"leg-item\\"><span class=\\"leg-dot\\" style=\\"background:" + e[1] + "\\"></span>" + e[0] + "</div>";
      }})
      .join("");
}}

renderLegend(currentKey);
</script>
</body>
</html>
"""


def build_html(umap_xy, coords_all, slide_ids, thumbs, meta_colors, n_total, pca_d):
    labels = [
        f"Slide: {slide_ids[i]} | ({coords_all[i,0]}, {coords_all[i,1]})"
        for i in range(len(slide_ids))
    ]
    color_opts = [{"key": col, "label": meta_colors[col]["label"]} for col in meta_colors]
    color_opts_html = "\n".join(
        f'<option value="{col}">{meta_colors[col]["label"]}</option>'
        for col in meta_colors
    )

    data = {
        "x":         umap_xy[:, 0].round(4).tolist(),
        "y":         umap_xy[:, 1].round(4).tolist(),
        "labels":    labels,
        "slide_ids": slide_ids,
        "slide_x":   coords_all[:, 0].tolist(),
        "slide_y":   coords_all[:, 1].tolist(),
        "thumbs":    thumbs,
        "meta":      {col: meta_colors[col]["values"] for col in meta_colors},
        "colors":    {col: meta_colors[col]["colors"] for col in meta_colors},
        "legends":   {col: meta_colors[col]["legend"] for col in meta_colors},
        "color_opts": color_opts,
    }

    return HTML_TEMPLATE.format(
        n_sampled=len(slide_ids),
        n_slides=len(set(slide_ids)),
        n_total=n_total,
        pca_d=pca_d,
        color_opts_html=color_opts_html,
        data_json=json.dumps(data, separators=(",", ":")),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    feat_dir = Path(args.feat_dir)
    tiff_dir = Path(args.tiff_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Slides to process
    all_h5 = sorted(feat_dir.glob("*.h5"))
    slides = [p.stem for p in all_h5]
    if args.n_slides:
        slides = slides[:args.n_slides]
    print(f"Processing {len(slides)} slide(s)  max_patches={args.max_patches:,}\n")

    # Metadata
    print("Loading metadata ...", flush=True)
    meta_df = load_metadata(Path(args.meta_csv))
    n_matched = sum(1 for s in slides if s in meta_df.index)
    print(f"  {n_matched}/{len(slides)} slides matched to metadata\n")

    # Exclusion list
    exclusion = None
    if args.exclusion_csv:
        excl_path = Path(args.exclusion_csv)
        exclusion = load_exclusion_set(excl_path)
        print(f"Exclusion list: {len(exclusion):,} patches loaded from {excl_path}\n")

    # Sample patches
    print("Sampling features ...")
    feats, coords, slide_ids, slide_patch_map, n_total = sample_patches(
        slides, feat_dir, args.max_patches, args.seed, exclusion)
    print()

    # UMAP
    print("Running UMAP ...")
    umap_xy, pca_d = run_umap(feats, args.pca_components, args.umap_neighbors,
                               args.umap_min_dist, args.seed)
    del feats
    print()

    # Metadata colours
    print("Building metadata colour arrays ...", flush=True)
    meta_colors = build_meta_colors(slide_ids, meta_df)
    print()

    # Thumbnails
    print("Extracting patch thumbnails ...")
    thumbs = extract_all_thumbs(slide_patch_map, coords, tiff_dir, args.thumb_px, args.jpeg_quality)
    print()

    # HTML
    print("Building HTML ...", end=" ", flush=True)
    html = build_html(umap_xy, coords, slide_ids, thumbs, meta_colors, n_total, pca_d)
    out_path = out_dir / args.out_name
    out_path.write_text(html)
    size_mb = out_path.stat().st_size / 1e6
    print(f"done → {out_path}  ({size_mb:.1f} MB)")

    # Also save a gzip-compressed copy for easier downloading
    import gzip
    gz_path = out_path.with_suffix(".html.gz")
    with gzip.open(gz_path, "wt", encoding="utf-8", compresslevel=6) as f:
        f.write(html)
    gz_mb = gz_path.stat().st_size / 1e6
    print(f"       {gz_path.name}  ({gz_mb:.1f} MB compressed — decompress with: gunzip {gz_path.name})")


if __name__ == "__main__":
    main()
