"""
Build an interactive HTML UMAP viewer for patch embeddings.

For each slide:
  - Loads Virchow2 features (2560-d) from features_virchow2/<slide>.h5
  - Runs PCA → 50-d then UMAP → 2-d
  - Extracts each patch thumbnail (224×224) from the pyramidal TIFF
  - Encodes thumbnails as base64 JPEG
  - Writes a self-contained HTML file with Plotly scatter +
    click-to-show-patch panel

Output:
  <OUT_DIR>/<slide>_umap.html   — one file per slide

Usage:
  python umap_patch_viewer.py                         # first 5 slides
  python umap_patch_viewer.py --slides 10407210HE1 10407220HE1
  python umap_patch_viewer.py --n_slides 20
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile
from PIL import Image
from sklearn.decomposition import PCA
from umap import UMAP

DEFAULT_FEAT_DIR = "/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/20x_224px_0px_overlap/features_virchow2"
TIFF_DIR  = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
OUT_DIR   = Path("/home/jovyan/kgbk271-ibd-volume/results/umap_patch_viewer")
PATCH_PX  = 672   # patch footprint in full-res pixels (page 0)
PYRLEVEL  = 1     # pyramid level to read (1 = 2× downsampled; faster than page 0)
PYRDIV    = 2     # divisor matching PYRLEVEL
THUMB_PX  = 224   # thumbnail size for hover/panel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir", type=str, default=DEFAULT_FEAT_DIR,
                   help="Directory containing per-slide virchow2 .h5 files.")
    p.add_argument("--slides", nargs="+", default=None,
                   help="Slide stems to process (without .h5/.tiff). Default: first --n_slides.")
    p.add_argument("--n_slides", type=int, default=5,
                   help="Number of slides to process when --slides is not set.")
    p.add_argument("--pca_components", type=int, default=50)
    p.add_argument("--umap_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--thumb_px", type=int, default=THUMB_PX)
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    return p.parse_args()


def load_features(h5_path: Path):
    with h5py.File(h5_path) as f:
        feats  = f["features"][:]   # (N, 2560)
        coords = f["coords"][:]     # (N, 2) — (x, y) in full-res pixels
    return feats, coords


def run_umap(feats: np.ndarray, pca_components: int, n_neighbors: int, min_dist: float):
    n = feats.shape[0]
    # PCA first for speed and noise reduction
    n_pca = min(pca_components, n - 1, feats.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    reduced = pca.fit_transform(feats)
    # UMAP needs n_neighbors < n_samples
    nn = min(n_neighbors, n - 1)
    um = UMAP(n_components=2, n_neighbors=nn, min_dist=min_dist, random_state=42)
    return um.fit_transform(reduced)


def load_slide_array(tiff_path: Path) -> np.ndarray:
    """Read pyramid level PYRLEVEL into memory once per slide."""
    t = tifffile.TiffFile(tiff_path)
    return t.pages[PYRLEVEL].asarray()


def extract_thumb_from_array(slide_arr: np.ndarray, x: int, y: int,
                              patch_px: int, thumb_px: int) -> str:
    """Slice patch from pre-loaded array (scaled coords), encode as base64 JPEG."""
    x2 = x // PYRDIV
    y2 = y // PYRDIV
    sz = patch_px // PYRDIV
    patch = slide_arr[y2 : y2 + sz, x2 : x2 + sz]
    if patch.shape[0] == 0 or patch.shape[1] == 0:
        patch = np.full((sz, sz, 3), 220, dtype=np.uint8)
    img = Image.fromarray(patch).resize((thumb_px, thumb_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


def color_by_position(coords: np.ndarray):
    """Return a colour string per patch based on normalised (x+y) position on slide."""
    xy_sum = coords[:, 0] + coords[:, 1]
    norm = (xy_sum - xy_sum.min()) / (xy_sum.max() - xy_sum.min() + 1e-9)
    # Map to a blue→red sequential scale via hex
    colors = []
    for v in norm:
        r = int(50  + v * 200)
        g = int(100 - v * 60)
        b = int(220 - v * 180)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors, norm.tolist()


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UMAP — {slide}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #111; color: #e8e8e8;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
  header {{ padding: 10px 16px; background: #1a1a1a; border-bottom: 1px solid #333;
            display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
  header h1 {{ font-size: 15px; font-weight: 600; letter-spacing: 0.03em; }}
  header span {{ font-size: 12px; color: #888; }}
  #main {{ display: flex; flex: 1; overflow: hidden; }}
  #plot {{ flex: 1; min-width: 0; }}
  #panel {{ width: 300px; flex-shrink: 0; background: #1a1a1a;
            border-left: 1px solid #2e2e2e; display: flex; flex-direction: column;
            align-items: center; padding: 16px; gap: 12px; overflow-y: auto; }}
  #panel .hint {{ font-size: 12px; color: #666; text-align: center; margin-top: 40px; }}
  #patch-img {{ width: 256px; height: 256px; border-radius: 6px;
                border: 1px solid #333; object-fit: cover; display: none; }}
  #patch-meta {{ font-size: 11px; color: #aaa; line-height: 1.7; display: none;
                 width: 100%; background: #242424; border-radius: 6px; padding: 10px 12px; }}
  #patch-meta b {{ color: #e0e0e0; }}
  #legend {{ margin-top: auto; padding-top: 12px; font-size: 11px; color: #666;
             border-top: 1px solid #2e2e2e; width: 100%; text-align: center; }}
  .grad {{ height: 8px; border-radius: 4px; margin: 4px 0;
           background: linear-gradient(to right, #3264dc, #7e5c82, #dc3250); }}
  .grad-labels {{ display: flex; justify-content: space-between; color: #555; font-size: 10px; }}
</style>
</head>
<body>
<header>
  <h1>UMAP &mdash; {slide}</h1>
  <span>{n_patches} patches &nbsp;·&nbsp; Virchow2 2560-d → PCA {pca_d}-d → UMAP 2-d</span>
</header>
<div id="main">
  <div id="plot"></div>
  <div id="panel">
    <div class="hint" id="hint">Click any dot to see the patch</div>
    <img id="patch-img" alt="patch">
    <div id="patch-meta"></div>
    <div id="legend">
      <div>Colour: slide position (top-left → bottom-right)</div>
      <div class="grad"></div>
      <div class="grad-labels"><span>top-left</span><span>bottom-right</span></div>
    </div>
  </div>
</div>
<script>
const DATA = {data_json};

const trace = {{
  type: "scattergl",
  mode: "markers",
  x: DATA.x,
  y: DATA.y,
  marker: {{
    size: 7,
    color: DATA.colors,
    line: {{ width: 0.5, color: "rgba(255,255,255,0.12)" }},
    opacity: 0.85,
  }},
  text: DATA.labels,
  hovertemplate: "%{{text}}<extra></extra>",
  customdata: DATA.idx,
}};

const layout = {{
  paper_bgcolor: "#111",
  plot_bgcolor: "#111",
  margin: {{ l: 40, r: 10, t: 10, b: 40 }},
  xaxis: {{ title: {{ text: "UMAP 1", font: {{ size: 11, color: "#666" }} }},
            gridcolor: "#222", zerolinecolor: "#333", tickfont: {{ color: "#555", size: 10 }} }},
  yaxis: {{ title: {{ text: "UMAP 2", font: {{ size: 11, color: "#666" }} }},
            gridcolor: "#222", zerolinecolor: "#333", tickfont: {{ color: "#555", size: 10 }} }},
  hovermode: "closest",
  dragmode: "pan",
}};

const config = {{ scrollZoom: true, displaylogo: false,
  modeBarButtonsToRemove: ["select2d","lasso2d","autoScale2d"],
  responsive: true }};

const gd = document.getElementById("plot");
Plotly.newPlot(gd, [trace], layout, config);

gd.on("plotly_click", function(ev) {{
  const pt  = ev.points[0];
  const idx = pt.customdata;
  const b64 = DATA.thumbs[idx];
  const x   = DATA.slide_x[idx];
  const y   = DATA.slide_y[idx];

  document.getElementById("hint").style.display = "none";

  const img = document.getElementById("patch-img");
  img.src = "data:image/jpeg;base64," + b64;
  img.style.display = "block";

  const meta = document.getElementById("patch-meta");
  meta.innerHTML = `
    <b>Patch index</b>: ${{idx}}<br>
    <b>Slide coord</b>: (${{x}}, ${{y}}) px<br>
    <b>UMAP</b>: (${{pt.x.toFixed(3)}}, ${{pt.y.toFixed(3)}})
  `;
  meta.style.display = "block";

  // Highlight selected point
  Plotly.restyle(gd, {{
    "marker.size": [DATA.idx.map(i => i === idx ? 14 : 7)],
    "marker.line.width": [DATA.idx.map(i => i === idx ? 2 : 0.5)],
    "marker.line.color": [DATA.idx.map(i => i === idx ? "#fff" : "rgba(255,255,255,0.12)")],
  }});
}});

// Hover: show thumb in tooltip via custom approach — Plotly handles it natively
</script>
</body>
</html>
"""


def build_html(slide: str, umap_xy: np.ndarray, coords: np.ndarray,
               thumbs: list[str], pca_d: int, args) -> str:
    n = len(thumbs)
    colors, _ = color_by_position(coords)
    labels = [f"patch {i} | ({coords[i,0]}, {coords[i,1]})" for i in range(n)]

    data = {
        "x":       umap_xy[:, 0].round(4).tolist(),
        "y":       umap_xy[:, 1].round(4).tolist(),
        "colors":  colors,
        "labels":  labels,
        "idx":     list(range(n)),
        "thumbs":  thumbs,
        "slide_x": coords[:, 0].tolist(),
        "slide_y": coords[:, 1].tolist(),
    }

    return HTML_TEMPLATE.format(
        slide=slide,
        n_patches=n,
        pca_d=pca_d,
        data_json=json.dumps(data),
    )


def process_slide(slide: str, args, out_dir: Path):
    h5_path   = Path(args.feat_dir) / f"{slide}.h5"
    tiff_path = TIFF_DIR / f"{slide}.tiff"
    out_path  = out_dir / f"{slide}_umap.html"

    if not h5_path.exists():
        print(f"[SKIP] {slide}: no features h5", file=sys.stderr)
        return
    if not tiff_path.exists():
        print(f"[SKIP] {slide}: no tiff", file=sys.stderr)
        return

    print(f"  Loading features ...", end=" ", flush=True)
    feats, coords = load_features(h5_path)
    n = feats.shape[0]
    print(f"{n} patches")

    print(f"  Running UMAP ...", end=" ", flush=True)
    pca_d = min(args.pca_components, n - 1, feats.shape[1])
    umap_xy = run_umap(feats, args.pca_components, args.umap_neighbors, args.umap_min_dist)
    print("done")

    print(f"  Loading TIFF pyramid level {PYRLEVEL} ...", end=" ", flush=True)
    slide_arr = load_slide_array(tiff_path)
    print(f"done ({slide_arr.shape[1]}×{slide_arr.shape[0]})")

    print(f"  Extracting {n} patch thumbnails ...", end=" ", flush=True)
    thumbs = []
    for x, y in coords:
        thumbs.append(extract_thumb_from_array(slide_arr, int(x), int(y), PATCH_PX, args.thumb_px))
    del slide_arr
    print("done")

    print(f"  Building HTML ...", end=" ", flush=True)
    html = build_html(slide, umap_xy, coords, thumbs, pca_d, args)
    out_path.write_text(html)
    size_mb = out_path.stat().st_size / 1e6
    print(f"done → {out_path.name} ({size_mb:.1f} MB)")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.slides:
        slides = args.slides
    else:
        all_h5 = sorted(Path(args.feat_dir).glob("*.h5"))
        slides = [p.stem for p in all_h5[: args.n_slides]]

    print(f"Processing {len(slides)} slide(s) → {out_dir}\n")
    for slide in slides:
        print(f"[{slide}]")
        try:
            process_slide(slide, args, out_dir)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
        print()

    print("All done.")


if __name__ == "__main__":
    main()
