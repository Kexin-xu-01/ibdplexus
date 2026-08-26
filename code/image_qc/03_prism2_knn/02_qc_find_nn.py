#!/usr/bin/env python3
"""
Given a CSV of bad-quality patches exported from the UMAP viewer, find their
K nearest neighbours in the *full* patch pool (every patch in every H5 file)
using the original Virchow2 feature vectors, and write an expanded exclusion
list CSV.  Optionally generates a gallery HTML so you can visually inspect
the neighbours.

Typical workflow
----------------
1. Open all_slides_umap.html, lasso-select poor-quality patches, click
   "Mark Bad", then "Export CSV" → saves bad_patches.csv
2. Run this script:
       python qc_find_nn.py --bad_csv bad_patches.csv
   Or with gallery visualisation:
       python qc_find_nn.py --bad_csv bad_patches.csv --html_out nn_gallery.html
3. Feed exclusion_list.csv to downstream training / filtering code.

Memory note
-----------
Loading all H5 features can be large (e.g. 500 slides × 3000 patches × 2560-d
float32 ≈ 15 GB).  Use --max_slides or --slides_file to restrict the search
pool if needed, or run on a high-memory node.

Usage
-----
  python qc_find_nn.py --bad_csv bad_patches.csv
  python qc_find_nn.py --bad_csv bad_patches.csv --k 20 --out_csv exclusion_list.csv
  python qc_find_nn.py --bad_csv bad_patches.csv --html_out nn_gallery.html --max_vis 30
"""

import argparse
import base64
import io
import json
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors

FEAT_DIR = Path(
    "/home/jovyan/kgbk271-ibd-volume/data/processed/"
    "tissue_threshold_15/20x_224px_0px_overlap/features_virchow2"
)
TIFF_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")

PATCH_PX = 672   # patch footprint at full resolution
PYRLEVEL = 1     # pyramid level to read (level 1 = 2× downsampled)
PYRDIV   = 2     # divisor matching PYRLEVEL
THUMB_PX = 160   # thumbnail size in gallery


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bad_csv",    required=True,
                   help="CSV exported from the UMAP viewer (bad_patches.csv)")
    p.add_argument("--feat_dir",   default=str(FEAT_DIR),
                   help="Directory containing per-slide .h5 feature files")
    p.add_argument("--tiff_dir",   default=str(TIFF_DIR),
                   help="Directory containing per-slide .tiff files (for gallery)")
    p.add_argument("--k",             type=int,   default=10,
                   help="Nearest neighbours per bad patch (default: 10); used when --dist_threshold is not set, and as display cap in the gallery when it is")
    p.add_argument("--dist_threshold", type=float, default=None,
                   help="If set, include ALL neighbours within this L2 feature-space distance instead of fixed K")
    p.add_argument("--out_csv",    default="exclusion_list.csv",
                   help="Output exclusion list CSV (default: exclusion_list.csv)")
    p.add_argument("--html_out",   default=None,
                   help="If set, write a gallery HTML showing each bad patch + its NNs")
    p.add_argument("--max_vis",    type=int, default=50,
                   help="Max bad patches to show in gallery (default: 50)")
    p.add_argument("--thumb_px",   type=int, default=THUMB_PX,
                   help="Thumbnail size in gallery HTML (default: 160)")
    p.add_argument("--n_intervals",          type=int,   default=5,
                   help="Number of distance bins in the interval gallery (default: 5); "
                        "used when --dist_threshold is set")
    p.add_argument("--samples_per_interval", type=int,   default=5,
                   help="Patches to display per distance bin (default: 5)")
    p.add_argument("--thresholds_csv", default=None,
                   help="CSV with slide_id,x,y,threshold columns (exported from gallery UI) "
                        "to apply per-patch distance thresholds instead of a single global one")
    p.add_argument("--max_slides", type=int, default=None,
                   help="Limit pool to first N slides (for testing / memory)")
    p.add_argument("--slides_file", default=None,
                   help="Text file listing slide IDs (one per line) to include in pool")
    return p.parse_args()


# ── Feature loading ────────────────────────────────────────────────────────────

def load_bad_features(bad_df: pd.DataFrame, feat_dir: Path):
    """Return feature matrix and record list for all matched bad patches."""
    bad_feats, bad_records = [], []

    for slide_id, grp in bad_df.groupby("slide_id"):
        h5_path = feat_dir / f"{slide_id}.h5"
        if not h5_path.exists():
            print(f"  [SKIP] {slide_id}: H5 not found")
            continue
        with h5py.File(h5_path) as f:
            feats  = f["features"][:]
            coords = f["coords"][:]

        for _, row in grp.iterrows():
            tx, ty = int(row["x"]), int(row["y"])
            mask = (coords[:, 0] == tx) & (coords[:, 1] == ty)
            hit = np.where(mask)[0]
            if len(hit) == 0:
                print(f"  [WARN] {slide_id} ({tx},{ty}): coord not found in H5, skipping")
                continue
            bad_feats.append(feats[hit[0]])
            bad_records.append({"slide_id": slide_id, "x": tx, "y": ty,
                                 "reason": "manually_marked_bad"})

    return np.stack(bad_feats) if bad_feats else None, bad_records


def load_pool(feat_dir: Path, allowed_slides, max_slides):
    """Load features + coordinate records from all (or a subset of) H5 files."""
    all_h5 = sorted(feat_dir.glob("*.h5"))
    if allowed_slides is not None:
        allowed = set(allowed_slides)
        all_h5 = [h for h in all_h5 if h.stem in allowed]
    if max_slides is not None:
        all_h5 = all_h5[:max_slides]

    feats_list, records = [], []
    total = len(all_h5)
    for i, h5_path in enumerate(all_h5):
        if i % 50 == 0:
            print(f"  [{i+1}/{total}] {h5_path.stem} ...", flush=True)
        with h5py.File(h5_path) as f:
            feats  = f["features"][:]
            coords = f["coords"][:]
        feats_list.append(feats)
        for cx, cy in coords:
            records.append({"slide_id": h5_path.stem, "x": int(cx), "y": int(cy)})

    pool_feats = np.concatenate(feats_list, axis=0) if feats_list else np.empty((0, 0))
    return pool_feats, pd.DataFrame(records)


# ── Thumbnail extraction ───────────────────────────────────────────────────────

def _thumb_b64(tiff_dir: Path, slide_id: str, x: int, y: int, thumb_px: int) -> str:
    tiff = tiff_dir / f"{slide_id}.tiff"
    if not tiff.exists():
        return _placeholder_b64(thumb_px)
    try:
        with openslide.OpenSlide(tiff) as sl:
            sz = PATCH_PX // PYRDIV
            region = sl.read_region((x, y), PYRLEVEL, (sz, sz))
            img = region.convert("RGB").resize((thumb_px, thumb_px), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return _placeholder_b64(thumb_px)


def _placeholder_b64(thumb_px: int) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.full((thumb_px, thumb_px, 3), 40, dtype=np.uint8)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Gallery HTML ───────────────────────────────────────────────────────────────

GALLERY_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>QC — KNN Gallery</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #111; color: #ddd; padding: 20px; }}
  h1 {{ font-size: 16px; font-weight: 600; margin-bottom: 4px; color: #eee; }}
  .subtitle {{ font-size: 12px; color: #555; margin-bottom: 14px; }}
  /* ── global controls bar ── */
  #ctrl-bar {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
               background: #181818; border: 1px solid #2a2a2a; border-radius: 7px;
               padding: 10px 14px; margin-bottom: 22px; position: sticky; top: 10px; z-index: 50; }}
  #ctrl-bar .lbl {{ font-size: 11px; color: #555; white-space: nowrap; }}
  #ctrl-bar input[type=number] {{ width: 64px; background: #222; color: #ccc;
    border: 1px solid #3a3a3a; border-radius: 4px; padding: 4px 6px; font-size: 12px; text-align:center; }}
  #ctrl-bar button {{ background: #222; color: #bbb; border: 1px solid #3a3a3a;
    border-radius: 4px; padding: 4px 10px; font-size: 12px; cursor: pointer; white-space: nowrap; }}
  #ctrl-bar button:hover {{ background: #2c2c2c; border-color: #555; }}
  #ctrl-bar button.export-btn {{ background: #102010; border-color: #336633; color: #66cc66; }}
  #ctrl-bar button.export-btn:hover {{ background: #152615; }}
  #ctrl-bar .total-lbl {{ font-size: 12px; color: #cc8844; font-weight: 600; margin-left: auto; }}
  /* ── per-query block ── */
  .query-block {{ margin-bottom: 28px; border: 1px solid #222; border-radius: 8px; overflow: hidden; }}
  .query-header {{ background: #1a1a1a; padding: 8px 14px; font-size: 12px;
                    color: #999; border-bottom: 1px solid #222; display: flex;
                    align-items: center; gap: 14px; flex-wrap: wrap; }}
  .query-header b {{ color: #eee; }}
  .query-header .nn-counts {{ font-size: 11px; color: #444; }}
  /* ── per-query threshold control ── */
  .thresh-row {{ display: flex; align-items: center; gap: 10px; padding: 7px 14px;
                  background: #161616; border-bottom: 1px solid #1e1e1e; flex-wrap: wrap; }}
  .thresh-row .tlbl {{ font-size: 11px; color: #555; white-space: nowrap; }}
  .thresh-row input[type=range] {{ flex: 1; min-width: 160px; max-width: 400px;
    accent-color: #4a8a4a; cursor: pointer; }}
  .thresh-row input[type=number] {{ width: 60px; background: #1e1e1e; color: #ccc;
    border: 1px solid #333; border-radius: 4px; padding: 3px 5px; font-size: 12px; text-align:center; }}
  .excl-count {{ font-size: 11px; color: #ee8844; white-space: nowrap; min-width: 140px; }}
  /* ── thumbnails ── */
  .thumb-row {{ display: flex; align-items: flex-start; overflow-x: auto; padding: 12px; gap: 0; }}
  .thumb-cell {{ display: flex; flex-direction: column; align-items: center; gap: 4px; flex-shrink: 0; }}
  .thumb-cell img {{ border-radius: 5px; object-fit: cover; border: 2px solid transparent;
                      cursor: pointer; flex-shrink: 0; }}
  .thumb-cell img.query-img {{ border-color: #ff4444; }}
  .thumb-cell img.nn-img {{ border-color: #2a2a2a; }}
  .thumb-cell img.nn-img:hover {{ border-color: #666; }}
  .thumb-cell .lbl {{ font-size: 10px; color: #555; text-align: center;
                       max-width: {thumb_px}px; overflow: hidden;
                       text-overflow: ellipsis; white-space: nowrap; }}
  .thumb-cell .dist {{ font-size: 10px; color: #448844; }}
  .sep {{ width: 2px; background: #222; border-radius: 2px; margin: 0 8px;
           align-self: stretch; flex-shrink: 0; }}
  /* ── interval groups ── */
  .igroup {{ display: flex; flex-direction: column; gap: 6px; flex-shrink: 0;
              transition: opacity 0.2s; }}
  .igroup.inactive {{ opacity: 0.15; }}
  .igroup-header {{ font-size: 10px; font-weight: 600; padding: 3px 7px;
                     border-radius: 4px; white-space: nowrap; }}
  .igroup-thumbs {{ display: flex; gap: 5px; align-items: flex-start; }}
  .empty-bin {{ font-size: 10px; color: #333; padding: 4px 6px; align-self: center; }}
  /* ── lightbox ── */
  #lb {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,.88);
          align-items: center; justify-content: center; z-index: 999; flex-direction: column; gap: 14px; }}
  #lb.open {{ display: flex; }}
  #lb img {{ max-width: 80vw; max-height: 75vh; border-radius: 8px; }}
  #lb-caption {{ font-size: 13px; color: #bbb; background: rgba(0,0,0,.7);
                  padding: 6px 16px; border-radius: 20px; }}
</style>
</head>
<body>
<h1>KNN Gallery &mdash; {n_queries} bad patches &nbsp;&middot;&nbsp; {mode_label}</h1>
<div class="subtitle">
  Red border = query patch &nbsp;&middot;&nbsp;
  Each column = one distance bin &nbsp;&middot;&nbsp;
  Adjust threshold per patch with the slider &nbsp;&middot;&nbsp;
  Dimmed bins are excluded &nbsp;&middot;&nbsp;
  {samples_per_interval} samples shown per bin &nbsp;&middot;&nbsp; Click to enlarge
</div>

<!-- Global controls -->
<div id="ctrl-bar">
  <span class="lbl">Global threshold d ≤</span>
  <input type="number" id="global-thresh" value="{default_thresh}" min="0" max="{max_dist}" step="0.1">
  <button onclick="applyGlobal()">Apply to all</button>
  <div style="width:1px;height:18px;background:#2a2a2a;margin:0 4px"></div>
  <button class="export-btn" onclick="exportThresholds()">&#8595; Export thresholds CSV</button>
  <span class="total-lbl" id="global-total"></span>
</div>

{blocks}

<div id="lb" onclick="this.classList.remove('open')">
  <img id="lb-img" src="" alt="">
  <div id="lb-caption"></div>
</div>

<script>
const QUERIES = {queries_json};

function updateRow(qi) {{
  var thresh = parseFloat(document.getElementById('input-' + qi).value) || 0;
  var q = QUERIES[qi - 1];
  var excl = 0;
  q.bins.forEach(function(bin, bi) {{
    var active = thresh > bin.lo;
    var el = document.getElementById('bin-' + qi + '-' + bi);
    if (el) el.classList.toggle('inactive', !active);
    if (active) excl += bin.count;
  }});
  var cel = document.getElementById('excl-' + qi);
  if (cel) cel.textContent = excl.toLocaleString() + ' patches excluded';
  updateTotal();
}}

function syncInput(qi, val) {{
  document.getElementById('input-' + qi).value = parseFloat(val).toFixed(1);
  updateRow(qi);
}}
function syncSlider(qi, val) {{
  document.getElementById('slider-' + qi).value = val;
  updateRow(qi);
}}

function applyGlobal() {{
  var val = document.getElementById('global-thresh').value;
  QUERIES.forEach(function(q) {{
    document.getElementById('slider-' + q.qi).value = val;
    document.getElementById('input-' + q.qi).value = parseFloat(val).toFixed(1);
    updateRow(q.qi);
  }});
}}

function updateTotal() {{
  var total = 0;
  QUERIES.forEach(function(q) {{
    var thresh = parseFloat(document.getElementById('input-' + q.qi).value) || 0;
    q.bins.forEach(function(bin) {{
      if (thresh > bin.lo) total += bin.count;
    }});
  }});
  document.getElementById('global-total').textContent =
    'Est. ' + total.toLocaleString() + ' patches excluded (pre-dedup)';
}}

function exportThresholds() {{
  var lines = ['slide_id,x,y,threshold'];
  QUERIES.forEach(function(q) {{
    var thresh = parseFloat(document.getElementById('input-' + q.qi).value) || 0;
    lines.push(q.slide_id + ',' + q.x + ',' + q.y + ',' + thresh.toFixed(1));
  }});
  var blob = new Blob([lines.join('\\n')], {{type: 'text/csv'}});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'per_patch_thresholds.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
}}

function showLB(src, caption) {{
  document.getElementById('lb-img').src = 'data:image/jpeg;base64,' + src;
  document.getElementById('lb-caption').textContent = caption;
  document.getElementById('lb').classList.add('open');
}}

// Init
QUERIES.forEach(function(q) {{ updateRow(q.qi); }});
</script>
</body>
</html>
"""

# Colour gradient for interval headers: green (close) → red (far)
_INTERVAL_COLORS = [
    ("#1a3a1a", "#44bb44"),  # dark green bg, bright green text
    ("#2a2e10", "#aacc22"),  # yellow-green
    ("#2e2810", "#ddaa22"),  # amber
    ("#2e1e10", "#ee8833"),  # orange
    ("#3a1010", "#ee4444"),  # red
]


def _interval_color(i: int, n: int):
    """Return (bg, fg) hex for interval index i out of n."""
    idx = int(round(i / max(n - 1, 1) * (len(_INTERVAL_COLORS) - 1)))
    return _INTERVAL_COLORS[min(idx, len(_INTERVAL_COLORS) - 1)]


def build_gallery(bad_records, nn_results, tiff_dir: Path, thumb_px: int,
                  mode_label: str = "", interval_edges=None,
                  samples_per_interval: int = 5,
                  default_thresh: float | None = None) -> str:
    """
    bad_records      : list of dicts  {slide_id, x, y}
    nn_results       : list of lists  [{slide_id, x, y, dist}, ...]  — one list per query,
                       sorted by distance ascending
    interval_edges   : if set, list of float bin edges [0, d1, d2, ..., max_d];
                       gallery groups NNs into these bins and shows sampled thumbs per bin.
                       If None, flat display (all NNs in a single row).
    samples_per_interval : max thumbs to show per bin
    default_thresh   : initial slider value (defaults to max bin edge if interval mode)
    """
    blocks = []
    total = len(bad_records)
    query_data = []  # for QUERIES JS array

    max_dist = interval_edges[-1] if interval_edges is not None else 0.0
    if default_thresh is None:
        default_thresh = max_dist

    for qi, (rec, nns) in enumerate(zip(bad_records, nn_results)):
        qi1 = qi + 1  # 1-based index used in element IDs
        print(f"  Gallery [{qi1}/{total}] {rec['slide_id']} ...", flush=True)
        query_thumb = _thumb_b64(tiff_dir, rec["slide_id"], rec["x"], rec["y"], thumb_px)

        query_cell = (
            f'<div class="thumb-cell">'
            f'<img class="query-img" src="data:image/jpeg;base64,{query_thumb}" '
            f'width="{thumb_px}" height="{thumb_px}" '
            f'onclick="showLB(\'{query_thumb}\', \'Query: {rec["slide_id"]} ({rec["x"]},{rec["y"]})\')"> '
            f'<div class="lbl">{rec["slide_id"][-12:]}</div>'
            f'<div class="lbl" style="color:#ff6666">query</div>'
            f'</div>'
        )

        if interval_edges is not None:
            n_bins = len(interval_edges) - 1
            # Bucket NNs into bins
            bins = [[] for _ in range(n_bins)]
            for nn in nns:
                d = nn["dist"]
                for bi in range(n_bins):
                    if interval_edges[bi] <= d < interval_edges[bi + 1]:
                        bins[bi].append(nn)
                        break
                else:
                    bins[-1].append(nn)

            # Build interval group HTML (with id for JS dim-toggle)
            igroup_parts = []
            bins_js = []
            for bi, bin_nns in enumerate(bins):
                lo, hi = interval_edges[bi], interval_edges[bi + 1]
                bg, fg = _interval_color(bi, n_bins)
                bins_js.append({"lo": round(lo, 4), "hi": round(hi, 4), "count": len(bin_nns)})
                header = (
                    f'<div class="igroup-header" style="background:{bg};color:{fg}">'
                    f'd: {lo:.1f}–{hi:.1f} &nbsp;(n={len(bin_nns):,})</div>'
                )
                if not bin_nns:
                    body = '<div class="igroup-thumbs"><div class="empty-bin">no patches</div></div>'
                else:
                    sample_idx = np.linspace(0, len(bin_nns) - 1,
                                             min(samples_per_interval, len(bin_nns)),
                                             dtype=int)
                    sampled = [bin_nns[i] for i in sample_idx]
                    cells = []
                    for rank, nn in enumerate(sampled, 1):
                        t = _thumb_b64(tiff_dir, nn["slide_id"], nn["x"], nn["y"], thumb_px)
                        cells.append(
                            f'<div class="thumb-cell">'
                            f'<img class="nn-img" src="data:image/jpeg;base64,{t}" '
                            f'width="{thumb_px}" height="{thumb_px}" '
                            f'onclick="showLB(\'{t}\', \'bin {bi+1} #{rank}: {nn["slide_id"]} ({nn["x"]},{nn["y"]}) d={nn["dist"]:.2f}\')"> '
                            f'<div class="lbl">{nn["slide_id"][-12:]}</div>'
                            f'<div class="dist">d={nn["dist"]:.1f}</div>'
                            f'</div>'
                        )
                    body = '<div class="igroup-thumbs">' + "".join(cells) + '</div>'
                igroup_parts.append(
                    f'<div class="igroup" id="bin-{qi1}-{bi}">{header}{body}</div>'
                )

            query_data.append({
                "qi": qi1,
                "slide_id": rec["slide_id"],
                "x": rec["x"],
                "y": rec["y"],
                "bins": bins_js,
            })

            nn_count_summary = " &nbsp;|&nbsp; ".join(
                f'd{i+1}: {len(b):,}' for i, b in enumerate(bins)
            )
            thresh_row = (
                f'<div class="thresh-row">'
                f'<span class="tlbl">Threshold d ≤</span>'
                f'<input type="range" id="slider-{qi1}" min="0" max="{max_dist:.1f}" step="{max_dist/100:.2f}" '
                f'value="{default_thresh:.1f}" oninput="syncInput({qi1}, this.value)">'
                f'<input type="number" id="input-{qi1}" min="0" max="{max_dist:.1f}" step="0.1" '
                f'value="{default_thresh:.1f}" oninput="syncSlider({qi1}, this.value)">'
                f'<span class="excl-count" id="excl-{qi1}"></span>'
                f'</div>'
            )
            inner = (
                f'<div class="thumb-row">'
                f'{query_cell}'
                f'<div class="sep"></div>'
                + "".join(igroup_parts) +
                f'</div>'
            )
            header_html = (
                f'<div class="query-header">'
                f'<b>Query #{qi1}</b> &nbsp; {rec["slide_id"]} &nbsp; ({rec["x"]}, {rec["y"]}) px'
                f'<span class="nn-counts">{nn_count_summary}</span>'
                f'</div>'
                f'{thresh_row}'
            )

        else:
            # Flat display (no threshold slider in flat mode)
            cells = []
            for rank, nn in enumerate(nns, 1):
                t = _thumb_b64(tiff_dir, nn["slide_id"], nn["x"], nn["y"], thumb_px)
                cells.append(
                    f'<div class="thumb-cell">'
                    f'<img class="nn-img" src="data:image/jpeg;base64,{t}" '
                    f'width="{thumb_px}" height="{thumb_px}" '
                    f'onclick="showLB(\'{t}\', \'NN#{rank}: {nn["slide_id"]} ({nn["x"]},{nn["y"]}) d={nn["dist"]:.2f}\')"> '
                    f'<div class="lbl">{nn["slide_id"][-12:]}</div>'
                    f'<div class="dist">d={nn["dist"]:.2f}</div>'
                    f'</div>'
                )
            inner = (
                f'<div class="thumb-row">'
                f'{query_cell}'
                f'<div class="sep"></div>'
                + "".join(cells) +
                f'</div>'
            )
            header_html = (
                f'<div class="query-header">'
                f'<b>Query #{qi1}</b> &nbsp; {rec["slide_id"]} &nbsp; ({rec["x"]}, {rec["y"]}) px'
                f'</div>'
            )
            query_data.append({
                "qi": qi1,
                "slide_id": rec["slide_id"],
                "x": rec["x"],
                "y": rec["y"],
                "bins": [],
            })

        blocks.append(
            f'<div class="query-block">{header_html}{inner}</div>'
        )

    return GALLERY_HTML.format(
        n_queries=len(bad_records),
        mode_label=mode_label,
        samples_per_interval=samples_per_interval,
        thumb_px=thumb_px,
        blocks="\n".join(blocks),
        queries_json=json.dumps(query_data),
        max_dist=f"{max_dist:.1f}",
        default_thresh=f"{default_thresh:.1f}",
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    feat_dir = Path(args.feat_dir)
    tiff_dir = Path(args.tiff_dir)
    out_path = Path(args.out_csv)

    # Load bad patches CSV
    bad_df = pd.read_csv(args.bad_csv)
    print(f"Loaded {len(bad_df)} bad patch entries from {args.bad_csv}")
    print(f"  Slides: {sorted(bad_df['slide_id'].unique())[:10]} ...")

    # Load feature vectors for bad patches
    print("\nLoading features for bad patches ...")
    bad_feats, bad_records = load_bad_features(bad_df, feat_dir)
    if bad_feats is None:
        print("No bad patch features found — check slide IDs and coordinates. Exiting.")
        return
    print(f"  Matched {len(bad_records)}/{len(bad_df)} bad patches to H5 coords")

    # Load the full patch pool
    allowed_slides = None
    if args.slides_file:
        allowed_slides = Path(args.slides_file).read_text().split()
        print(f"\nSlides file: restricting pool to {len(allowed_slides)} slides")

    print(f"\nLoading pool features from {feat_dir} ...")
    pool_feats, pool_records = load_pool(feat_dir, allowed_slides, args.max_slides)
    print(f"  Pool size: {len(pool_feats):,} patches across {pool_records['slide_id'].nunique()} slides")

    if len(pool_feats) == 0:
        print("Empty pool — nothing to search. Exiting.")
        return

    # Load per-patch thresholds if provided
    per_patch_thresh = None
    if args.thresholds_csv:
        tdf = pd.read_csv(args.thresholds_csv)
        per_patch_thresh = {
            (str(r["slide_id"]), int(r["x"]), int(r["y"])): float(r["threshold"])
            for _, r in tdf.iterrows()
        }
        print(f"\nLoaded per-patch thresholds for {len(per_patch_thresh)} queries from {args.thresholds_csv}")

    # Nearest-neighbour search (keep distances for gallery)
    if per_patch_thresh is not None:
        # Use the max threshold as global search radius, then filter per-query
        global_radius = max(per_patch_thresh.values())
        print(f"\nBuilding NearestNeighbors index (per-patch thresholds, max radius={global_radius:.1f}) ...")
        nn_model = NearestNeighbors(algorithm="auto", n_jobs=-1)
        nn_model.fit(pool_feats)
        print("  Querying (radius_neighbors at max threshold) ...")
        distances_list, indices_list = nn_model.radius_neighbors(
            bad_feats, radius=global_radius, sort_results=True
        )
        # Filter each query's results to its own threshold
        distances, indices = [], []
        for qi, rec in enumerate(bad_records):
            key = (rec["slide_id"], rec["x"], rec["y"])
            thr = per_patch_thresh.get(key, global_radius)
            mask = distances_list[qi] <= thr
            distances.append(distances_list[qi][mask])
            indices.append(indices_list[qi][mask])
        per_query = [len(r) for r in indices]
        print(f"  Neighbours per query (after per-patch filter): min={min(per_query)}, median={int(np.median(per_query))}, max={max(per_query)}")
    elif args.dist_threshold is not None:
        print(f"\nBuilding NearestNeighbors index (radius threshold={args.dist_threshold}) ...")
        nn_model = NearestNeighbors(algorithm="auto", n_jobs=-1)
        nn_model.fit(pool_feats)
        print("  Querying (radius_neighbors) ...")
        distances_list, indices_list = nn_model.radius_neighbors(
            bad_feats, radius=args.dist_threshold, sort_results=True
        )
        distances = distances_list
        indices   = indices_list
        per_query = [len(r) for r in indices]
        print(f"  Neighbours per query: min={min(per_query)}, median={int(np.median(per_query))}, max={max(per_query)}")
    else:
        k_query = min(args.k + 1, len(pool_feats))  # +1: bad patch itself may be in pool
        print(f"\nBuilding NearestNeighbors index (k={args.k}) ...")
        nn_model = NearestNeighbors(n_neighbors=k_query, algorithm="auto", n_jobs=-1)
        nn_model.fit(pool_feats)
        print("  Querying ...")
        distances, indices = nn_model.kneighbors(bad_feats)

    # Collect unique neighbour pool indices for exclusion list
    neighbour_idx_set = set()
    for row in indices:
        for idx in row:
            neighbour_idx_set.add(int(idx))
    print(f"  Found {len(neighbour_idx_set)} unique pool indices (includes self-matches)")

    # Build exclusion list
    nn_records_df = pool_records.iloc[sorted(neighbour_idx_set)].copy()
    nn_records_df["reason"] = "nn_of_bad"

    bad_df_out = pd.DataFrame(bad_records)[["slide_id", "x", "y", "reason"]]
    exclusion = pd.concat([bad_df_out, nn_records_df], ignore_index=True)
    exclusion = exclusion.drop_duplicates(subset=["slide_id", "x", "y"], keep="first")
    exclusion = exclusion.sort_values(["slide_id", "x", "y"]).reset_index(drop=True)
    exclusion.to_csv(out_path, index=False)

    n_orig = len(bad_records)
    n_nn   = len(exclusion) - n_orig
    if per_patch_thresh is not None:
        mode_str = "per-patch threshold"
    elif args.dist_threshold is not None:
        mode_str = f"dist≤{args.dist_threshold}"
    else:
        mode_str = f"k={args.k}"
    print(f"\nExclusion list written → {out_path}")
    print(f"  Original bad patches : {n_orig:,}")
    print(f"  Added via NN ({mode_str}) : {n_nn:,}")
    print(f"  Total excluded        : {len(exclusion):,}")

    # Gallery HTML
    if args.html_out:
        vis_n = min(args.max_vis, len(bad_records))
        print(f"\nBuilding gallery HTML for first {vis_n} bad patches ...")

        # Per-query NN records with distances (skip self-matches).
        # In interval mode pass the full list so every distance bin can be populated;
        # in flat k-NN mode cap at args.k.
        use_intervals = args.dist_threshold is not None
        nn_results = []
        for qi in range(vis_n):
            nns = []
            for pool_idx, dist in zip(indices[qi], distances[qi]):
                pool_row = pool_records.iloc[int(pool_idx)]
                if (pool_row["slide_id"] == bad_records[qi]["slide_id"] and
                        pool_row["x"] == bad_records[qi]["x"] and
                        pool_row["y"] == bad_records[qi]["y"]):
                    continue
                nns.append({
                    "slide_id": pool_row["slide_id"],
                    "x": int(pool_row["x"]),
                    "y": int(pool_row["y"]),
                    "dist": float(dist),
                })
                if not use_intervals and len(nns) == args.k:
                    break
            nn_results.append(nns)

        interval_edges = None
        if args.dist_threshold is not None:
            interval_edges = list(np.linspace(0, args.dist_threshold, args.n_intervals + 1))
            print(f"  Distance bins: {[f'{e:.1f}' for e in interval_edges]}")

        html = build_gallery(
            bad_records[:vis_n], nn_results, tiff_dir, args.thumb_px,
            mode_label=mode_str,
            interval_edges=interval_edges,
            samples_per_interval=args.samples_per_interval,
            default_thresh=args.dist_threshold,
        )
        html_path = Path(args.html_out)
        html_path.write_text(html)
        size_mb = html_path.stat().st_size / 1e6
        print(f"Gallery written → {html_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
