#!/usr/bin/env python3
"""
Cluster all patch embeddings for QC — identify blurry, empty, smudged patches.

Pipeline:
  1. Fit PCA (50D) on a random 50k subsample of all patch features
  2. Transform all patches in per-slide batches
  3. MiniBatch K-means clustering
  4. For each cluster: extract 9 representative patches, compute blur score
     (Laplacian variance — low = blurry/empty)
  5. Save patch_clusters.parquet  (slide_id, x, y, cluster_id for every patch)
  6. Save cluster_qc.html  — interactive visual report; click to mark artifact
     clusters, then export their IDs

Usage:
  python cluster_qc.py
  python cluster_qc.py --n_clusters 30
  python cluster_qc.py --n_clusters 20 --n_reps 12
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
from scipy.ndimage import laplace
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

FEAT_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15/20x_224px_0px_overlap/features_virchow2")
TIFF_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
OUT_DIR  = Path("/home/jovyan/kgbk271-ibd-volume/results/cluster_qc")

PATCH_PX = 672
PYRLEVEL = 1
PYRDIV   = 2
THUMB_PX = 160   # px for representative patch thumbnails


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir",       default=str(FEAT_DIR))
    p.add_argument("--tiff_dir",       default=str(TIFF_DIR))
    p.add_argument("--out_dir",        default=str(OUT_DIR))
    p.add_argument("--n_clusters",     type=int, default=20,
                   help="Number of K-means clusters")
    p.add_argument("--n_reps",         type=int, default=9,
                   help="Representative patches per cluster shown in HTML")
    p.add_argument("--pca_components", type=int, default=50)
    p.add_argument("--pca_sample",     type=int, default=50000,
                   help="Patches used to fit PCA (random subsample)")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


# ── Feature loading ───────────────────────────────────────────────────────────

def list_slides(feat_dir: Path) -> list[str]:
    return sorted(p.stem for p in feat_dir.glob("*.h5"))


def patch_counts(slides: list[str], feat_dir: Path) -> dict[str, int]:
    counts = {}
    for slide in slides:
        with h5py.File(feat_dir / f"{slide}.h5") as f:
            counts[slide] = f["features"].shape[0]
    return counts


def sample_for_pca(slides, counts, feat_dir, n_sample, seed) -> np.ndarray:
    """Draw a stratified random sample of n_sample patches for PCA fitting.
    Uses HDF5 fancy indexing — reads only selected rows, not entire files."""
    rng = np.random.default_rng(seed)
    total = sum(counts.values())
    alloc = {s: max(1, round(counts[s] / total * n_sample)) for s in slides}

    chunks = []
    for slide in slides:
        n = counts[slide]
        k = min(alloc[slide], n)
        idx = np.sort(rng.choice(n, k, replace=False))
        with h5py.File(feat_dir / f"{slide}.h5") as f:
            chunks.append(f["features"][idx])   # reads only k rows
    return np.concatenate(chunks, axis=0)


def fit_pca(sample: np.ndarray, n_components: int, seed: int) -> PCA:
    print(f"  Fitting PCA {sample.shape[1]}→{n_components} on {len(sample):,} patches ...",
          end=" ", flush=True)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(sample)
    print(f"done  (explained var: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
    return pca


def load_and_transform(slides, counts, feat_dir, pca) -> tuple[np.ndarray, list, np.ndarray]:
    """
    Load all features, PCA-transform, return:
      reduced   — (N, 50) float32
      slide_ids — list of N slide_id strings
      coords    — (N, 2) int64 pixel coords
    """
    total = sum(counts.values())
    reduced   = np.empty((total, pca.n_components), dtype=np.float32)
    slide_ids = []
    coords    = np.empty((total, 2), dtype=np.int64)
    g = 0

    n = len(slides)
    for si, slide in enumerate(slides):
        print(f"\r  Transforming [{si+1}/{n}] {slide:<20}", end="", flush=True)
        with h5py.File(feat_dir / f"{slide}.h5") as f:
            feats  = f["features"][:]
            coords_s = f["coords"][:]
        k = len(feats)
        # Transform in one batch (2560→50 is cheap)
        reduced[g:g+k] = pca.transform(feats).astype(np.float32)
        slide_ids.extend([slide] * k)
        coords[g:g+k] = coords_s
        g += k

    print(f"\r  Transformed {g:,} patches across {n} slides" + " " * 30)
    return reduced, slide_ids, coords


# ── Clustering ────────────────────────────────────────────────────────────────

def run_kmeans(X: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    print(f"  MiniBatch K-means K={n_clusters} on {len(X):,} patches ...",
          end=" ", flush=True)
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(4096, len(X)),
        random_state=seed,
        n_init=3,
        max_iter=300,
    )
    labels = km.fit_predict(X)
    print("done  (inertia {:.3e})".format(km.inertia_))
    return labels, km


# ── Representative patches ────────────────────────────────────────────────────

def find_representatives(X, labels, centroids, n_reps) -> dict[int, list[int]]:
    """For each cluster, return indices of the n_reps patches closest to centroid."""
    reps = {}
    for k in range(len(centroids)):
        mask = np.where(labels == k)[0]
        if len(mask) == 0:
            reps[k] = []
            continue
        dists = np.linalg.norm(X[mask] - centroids[k], axis=1)
        order = np.argsort(dists)[:n_reps]
        reps[k] = mask[order].tolist()
    return reps


# ── Thumbnails & blur scores ──────────────────────────────────────────────────

def _thumb_b64(sl: openslide.OpenSlide, x: int, y: int) -> tuple[str, float]:
    sz = PATCH_PX // PYRDIV
    region = sl.read_region((x, y), PYRLEVEL, (sz, sz))
    img = region.convert("RGB").resize((THUMB_PX, THUMB_PX), Image.LANCZOS)

    # Blur score: Laplacian variance (higher = sharper)
    gray = np.array(img.convert("L")).astype(np.float32)
    blur_score = float(np.var(laplace(gray)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64, blur_score


def _placeholder() -> tuple[str, float]:
    buf = io.BytesIO()
    Image.fromarray(np.full((THUMB_PX, THUMB_PX, 3), 40, dtype=np.uint8)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode(), 0.0


def extract_rep_thumbnails(reps, slide_ids, coords, tiff_dir) -> dict[int, list]:
    """
    Returns {cluster_id: [{"b64": ..., "blur": ..., "slide": ..., "x": ..., "y": ...}]}
    """
    # Group all needed patches by slide to avoid re-opening TIFFs
    needed: dict[str, list[tuple[int, int, int]]] = {}  # slide → [(cluster, global_idx, pos_in_list)]
    for cluster_id, global_indices in reps.items():
        for gi in global_indices:
            slide = slide_ids[gi]
            if slide not in needed:
                needed[slide] = []
            needed[slide].append((cluster_id, gi))

    results: dict[int, list] = {k: [] for k in reps}
    ph_b64, ph_blur = _placeholder()

    n_slides = len(needed)
    for si, (slide, patches) in enumerate(needed.items()):
        print(f"\r  Thumbnails [{si+1}/{n_slides}] {slide:<20}", end="", flush=True)
        tiff = tiff_dir / f"{slide}.tiff"
        if not tiff.exists():
            for cluster_id, gi in patches:
                results[cluster_id].append(
                    {"b64": ph_b64, "blur": ph_blur,
                     "slide": slide, "x": int(coords[gi, 0]), "y": int(coords[gi, 1])})
            continue
        try:
            with openslide.OpenSlide(tiff) as sl:
                for cluster_id, gi in patches:
                    x, y = int(coords[gi, 0]), int(coords[gi, 1])
                    b64, blur = _thumb_b64(sl, x, y)
                    results[cluster_id].append(
                        {"b64": b64, "blur": blur, "slide": slide, "x": x, "y": y})
        except Exception as e:
            for cluster_id, gi in patches:
                results[cluster_id].append(
                    {"b64": ph_b64, "blur": ph_blur,
                     "slide": slide, "x": int(coords[gi, 0]), "y": int(coords[gi, 1])})
    print()
    return results


# ── Parquet output ────────────────────────────────────────────────────────────

def save_parquet(slide_ids, coords, labels, out_dir: Path):
    df = pd.DataFrame({
        "slide_id":  slide_ids,
        "x":         coords[:, 0],
        "y":         coords[:, 1],
        "cluster_id": labels.astype(np.int16),
    })
    path = out_dir / "patch_clusters.parquet"
    df.to_parquet(path, index=False)
    print(f"  Saved {path}  ({len(df):,} rows, {path.stat().st_size/1e6:.1f} MB)")
    return df


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Patch Cluster QC</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: system-ui, sans-serif; background: #0f0f0f; color: #d8d8d8;
        display: flex; flex-direction: column; min-height: 100vh; }}

header {{ padding: 12px 20px; background: #181818; border-bottom: 1px solid #2a2a2a;
           display: flex; align-items: center; gap: 18px; flex-wrap: wrap; flex-shrink: 0; }}
header h1 {{ font-size: 15px; font-weight: 600; letter-spacing: 0.03em; }}
.stats {{ font-size: 12px; color: #555; }}

.toolbar {{ padding: 10px 20px; background: #141414; border-bottom: 1px solid #222;
             display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
.toolbar label {{ font-size: 12px; color: #666; }}
select {{ background: #222; color: #ccc; border: 1px solid #3a3a3a; border-radius: 4px;
           padding: 4px 8px; font-size: 12px; cursor: pointer; }}
button {{ padding: 6px 14px; border-radius: 5px; border: none; font-size: 12px;
           cursor: pointer; font-weight: 500; }}
.btn-export {{ background: #2563eb; color: #fff; }}
.btn-export:hover {{ background: #1d4ed8; }}
.btn-clear  {{ background: #292929; color: #aaa; border: 1px solid #3a3a3a; }}
.btn-clear:hover {{ background: #333; }}
.badge {{ font-size: 11px; background: #e15759; color: #fff; border-radius: 4px;
           padding: 2px 7px; display: none; }}
#artifact-count {{ display: inline-block; }}

#grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
          gap: 14px; padding: 16px 20px; }}

.card {{ background: #181818; border: 1px solid #262626; border-radius: 8px;
          overflow: hidden; cursor: pointer; transition: border-color 0.15s; }}
.card:hover {{ border-color: #444; }}
.card.artifact {{ border-color: #e15759 !important; background: #200f0f; }}
.card.artifact .card-header {{ background: #2d0f0f; }}

.card-header {{ padding: 10px 12px; background: #1e1e1e; display: flex;
                 align-items: center; gap: 8px; user-select: none; }}
.cid {{ font-size: 13px; font-weight: 600; color: #d0d0d0; }}
.csize {{ font-size: 11px; color: #555; margin-left: auto; }}
.artifact-tag {{ font-size: 10px; background: #e15759; color: #fff; border-radius: 3px;
                  padding: 2px 6px; display: none; letter-spacing: 0.05em; }}
.card.artifact .artifact-tag {{ display: inline; }}

.blur-bar-wrap {{ padding: 6px 12px 2px; }}
.blur-label {{ font-size: 10px; color: #444; margin-bottom: 3px; display: flex;
                justify-content: space-between; }}
.blur-bar {{ height: 4px; background: #2a2a2a; border-radius: 2px; overflow: hidden; }}
.blur-fill {{ height: 100%; border-radius: 2px; transition: width 0.3s; }}

.patches {{ display: grid; padding: 8px 10px 10px;
             grid-template-columns: repeat(3, 1fr); gap: 4px; }}
.patch-wrap {{ position: relative; }}
.patch-wrap img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 3px;
                    display: block; }}
.patch-blur {{ position: absolute; bottom: 2px; right: 3px; font-size: 9px;
                background: rgba(0,0,0,0.6); color: #ccc; padding: 1px 3px;
                border-radius: 2px; }}
</style>
</head>
<body>
<header>
  <h1>Patch Cluster QC</h1>
  <span class="stats">{n_patches:,} patches &thinsp;&middot;&thinsp; {n_slides:,} slides
    &thinsp;&middot;&thinsp; K={n_clusters} clusters
    &thinsp;&middot;&thinsp; Virchow2 &rarr; PCA {pca_d}D &rarr; MiniBatch K-means</span>
</header>
<div class="toolbar">
  <label>Sort by</label>
  <select id="sort-sel" onchange="sortCards()">
    <option value="cluster">Cluster ID</option>
    <option value="blur_asc">Blur score ↑ (blurriest first)</option>
    <option value="blur_desc">Blur score ↓ (sharpest first)</option>
    <option value="size_desc">Cluster size ↓</option>
    <option value="size_asc">Cluster size ↑</option>
  </select>
  <span style="margin-left:8px; font-size:12px; color:#555">
    Click a card to mark as <span style="color:#e15759">ARTIFACT</span>
  </span>
  <span id="artifact-count" class="badge">0 marked</span>
  <button class="btn-export" onclick="exportArtifacts()">Export artifact cluster IDs</button>
  <button class="btn-clear"  onclick="clearAll()">Clear all</button>
</div>
<div id="grid"></div>

<script>
const DATA = {data_json};

// Pre-compute per-cluster blur stats from representative patches
DATA.clusters.forEach(c => {{
  const scores = c.patches.map(p => p.blur).filter(v => v > 0);
  c.avg_blur = scores.length ? scores.reduce((a,b)=>a+b,0)/scores.length : 0;
}});
const maxBlur = Math.max(...DATA.clusters.map(c => c.avg_blur), 1);

const artifacts = new Set();

function blurColor(v, max) {{
  const t = Math.min(v / max, 1);
  const r = Math.round(229 * (1-t) + 89 * t);
  const g = Math.round(87  * (1-t) + 161 * t);
  const b = Math.round(89  * (1-t) + 79 * t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function buildCard(c) {{
  const div = document.createElement("div");
  div.className = "card" + (artifacts.has(c.id) ? " artifact" : "");
  div.dataset.cid    = c.id;
  div.dataset.blur   = c.avg_blur;
  div.dataset.size   = c.size;

  const pct  = Math.round(c.avg_blur / maxBlur * 100);
  const col  = blurColor(c.avg_blur, maxBlur);

  div.innerHTML = `
    <div class="card-header">
      <span class="cid">Cluster ${{c.id}}</span>
      <span class="artifact-tag">ARTIFACT</span>
      <span class="csize">${{c.size.toLocaleString()}} patches</span>
    </div>
    <div class="blur-bar-wrap">
      <div class="blur-label">
        <span>Blur score (Laplacian var)</span>
        <span>${{c.avg_blur.toFixed(1)}}</span>
      </div>
      <div class="blur-bar"><div class="blur-fill" style="width:${{pct}}%;background:${{col}}"></div></div>
    </div>
    <div class="patches">
      ${{c.patches.map(p => `
        <div class="patch-wrap">
          <img src="data:image/jpeg;base64,${{p.b64}}" title="${{p.slide}} (${{p.x}},${{p.y}})">
          <span class="patch-blur">${{p.blur.toFixed(0)}}</span>
        </div>`).join("")}}
    </div>`;

  div.addEventListener("click", () => toggleArtifact(div, c.id));
  return div;
}}

function renderGrid(clusters) {{
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  clusters.forEach(c => grid.appendChild(buildCard(c)));
}}

function sortCards() {{
  const sel  = document.getElementById("sort-sel").value;
  const copy = [...DATA.clusters];
  if      (sel === "blur_asc")   copy.sort((a,b) => a.avg_blur - b.avg_blur);
  else if (sel === "blur_desc")  copy.sort((a,b) => b.avg_blur - a.avg_blur);
  else if (sel === "size_desc")  copy.sort((a,b) => b.size - a.size);
  else if (sel === "size_asc")   copy.sort((a,b) => a.size - b.size);
  else                           copy.sort((a,b) => a.id - b.id);
  renderGrid(copy);
}}

function toggleArtifact(card, id) {{
  if (artifacts.has(id)) {{ artifacts.delete(id); card.classList.remove("artifact"); }}
  else                   {{ artifacts.add(id);    card.classList.add("artifact"); }}
  const badge = document.getElementById("artifact-count");
  badge.textContent = artifacts.size + " marked";
  badge.style.display = artifacts.size ? "inline" : "none";
}}

function exportArtifacts() {{
  if (!artifacts.size) {{ alert("No clusters marked as artifacts."); return; }}
  const ids = [...artifacts].sort((a,b)=>a-b);
  const json = JSON.stringify(ids);
  navigator.clipboard.writeText(json).then(() => {{
    alert("Copied to clipboard:\\n" + json + "\\n\\nPaste into filter_patches.py");
  }}).catch(() => {{
    prompt("Copy this:", json);
  }});
}}

function clearAll() {{
  artifacts.clear();
  document.querySelectorAll(".card.artifact").forEach(c => c.classList.remove("artifact"));
  const badge = document.getElementById("artifact-count");
  badge.textContent = "0 marked";
  badge.style.display = "none";
}}

// Initial render sorted by blur (blurriest first — most likely artifacts on top)
document.getElementById("sort-sel").value = "blur_asc";
sortCards();
</script>
</body>
</html>
"""


def build_html(cluster_data, n_patches, n_slides, n_clusters, pca_d) -> str:
    data = {
        "clusters": [
            {
                "id":   k,
                "size": int(cluster_data[k]["size"]),
                "patches": [
                    {"b64":   p["b64"],
                     "blur":  round(p["blur"], 2),
                     "slide": p["slide"],
                     "x":     p["x"],
                     "y":     p["y"]}
                    for p in cluster_data[k]["patches"]
                ],
            }
            for k in sorted(cluster_data)
        ]
    }
    return HTML.format(
        n_patches=n_patches,
        n_slides=n_slides,
        n_clusters=n_clusters,
        pca_d=pca_d,
        data_json=json.dumps(data, separators=(",", ":")),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    feat_dir = Path(args.feat_dir)
    tiff_dir = Path(args.tiff_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slides = list_slides(feat_dir)
    print(f"Found {len(slides)} slides\n")

    # Patch counts
    print("Scanning feature files ...")
    counts = patch_counts(slides, feat_dir)
    total  = sum(counts.values())
    print(f"  {total:,} total patches\n")

    # PCA
    print(f"Sampling {args.pca_sample:,} patches for PCA ...")
    sample = sample_for_pca(slides, counts, feat_dir, args.pca_sample, args.seed)
    print()
    pca = fit_pca(sample, args.pca_components, args.seed)
    del sample
    print()

    # Load + transform all
    print("Loading and transforming all patches ...")
    X, slide_ids, coords = load_and_transform(slides, counts, feat_dir, pca)
    print()

    # Cluster
    print("Clustering ...")
    labels, km = run_kmeans(X, args.n_clusters, args.seed)
    print()

    # Save parquet
    print("Saving patch_clusters.parquet ...")
    save_parquet(slide_ids, coords, labels, out_dir)
    print()

    # Representatives
    print("Finding representative patches per cluster ...")
    reps = find_representatives(X, labels, km.cluster_centers_, args.n_reps)
    del X

    # Thumbnails + blur
    print("Extracting thumbnails & computing blur scores ...")
    rep_thumbs = extract_rep_thumbnails(reps, slide_ids, coords, tiff_dir)
    print()

    # Cluster summary
    cluster_data = {}
    for k in range(args.n_clusters):
        size = int(np.sum(labels == k))
        cluster_data[k] = {"size": size, "patches": rep_thumbs.get(k, [])}

    # Print summary table
    print(f"{'Cluster':>8}  {'Size':>8}  {'Avg Blur':>10}")
    print("-" * 34)
    for k in sorted(cluster_data, key=lambda k: np.mean([p["blur"] for p in cluster_data[k]["patches"]] or [0])):
        blurs = [p["blur"] for p in cluster_data[k]["patches"]]
        avg = np.mean(blurs) if blurs else 0
        print(f"{k:>8}  {cluster_data[k]['size']:>8,}  {avg:>10.1f}")
    print()

    # HTML
    print("Building cluster_qc.html ...", end=" ", flush=True)
    html = build_html(cluster_data, total, len(slides), args.n_clusters, pca.n_components)
    out_path = out_dir / "cluster_qc.html"
    out_path.write_text(html)
    size_mb = out_path.stat().st_size / 1e6
    print(f"done → {out_path}  ({size_mb:.1f} MB)")
    print(f"\nWorkflow:")
    print(f"  1. Open {out_path}")
    print(f"  2. Patches are sorted blurriest-first — inspect and click artifact clusters")
    print(f"  3. Click 'Export artifact cluster IDs' → copy JSON list")
    print(f"  4. Use patch_clusters.parquet + that list to filter patches from your pipeline")


if __name__ == "__main__":
    main()
