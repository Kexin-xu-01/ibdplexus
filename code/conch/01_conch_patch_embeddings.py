"""
CONCH patch description pipeline.

Aggregates PRISM2 attention scores onto the CONCH v1.5 tile grid by mapping
each PRISM2 tile's center to its containing CONCH tile, then selects the top-K
CONCH tiles and saves their precomputed embeddings.

No model inference required — uses precomputed embeddings from both:
    features_virchow2  (via prism2_attention_heatmap .h5 files)
    features_conch_v15 (precomputed 768-dim CONCH v1.5 embeddings)

Spatial join:
    PRISM2 tile  672 level-0 px at 20x  (patch_size_level0 = 672)
    CONCH tile  1536 level-0 px at 20x  (patch_size_level0 = 1536)

    For each PRISM2 tile at (px, py):
        center = (px + 336, py + 336)
        → assigned to the CONCH tile whose rect contains that center

    Aggregated attention per CONCH tile = mean PRISM2 attn of assigned tiles
    (CONCH tiles with no assigned PRISM2 tiles receive score 0)

Output H5 schema per slide (prism2_conch_patch_embeddings/<slide>.h5):
    embeddings        (K, 768)  float32  CONCH v1.5 embeddings, attn-rank order
    coords            (K, 2)    int64    level-0 (x, y) of each CONCH tile
    attn_aggregated   (K,)      float32  mean PRISM2 attention [0, 1]
    patch_rank        (K,)      int32    0 = highest-attention CONCH tile
    n_prism2_assigned (K,)      int32    number of PRISM2 tiles mapped to each
    attrs:
        slide                  str
        n_patches              int   K actually written
        encoder                str   "conch_v15"
        prism2_patch_size_l0   int   PRISM2 tile size in level-0 pixels
        conch_patch_size_l0    int   CONCH tile size in level-0 pixels

Usage:
    python 17_conch_patch_description.py --slide 10407210HE1
    python 17_conch_patch_description.py --all
    python 17_conch_patch_description.py --all --n_patches 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

HEATMAP_DIR  = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/prism2_attention_heatmap")
CONCH_DIR    = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/20x_512px_0px_overlap/features_conch_v15")
OUT_DIR      = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/prism2_conch_patch_embeddings")

N_TOP_PATCHES       = 8
PRISM2_PATCH_SIZE   = 672   # level-0 pixels for the 20x 224px PRISM2 grid
CONCH_PATCH_SIZE    = 1536  # level-0 pixels for the 20x 512px CONCH grid


# ---------------------------------------------------------------------------
# Spatial join
# ---------------------------------------------------------------------------

def aggregate_prism2_onto_conch(
    prism2_coords: np.ndarray,  # (N, 2) int  level-0 (x, y), PRISM2 tiles
    prism2_heatmap: np.ndarray, # (N,)   float  normalised [0,1]
    conch_coords: np.ndarray,   # (M, 2) int  level-0 (x, y), CONCH tiles
    prism2_half: int,           # PRISM2_PATCH_SIZE // 2  (center offset)
    conch_size: int,            # CONCH_PATCH_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Return (agg_scores, n_assigned) both shape (M,).

    Maps each PRISM2 tile's center to the CONCH tile whose axis-aligned
    rectangle contains that center, then averages attn scores per CONCH tile.
    """
    M = len(conch_coords)
    agg  = np.zeros(M, dtype=np.float64)
    cnt  = np.zeros(M, dtype=np.int32)

    # Precompute PRISM2 centers
    cx = prism2_coords[:, 0] + prism2_half  # (N,)
    cy = prism2_coords[:, 1] + prism2_half

    # Vectorised: for each PRISM2 center find which CONCH col/row it falls in
    # CONCH tiles are on a regular grid; col/row index is deterministic
    # (works even when CONCH tile coords are not at 0-origin)
    col = (cx - conch_coords[:, 0].min()) // conch_size
    row = (cy - conch_coords[:, 1].min()) // conch_size

    cx0_grid = conch_coords[:, 0].min() + col * conch_size
    cy0_grid = conch_coords[:, 1].min() + row * conch_size

    # Build lookup: (cx0, cy0) → CONCH index
    coord_to_idx = {(int(c[0]), int(c[1])): i for i, c in enumerate(conch_coords)}

    for j in range(len(prism2_coords)):
        key = (int(cx0_grid[j]), int(cy0_grid[j]))
        m   = coord_to_idx.get(key, -1)
        if m >= 0:
            agg[m] += prism2_heatmap[j]
            cnt[m] += 1

    # Average (tiles with no PRISM2 tiles assigned keep score 0)
    mask = cnt > 0
    agg[mask] /= cnt[mask]
    return agg.astype(np.float32), cnt


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------

def process_slide(
    slide: str,
    heatmap_dir: Path,
    conch_dir: Path,
    out_dir: Path,
    n_patches: int,
    skip_existing: bool,
):
    out_h5 = out_dir / f"{slide}.h5"
    if skip_existing and out_h5.exists():
        print(f"  [skip] {slide}")
        return

    heatmap_h5 = heatmap_dir / f"{slide}.h5"
    conch_h5   = conch_dir   / f"{slide}.h5"

    if not heatmap_h5.exists():
        print(f"  [warn] heatmap not found: {heatmap_h5.name}")
        return
    if not conch_h5.exists():
        print(f"  [warn] CONCH features not found: {conch_h5.name}")
        return

    with h5py.File(heatmap_h5) as f:
        prism2_coords  = f["coords"][:]    # (N, 2)
        prism2_heatmap = f["heatmap"][:]   # (N,) normalised [0, 1]

    with h5py.File(conch_h5) as f:
        conch_coords   = f["coords"][:]    # (M, 2)
        conch_features = f["features"][:]  # (M, 768)
        conch_attrs    = dict(f["coords"].attrs)

    conch_patch_size = int(conch_attrs.get("patch_size_level0", CONCH_PATCH_SIZE))
    prism2_half      = PRISM2_PATCH_SIZE // 2

    agg_scores, n_assigned = aggregate_prism2_onto_conch(
        prism2_coords, prism2_heatmap, conch_coords, prism2_half, conch_patch_size
    )

    # Select top-K CONCH tiles by aggregated attention
    M = len(conch_coords)
    k = min(n_patches, M)
    top_idx = np.argsort(agg_scores)[::-1][:k]

    sel_emb    = conch_features[top_idx].astype(np.float32)
    sel_coords = conch_coords[top_idx].astype(np.int64)
    sel_scores = agg_scores[top_idx]
    sel_cnt    = n_assigned[top_idx]

    prism2_assigned_total = int((n_assigned > 0).sum())
    print(
        f"  {slide}: {M} CONCH tiles, {len(prism2_heatmap)} PRISM2 tiles, "
        f"{prism2_assigned_total}/{M} CONCH tiles with PRISM2 coverage, "
        f"writing top-{k}",
        flush=True,
    )

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("embeddings",        data=sel_emb)
        f.create_dataset("coords",            data=sel_coords)
        f.create_dataset("attn_aggregated",   data=sel_scores)
        f.create_dataset("patch_rank",        data=np.arange(k, dtype=np.int32))
        f.create_dataset("n_prism2_assigned", data=sel_cnt)
        f.attrs["slide"]                 = slide
        f.attrs["n_patches"]             = k
        f.attrs["encoder"]               = "conch_v15"
        f.attrs["prism2_patch_size_l0"]  = PRISM2_PATCH_SIZE
        f.attrs["conch_patch_size_l0"]   = conch_patch_size
    print(f"  saved → {out_h5.name}  embeddings {sel_emb.shape}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slide",         help="Slide stem (e.g. 10407210HE1)")
    p.add_argument("--all",           action="store_true", help="Process all slides")
    p.add_argument("--heatmap_dir",   default=str(HEATMAP_DIR),
                   help="PRISM2 attention heatmap .h5 directory  (default: %(default)s)")
    p.add_argument("--conch_dir",     default=str(CONCH_DIR),
                   help="CONCH v1.5 feature .h5 directory  (default: %(default)s)")
    p.add_argument("--out_dir",       default=str(OUT_DIR),
                   help="Output directory  (default: %(default)s)")
    p.add_argument("--n_patches",     type=int, default=N_TOP_PATCHES,
                   help="Top-K CONCH tiles per slide  (default: %(default)s)")
    p.add_argument("--skip_existing", action="store_true", default=True)
    args = p.parse_args()

    if not args.slide and not args.all:
        p.error("Specify --slide STEM or --all")

    heatmap_dir = Path(args.heatmap_dir)
    conch_dir   = Path(args.conch_dir)
    out_root    = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.slide:
        slides = [args.slide]
    else:
        h5s    = sorted(heatmap_dir.glob("*.h5"))
        slides = [p.stem for p in h5s]
        print(f"Found {len(slides)} heatmap files.\n")

    for i, slide in enumerate(slides, 1):
        print(f"[{i}/{len(slides)}]", end=" ")
        try:
            process_slide(
                slide, heatmap_dir, conch_dir, out_root,
                n_patches=args.n_patches,
                skip_existing=args.skip_existing,
            )
        except Exception as e:
            print(f"  [ERROR] {slide}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    print(f"\nDone. Output: {out_root}")


if __name__ == "__main__":
    main()
