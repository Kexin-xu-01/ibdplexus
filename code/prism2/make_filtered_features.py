#!/usr/bin/env python3
"""
Build tissue_threshold_15_filtered: features_virchow2 h5 files keeping only patches
that pass BOTH the Laplacian t100 filter AND the faint-patch intensity filter.

Laplacian t100 (blur filter):
  /data/processed/tissue_threshold_15_remove_artifact/laplacien_t100/{slide}_patches.h5
  → key 'coords' (N,2): coords of patches that PASSED blur threshold

Intensity filter:
  /results/cluster_qc/patch_intensity.parquet  (slide_id, x, y, mean_intensity)
  → keep mean_intensity < INTENSITY_THRESHOLD (p98 = 211.8)

Output:
  /data/processed/tissue_threshold_15_filtered/20x_224px_0px_overlap/features_virchow2/{slide}.h5
  → keys 'coords' (M,2) and 'features' (M,2560)
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

ORIG_DIR   = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/"
                  "tissue_threshold_15/20x_224px_0px_overlap/features_virchow2")
LAP_DIR    = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/"
                  "tissue_threshold_15_remove_artifact/laplacien_t100")
INTENS_PAR = Path("/home/jovyan/kgbk271-ibd-volume/results/cluster_qc/patch_intensity.parquet")
OUT_DIR    = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/"
                  "tissue_threshold_15_filtered/20x_224px_0px_overlap/features_virchow2")

INTENSITY_THRESHOLD = 211.8   # p98; patches with mean_intensity >= this are faint/white


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load intensity data into memory-friendly lookup: {slide_id -> set of (x,y)}
    print("Loading intensity parquet ...", flush=True)
    intens_df = pd.read_parquet(INTENS_PAR)
    print(f"  {len(intens_df):,} rows loaded", flush=True)

    # Build per-slide sets of intensity-passing (x,y) coords
    passing_intens = {}
    for slide_id, grp in intens_df.groupby("slide_id"):
        mask = grp["mean_intensity"] < INTENSITY_THRESHOLD
        coords = set(zip(grp.loc[mask, "x"].astype(int), grp.loc[mask, "y"].astype(int)))
        passing_intens[slide_id] = coords
    print(f"  {sum(len(v) for v in passing_intens.values()):,} intensity-passing patches "
          f"across {len(passing_intens)} slides", flush=True)
    del intens_df   # free RAM

    # Process each slide
    slides = sorted(p.stem for p in ORIG_DIR.glob("*.h5"))
    n_slides = len(slides)
    print(f"\n{n_slides} slides to process\n", flush=True)

    total_in  = 0
    total_out = 0
    n_lap_missing = 0

    for si, slide in enumerate(slides):
        orig_h5 = ORIG_DIR / f"{slide}.h5"
        lap_h5  = LAP_DIR  / f"{slide}_patches.h5"
        out_h5  = OUT_DIR  / f"{slide}.h5"

        with h5py.File(orig_h5) as f:
            orig_coords  = f["coords"][:]      # (N, 2) int64
            orig_features = None               # lazy load below

        n_orig = len(orig_coords)
        total_in += n_orig

        # --- Laplacian filter ---
        if lap_h5.exists():
            with h5py.File(lap_h5) as f:
                lap_coords_arr = f["coords"][:]
            lap_set = set(map(tuple, lap_coords_arr.tolist()))
        else:
            # No Laplacian file for this slide: skip the blur filter
            lap_set = None
            n_lap_missing += 1

        # --- Intensity filter ---
        intens_set = passing_intens.get(slide, None)

        # --- Compute keep mask ---
        if lap_set is None and intens_set is None:
            # No filter data at all: keep everything
            keep_idx = np.arange(n_orig)
        else:
            keep_idx = []
            for i, (x, y) in enumerate(orig_coords.tolist()):
                coord = (int(x), int(y))
                lap_ok    = (lap_set is None)  or (coord in lap_set)
                intens_ok = (intens_set is None) or (coord in intens_set)
                if lap_ok and intens_ok:
                    keep_idx.append(i)
            keep_idx = np.array(keep_idx, dtype=np.int64)

        n_keep = len(keep_idx)
        total_out += n_keep

        if n_keep == 0:
            # Skip slides where every patch is filtered out
            if (si + 1) % 500 == 0 or n_keep == 0:
                print(f"  [{si+1}/{n_slides}] {slide}: {n_orig} → {n_keep} (SKIPPED — all filtered)",
                      flush=True)
            continue

        # --- Load features for kept rows ---
        with h5py.File(orig_h5) as f:
            if n_keep == n_orig:
                kept_coords   = orig_coords
                kept_features = f["features"][:]
            else:
                kept_coords   = orig_coords[keep_idx]
                kept_features = f["features"][keep_idx]

        # --- Write output h5 ---
        with h5py.File(out_h5, "w") as f:
            f.create_dataset("coords",   data=kept_coords,   compression="gzip", compression_opts=4)
            f.create_dataset("features", data=kept_features, compression="gzip", compression_opts=4)

        if (si + 1) % 500 == 0:
            print(f"  [{si+1}/{n_slides}]  in={total_in:,}  out={total_out:,}  "
                  f"kept={total_out/total_in*100:.1f}%", flush=True)

    print(f"\nDone.")
    print(f"  Input patches : {total_in:,}")
    print(f"  Output patches: {total_out:,}  ({total_out/total_in*100:.1f}% kept)")
    print(f"  Slides with no Laplacian file: {n_lap_missing}")
    print(f"  Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()