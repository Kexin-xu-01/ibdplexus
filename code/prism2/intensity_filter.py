#!/usr/bin/env python3
"""
Compute mean RGB intensity for every patch and save to patch_intensity.parquet.
Uses openslide read_region at pyramid level 5 (32x down) — tiny 21x21 px reads,
minimal memory usage.

After running, use the parquet to threshold faint patches:
    df = pd.read_parquet("patch_intensity.parquet")
    clean = df[df.mean_intensity < THRESHOLD]
"""

import h5py
import numpy as np
import openslide
import pandas as pd
from pathlib import Path

FEAT_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/"
                "tissue_threshold_15/20x_224px_0px_overlap/features_virchow2")
TIFF_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
OUT_DIR  = Path("/home/jovyan/kgbk271-ibd-volume/results/cluster_qc")
PATCH_PX = 672


def main():
    slides = sorted(p.stem for p in FEAT_DIR.glob("*.h5"))
    print(f"{len(slides)} slides", flush=True)

    rows = []   # (slide_id, x, y, mean_intensity)

    for si, slide in enumerate(slides):
        tiff = TIFF_DIR / f"{slide}.tiff"
        h5   = FEAT_DIR / f"{slide}.h5"
        if not tiff.exists():
            continue
        with h5py.File(h5) as f:
            coords = f["coords"][:]          # (N, 2) full-res (x, y)

        try:
            with openslide.OpenSlide(tiff) as sl:
                lvl = min(5, sl.level_count - 1)
                div = int(round(sl.level_downsamples[lvl]))
                sz  = max(1, PATCH_PX // div)  # ~21 px at level 5
                for x, y in coords:
                    region = sl.read_region((int(x), int(y)), lvl, (sz, sz))
                    arr = np.frombuffer(region.tobytes(), dtype=np.uint8)
                    # RGBA → take R,G,B channels only
                    mean_val = arr.reshape(-1, 4)[:, :3].mean()
                    rows.append((slide, int(x), int(y), float(mean_val)))
        except Exception:
            for x, y in coords:
                rows.append((slide, int(x), int(y), float("nan")))

        if (si + 1) % 100 == 0:
            print(f"  {si+1}/{len(slides)}", flush=True)

    print("Building dataframe ...", flush=True)
    df = pd.DataFrame(rows, columns=["slide_id", "x", "y", "mean_intensity"])
    out = OUT_DIR / "patch_intensity.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows → {out}")

    v = df["mean_intensity"].dropna()
    print(f"\nIntensity distribution (0=black, 255=white):")
    for p in [50, 75, 90, 95, 99, 99.5]:
        print(f"  p{p:5.1f}: {np.percentile(v, p):.1f}")


if __name__ == "__main__":
    main()
