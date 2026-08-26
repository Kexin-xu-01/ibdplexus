#!/usr/bin/env python3
"""
Apply the manual KNN exclusion list to filtered patch H5 files.

Reads each H5 from tissue_threshold_15_filtered_no_darkspot (which has already
had Laplacian blur, faint-patch, and GrandQC dark-spot filters applied), removes
any patch whose (slide_id, x, y) appears in the merged exclusion list produced
by the prism2_knn QC pipeline (iter1 + iter2), and writes the result to
tissue_threshold_15_filtered_no_darkspot_manual_knn.

Output H5 files are drop-in replacements: same key layout (coords, features),
same gzip-4 compression, same dtype.

Only features_virchow2/ is processed (per-patch coords + features).
prism2_base/ and prism2_diagnostic/ are slide-level aggregations and are not
affected by patch-level exclusion.

Usage
-----
  python apply_knn_exclusion.py
  python apply_knn_exclusion.py --workers 16
  python apply_knn_exclusion.py --start 0 --end 1000   # chunk for cluster jobs
"""

import argparse
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SRC_DIR = Path(
    "/home/jovyan/kgbk271-ibd-volume/data/processed/"
    "tissue_threshold_15_filtered_no_darkspot/20x_224px_0px_overlap/features_virchow2"
)
OUT_DIR = Path(
    "/home/jovyan/kgbk271-ibd-volume/data/processed/"
    "tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/features_virchow2"
)
EXCL_CSV = Path(
    "/home/jovyan/kgbk271-ibd-volume/image_preprocessing/patch_qc/"
    "prism2_knn/exclusion_list_merged.csv"
)


def filter_slide(h5_path: Path, excl_by_slide: dict, out_dir: Path) -> dict:
    """Filter one slide's H5, writing the result to out_dir. Returns stats dict."""
    slide_id = h5_path.stem
    out_path = out_dir / h5_path.name

    if out_path.exists():
        with h5py.File(out_path) as f:
            n_kept = len(f["coords"])
        with h5py.File(h5_path) as f:
            n_total = len(f["coords"])
        return dict(slide=slide_id, total=n_total, kept=n_kept, skipped=True)

    try:
        with h5py.File(h5_path) as f:
            coords  = f["coords"][:]
            features = f["features"][:]

        bad_coords = excl_by_slide.get(slide_id, set())
        if bad_coords:
            keep = np.array([
                (int(cx), int(cy)) not in bad_coords
                for cx, cy in coords
            ])
        else:
            keep = np.ones(len(coords), dtype=bool)

        coords_out   = coords[keep]
        features_out = features[keep]

        with h5py.File(out_path, "w") as f:
            f.create_dataset("coords",   data=coords_out,
                             compression="gzip", compression_opts=4)
            f.create_dataset("features", data=features_out,
                             compression="gzip", compression_opts=4)

        return dict(slide=slide_id,
                    total=len(coords),
                    kept=len(coords_out),
                    removed=int(keep.sum() != len(coords)))

    except Exception as e:
        return dict(slide=slide_id, total=0, kept=0, error=str(e))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir",  default=str(SRC_DIR))
    p.add_argument("--out_dir",  default=str(OUT_DIR))
    p.add_argument("--excl_csv", default=str(EXCL_CSV),
                   help="Merged exclusion list CSV (slide_id, x, y, reason)")
    p.add_argument("--workers",  type=int, default=min(8, cpu_count()))
    p.add_argument("--start",    type=int, default=0,
                   help="First slide index (for chunked cluster jobs)")
    p.add_argument("--end",      type=int, default=-1,
                   help="Last slide index exclusive (-1 = all)")
    return p.parse_args()


def main():
    args = parse_args()
    src_dir  = Path(args.src_dir)
    out_dir  = Path(args.out_dir)
    excl_csv = Path(args.excl_csv)

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Source  : {src_dir}")
    log.info(f"Output  : {out_dir}")
    log.info(f"Excl CSV: {excl_csv}")

    # Load exclusion list → per-slide set of (x, y) tuples for O(1) lookup
    excl_df = pd.read_csv(excl_csv)
    excl_by_slide: dict[str, set] = {}
    for _, row in excl_df.iterrows():
        excl_by_slide.setdefault(str(row["slide_id"]), set()).add(
            (int(row["x"]), int(row["y"]))
        )
    total_excl = sum(len(v) for v in excl_by_slide.values())
    log.info(f"Exclusion list: {total_excl:,} patches across "
             f"{len(excl_by_slide):,} slides")

    # Enumerate source H5 files
    all_h5 = sorted(src_dir.glob("*.h5"))
    end = args.end if args.end > 0 else len(all_h5)
    all_h5 = all_h5[args.start:end]
    log.info(f"Slides to process: {len(all_h5):,}  "
             f"[{args.start}:{end}]  workers={args.workers}")

    worker = partial(filter_slide, excl_by_slide=excl_by_slide, out_dir=out_dir)

    total_in = total_out = n_skipped = n_affected = 0
    errors = []

    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, all_h5), 1):
            if "error" in result:
                errors.append(result)
            else:
                total_in  += result["total"]
                total_out += result["kept"]
                if result.get("skipped"):
                    n_skipped += 1
                elif result.get("removed"):
                    n_affected += 1
            if i % 500 == 0 or i == len(all_h5):
                pct = 100 * total_out / max(total_in, 1)
                log.info(f"  [{i:>5}/{len(all_h5)}]  "
                         f"{total_out:,}/{total_in:,} kept ({pct:.2f}%)  "
                         f"skipped={n_skipped}  affected={n_affected}")

    removed = total_in - total_out
    pct_removed = 100 * removed / max(total_in, 1)
    log.info("")
    log.info(f"Done.")
    log.info(f"  Total patches in  : {total_in:>10,}")
    log.info(f"  Total patches out : {total_out:>10,}")
    log.info(f"  Removed           : {removed:>10,}  ({pct_removed:.2f}%)")
    log.info(f"  Slides affected   : {n_affected:>10,}")
    log.info(f"  Slides unchanged  : {len(all_h5) - n_affected - len(errors):>10,}")
    if errors:
        log.warning(f"  Errors            : {len(errors)}")
        for e in errors[:5]:
            log.warning(f"    {e}")


if __name__ == "__main__":
    main()
