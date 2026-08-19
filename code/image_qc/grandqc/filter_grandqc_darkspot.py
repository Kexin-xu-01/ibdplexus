"""
Filter patch features using GrandQC dark-spots masks (class 3).

Reads tissue_threshold_15_filtered h5 files (coords + features) and removes
patches where the dark-spots (mask value 3) pixel fraction exceeds
`--threshold` (default 0.10 = 10%).

Output h5 files have the same format (coords + features, gzip-compressed).

Usage:
    python filter_grandqc_darkspot.py --threshold 0.10 --workers 8
"""

import argparse, warnings, logging
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import h5py
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

BASE_DIR   = Path("/home/jovyan/kgbk271-ibd-volume/data/processed")
IN_DIR     = BASE_DIR / "tissue_threshold_15_filtered" / "20x_224px_0px_overlap" / "features_virchow2"
MASKS_DIR  = BASE_DIR / "tissue_threshold_15_remove_artifact" / "grandqc" / "mpp1" / "grandqc_masks"
OUT_DIR    = BASE_DIR / "tissue_threshold_15_filtered_no_darkspot" / "20x_224px_0px_overlap" / "features_virchow2"

M_P_S       = 512   # GrandQC model patch size in pixels
DARK_CLASS  = 3     # GrandQC class 3 = dark spots

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def filter_slide(h5_path: Path, threshold: float, out_dir: Path) -> dict:
    stem     = h5_path.stem
    out_path = out_dir / h5_path.name

    if out_path.exists():
        with h5py.File(str(out_path), "r") as f:
            n_kept = len(f["coords"])
        with h5py.File(str(h5_path), "r") as f:
            n_total = len(f["coords"])
        return dict(slide=stem, total=n_total, kept=n_kept, skipped=True)

    mask_path = MASKS_DIR / f"{stem}_artifact_mask.png"
    ps_path   = MASKS_DIR / f"{stem}_p_s.npy"

    if not mask_path.exists() or not ps_path.exists():
        # No GrandQC mask: copy everything through unchanged
        with h5py.File(str(h5_path), "r") as f:
            coords   = f["coords"][:]
            features = f["features"][:]
        with h5py.File(str(out_path), "w") as f:
            f.create_dataset("coords",   data=coords,   compression="gzip", compression_opts=4)
            f.create_dataset("features", data=features, compression="gzip", compression_opts=4)
        return dict(slide=stem, total=len(coords), kept=len(coords), error="no mask – kept all")

    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        p_s  = int(np.load(str(ps_path))[0])
        scale = M_P_S / p_s

        with h5py.File(str(h5_path), "r") as f:
            coords   = f["coords"][:]
            features = f["features"][:]

        keep = []
        for coord in coords:
            mx = int(coord[0] * scale)
            my = int(coord[1] * scale)
            patch_l0 = p_s
            mw = mh = max(1, int(patch_l0 * scale))
            region = mask[my:min(my + mh, mask.shape[0]),
                          mx:min(mx + mw, mask.shape[1])]
            if region.size == 0:
                keep.append(True)
                continue
            dark_frac = float((region == DARK_CLASS).mean())
            keep.append(dark_frac < threshold)

        keep_arr = np.array(keep, dtype=bool)
        kept_coords   = coords[keep_arr]
        kept_features = features[keep_arr]

        with h5py.File(str(out_path), "w") as f:
            f.create_dataset("coords",   data=kept_coords,   compression="gzip", compression_opts=4)
            f.create_dataset("features", data=kept_features, compression="gzip", compression_opts=4)

        return dict(slide=stem, total=len(coords), kept=int(keep_arr.sum()))

    except Exception as e:
        return dict(slide=stem, total=0, kept=0, error=str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Max allowed dark-spot pixel fraction per patch (default 0.10)")
    parser.add_argument("--workers",   type=int, default=min(8, cpu_count()))
    parser.add_argument("--start",     type=int, default=0)
    parser.add_argument("--end",       type=int, default=-1)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Input:     {IN_DIR}")
    log.info(f"Masks:     {MASKS_DIR}")
    log.info(f"Output:    {OUT_DIR}")
    log.info(f"Threshold: {args.threshold}  |  Workers: {args.workers}")

    h5_files = sorted(IN_DIR.glob("*.h5"))
    end      = args.end if args.end > 0 else len(h5_files)
    h5_files = h5_files[args.start:end]
    log.info(f"Slides to process: {len(h5_files)} [{args.start}:{end}]")

    worker = partial(filter_slide, threshold=args.threshold, out_dir=OUT_DIR)

    total_in = total_out = n_no_mask = n_errors = 0

    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, h5_files), 1):
            if "error" in result:
                if "no mask" in result["error"]:
                    n_no_mask += 1
                    total_in  += result["total"]
                    total_out += result["kept"]
                else:
                    n_errors += 1
                    log.warning(f"  Error [{result['slide']}]: {result['error']}")
            else:
                total_in  += result["total"]
                total_out += result["kept"]

            if i % 200 == 0 or i == len(h5_files):
                pct = total_out / max(total_in, 1) * 100
                log.info(f"  {i}/{len(h5_files)} slides  |  "
                         f"{total_out:,}/{total_in:,} patches kept ({pct:.1f}%)  "
                         f"[{n_no_mask} no mask, {n_errors} errors]")

    pct = total_out / max(total_in, 1) * 100
    log.info(f"\nDone.")
    log.info(f"  Kept    {total_out:,} / {total_in:,} patches ({pct:.1f}%)")
    log.info(f"  Removed {total_in - total_out:,} patches ({100-pct:.1f}%) as dark spots")
    log.info(f"  Slides with no GrandQC mask (kept all): {n_no_mask}")
    if n_errors:
        log.warning(f"  Slides with errors: {n_errors}")


if __name__ == "__main__":
    main()
