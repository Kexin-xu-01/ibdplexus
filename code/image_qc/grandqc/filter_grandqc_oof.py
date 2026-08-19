"""
Filter patch coordinates using GrandQC out-of-focus (OOF) masks.

Reads GrandQC artifact masks already produced by run_grandqc.py and
filters patch .h5 files from laplacien_t100/ (Laplacian pre-filtered),
keeping only patches where the OOF class (mask value 5) fraction is
below `--oof_threshold` (default 0.10 = 10%).

Output h5 files are drop-in replacements — same format, same attrs.

Usage:
    python filter_grandqc_oof.py --oof_threshold 0.10 --workers 8
"""

import argparse, warnings, logging
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import h5py
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/jovyan/kgbk271-ibd-volume/data/processed"
                   "/tissue_threshold_15_remove_artifact")
IN_PATCHES  = BASE_DIR / "laplacien_t100"            # Laplacian pre-filtered coords
MASKS_DIR   = BASE_DIR / "grandqc_masks"
OUT_DIR     = BASE_DIR / "grandqc_oof"

M_P_S = 512   # GrandQC model patch size in pixels
OOF_CLASS = 6  # GrandQC class 6 = out-of-focus (class 1=clean, 2=fold, 3=dark, 4=pen, 5=bubble, 6=oof, 7=bg)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def filter_slide(h5_path: Path, oof_threshold: float, out_dir: Path) -> dict:
    """Filter one slide's h5 coords by OOF fraction. Returns stats dict."""
    stem     = h5_path.stem.replace("_patches", "")
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
        return dict(slide=stem, total=0, kept=0, error="no mask")

    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        p_s  = int(np.load(str(ps_path))[0])

        with h5py.File(str(h5_path), "r") as f:
            coords = f["coords"][:]
            attrs  = dict(f["coords"].attrs)

        patch_l0 = int(round(float(attrs["patch_size_level0"])))
        scale    = M_P_S / p_s

        kept = []
        for coord in coords:
            mx = int(coord[0] * scale)
            my = int(coord[1] * scale)
            mw = mh = max(1, int(patch_l0 * scale))
            region = mask[my:min(my + mh, mask.shape[0]),
                          mx:min(mx + mw, mask.shape[1])]
            if region.size == 0:
                kept.append(coord)
                continue
            oof_frac = float((region == OOF_CLASS).mean())
            if oof_frac < oof_threshold:
                kept.append(coord)

        kept_arr = np.array(kept, dtype=np.int64) if kept else np.zeros((0, 2), dtype=np.int64)

        with h5py.File(str(out_path), "w") as f:
            ds = f.create_dataset("coords", data=kept_arr)
            for k, v in attrs.items():
                ds.attrs[k] = v

        return dict(slide=stem, total=len(coords), kept=len(kept))

    except Exception as e:
        return dict(slide=stem, total=0, kept=0, error=str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof_threshold", type=float, default=0.10,
                        help="Max allowed OOF pixel fraction (default: 0.10)")
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()))
    parser.add_argument("--start",   type=int, default=0)
    parser.add_argument("--end",     type=int, default=-1)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Input coords: {IN_PATCHES}")
    log.info(f"Output dir:   {OUT_DIR}")
    log.info(f"OOF threshold: {args.oof_threshold}  |  Workers: {args.workers}")

    h5_files = sorted(IN_PATCHES.glob("*_patches.h5"))
    end      = args.end if args.end > 0 else len(h5_files)
    h5_files = h5_files[args.start:end]
    log.info(f"Slides to process: {len(h5_files)} [{args.start}:{end}]")

    worker = partial(filter_slide, oof_threshold=args.oof_threshold, out_dir=OUT_DIR)

    total_in = total_out = n_no_mask = 0
    errors = []

    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, h5_files), 1):
            if "error" in result:
                if result["error"] == "no mask":
                    n_no_mask += 1
                else:
                    errors.append(result)
            else:
                total_in  += result["total"]
                total_out += result["kept"]
            if i % 100 == 0 or i == len(h5_files):
                pct = total_out / max(total_in, 1) * 100
                log.info(f"  {i}/{len(h5_files)} slides  |  "
                         f"{total_out:,}/{total_in:,} patches kept ({pct:.1f}%)  "
                         f"[{n_no_mask} no mask]")

    pct = total_out / max(total_in, 1) * 100
    log.info(f"\nDone. Kept {total_out:,} / {total_in:,} patches ({pct:.1f}%)")
    log.info(f"Removed (OOF) {total_in - total_out:,} patches ({100-pct:.1f}%)")
    log.info(f"Slides skipped (no GrandQC mask): {n_no_mask}")
    if errors:
        log.warning(f"{len(errors)} slides with errors:")
        for e in errors[:10]:
            log.warning(f"  {e}")


if __name__ == "__main__":
    main()
