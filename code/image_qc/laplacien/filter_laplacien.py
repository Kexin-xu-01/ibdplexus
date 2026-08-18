"""
Filter patch coordinates using Laplacian variance (sharpness) threshold.

Reads every patch from its WSI, computes Laplacian variance, and writes
a new .h5 file containing only coordinates that pass the threshold.
Output h5 files are drop-in replacements for the originals — same format,
same attrs, filtered coords.

Usage:
    python filter_laplacien.py --threshold 100 --workers 8
    python filter_laplacien.py --threshold 100 --workers 8 --start 0 --end 500
"""

import argparse, warnings, logging
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import h5py
from pathlib import Path
from openslide import OpenSlide
from multiprocessing import Pool, cpu_count
from functools import partial

# ── Paths ─────────────────────────────────────────────────────────────────────
PATCHES_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/processed"
                   "/tissue_threshold_15/20x_224px_0px_overlap/patches")
WSI_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
OUT_BASE    = Path("/home/jovyan/kgbk271-ibd-volume/data/processed"
                   "/tissue_threshold_15_remove_artifact")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def process_slide(h5_path: Path, threshold: float, out_dir: Path) -> dict:
    """Filter one slide's patch coords by Laplacian variance. Returns stats dict."""
    stem     = h5_path.stem.replace("_patches", "")
    out_path = out_dir / h5_path.name

    if out_path.exists():
        # Already done — read stats from existing file
        f = h5py.File(str(out_path), "r")
        n_kept = len(f["coords"])
        f.close()
        f = h5py.File(str(h5_path), "r")
        n_total = len(f["coords"])
        f.close()
        return dict(slide=stem, total=n_total, kept=n_kept, skipped=True)

    wsi = next(iter(WSI_DIR.glob(f"{stem}.*")), None)
    if wsi is None:
        return dict(slide=stem, total=0, kept=0, error="no WSI")

    try:
        f      = h5py.File(str(h5_path), "r")
        coords = f["coords"][:]
        attrs  = dict(f["coords"].attrs)
        f.close()

        psz    = int(attrs["patch_size"])
        psz_l0 = int(round(float(attrs["patch_size_level0"])))

        slide  = OpenSlide(str(wsi))
        kept   = []
        for coord in coords:
            reg  = slide.read_region(
                (int(coord[0]), int(coord[1])), 0, (psz_l0, psz_l0)
            )
            arr  = np.array(reg)[:, :, :3]
            if psz_l0 != psz:
                arr = cv2.resize(arr, (psz, psz), interpolation=cv2.INTER_LANCZOS4)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            lv   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if lv >= threshold:
                kept.append(coord)
        slide.close()

        kept_arr = np.array(kept, dtype=np.int64) if kept else np.zeros((0, 2), dtype=np.int64)

        # Write filtered h5 — same format, same attrs
        f = h5py.File(str(out_path), "w")
        ds = f.create_dataset("coords", data=kept_arr)
        for k, v in attrs.items():
            ds.attrs[k] = v
        f.close()

        return dict(slide=stem, total=len(coords), kept=len(kept))

    except Exception as e:
        return dict(slide=stem, total=0, kept=0, error=str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--workers",   type=int,   default=min(8, cpu_count()))
    parser.add_argument("--start",     type=int,   default=0)
    parser.add_argument("--end",       type=int,   default=-1)
    args = parser.parse_args()

    out_dir = OUT_BASE / f"laplacien_t{int(args.threshold)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output → {out_dir}")
    log.info(f"Threshold = {args.threshold}  |  Workers = {args.workers}")

    h5_files = sorted(PATCHES_DIR.glob("*_patches.h5"))
    end      = args.end if args.end > 0 else len(h5_files)
    h5_files = h5_files[args.start:end]
    log.info(f"Slides to process: {len(h5_files)} [{args.start}:{end}]")

    worker = partial(process_slide, threshold=args.threshold, out_dir=out_dir)

    total_in = total_out = 0
    errors   = []

    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, h5_files), 1):
            if "error" in result:
                errors.append(result)
            else:
                total_in  += result["total"]
                total_out += result["kept"]
            if i % 100 == 0 or i == len(h5_files):
                pct = total_out / max(total_in, 1) * 100
                log.info(f"  {i}/{len(h5_files)} slides  |  "
                         f"{total_out:,}/{total_in:,} patches kept ({pct:.1f}%)")

    pct = total_out / max(total_in, 1) * 100
    log.info(f"\nDone. Kept {total_out:,} / {total_in:,} patches ({pct:.1f}%)")
    log.info(f"Removed {total_in - total_out:,} patches ({100-pct:.1f}%)")
    if errors:
        log.warning(f"{len(errors)} slides with errors:")
        for e in errors[:10]:
            log.warning(f"  {e}")


if __name__ == "__main__":
    main()
