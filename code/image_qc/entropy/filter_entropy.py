"""
Filter patch coordinates by Shannon entropy.

Keeps patches where the grayscale histogram entropy >= --min_entropy (bits).
Low-entropy patches are homogeneous (blank, pen line, empty tissue).

Output h5 files are drop-in replacements — same format, same attrs.
Output folder: tissue_threshold_15_remove_artifact/entropy_t{min_entropy}/

Usage:
    python filter_entropy.py --min_entropy 4.0 --workers 8
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

PATCHES_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/processed"
                   "/tissue_threshold_15/20x_224px_0px_overlap/patches")
WSI_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
OUT_BASE    = Path("/home/jovyan/kgbk271-ibd-volume/data/processed"
                   "/tissue_threshold_15_remove_artifact")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def shannon_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def process_slide(h5_path: Path, min_entropy: float, out_dir: Path) -> dict:
    stem     = h5_path.stem.replace("_patches", "")
    out_path = out_dir / h5_path.name

    if out_path.exists():
        with h5py.File(str(out_path), "r") as f: n_kept = len(f["coords"])
        with h5py.File(str(h5_path),  "r") as f: n_total = len(f["coords"])
        return dict(slide=stem, total=n_total, kept=n_kept, skipped=True)

    wsi = next(iter(WSI_DIR.glob(f"{stem}.*")), None)
    if wsi is None:
        return dict(slide=stem, total=0, kept=0, error="no WSI")

    try:
        with h5py.File(str(h5_path), "r") as f:
            coords = f["coords"][:]
            attrs  = dict(f["coords"].attrs)
        psz    = int(attrs["patch_size"])
        psz_l0 = int(round(float(attrs["patch_size_level0"])))
        slide  = OpenSlide(str(wsi))

        kept = []
        for coord in coords:
            reg = slide.read_region((int(coord[0]), int(coord[1])), 0, (psz_l0, psz_l0))
            arr = np.array(reg)[:, :, :3]
            if psz_l0 != psz:
                arr = cv2.resize(arr, (psz, psz), interpolation=cv2.INTER_LANCZOS4)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            if shannon_entropy(gray) >= min_entropy:
                kept.append(coord)
        slide.close()

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
    parser.add_argument("--min_entropy", type=float, default=4.0)
    parser.add_argument("--workers",     type=int,   default=min(8, cpu_count()))
    parser.add_argument("--start",       type=int,   default=0)
    parser.add_argument("--end",         type=int,   default=-1)
    args = parser.parse_args()

    tag     = f"{args.min_entropy}".replace(".", "p")
    out_dir = OUT_BASE / f"entropy_t{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Output → {out_dir}")
    log.info(f"Min entropy (bits): {args.min_entropy}  |  Workers: {args.workers}")

    h5_files = sorted(PATCHES_DIR.glob("*_patches.h5"))
    end      = args.end if args.end > 0 else len(h5_files)
    h5_files = h5_files[args.start:end]
    log.info(f"Slides: {len(h5_files)} [{args.start}:{end}]")

    worker = partial(process_slide, min_entropy=args.min_entropy, out_dir=out_dir)

    total_in = total_out = 0
    errors   = []

    with Pool(processes=args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, h5_files), 1):
            if "error" in r:
                errors.append(r)
            else:
                total_in  += r["total"]
                total_out += r["kept"]
            if i % 100 == 0 or i == len(h5_files):
                pct = total_out / max(total_in, 1) * 100
                log.info(f"  {i}/{len(h5_files)} slides  |  "
                         f"{total_out:,}/{total_in:,} kept ({pct:.1f}%)")

    pct = total_out / max(total_in, 1) * 100
    log.info(f"\nDone. Kept {total_out:,} / {total_in:,} ({pct:.1f}%)")
    if errors:
        log.warning(f"{len(errors)} errors:")
        for e in errors[:10]: log.warning(f"  {e}")


if __name__ == "__main__":
    main()
