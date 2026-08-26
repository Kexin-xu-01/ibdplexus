"""
Inspect patches rejected by the Laplacian filter.

Samples slides, computes LV for all patches, collects those below the
threshold, and:
  1. Saves individual patch PNGs organised into LV-range subfolders
     so you can browse them like a file explorer.
  2. Generates a ranked grid plot (worst → near-threshold), showing
     all rejected patches colour-coded by LV bucket.
  3. Generates a per-bucket summary grid (one page per bucket).

Output structure:
  laplacien_t100_rejected/
    plots/
      ranked_grid.png          -- all rejected patches in LV order
      bucket_00_0-10.png       -- 6×N grid for LV 0-10
      bucket_01_10-25.png
      bucket_02_25-50.png
      bucket_03_50-75.png
      bucket_04_75-100.png
    patches/
      lv_000-010/slide__x_y__lv12.3.png
      lv_010-025/...
      ...

Usage:
    python inspect_rejected.py --threshold 100 --n_slides 80 --workers 8
"""

import argparse, warnings, random, logging
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from openslide import OpenSlide
from multiprocessing import Pool, cpu_count
from functools import partial

# ── Paths ─────────────────────────────────────────────────────────────────────
PATCHES_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/processed"
                   "/tissue_threshold_15/20x_224px_0px_overlap/patches")
WSI_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
OUT_BASE    = Path("/home/jovyan/kgbk271-ibd-volume/data/processed"
                   "/tissue_threshold_15_remove_artifact/laplacien_t100_rejected")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# LV range buckets for organisation
BUCKETS = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 100)]
BUCKET_COLORS = ["#ef5350", "#ff7043", "#ffa726", "#ffcc02", "#c5e1a5"]


def bucket_name(lo, hi):
    return f"lv_{lo:03d}-{hi:03d}"


def bucket_label(lo, hi):
    return f"LV {lo}–{hi}"


def collect_slide(args_tuple):
    """Worker: collect rejected patches from one slide. Returns list of (lv, slide_stem, coord, patch_rgb)."""
    h5_path, threshold = args_tuple
    stem = h5_path.stem.replace("_patches", "")
    wsi  = next(iter(WSI_DIR.glob(f"{stem}.*")), None)
    if wsi is None:
        return []
    try:
        slide = OpenSlide(str(wsi))
        with h5py.File(str(h5_path), "r") as f:
            coords = f["coords"][:]
            attrs  = dict(f["coords"].attrs)
        psz    = int(attrs["patch_size"])
        psz_l0 = int(round(float(attrs["patch_size_level0"])))

        rejected = []
        for coord in coords:
            reg  = slide.read_region((int(coord[0]), int(coord[1])), 0, (psz_l0, psz_l0))
            arr  = np.array(reg)[:, :, :3]
            if psz_l0 != psz:
                arr = cv2.resize(arr, (psz, psz), interpolation=cv2.INTER_LANCZOS4)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            lv   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if lv < threshold:
                rejected.append((lv, stem, (int(coord[0]), int(coord[1])), arr))
        slide.close()
        return rejected
    except Exception as e:
        log.warning(f"  [skip] {stem}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--n_slides",  type=int,   default=80)
    parser.add_argument("--workers",   type=int,   default=min(8, cpu_count()))
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--max_per_bucket", type=int, default=300,
                        help="Max individual PNGs saved per bucket folder")
    args = parser.parse_args()

    random.seed(args.seed)

    out_plots   = OUT_BASE / "plots"
    out_patches = OUT_BASE / "patches"
    out_plots.mkdir(parents=True, exist_ok=True)
    for lo, hi in BUCKETS:
        (out_patches / bucket_name(lo, hi)).mkdir(parents=True, exist_ok=True)

    # ── Sample slides ─────────────────────────────────────────────────────────
    all_h5 = sorted(PATCHES_DIR.glob("*_patches.h5"))
    selected = random.sample(all_h5, min(args.n_slides, len(all_h5)))
    log.info(f"Sampling {len(selected)} slides for rejected-patch inspection")

    worker_args = [(p, args.threshold) for p in selected]

    all_rejected = []   # list of (lv, stem, coord, patch_rgb)
    with Pool(processes=args.workers) as pool:
        for i, results in enumerate(pool.imap_unordered(collect_slide, worker_args), 1):
            all_rejected.extend(results)
            if i % 10 == 0 or i == len(selected):
                log.info(f"  {i}/{len(selected)} slides  |  {len(all_rejected):,} rejected so far")

    log.info(f"\nTotal rejected patches collected: {len(all_rejected):,}")
    if not all_rejected:
        log.warning("No rejected patches found — check threshold or slide paths.")
        return

    # Sort by LV ascending (blurriest first)
    all_rejected.sort(key=lambda x: x[0])

    # ── Save individual PNGs into bucket folders ───────────────────────────────
    bucket_lists = {(lo, hi): [] for lo, hi in BUCKETS}
    for lv, stem, (cx, cy), patch in all_rejected:
        for lo, hi in BUCKETS:
            if lo <= lv < hi:
                bucket_lists[(lo, hi)].append((lv, stem, cx, cy, patch))
                break

    saved_counts = {}
    for (lo, hi), items in bucket_lists.items():
        bdir = out_patches / bucket_name(lo, hi)
        saved = 0
        for lv, stem, cx, cy, patch in items[:args.max_per_bucket]:
            fname = bdir / f"{stem}__{cx}_{cy}__lv{lv:.1f}.png"
            cv2.imwrite(str(fname), patch[:, :, ::-1])
            saved += 1
        saved_counts[(lo, hi)] = saved
        log.info(f"  bucket {bucket_name(lo,hi)}: {len(items):,} patches, saved {saved} PNGs")

    # ── Ranked grid plot (all rejected, LV ascending) ─────────────────────────
    log.info("Generating ranked grid plot …")
    patches_only = [p for _, _, _, p in all_rejected]
    lvs_only     = [lv for lv, _, _, _ in all_rejected]

    n_show   = min(len(patches_only), 600)
    indices  = np.linspace(0, len(patches_only) - 1, n_show, dtype=int)
    ncols    = 30
    nrows    = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 0.9, nrows * 1.1))
    fig.patch.set_facecolor("#12121f")
    axes = np.array(axes).reshape(-1)

    for ax_i, idx in enumerate(indices):
        ax = axes[ax_i]
        ax.imshow(patches_only[idx])
        ax.axis("off")
        lv = lvs_only[idx]
        # colour by bucket
        col = "#ef5350"
        for (lo, hi), c in zip(BUCKETS, BUCKET_COLORS):
            if lo <= lv < hi:
                col = c; break
        ax.set_title(f"{lv:.0f}", fontsize=5, color=col, pad=1)

    for ax in axes[len(indices):]:
        ax.axis("off"); ax.set_facecolor("#12121f")

    fig.suptitle(
        f"Rejected patches — LV < {args.threshold:.0f}  "
        f"({len(all_rejected):,} total from {len(selected)} slides, showing {n_show} evenly spaced)\n"
        f"Sorted blurriest → sharpest (left→right, top→bottom).  "
        f"Score colour: red=0-10  orange=10-25  amber=25-50  yellow=50-75  green=75-100",
        fontsize=8, color="white", y=1.002
    )
    plt.tight_layout(pad=0.2)
    out = out_plots / "ranked_grid.png"
    plt.savefig(str(out), dpi=130, bbox_inches="tight", facecolor="#12121f")
    plt.close()
    log.info(f"  Saved → {out}")

    # ── Per-bucket summary grids ───────────────────────────────────────────────
    log.info("Generating per-bucket plots …")
    for bi, ((lo, hi), color) in enumerate(zip(BUCKETS, BUCKET_COLORS)):
        items = bucket_lists[(lo, hi)]
        if not items:
            continue
        n_show_b = min(len(items), 180)
        indices_b = np.linspace(0, len(items) - 1, n_show_b, dtype=int)
        ncols_b   = 18
        nrows_b   = (n_show_b + ncols_b - 1) // ncols_b
        fig, axes = plt.subplots(nrows_b, ncols_b, figsize=(ncols_b * 1.1, nrows_b * 1.3))
        fig.patch.set_facecolor("#12121f")
        axes = np.array(axes).reshape(-1)
        for ax_i, idx in enumerate(indices_b):
            lv, stem, cx, cy, patch = items[idx]
            ax = axes[ax_i]
            ax.imshow(patch)
            ax.axis("off")
            ax.set_title(f"{lv:.1f}", fontsize=6, color=color, pad=1)
        for ax in axes[len(indices_b):]:
            ax.axis("off"); ax.set_facecolor("#12121f")
        fig.suptitle(
            f"{bucket_label(lo, hi)}  |  {len(items):,} patches from {len(selected)} slides  "
            f"|  showing {n_show_b} evenly spaced",
            fontsize=9, color=color, y=1.002
        )
        plt.tight_layout(pad=0.2)
        out = out_plots / f"bucket_{bi:02d}_{bucket_name(lo,hi)}.png"
        plt.savefig(str(out), dpi=130, bbox_inches="tight", facecolor="#12121f")
        plt.close()
        log.info(f"  Saved → {out}")

    log.info(f"\nAll outputs in {OUT_BASE}")
    log.info(f"  plots/           — summary grids")
    log.info(f"  patches/         — individual PNGs organised by LV bucket")


if __name__ == "__main__":
    main()
