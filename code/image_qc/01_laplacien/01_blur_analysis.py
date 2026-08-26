"""
Laplacian variance blur analysis for patch QC.

Samples patches from many slides, computes sharpness scores, then
generates a grid showing patch examples around each candidate threshold
to help visually select a cut-off.
"""

import random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from openslide import OpenSlide

# ── Paths ─────────────────────────────────────────────────────────────────────
PATCHES_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15/20x_224px_0px_overlap/patches")
WSI_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
OUT_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_remove_artifact/laplacien")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Sampling ──────────────────────────────────────────────────────────────────
N_SLIDES         = 60
PATCHES_PER_SLIDE = 40
RANDOM_SEED      = 42

# Candidate thresholds to visualise
THRESHOLDS = [50, 75, 100, 125, 150, 200, 300, 500]
N_EXAMPLES = 8   # patches to show on each side of every threshold

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def laplacian_variance(patch_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_patch(slide, coord, patch_size_l0: int, patch_size: int) -> np.ndarray:
    region = slide.read_region(
        (int(coord[0]), int(coord[1])), 0, (patch_size_l0, patch_size_l0)
    )
    arr = np.array(region)[:, :, :3]
    if patch_size_l0 != patch_size:
        arr = cv2.resize(arr, (patch_size, patch_size), interpolation=cv2.INTER_LANCZOS4)
    return arr


# ── Sample patches ─────────────────────────────────────────────────────────────
h5_files = sorted(PATCHES_DIR.glob("*_patches.h5"))
selected = random.sample(h5_files, min(N_SLIDES, len(h5_files)))

print(f"Sampling {N_SLIDES} slides …")
scores = []   # list of (lv, patch_rgb)

for h5_path in selected:
    stem = h5_path.stem.replace("_patches", "")
    wsi  = next(iter(WSI_DIR.glob(f"{stem}.*")), None)
    if wsi is None:
        continue
    try:
        slide = OpenSlide(str(wsi))
        f = h5py.File(str(h5_path), "r")
        coords   = f["coords"][:]
        attrs    = dict(f["coords"].attrs)
        psz      = int(attrs["patch_size"])
        psz_l0   = int(round(float(attrs["patch_size_level0"])))
        f.close()

        chosen = np.random.choice(len(coords), min(PATCHES_PER_SLIDE, len(coords)), replace=False)
        for idx in chosen:
            patch = read_patch(slide, coords[idx], psz_l0, psz)
            scores.append((laplacian_variance(patch), patch))
        slide.close()
    except Exception as e:
        print(f"  [skip] {stem}: {e}")

print(f"Total patches: {len(scores)}")
lv_all = np.array([s[0] for s in scores])
print(f"LV  min={lv_all.min():.1f}  p5={np.percentile(lv_all,5):.1f}  "
      f"median={np.median(lv_all):.1f}  p95={np.percentile(lv_all,95):.1f}  max={lv_all.max():.1f}")

# Sort by score
order   = np.argsort(lv_all)
lv_sorted = lv_all[order]
patches_sorted = [scores[i][1] for i in order]

# ── Print removal stats ────────────────────────────────────────────────────────
print("\nRemoval at each threshold:")
for t in THRESHOLDS:
    pct = (lv_all < t).mean() * 100
    print(f"  t={t:4d}: {pct:5.1f}% removed")


# ── Plot ──────────────────────────────────────────────────────────────────────
# Layout: distribution histogram on top, then one row per threshold.
# Each row: N_EXAMPLES patches below threshold | divider | N_EXAMPLES above threshold.

N_COLS = N_EXAMPLES * 2 + 1
N_ROWS = len(THRESHOLDS)
FIG_W  = N_COLS * 1.9
FIG_H  = N_ROWS * 2.2 + 2.5   # +2.5 for histogram

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#12121f")

# ── Histogram ─────────────────────────────────────────────────────────────────
hist_h = 2.0 / FIG_H
ax_hist = fig.add_axes([0.04, 1 - hist_h + 0.005, 0.92, hist_h - 0.02])
ax_hist.set_facecolor("#0d1b2a")
ax_hist.hist(lv_sorted, bins=100, color="#4fc3f7", edgecolor="none", alpha=0.85)
ax_hist.set_xlabel("Laplacian variance", color="white", fontsize=9)
ax_hist.set_ylabel("Count", color="white", fontsize=9)
ax_hist.set_title(
    f"Sharpness distribution  ({len(scores)} patches from {N_SLIDES} slides)",
    color="white", fontsize=10, pad=4
)
ax_hist.tick_params(colors="white", labelsize=8)
for sp in ax_hist.spines.values():
    sp.set_edgecolor("#333")

palette = plt.cm.tab10(np.linspace(0, 0.9, len(THRESHOLDS)))
for t, c in zip(THRESHOLDS, palette):
    pct = (lv_all < t).mean() * 100
    ax_hist.axvline(t, color=c, lw=1.5, ls="--", alpha=0.9,
                    label=f"t={t}  ({pct:.1f}% removed)")
ax_hist.legend(fontsize=7.5, loc="upper right",
               facecolor="#0d1b2a", edgecolor="#444", labelcolor="white",
               ncol=2, framealpha=0.9)

# ── Patch rows ────────────────────────────────────────────────────────────────
row_h = (1 - hist_h) / N_ROWS
col_w = 0.92 / N_COLS
x0    = 0.04

for row_i, (thresh, color) in enumerate(zip(THRESHOLDS, palette)):
    y_bot = 1 - hist_h - (row_i + 1) * row_h + 0.005

    below = np.where(lv_sorted <  thresh)[0]
    above = np.where(lv_sorted >= thresh)[0]

    # Closest to threshold on each side
    b_idx = below[-N_EXAMPLES:] if len(below) >= N_EXAMPLES else below
    a_idx = above[:N_EXAMPLES]  if len(above) >= N_EXAMPLES else above

    # Label column
    ax_lbl = fig.add_axes([x0 - 0.035, y_bot, 0.034, row_h - 0.008])
    ax_lbl.axis("off")
    pct = (lv_all < thresh).mean() * 100
    ax_lbl.text(0.5, 0.5, f"t={thresh}\n{pct:.1f}%\nremoved",
                ha="center", va="center", fontsize=8, color=color,
                fontweight="bold", transform=ax_lbl.transAxes)

    for col_i in range(N_EXAMPLES):
        ax = fig.add_axes([x0 + col_i * col_w, y_bot, col_w - 0.003, row_h - 0.01])
        ax.axis("off")
        if col_i < len(b_idx):
            ax.imshow(patches_sorted[b_idx[col_i]])
            ax.set_title(f"{lv_sorted[b_idx[col_i]]:.0f}", fontsize=6.5,
                         color="#ef9a9a", pad=1)
        else:
            ax.set_facecolor("#0d1b2a")

    # Divider
    ax_div = fig.add_axes([x0 + N_EXAMPLES * col_w, y_bot, col_w - 0.003, row_h - 0.01])
    ax_div.set_facecolor("#12121f")
    ax_div.axis("off")
    ax_div.axvline(0.5, color=color, lw=2)
    ax_div.text(0.5, 0.5, "threshold", ha="center", va="center",
                fontsize=6, color=color, rotation=90, transform=ax_div.transAxes)

    for col_i in range(N_EXAMPLES):
        ax = fig.add_axes([x0 + (N_EXAMPLES + 1 + col_i) * col_w, y_bot,
                           col_w - 0.003, row_h - 0.01])
        ax.axis("off")
        if col_i < len(a_idx):
            ax.imshow(patches_sorted[a_idx[col_i]])
            ax.set_title(f"{lv_sorted[a_idx[col_i]]:.0f}", fontsize=6.5,
                         color="#a5d6a7", pad=1)
        else:
            ax.set_facecolor("#0d1b2a")

# Legend strip
fig.text(0.04, 0.002, "Red score = REMOVED (below threshold)   |   "
         "Green score = KEPT (above threshold)   |   "
         "Patches closest to each threshold boundary are shown",
         color="#aaaaaa", fontsize=8, va="bottom")

out = OUT_DIR / "blur_threshold_analysis.png"
plt.savefig(str(out), dpi=120, bbox_inches="tight",
            facecolor="#12121f", edgecolor="none")
plt.close()
print(f"\nPlot saved → {out}")
