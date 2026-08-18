"""
Sample patches from multiple slides, compute Laplacian variance (blur score),
and generate a diagnostic plot showing patches at different threshold levels.
"""

import os
import random
import numpy as np
import cv2
import h5py
import openslide
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

PATCHES_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15/20x_224px_0px_overlap/patches")
WSI_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
OUT_PLOT = Path("/home/jovyan/blur_threshold_analysis.png")

# ── Sampling config ──────────────────────────────────────────────────────────
SLIDES_TO_SAMPLE = 30          # number of slides
PATCHES_PER_SLIDE = 30         # patches per slide
RANDOM_SEED = 42


def laplacian_variance(patch_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_patch(slide, coord, patch_size_level0, patch_size) -> np.ndarray:
    region = slide.read_region(
        (int(coord[0]), int(coord[1])), 0, (patch_size_level0, patch_size_level0)
    )
    arr = np.array(region)[:, :, :3]
    if patch_size_level0 != patch_size:
        arr = cv2.resize(arr, (patch_size, patch_size), interpolation=cv2.INTER_LANCZOS4)
    return arr


# ── Sample patches ───────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

h5_files = sorted(PATCHES_DIR.glob("*_patches.h5"))
selected_slides = random.sample(h5_files, min(SLIDES_TO_SAMPLE, len(h5_files)))

print(f"Sampling from {len(selected_slides)} slides …")

scores = []     # (lap_var, patch_rgb)
for h5_path in selected_slides:
    stem = h5_path.stem.replace("_patches", "")
    # find matching WSI
    wsi_candidates = list(WSI_DIR.glob(f"{stem}.*"))
    if not wsi_candidates:
        print(f"  [skip] no WSI for {stem}")
        continue

    try:
        slide = openslide.OpenSlide(str(wsi_candidates[0]))
        f = h5py.File(str(h5_path), "r")
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)
        patch_size = int(attrs["patch_size"])
        patch_size_level0 = int(round(float(attrs["patch_size_level0"])))
        f.close()

        chosen = np.random.choice(len(coords), min(PATCHES_PER_SLIDE, len(coords)), replace=False)
        for idx in chosen:
            patch = read_patch(slide, coords[idx], patch_size_level0, patch_size)
            lv = laplacian_variance(patch)
            scores.append((lv, patch))

        slide.close()
        print(f"  {stem}: {len(chosen)} patches")
    except Exception as e:
        print(f"  [error] {stem}: {e}")

print(f"\nTotal patches sampled: {len(scores)}")

lap_values = np.array([s[0] for s in scores])
print(f"Laplacian variance  min={lap_values.min():.1f}  "
      f"max={lap_values.max():.1f}  "
      f"median={np.median(lap_values):.1f}  "
      f"mean={lap_values.mean():.1f}")

# ── Candidate thresholds to visualise ────────────────────────────────────────
thresholds = [50, 100, 150, 200, 300, 500]

# Sort all sampled patches by blur score
scores_sorted = sorted(scores, key=lambda x: x[0])
all_lv = np.array([s[0] for s in scores_sorted])
all_patches = [s[1] for s in scores_sorted]

# For each threshold, pick ~4 example patches just BELOW the threshold (blurry)
# and ~4 just above (sharp), so the reader sees the decision boundary.
EXAMPLES_EACH = 4

fig = plt.figure(figsize=(20, len(thresholds) * 4 + 3))
fig.patch.set_facecolor("#1a1a2e")

title_ax = fig.add_axes([0, 0.97, 1, 0.03])
title_ax.axis("off")
title_ax.text(
    0.5, 0.5,
    "Laplacian Variance Threshold Analysis — blurry (left) vs sharp (right) at each threshold",
    ha="center", va="center", fontsize=13, color="white", fontweight="bold"
)

# Distribution histogram axes
hist_ax = fig.add_axes([0.05, 0.88, 0.9, 0.08])
hist_ax.hist(all_lv, bins=80, color="#4fc3f7", edgecolor="none", alpha=0.8)
hist_ax.set_facecolor("#0d1b2a")
hist_ax.tick_params(colors="white")
hist_ax.set_xlabel("Laplacian variance", color="white", fontsize=9)
hist_ax.set_ylabel("Count", color="white", fontsize=9)
hist_ax.set_title("Distribution of blur scores across sampled patches", color="white", fontsize=10)
for spine in hist_ax.spines.values():
    spine.set_edgecolor("#444")
for thresh, color in zip(thresholds, ["#ef5350","#ff7043","#ffca28","#66bb6a","#26c6da","#ab47bc"]):
    hist_ax.axvline(thresh, color=color, linewidth=1.5, linestyle="--", alpha=0.9,
                    label=f"t={thresh}  ({(all_lv < thresh).mean()*100:.1f}% removed)")
hist_ax.legend(fontsize=8, loc="upper right",
               facecolor="#0d1b2a", edgecolor="#444", labelcolor="white")

# One row per threshold
n_cols = EXAMPLES_EACH * 2 + 1   # blurry | divider | sharp
row_height = 0.83 / len(thresholds)

for row_i, (thresh, row_color) in enumerate(
    zip(thresholds, ["#ef5350","#ff7043","#ffca28","#66bb6a","#26c6da","#ab47bc"])
):
    y_top = 0.87 - row_i * row_height
    y_bottom = y_top - row_height + 0.005

    # label
    lbl_ax = fig.add_axes([0.0, y_bottom, 0.05, row_height - 0.005])
    lbl_ax.set_facecolor("#0d1b2a")
    lbl_ax.axis("off")
    pct_removed = (all_lv < thresh).mean() * 100
    lbl_ax.text(0.5, 0.5,
                f"t = {thresh}\n({pct_removed:.1f}%\nremoved)",
                ha="center", va="center", fontsize=9,
                color=row_color, fontweight="bold", transform=lbl_ax.transAxes)

    below_idx = np.where(all_lv < thresh)[0]
    above_idx = np.where(all_lv >= thresh)[0]

    # pick the highest-LV examples from below (closest to threshold = most ambiguous)
    if len(below_idx) >= EXAMPLES_EACH:
        blurry_idx = below_idx[-EXAMPLES_EACH:]
    else:
        blurry_idx = below_idx

    # pick the lowest-LV examples from above (closest to threshold)
    if len(above_idx) >= EXAMPLES_EACH:
        sharp_idx = above_idx[:EXAMPLES_EACH]
    else:
        sharp_idx = above_idx

    col_width = 0.9 / n_cols
    col_start = 0.05

    for col_i in range(EXAMPLES_EACH):
        ax = fig.add_axes([col_start + col_i * col_width, y_bottom,
                           col_width - 0.003, row_height - 0.01])
        ax.axis("off")
        if col_i < len(blurry_idx):
            ax.imshow(all_patches[blurry_idx[col_i]])
            ax.set_title(f"LV={all_lv[blurry_idx[col_i]]:.0f}", fontsize=7,
                         color="#ef9a9a", pad=1)
        else:
            ax.set_facecolor("#0d1b2a")

    # divider
    div_ax = fig.add_axes([col_start + EXAMPLES_EACH * col_width, y_bottom,
                           col_width - 0.003, row_height - 0.01])
    div_ax.set_facecolor("#0d1b2a")
    div_ax.axis("off")
    div_ax.axvline(0.5, color=row_color, linewidth=2)
    div_ax.text(0.5, 0.5, "│\nthresh", ha="center", va="center",
                fontsize=7, color=row_color, transform=div_ax.transAxes)

    for col_i in range(EXAMPLES_EACH):
        ax = fig.add_axes([col_start + (EXAMPLES_EACH + 1 + col_i) * col_width, y_bottom,
                           col_width - 0.003, row_height - 0.01])
        ax.axis("off")
        if col_i < len(sharp_idx):
            ax.imshow(all_patches[sharp_idx[col_i]])
            ax.set_title(f"LV={all_lv[sharp_idx[col_i]]:.0f}", fontsize=7,
                         color="#a5d6a7", pad=1)
        else:
            ax.set_facecolor("#0d1b2a")

plt.savefig(str(OUT_PLOT), dpi=130, bbox_inches="tight",
            facecolor="#1a1a2e", edgecolor="none")
plt.close()
print(f"\nPlot saved → {OUT_PLOT}")
print("\nPercentages removed at each threshold:")
for t in thresholds:
    pct = (all_lv < t).mean() * 100
    n_kept = (all_lv >= t).sum()
    print(f"  t={t:4d}:  {pct:5.1f}% removed,  ~{n_kept} kept (from sample)")
