"""
Diagnostic analysis for Shannon entropy filter.

Samples patches, computes grayscale histogram entropy, and generates a
threshold-strip plot to help choose a minimum entropy cutoff.

Low entropy = homogeneous patch (blank tissue, uniform stain, background).
High entropy = complex texture, more information content.

Output: tissue_threshold_15_remove_artifact/entropy/analysis_entropy.png

Usage:
    python analysis.py
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
PATCHES_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/processed"
                   "/tissue_threshold_15/20x_224px_0px_overlap/patches")
WSI_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
OUT_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/processed"
                   "/tissue_threshold_15_remove_artifact/entropy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SLIDES          = 60
PATCHES_PER_SLIDE = 40
RANDOM_SEED       = 42
N_EXAMPLES        = 8

THRESHOLDS = [3.0, 4.0, 5.0, 5.5, 6.0, 6.5]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def shannon_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def read_patch(slide, coord, psz_l0, psz):
    reg = slide.read_region((int(coord[0]), int(coord[1])), 0, (psz_l0, psz_l0))
    arr = np.array(reg)[:, :, :3]
    if psz_l0 != psz:
        arr = cv2.resize(arr, (psz, psz), interpolation=cv2.INTER_LANCZOS4)
    return arr


# ── Sample ────────────────────────────────────────────────────────────────────
h5_files = sorted(PATCHES_DIR.glob("*_patches.h5"))
selected = random.sample(h5_files, min(N_SLIDES, len(h5_files)))
print(f"Sampling {len(selected)} slides …")

scores  = []
patches = []

for h5_path in selected:
    stem = h5_path.stem.replace("_patches", "")
    wsi  = next(iter(WSI_DIR.glob(f"{stem}.*")), None)
    if wsi is None:
        continue
    try:
        slide = OpenSlide(str(wsi))
        with h5py.File(str(h5_path), "r") as f:
            coords = f["coords"][:]
            attrs  = dict(f["coords"].attrs)
        psz    = int(attrs["patch_size"])
        psz_l0 = int(round(float(attrs["patch_size_level0"])))
        chosen = np.random.choice(len(coords), min(PATCHES_PER_SLIDE, len(coords)), replace=False)
        for idx in chosen:
            patch = read_patch(slide, coords[idx], psz_l0, psz)
            gray  = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            scores.append(shannon_entropy(gray))
            patches.append(patch)
        slide.close()
    except Exception as e:
        print(f"  [skip] {stem}: {e}")

scores  = np.array(scores)
print(f"Total patches: {len(scores)}")
print(f"Entropy (bits): min={scores.min():.2f}  p5={np.percentile(scores,5):.2f}  "
      f"median={np.median(scores):.2f}  p95={np.percentile(scores,95):.2f}  max={scores.max():.2f}")
print("\nRemoval at each threshold:")
for t in THRESHOLDS:
    print(f"  entropy < {t:.1f}: {(scores < t).mean()*100:.1f}% removed")

# ── Plot ──────────────────────────────────────────────────────────────────────
order   = np.argsort(scores)
s_sort  = scores[order]
p_sort  = [patches[i] for i in order]
palette = plt.cm.tab10(np.linspace(0, 0.9, len(THRESHOLDS)))

N_COLS = N_EXAMPLES * 2 + 1
N_ROWS = len(THRESHOLDS)
FIG_W  = N_COLS * 1.9
FIG_H  = N_ROWS * 2.2 + 2.5

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#12121f")

hist_h = 2.0 / FIG_H
ax_h   = fig.add_axes([0.04, 1 - hist_h + 0.005, 0.92, hist_h - 0.02])
ax_h.set_facecolor("#0d1b2a")
ax_h.hist(s_sort, bins=80, color="#ce93d8", edgecolor="none", alpha=0.85)
ax_h.set_xlabel("Shannon entropy (bits)", color="white", fontsize=9)
ax_h.set_ylabel("Count", color="white", fontsize=9)
ax_h.set_title(f"Patch entropy distribution  ({len(scores)} patches from {N_SLIDES} slides)",
               color="white", fontsize=10, pad=4)
ax_h.tick_params(colors="white", labelsize=8)
for sp in ax_h.spines.values():
    sp.set_edgecolor("#333")
for t, c in zip(THRESHOLDS, palette):
    pct = (scores < t).mean() * 100
    ax_h.axvline(t, color=c, lw=1.5, ls="--", alpha=0.9,
                 label=f"t={t}  ({pct:.1f}% removed)")
ax_h.legend(fontsize=7.5, loc="upper left", facecolor="#0d1b2a",
            edgecolor="#444", labelcolor="white", ncol=2, framealpha=0.9)

row_h = (1 - hist_h) / N_ROWS
col_w = 0.92 / N_COLS
x0    = 0.04

for row_i, (thresh, color) in enumerate(zip(THRESHOLDS, palette)):
    y_bot = 1 - hist_h - (row_i + 1) * row_h + 0.005
    below = np.where(s_sort <  thresh)[0]
    above = np.where(s_sort >= thresh)[0]
    b_idx = below[-N_EXAMPLES:] if len(below) >= N_EXAMPLES else below
    a_idx = above[:N_EXAMPLES]  if len(above) >= N_EXAMPLES else above

    ax_lbl = fig.add_axes([x0 - 0.035, y_bot, 0.034, row_h - 0.008])
    ax_lbl.axis("off")
    pct = (scores < thresh).mean() * 100
    ax_lbl.text(0.5, 0.5, f"t={thresh}\n{pct:.1f}%\nremoved",
                ha="center", va="center", fontsize=8, color=color,
                fontweight="bold", transform=ax_lbl.transAxes)

    for col_i in range(N_EXAMPLES):
        ax = fig.add_axes([x0 + col_i * col_w, y_bot, col_w - 0.003, row_h - 0.01])
        ax.axis("off")
        if col_i < len(b_idx):
            ax.imshow(p_sort[b_idx[col_i]])
            ax.set_title(f"{s_sort[b_idx[col_i]]:.2f}", fontsize=6.5, color="#ef9a9a", pad=1)
        else:
            ax.set_facecolor("#0d1b2a")

    ax_div = fig.add_axes([x0 + N_EXAMPLES * col_w, y_bot, col_w - 0.003, row_h - 0.01])
    ax_div.set_facecolor("#12121f"); ax_div.axis("off")
    ax_div.axvline(0.5, color=color, lw=2)
    ax_div.text(0.5, 0.5, "threshold", ha="center", va="center",
                fontsize=6, color=color, rotation=90, transform=ax_div.transAxes)

    for col_i in range(N_EXAMPLES):
        ax = fig.add_axes([x0 + (N_EXAMPLES + 1 + col_i) * col_w, y_bot,
                           col_w - 0.003, row_h - 0.01])
        ax.axis("off")
        if col_i < len(a_idx):
            ax.imshow(p_sort[a_idx[col_i]])
            ax.set_title(f"{s_sort[a_idx[col_i]]:.2f}", fontsize=6.5, color="#a5d6a7", pad=1)
        else:
            ax.set_facecolor("#0d1b2a")

fig.text(0.04, 0.002,
         "Red = REMOVED (below threshold)   |   Green = KEPT   |   "
         "Patches closest to threshold boundary shown",
         color="#aaaaaa", fontsize=8, va="bottom")

out = OUT_DIR / "analysis_entropy.png"
plt.savefig(str(out), dpi=120, bbox_inches="tight", facecolor="#12121f", edgecolor="none")
plt.close()
print(f"\nSaved → {out}")
