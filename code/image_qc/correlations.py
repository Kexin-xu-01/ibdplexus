"""
Pairwise correlation between Laplacian variance, Shannon entropy, and Otsu
tissue fraction across sampled patches.

Outputs a scatter-matrix plot and prints Pearson + Spearman correlations.
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
from scipy.stats import pearsonr, spearmanr
from openslide import OpenSlide

PATCHES_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/processed"
                   "/tissue_threshold_15/20x_224px_0px_overlap/patches")
WSI_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
OUT_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/processed"
                   "/tissue_threshold_15_remove_artifact")

N_SLIDES          = 60
PATCHES_PER_SLIDE = 40
RANDOM_SEED       = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def laplacian_variance(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def shannon_entropy(gray):
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    p = hist / hist.sum(); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def otsu_tissue_fraction(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(binary.mean() / 255.0)


h5_files = sorted(PATCHES_DIR.glob("*_patches.h5"))
selected = random.sample(h5_files, min(N_SLIDES, len(h5_files)))
print(f"Sampling {len(selected)} slides …")

lv_scores, ent_scores, otsu_scores = [], [], []

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
            reg = slide.read_region((int(coords[idx][0]), int(coords[idx][1])), 0, (psz_l0, psz_l0))
            arr = np.array(reg)[:, :, :3]
            if psz_l0 != psz:
                arr = cv2.resize(arr, (psz, psz), interpolation=cv2.INTER_LANCZOS4)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            lv_scores.append(laplacian_variance(gray))
            ent_scores.append(shannon_entropy(gray))
            otsu_scores.append(otsu_tissue_fraction(gray))
        slide.close()
    except Exception as e:
        print(f"  [skip] {stem}: {e}")

lv   = np.array(lv_scores)
ent  = np.array(ent_scores)
otsu = np.array(otsu_scores)
print(f"Patches: {len(lv)}")

# ── Correlations ──────────────────────────────────────────────────────────────
pairs = [
    ("Laplacian", lv,   "Entropy",  ent),
    ("Laplacian", lv,   "Otsu",     otsu),
    ("Entropy",   ent,  "Otsu",     otsu),
]
print("\nPearson / Spearman correlations:")
for n1, a, n2, b in pairs:
    pr, _ = pearsonr(a, b)
    sr, _ = spearmanr(a, b)
    print(f"  {n1:12s} vs {n2:12s}:  Pearson r={pr:+.3f}  Spearman ρ={sr:+.3f}")

# ── Scatter matrix ─────────────────────────────────────────────────────────────
metrics = [
    ("Laplacian variance (log)", np.log1p(lv)),
    ("Shannon entropy (bits)",   ent),
    ("Otsu tissue fraction",     otsu),
]
n = len(metrics)
fig, axes = plt.subplots(n, n, figsize=(11, 10))
fig.patch.set_facecolor("#12121f")

for i, (ylabel, y) in enumerate(metrics):
    for j, (xlabel, x) in enumerate(metrics):
        ax = axes[i][j]
        ax.set_facecolor("#0d1b2a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.tick_params(colors="white", labelsize=7)

        if i == j:
            ax.hist(x, bins=50, color="#4fc3f7", edgecolor="none", alpha=0.85)
            ax.set_ylabel("Count", color="white", fontsize=8)
        else:
            pr, _ = pearsonr(x, y)
            sr, _ = spearmanr(x, y)
            ax.scatter(x, y, s=3, alpha=0.3, c="#4fc3f7")
            ax.text(0.05, 0.92, f"r={pr:+.2f}  ρ={sr:+.2f}",
                    transform=ax.transAxes, fontsize=8, color="white",
                    va="top", bbox=dict(boxstyle="round,pad=0.2",
                                        facecolor="#1a1a2e", alpha=0.8))
        if i == n - 1:
            ax.set_xlabel(xlabel, color="white", fontsize=8)
        if j == 0:
            ax.set_ylabel(ylabel, color="white", fontsize=8)

fig.suptitle(f"QC metric correlations  ({len(lv)} patches from {N_SLIDES} slides)",
             color="white", fontsize=11, y=1.01)
plt.tight_layout(pad=0.5)
out = OUT_DIR / "qc_metric_correlations.png"
plt.savefig(str(out), dpi=130, bbox_inches="tight", facecolor="#12121f")
plt.close()
print(f"\nPlot saved → {out}")
