"""
Visualise GrandQC artifact examples from already-computed masks.

For each artifact class, samples patches flagged by that class and plots
them side-by-side with clean-tissue examples.

Run after run_grandqc.py has produced masks in grandqc_masks/.
"""

import sys, warnings, random
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
OUT_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_remove_artifact")
MASKS_DIR   = OUT_DIR / "grandqc_masks"

# ── Config ────────────────────────────────────────────────────────────────────
M_P_S        = 512
ARTIFACT_THR = 0.10   # flag patch if >=10% of mask region is artifact
N_EX         = 6      # examples per class
RANDOM_SEED  = 42
N_SLIDES     = 30     # number of slides to sample from for examples

# Mask is 0-indexed: 0=clean, 1-6=artifacts, 7=background
ARTIFACT_NAMES = {
    0: "Clean tissue",  1: "Tissue fold",   2: "Dark spots / foreign",
    3: "Pen markings",  4: "Air bubble / edge",  5: "Out of focus",
    6: "Other artifact",
}
ARTIFACT_COLORS = {
    0: "#4CAF50", 1: "#FF5722", 2: "#388E3C",
    3: "#2196F3", 4: "#FF9800", 5: "#7B1FA2", 6: "#F57C00",
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def patch_artifact_info(mask: np.ndarray, coord, patch_size_l0: int, p_s: int):
    """Map level-0 patch coord → mask region, return (artifact_frac, dominant_class).
    Mask is 0-indexed: 0=clean tissue, 1-6=artifacts, 7=background."""
    scale  = M_P_S / p_s
    mx, my = int(coord[0] * scale), int(coord[1] * scale)
    mw = mh = max(1, int(patch_size_l0 * scale))
    region  = mask[my:min(my + mh, mask.shape[0]), mx:min(mx + mw, mask.shape[1])]
    if region.size == 0:
        return 0.0, 0
    art     = (region >= 1) & (region <= 6)
    frac    = float(art.mean())
    dom     = 0
    if frac > 0:
        cls, cnt = np.unique(region[art], return_counts=True)
        dom = int(cls[cnt.argmax()])
    return frac, dom


def read_patch(slide: OpenSlide, coord, patch_size_l0: int, patch_size: int) -> np.ndarray:
    region = slide.read_region((int(coord[0]), int(coord[1])), 0, (patch_size_l0, patch_size_l0))
    arr = np.array(region)[:, :, :3]
    if patch_size_l0 != patch_size:
        arr = cv2.resize(arr, (patch_size, patch_size), interpolation=cv2.INTER_LANCZOS4)
    return arr


# ── Collect examples ──────────────────────────────────────────────────────────
# Only iterate slides that have both a mask and an h5 file
mask_stems = {p.stem.replace("_artifact_mask", "") for p in MASKS_DIR.glob("*_artifact_mask.png")}
h5_map     = {p.stem.replace("_patches", ""): p for p in PATCHES_DIR.glob("*_patches.h5")}
available  = sorted(mask_stems & set(h5_map.keys()))
print(f"Slides with masks: {len(mask_stems)}  |  with h5: {len(h5_map)}  |  overlap: {len(available)}")

selected = random.sample(available, min(N_SLIDES, len(available)))

artifact_examples = {c: [] for c in range(0, 7)}  # 0=clean, 1-6=artifacts
slide_stats = []

for stem in selected:
    wsi_candidates = list(WSI_DIR.glob(f"{stem}.*"))
    if not wsi_candidates:
        continue

    mask = cv2.imread(str(MASKS_DIR / f"{stem}_artifact_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue

    p_s_path = MASKS_DIR / f"{stem}_p_s.npy"
    if not p_s_path.exists():
        continue
    p_s = int(np.load(str(p_s_path))[0])

    slide = OpenSlide(str(wsi_candidates[0]))
    f     = h5py.File(str(h5_map[stem]), "r")
    coords   = f["coords"][:]
    attrs    = dict(f["coords"].attrs)
    patch_sz = int(attrs["patch_size"])
    patch_l0 = int(round(float(attrs["patch_size_level0"])))
    f.close()

    n_art = 0
    art_counts = {c: 0 for c in range(2, 7)}
    for coord in coords:
        frac, dom = patch_artifact_info(mask, coord, patch_l0, p_s)
        if frac >= ARTIFACT_THR:
            n_art += 1
            art_counts[dom] = art_counts.get(dom, 0) + 1
            if len(artifact_examples[dom]) < N_EX:
                artifact_examples[dom].append(
                    (read_patch(slide, coord, patch_l0, patch_sz), frac, stem)
                )
        else:
            if len(artifact_examples[0]) < N_EX:
                artifact_examples[0].append(
                    (read_patch(slide, coord, patch_l0, patch_sz), frac, stem)
                )

    pct = n_art / max(len(coords), 1) * 100
    slide_stats.append(dict(slide=stem, total=len(coords), n_art=n_art, pct=pct))
    slide.close()

    if all(len(v) >= N_EX for v in artifact_examples.values()):
        break

# ── Plot ──────────────────────────────────────────────────────────────────────
classes_to_show = [0] + [c for c in range(1, 7) if artifact_examples[c]]
n_rows = len(classes_to_show)
fig, axes = plt.subplots(n_rows, N_EX, figsize=(N_EX * 2.5, n_rows * 2.8))
fig.patch.set_facecolor("#1a1a2e")
if n_rows == 1:
    axes = axes[np.newaxis, :]

for row_i, cls in enumerate(classes_to_show):
    examples = artifact_examples[cls]
    color    = ARTIFACT_COLORS[cls]
    label    = ARTIFACT_NAMES[cls]
    for col_i, ax in enumerate(axes[row_i]):
        ax.axis("off")
        if col_i < len(examples):
            img, frac, _ = examples[col_i]
            ax.imshow(img)
            ax.set_title(f"frac={frac:.2f}", fontsize=7, color=color, pad=1)
        else:
            ax.set_facecolor("#0d1b2a")
    axes[row_i, 0].set_ylabel(
        f"{'Class ' + str(cls) + chr(10) if cls > 0 else ''}{label}",
        color=color, fontsize=9, fontweight="bold", rotation=90, labelpad=4
    )

for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_visible(False)

total_patches = sum(s["total"] for s in slide_stats)
total_art     = sum(s["n_art"]  for s in slide_stats)
pct_overall   = total_art / max(total_patches, 1) * 100
header = (f"GrandQC artifact examples  |  MPP=1.5  |  "
          f"flag threshold ≥{ARTIFACT_THR*100:.0f}% of patch\n"
          f"Sampled {len(slide_stats)} slides: "
          f"{total_art}/{total_patches} patches flagged ({pct_overall:.1f}%)")
fig.suptitle(header, fontsize=9, color="white", fontweight="bold",
             y=1.01, ha="left", x=0.02)

plt.tight_layout(pad=0.4)
out_plot = OUT_DIR / "grandqc_artifact_examples.png"
plt.savefig(str(out_plot), dpi=130, bbox_inches="tight",
            facecolor="#1a1a2e", edgecolor="none")
plt.close()
print(f"Plot saved → {out_plot}")
print(f"Overall: {total_art}/{total_patches} ({pct_overall:.1f}%) patches flagged")
