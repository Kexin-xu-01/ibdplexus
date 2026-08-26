"""
GrandQC-based artifact detection for patch filtering.

Uses existing tissue segmentation GeoJSON contours (already computed) to
skip the tissue-detection step, then runs GrandQC artifact segmentation
at MPP=1.5 on each slide.

Artifact class labels (mask pixel values):
  1 – clean tissue
  2 – tissue fold
  3 – dark spots / foreign objects
  4 – pen markings
  5 – air bubble / slide edge
  6 – out of focus
  7 – background
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

GRANDQC_DIR = "/tmp/grandqc/01_WSI_inference_OPENSLIDE_QC"
sys.path.insert(0, GRANDQC_DIR)

import json, random
import numpy as np
import cv2
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from openslide import OpenSlide
import segmentation_models_pytorch as smp

from wsi_slide_info import slide_info
from wsi_process import slide_process_single
from wsi_colors import colors_QC7 as COLORS_QC

# ── Paths ────────────────────────────────────────────────────────────────────
PATCHES_DIR  = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15/20x_224px_0px_overlap/patches")
WSI_DIR      = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
GEOJSON_DIR  = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/patch_qc/trident_processed/contours_geojson")
MODEL_QC     = "/home/jovyan/shared-data/users/kexin/models/histology/grandqc/GrandQC_MPP15.pth"
MODEL_QC_SD  = "/tmp/grandqc_mpp15_state_dict.pth"   # state-dict extracted earlier
OUT_DIR      = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_remove_artifact")
OUT_MASKS    = OUT_DIR / "grandqc_masks"
OUT_MASKS.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE        = "cpu"
MPP_MODEL_TD  = 10.0    # tissue-detection mask resolution
MPP_MODEL_QC  = 1.5     # artifact-detection model resolution
M_P_S         = 512
ENCODER       = "timm-efficientnet-b0"
BACK_CLASS    = 7
ARTIFACT_THR  = 0.1     # flag if >=10 % of patch region is artifact
N_EX          = 6       # example patches per class in the visualization
RANDOM_SEED   = 42

ARTIFACT_NAMES = {
    1: "Clean tissue", 2: "Tissue fold",   3: "Dark spot / foreign",
    4: "Pen marking",  5: "Air bubble / edge", 6: "Out of focus", 7: "Background",
}
ARTIFACT_COLORS = {
    1: "#4CAF50", 2: "#FF5722", 3: "#9C27B0",
    4: "#2196F3", 5: "#FF9800", 6: "#00BCD4", 7: "#9E9E9E",
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Load GrandQC model from extracted state dict ─────────────────────────────
print("Loading GrandQC artifact model (MPP=1.5) from state dict …")
model_qc = smp.Unet(encoder_name=ENCODER, encoder_weights=None, classes=8, activation="softmax")
model_qc.load_state_dict(torch.load(MODEL_QC_SD, map_location=DEVICE, weights_only=True))
model_qc.to(DEVICE); model_qc.eval()
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, "imagenet")

# Patch wsi_process.get_preprocessing to accept the smp preprocessing fn
import wsi_process
_orig_get_prep = wsi_process.get_preprocessing
def _get_prep_patched(image, preprocessing_fn_arg, model_size):
    if image.size != model_size:
        image = image.resize(model_size)
    img = np.array(image)
    x   = preprocessing_fn(img)       # use module-level preprocessing_fn
    return x.transpose(2, 0, 1).astype("float32")
wsi_process.get_preprocessing = _get_prep_patched


# ── Build tissue mask from GeoJSON contours ───────────────────────────────────
def tissue_mask_from_geojson(geojson_path: Path, w_l0: int, h_l0: int, mpp: float) -> np.ndarray:
    """
    Rasterize tissue contours (level-0 pixel coords) into a binary mask at
    MPP=10 (0=tissue, 1=background), matching what GrandQC tissue detection produces.
    """
    rf      = MPP_MODEL_TD / mpp
    mask_w  = int(w_l0 / rf)
    mask_h  = int(h_l0 / rf)
    mask    = np.ones((mask_h, mask_w), dtype=np.uint8)  # 1=background

    with open(geojson_path) as f:
        g = json.load(f)

    for feat in g["features"]:
        coords = np.array(feat["geometry"]["coordinates"][0])
        pts = (coords / rf).astype(np.int32)
        # geojson coords may be (x, y); cv2 fillPoly expects (x, y) in each row
        cv2.fillPoly(mask, [pts], color=0)   # 0 = tissue

    return mask


# ── Artifact detection ────────────────────────────────────────────────────────
def run_artifact_detection(slide: OpenSlide, tis_mask: np.ndarray, mpp: float):
    """Return (full_artifact_mask, p_s)."""
    p_s, patch_n_w, patch_n_h, _, w_l0, h_l0, _ = slide_info(slide, M_P_S, MPP_MODEL_QC)
    tis_resized = np.array(
        Image.fromarray(tis_mask).resize(
            (int(w_l0 * mpp / MPP_MODEL_QC), int(h_l0 * mpp / MPP_MODEL_QC)),
            Image.Resampling.LANCZOS,
        )
    )
    _, full_mask = slide_process_single(
        model_qc, tis_resized, slide,
        patch_n_w, patch_n_h, p_s, M_P_S,
        COLORS_QC, ENCODER, "imagenet",
        DEVICE, BACK_CLASS, MPP_MODEL_QC, mpp, w_l0, h_l0,
    )
    return full_mask, p_s


# ── Patch ↔ mask lookup ───────────────────────────────────────────────────────
def patch_artifact_info(mask: np.ndarray, coord, patch_size_l0: int, p_s: int):
    """
    Map a level-0 patch coordinate into the artifact mask coordinate space.
    Mask pixel (mx, my) covers level-0 region starting at (mx * p_s/M_P_S, my * p_s/M_P_S).
    Returns (artifact_fraction, dominant_artifact_class).
    """
    scale = M_P_S / p_s
    mx    = int(coord[0] * scale)
    my    = int(coord[1] * scale)
    mw    = max(1, int(patch_size_l0 * scale))
    mh    = max(1, int(patch_size_l0 * scale))
    region = mask[my:min(my+mh, mask.shape[0]), mx:min(mx+mw, mask.shape[1])]
    if region.size == 0:
        return 0.0, 1
    art_pix  = (region >= 2) & (region <= 6)
    frac     = float(art_pix.mean())
    dominant = 1
    if frac > 0:
        classes, counts = np.unique(region[art_pix], return_counts=True)
        dominant = int(classes[counts.argmax()])
    return frac, dominant


def read_patch(slide: OpenSlide, coord, patch_size_l0: int, patch_size: int) -> np.ndarray:
    region = slide.read_region((int(coord[0]), int(coord[1])), 0, (patch_size_l0, patch_size_l0))
    arr = np.array(region)[:, :, :3]
    if patch_size_l0 != patch_size:
        arr = cv2.resize(arr, (patch_size, patch_size), interpolation=cv2.INTER_LANCZOS4)
    return arr


# ── Discover slides that have both geojson + h5 coords ───────────────────────
geojson_stems = {p.stem for p in GEOJSON_DIR.glob("*.geojson")}
h5_stems      = {p.stem.replace("_patches", ""): p for p in PATCHES_DIR.glob("*_patches.h5")}
overlap       = sorted(geojson_stems & set(h5_stems.keys()))
print(f"Slides with both geojson and h5 coords: {len(overlap)}")

artifact_examples = {c: [] for c in range(2, 7)}
clean_examples    = []
slide_stats       = []

for stem in overlap:
    wsi_candidates = list(WSI_DIR.glob(f"{stem}.*"))
    if not wsi_candidates:
        print(f"  [skip] no WSI for {stem}"); continue

    print(f"\n{'─'*60}\n  Slide: {stem}")
    slide = OpenSlide(str(wsi_candidates[0]))

    f        = h5py.File(str(h5_stems[stem]), "r")
    coords   = f["coords"][:]
    attrs    = dict(f["coords"].attrs)
    patch_sz = int(attrs["patch_size"])
    patch_l0 = int(round(float(attrs["patch_size_level0"])))
    f.close()

    w_l0, h_l0 = slide.level_dimensions[0]
    mpp = float(slide.properties["openslide.mpp-x"])
    print(f"  MPP={mpp:.4f}  patches={len(coords)}")

    print("  Building tissue mask from geojson …")
    tis_mask = tissue_mask_from_geojson(GEOJSON_DIR / f"{stem}.geojson", w_l0, h_l0, mpp)

    print("  Running artifact detection …")
    art_mask, p_s = run_artifact_detection(slide, tis_mask, mpp)

    mask_path = OUT_MASKS / f"{stem}_artifact_mask.png"
    cv2.imwrite(str(mask_path), art_mask)
    print(f"  Mask saved → {mask_path.name}")

    n_artifact = 0
    art_class_counts = {c: 0 for c in range(2, 7)}
    for coord in coords:
        frac, dom = patch_artifact_info(art_mask, coord, patch_l0, p_s)
        if frac >= ARTIFACT_THR:
            n_artifact += 1
            art_class_counts[dom] = art_class_counts.get(dom, 0) + 1
            if len(artifact_examples[dom]) < N_EX:
                artifact_examples[dom].append(
                    (read_patch(slide, coord, patch_l0, patch_sz), frac, stem)
                )
        else:
            if len(clean_examples) < N_EX:
                clean_examples.append(
                    (read_patch(slide, coord, patch_l0, patch_sz), frac, stem)
                )

    pct = n_artifact / max(len(coords), 1) * 100
    print(f"  Flagged: {n_artifact}/{len(coords)} ({pct:.1f}%)")
    print(f"  By class: { {ARTIFACT_NAMES[c]: v for c, v in art_class_counts.items() if v > 0} }")
    slide_stats.append(dict(slide=stem, total=len(coords), n_artifact=n_artifact,
                            pct=pct, by_class=art_class_counts))
    slide.close()


# ── Visualise artifact examples ────────────────────────────────────────────────
print("\nGenerating artifact examples plot …")

art_classes_present = [c for c in range(2, 7) if artifact_examples[c]]
total_rows = 1 + len(art_classes_present)

fig, axes = plt.subplots(total_rows, N_EX, figsize=(N_EX * 2.5, total_rows * 2.8))
fig.patch.set_facecolor("#1a1a2e")
if total_rows == 1:
    axes = axes[np.newaxis, :]


def render_row(row_axes, examples, label, color):
    for col_i, ax in enumerate(row_axes):
        ax.axis("off")
        if col_i < len(examples):
            img, frac, _ = examples[col_i]
            ax.imshow(img)
            ax.set_title(f"frac={frac:.2f}", fontsize=7, color=color, pad=1)
        else:
            ax.set_facecolor("#0d1b2a")
    row_axes[0].set_ylabel(label, color=color, fontsize=9, fontweight="bold",
                           rotation=90, labelpad=4)


render_row(axes[0], clean_examples, "Clean\ntissue", ARTIFACT_COLORS[1])
for row_i, cls in enumerate(art_classes_present, start=1):
    render_row(axes[row_i], artifact_examples[cls],
               f"Class {cls}\n{ARTIFACT_NAMES[cls]}", ARTIFACT_COLORS[cls])

for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_visible(False)

header = (f"GrandQC artifact examples  |  MPP=1.5 model  |  "
          f"flag threshold≥{ARTIFACT_THR*100:.0f}% of patch\n"
          f"{len(overlap)} slides with existing tissue segmentation\n")
for s in slide_stats:
    header += f"  {s['slide']}: {s['n_artifact']}/{s['total']} ({s['pct']:.1f}%) flagged\n"
fig.suptitle(header.strip(), fontsize=8, color="white", fontweight="bold",
             y=1.01, ha="left", x=0.02)

plt.tight_layout(pad=0.4)
out_plot = OUT_DIR / "grandqc_artifact_examples.png"
plt.savefig(str(out_plot), dpi=130, bbox_inches="tight",
            facecolor="#1a1a2e", edgecolor="none")
plt.close()
print(f"Plot saved → {out_plot}")
print("Done.")
