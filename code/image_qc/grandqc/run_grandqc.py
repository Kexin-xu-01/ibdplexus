"""
GrandQC artifact segmentation over all slides that have tissue contours.

Pipeline per slide:
  1. Rasterize tissue GeoJSON contours → binary tissue mask at MPP=10
  2. Run GrandQC artifact model (MPP=1.5) using the tissue mask to skip background
  3. Save raw mask (PNG, values 1-7) for downstream patch filtering
  4. Save colorized overlay for visual QC

Mask values (saved PNG, uint8) — 0-indexed argmax of 8-class softmax:
  0 – clean tissue
  1 – tissue fold
  2 – dark spots / foreign objects
  3 – pen markings
  4 – air bubble / slide edge
  5 – out of focus
  6 – (7th artifact class, treat as artifact)
  7 – background (forced for all background pixels)

Usage (GPU job):
  python run_grandqc.py --start 0 --end 500
  python run_grandqc.py --start 500 --end 1000
  ...

Supports restart: skips slides whose mask already exists in OUT_MASKS.
"""

import sys, os, warnings, argparse, json
warnings.filterwarnings("ignore")

GRANDQC_DIR = "/tmp/grandqc/01_WSI_inference_OPENSLIDE_QC"
if not os.path.isdir(GRANDQC_DIR):
    raise RuntimeError(f"Clone grandqc repo first: git clone --depth=1 "
                       f"https://github.com/cpath-ukk/grandqc {GRANDQC_DIR}")
sys.path.insert(0, GRANDQC_DIR)

import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from openslide import OpenSlide
import segmentation_models_pytorch as smp
from tqdm import tqdm

from wsi_slide_info import slide_info
from wsi_colors import colors_QC7 as COLORS_QC

# ── Paths ─────────────────────────────────────────────────────────────────────
WSI_DIR      = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
GEOJSON_DIR  = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15/contours_geojson")
MODEL_QC_SD  = Path("/home/jovyan/shared-data/users/kexin/models/histology/grandqc/GrandQC_MPP1_state_dict.pth")
MODEL_QC_FULL= Path("/home/jovyan/shared-data/users/kexin/models/histology/grandqc/GrandQC_MPP15.pth")
OUT_DIR      = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_remove_artifact")
OUT_MASKS    = OUT_DIR / "grandqc_masks"
OUT_OVERLAY  = OUT_DIR / "grandqc_overlays"
OUT_MASKS.mkdir(parents=True, exist_ok=True)
OUT_OVERLAY.mkdir(parents=True, exist_ok=True)

# ── GrandQC config ────────────────────────────────────────────────────────────
MPP_MODEL_TD = 10.0   # tissue mask resolution
MPP_MODEL_QC = 1.5    # artifact model resolution (7x)
M_P_S        = 512    # model patch size
ENCODER      = "timm-efficientnet-b0"
BACK_CLASS   = 7      # background class index in mask

# Colormap: mask values 0-7 → RGB  (index = mask_value)
COLORMAP = np.array([
    [128, 128, 128],   # 0: clean tissue – gray
    [255,  99,  71],   # 1: tissue fold – orange
    [  0, 200,   0],   # 2: dark spots – green
    [255,   0,   0],   # 3: pen marks – red
    [255,   0, 255],   # 4: air bubble/edge – pink
    [ 75,   0, 130],   # 5: out of focus – violet
    [255, 165,   0],   # 6: other artifact – amber
    [  0,   0,   0],   # 7: background – black (transparent in overlay)
], dtype=np.uint8)

ARTIFACT_NAMES = {
    0: "clean tissue", 1: "tissue fold",   2: "dark spots",
    3: "pen marks",    4: "air bubble/edge", 5: "out of focus",
    6: "other artifact",
}


# ── Model setup ───────────────────────────────────────────────────────────────
def _install_timm_shims():
    """Create missing timm submodule shims so old pickled models can be unpickled."""
    import importlib, pathlib
    import timm.models.layers as _layers_mod
    layers_dir = pathlib.Path(_layers_mod.__file__).parent
    models_dir = layers_dir.parent
    shims = [
        "activations", "adaptive_avgmax_pool", "attention_pool2d", "blur_pool",
        "cbam", "cond_conv2d", "conv2d_same", "conv_bn_act", "drop", "eca",
        "filter_response_norm", "gather_excite", "global_context", "helpers",
        "inplace_abn", "linear", "mlp", "mixed_conv2d", "non_local_attn",
        "norm_act", "patch_embed", "pool2d_same", "selective_kernel",
        "separable_conv", "space_to_depth", "split_attn", "split_batchnorm",
        "std_conv", "test_time_pool", "trace_utils", "weight_init",
    ]
    for name in shims:
        p = layers_dir / f"{name}.py"
        if not p.exists():
            p.write_text("from timm.models.layers import *\n")
    eb = models_dir / "efficientnet_blocks.py"
    if not eb.exists():
        eb.write_text("from timm.models._efficientnet_blocks import *\n")


def build_model(device: str):
    """Load GrandQC MPP1.5 model from pre-extracted state dict (or extract it first)."""
    sd_path = OUT_DIR / "grandqc_mpp15_state_dict.pth"
    if not sd_path.exists():
        _install_timm_shims()
        import warnings; warnings.filterwarnings("ignore")
        full = torch.load(str(MODEL_QC_FULL), map_location="cpu", weights_only=False)
        torch.save(full.state_dict(), str(sd_path))
        del full
        print(f"  State dict extracted → {sd_path.name}")
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, classes=8, activation="softmax")
    model.load_state_dict(torch.load(str(sd_path), map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


# ── Tissue mask from GeoJSON ───────────────────────────────────────────────────
def tissue_mask_from_geojson(geojson_path: Path, w_l0: int, h_l0: int, mpp: float) -> np.ndarray:
    """
    Rasterize tissue contours (level-0 px coords) to a binary mask at MPP=10.
    Returns array of shape (mask_h, mask_w) with 0=tissue, 1=background.
    """
    rf     = MPP_MODEL_TD / mpp
    mask_w = max(1, int(w_l0 / rf))
    mask_h = max(1, int(h_l0 / rf))
    mask   = np.ones((mask_h, mask_w), dtype=np.uint8)   # default: background

    with open(geojson_path) as f:
        g = json.load(f)

    for feat in g["features"]:
        geom = feat["geometry"]
        rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" \
                else [r[0] for r in geom["coordinates"]]
        for ring in rings:
            pts = (np.array(ring) / rf).astype(np.int32)
            cv2.fillPoly(mask, [pts], color=0)   # 0 = tissue

    return mask


# ── Colorize raw mask ─────────────────────────────────────────────────────────
def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map uint8 mask values 0-7 to RGB using COLORMAP."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for val in range(8):
        rgb[mask == val] = COLORMAP[val]
    return rgb


# ── Artifact inference on one slide ──────────────────────────────────────────
def run_slide(model, preprocessing_fn, slide: OpenSlide, tis_mask: np.ndarray,
              mpp: float, device: str):
    """
    Run artifact segmentation and return (raw_mask, p_s).

    raw_mask: uint8 array, values 1-7
    p_s: level-0 pixels per model cell (needed for patch mapping downstream)
    """
    p_s, patch_n_w, patch_n_h, _, w_l0, h_l0, _ = slide_info(slide, M_P_S, MPP_MODEL_QC)

    # Resize tissue mask from MPP=10 to MPP=1.5 space
    tis_mpp = np.array(
        Image.fromarray(tis_mask).resize(
            (int(w_l0 * mpp / MPP_MODEL_QC), int(h_l0 * mpp / MPP_MODEL_QC)),
            Image.Resampling.LANCZOS,
        )
    )

    full_mask = None

    for hi in tqdm(range(patch_n_h), desc="  rows", leave=False):
        row = None
        for wi in range(patch_n_w):
            x = wi * p_s if wi > 0 else 0
            y = hi * p_s if hi > 0 else 0

            # Check tissue mask cell
            td_cell = tis_mpp[hi * M_P_S:(hi + 1) * M_P_S,
                               wi * M_P_S:(wi + 1) * M_P_S]
            if td_cell.shape != (M_P_S, M_P_S):
                pad = [(0, M_P_S - td_cell.shape[0]), (0, M_P_S - td_cell.shape[1])]
                td_cell = np.pad(td_cell, pad, constant_values=1)

            if np.count_nonzero(td_cell == 0) > 50:
                # Tissue present – run model
                patch = slide.read_region((x, y), 0, (p_s, p_s)).convert("RGB")
                patch = patch.resize((M_P_S, M_P_S), Image.Resampling.LANCZOS)
                img   = preprocessing_fn(np.array(patch))
                x_t   = torch.from_numpy(img.transpose(2, 0, 1).astype("float32"))
                with torch.no_grad():
                    pred = model(x_t.unsqueeze(0).to(device))
                raw = np.argmax(pred.squeeze().cpu().numpy(), axis=0).astype(np.uint8)
                # Keep 0-indexed: 0=clean tissue, 1-6=artifacts, 7=background
                # Force background pixels (tissue mask == 1) to BACK_CLASS
                cell = np.where(td_cell == 1, BACK_CLASS, raw).astype(np.uint8)
            else:
                cell = np.full((M_P_S, M_P_S), BACK_CLASS, dtype=np.uint8)

            row = cell if row is None else np.concatenate([row, cell], axis=1)

        full_mask = row if full_mask is None else np.concatenate([full_mask, row], axis=0)

    return full_mask, p_s


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=-1)
    parser.add_argument("--gpu",   type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading GrandQC model …")
    model = build_model(device)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, "imagenet")

    geojson_files = sorted(GEOJSON_DIR.glob("*.geojson"))
    end = args.end if args.end > 0 else len(geojson_files)
    geojson_files = geojson_files[args.start:end]
    print(f"Processing {len(geojson_files)} slides [{args.start}:{end}]")

    for gj_path in geojson_files:
        stem = gj_path.stem
        mask_path = OUT_MASKS / f"{stem}_artifact_mask.png"
        if mask_path.exists():
            continue  # already done

        wsi_candidates = list(WSI_DIR.glob(f"{stem}.*"))
        if not wsi_candidates:
            print(f"  [skip] no WSI for {stem}")
            continue

        print(f"\n  {stem}")
        try:
            slide = OpenSlide(str(wsi_candidates[0]))
            w_l0, h_l0 = slide.level_dimensions[0]
            mpp = float(slide.properties["openslide.mpp-x"])

            tis_mask = tissue_mask_from_geojson(gj_path, w_l0, h_l0, mpp)
            raw_mask, p_s = run_slide(model, preprocessing_fn, slide, tis_mask, mpp, device)

            # Save raw mask (values 1-7) for downstream filtering
            cv2.imwrite(str(mask_path), raw_mask)

            # Save p_s alongside so filter script can map coordinates correctly
            np.save(str(OUT_MASKS / f"{stem}_p_s.npy"), np.array([p_s]))

            # Save colorized overlay — read the exact slide region covered by the mask
            # mask covers (patch_n_w*p_s) × (patch_n_h*p_s) level-0 pixels
            color_mask = colorize_mask(raw_mask)
            mask_h, mask_w = raw_mask.shape
            best_lvl = slide.get_best_level_for_downsample(p_s / M_P_S)
            lvl_ds   = slide.level_downsamples[best_lvl]
            read_w   = int(mask_w * p_s / M_P_S / lvl_ds)
            read_h   = int(mask_h * p_s / M_P_S / lvl_ds)
            region   = slide.read_region((0, 0), best_lvl, (read_w, read_h)).convert("RGB")
            thumb_np = cv2.resize(np.array(region), (mask_w, mask_h), interpolation=cv2.INTER_AREA)
            overlay  = cv2.addWeighted(thumb_np, 0.6, color_mask, 0.4, 0)
            cv2.imwrite(str(OUT_OVERLAY / f"{stem}_overlay.jpg"), overlay[:, :, ::-1])

            # Stats (artifacts = values 1-6)
            vals, cnts = np.unique(raw_mask, return_counts=True)
            art_pct = cnts[(vals >= 1) & (vals <= 6)].sum() / raw_mask.size * 100
            print(f"    artifact pixels: {art_pct:.1f}%  "
                  f"{ {ARTIFACT_NAMES.get(int(v), str(v)): int(c) for v, c in zip(vals, cnts) if 1 <= v <= 6} }")
            slide.close()
        except Exception as e:
            print(f"    [error] {e}")


if __name__ == "__main__":
    main()
