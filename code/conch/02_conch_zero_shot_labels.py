"""
CONCH v1 zero-shot patch labelling.

Uses CONCH v1's CLIP-style vision+text encoders to score each high-attention
patch image against a set of pathology text prompts, producing ranked labels
for every patch.

Architecture note
-----------------
CONCH v1 is a CoCa-style model but the local checkpoint ONLY contains the
contrastive (CLIP) weights — the captioning text-decoder is absent. Zero-shot
scoring (image ↔ text cosine similarity) is therefore the only available text
output, not free-form caption generation. For free-form generation use
03_prism2_patch_describe.py instead.

Requires the CONCH package:
    pip install git+https://github.com/Mahmoodlab/CONCH.git

Pipeline
--------
    prism2_attention_heatmap/<slide>.h5   → top-K patch coords
    tiff_mpp_corrected/<slide>.tiff       → raw patch images via OpenSlide
    CONCH v1 visual encoder               → 512-dim contrast embeddings
    CONCH v1 text encoder                 → 512-dim text embeddings for prompts
    cosine similarity                     → per-patch ranked label list

Output JSON per slide (prism2_conch_zero_shot/<slide>.json):
    [
      {
        "patch_rank": 0,
        "coord_x": 28896, "coord_y": 38304,
        "attn_score": 0.87,
        "labels": [
          {"label": "neutrophilic infiltration", "score": 0.34},
          ...
        ]
      },
      ...
    ]

Usage:
    python 02_conch_zero_shot_labels.py --slide 10407210HE1
    python 02_conch_zero_shot_labels.py --all
    python 02_conch_zero_shot_labels.py --all --n_patches 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

try:
    from conch.open_clip_custom import create_model_from_pretrained
    from conch.open_clip_custom.custom_tokenizer import get_tokenizer, tokenize
    _CONCH_AVAILABLE = True
except ImportError:
    _CONCH_AVAILABLE = False

CONCH_V1_PATH   = "/home/jovyan/shared-data/users/kexin/models/histology/conch_v1/pytorch_model.bin"
HEATMAP_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/prism2_attention_heatmap")
WSI_DIR         = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")
OUT_DIR         = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/prism2_conch_zero_shot")

N_TOP_PATCHES = 8
# CONCH input = 448px at 20x (0.5 MPP).  WSI MPP ≈ 0.171 → 448×0.5/0.171 ≈ 1311px at level-0.
# PRISM2 used 224px @ 20x ≈ 655px at level-0 (half the linear size).
PRISM2_PATCH_SIZE = 1311  # level-0 pixels — CONCH native 20x scale

# ---------------------------------------------------------------------------
# Prompt design — matches CONCH's own zero-shot classification protocol
#
# The official CONCH zero-shot approach (from their CRC100K / NSCLC notebooks):
#   1. SHORT classnames  (the concept, not a long sentence)
#   2. 22 template variants wrapping CLASSNAME
#   3. Average embedding over all (classname × template) pairs
#
# Pre-training captions come from pathologist Twitter posts and educational
# slides — short, natural pathology language like "active colitis, H&E" or
# "cryptitis with abscess formation" — NOT "a histology patch showing X..."
# Long descriptive sentences misalign with that distribution.
# ---------------------------------------------------------------------------

# 22 templates from CONCH's official prompt files (nsclc / crc100k)
TEMPLATES = [
    "CLASSNAME.",
    "a photomicrograph showing CLASSNAME.",
    "a photomicrograph of CLASSNAME.",
    "an image of CLASSNAME.",
    "an image showing CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "there is CLASSNAME.",
    "a histopathological image showing CLASSNAME.",
    "a histopathological image of CLASSNAME.",
    "a histopathological photograph of CLASSNAME.",
    "a histopathological photograph showing CLASSNAME.",
    "shows CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "an H&E stained image of CLASSNAME.",
    "an H&E stained image showing CLASSNAME.",
    "an H&E image showing CLASSNAME.",
    "an H&E image of CLASSNAME.",
    "CLASSNAME, H&E stain.",
    "CLASSNAME, H&E.",
]

# Each classname is the exact UAMP term label (underscores → spaces).
# build_text_classifier() averages embeddings over all 22 templates.
TERM_CLASSNAMES = {
    "inflammation_involvement":             ["inflammation involvement"],
    "crypt_architectural_distortion":       ["crypt architectural distortion"],
    "neutrophil_granulocytic_infiltration": ["neutrophil granulocytic infiltration"],
    "crypt_abscesses":                      ["crypt abscesses"],
    "lymphoid_aggregates":                  ["lymphoid aggregates"],
    "histiocytic_granulomas":               ["histiocytic granulomas"],
    "mucin_depletion":                      ["mucin depletion"],
    "pyloric_gland_metaplasia":             ["pyloric gland metaplasia"],
    "paneth_cell_metaplasia":               ["paneth cell metaplasia"],
    "neuronal_hyperplasia":                 ["neuronal hyperplasia"],
    "muscular_hypertrophy":                 ["muscular hypertrophy"],
    "normal_mucosa":                        ["normal mucosa"],
}

LABEL_NAMES = list(TERM_CLASSNAMES.keys())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_conch_v1(device: torch.device):
    if not _CONCH_AVAILABLE:
        raise ImportError(
            "CONCH package not installed. Run:\n"
            "  pip install git+https://github.com/Mahmoodlab/CONCH.git"
        )
    model, transform = create_model_from_pretrained(
        'conch_ViT-B-16', checkpoint_path=CONCH_V1_PATH
    )
    model = model.to(device).eval()
    tokenizer = get_tokenizer()
    return model, transform, tokenizer


@torch.no_grad()
def build_text_classifier(model, tokenizer, device: torch.device) -> np.ndarray:
    """
    Build a (n_labels, 512) classifier matrix using the official CONCH
    template-ensemble approach:
      for each label → average L2-normalised embeddings over all
      (classname × template) pairs, then re-normalise.

    Returns float32 numpy array, shape (n_labels, 512).
    """
    import torch.nn.functional as F
    weights = []
    for label in LABEL_NAMES:
        classnames = TERM_CLASSNAMES[label]
        embs = []
        for cname in classnames:
            prompts = [t.replace("CLASSNAME", cname) for t in TEMPLATES]
            tokens = tokenize(tokenizer, prompts).to(device)
            e = model.encode_text(tokens, normalize=True)   # (22, 512)
            embs.append(e)
        # Stack: (n_classnames, 22, 512) → mean over both dims → (512,)
        class_emb = torch.stack(embs).mean(dim=(0, 1))
        class_emb = F.normalize(class_emb, dim=-1)
        weights.append(class_emb)
    # (n_labels, 512)
    return torch.stack(weights).cpu().float().numpy()


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_top_patches(
    wsi_path: Path,
    coords: np.ndarray,
    heatmap_norm: np.ndarray,
    n: int,
    patch_px: int,
) -> tuple[list[Image.Image], np.ndarray, np.ndarray, np.ndarray]:
    import openslide
    top_idx = np.argsort(heatmap_norm)[::-1][:n]
    sl = openslide.OpenSlide(str(wsi_path))
    images, sel_coords, sel_scores = [], [], []
    for i in top_idx:
        x, y = int(coords[i, 0]), int(coords[i, 1])
        region = sl.read_region((x, y), 0, (patch_px, patch_px))
        images.append(region.convert("RGB"))
        sel_coords.append([x, y])
        sel_scores.append(float(heatmap_norm[i]))
    sl.close()
    return images, np.array(sel_coords, dtype=np.int64), np.array(sel_scores, dtype=np.float32), top_idx


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_scores(
    model,
    transform,
    images: list[Image.Image],
    classifier: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Return (n_patches, n_labels) cosine similarity matrix, float32.

    classifier : (n_labels, 512) pre-built from build_text_classifier().
                 Each row is already L2-normalised, so dot-product = cosine sim.
    """
    img_tensors = torch.stack([transform(img) for img in images]).to(device)
    image_embs = model.encode_image(img_tensors, normalize=True, proj_contrast=True)  # (K, 512)
    clf_t = torch.from_numpy(classifier).to(device)                                   # (L, 512)
    scores = (image_embs @ clf_t.T).cpu().float().numpy()                             # (K, L)
    return scores


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------

def process_slide(
    slide: str,
    model,
    transform,
    classifier: np.ndarray,
    device: torch.device,
    n_patches: int,
    skip_existing: bool,
):
    out_json = OUT_DIR / f"{slide}.json"
    if skip_existing and out_json.exists():
        print(f"  [skip] {slide}")
        return

    heatmap_h5 = HEATMAP_DIR / f"{slide}.h5"
    wsi_path   = WSI_DIR / f"{slide}.tiff"

    if not heatmap_h5.exists():
        print(f"  [warn] heatmap not found: {heatmap_h5.name}")
        return
    if not wsi_path.exists():
        print(f"  [warn] WSI not found: {wsi_path.name}")
        return

    with h5py.File(heatmap_h5) as f:
        coords       = f["coords"][:]
        heatmap_norm = f["heatmap"][:]

    k = min(n_patches, len(coords))
    images, sel_coords, sel_scores, top_idx = extract_top_patches(
        wsi_path, coords, heatmap_norm, k, PRISM2_PATCH_SIZE
    )
    print(f"  {slide}: {len(coords)} tiles, scoring top-{k}", flush=True)

    sim = compute_scores(model, transform, images, classifier, device)
    # sim: (k, n_labels)

    results = []
    for rank, (coord, attn, patch_sim) in enumerate(zip(sel_coords, sel_scores, sim)):
        ranked = sorted(zip(LABEL_NAMES, patch_sim.tolist()), key=lambda x: -x[1])
        results.append({
            "patch_rank":  rank,
            "coord_x":     int(coord[0]),
            "coord_y":     int(coord[1]),
            "attn_score":  float(attn),
            "labels":      [{"label": l, "score": round(s, 4)} for l, s in ranked],
        })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"  saved → {out_json.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slide",         help="Slide stem (e.g. 10407210HE1)")
    p.add_argument("--all",           action="store_true")
    p.add_argument("--n_patches",     type=int, default=N_TOP_PATCHES)
    p.add_argument("--gpu",           type=int, default=0)
    p.add_argument("--skip_existing", action="store_true", default=False)
    args = p.parse_args()

    if not args.slide and not args.all:
        p.error("Specify --slide STEM or --all")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Loading CONCH v1 ...")
    model, transform, tokenizer = load_conch_v1(device)
    print(f"Model ready on {device}.")

    print("Building text classifier (22 templates × classnames per label) ...")
    classifier = build_text_classifier(model, tokenizer, device)
    print(f"  Classifier shape: {classifier.shape}  "
          f"[{len(LABEL_NAMES)} labels × {classifier.shape[1]}-dim]\n")

    if args.slide:
        slides = [args.slide]
    else:
        slides = [p.stem for p in sorted(HEATMAP_DIR.glob("*.h5"))]
        print(f"Found {len(slides)} slides.\n")

    for i, slide in enumerate(slides, 1):
        print(f"[{i}/{len(slides)}]", end=" ")
        try:
            process_slide(slide, model, transform, classifier, device,
                          args.n_patches, args.skip_existing)
        except Exception as e:
            print(f"  [ERROR] {slide}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
