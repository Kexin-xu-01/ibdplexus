"""
PRISM2 free-form description of high-attention patches.

Uses PRISM2's Phi-3-mini text decoder to generate natural-language descriptions
of the histological content visible in the top-K high-attention tile regions
identified by the PRISM2 attention heatmap.

How it works
------------
The PRISM2 attention heatmap ranks all Virchow2 tiles in a slide by importance.
Here we subset the Virchow2 feature matrix to only the top-K tiles, present
that mini-slide to PRISM2, and ask it to describe what it sees in those
specific regions. Because PRISM2 is conditioned on the full tile set, using
a small focused subset shifts its attention to just those high-signal areas.

PRISM2 `get_response()` API:
    model.get_response(tile_embeddings, attention_mask, prompt, **gen_kwargs)
    → list[str]  (one response per slide in the batch)

Note on caption generation capability
--------------------------------------
| Model     | Free-form generation | Zero-shot scoring |
|-----------|---------------------|-------------------|
| CONCH v1  | NO (decoder weights absent locally) | YES (CLIP) |
| CONCH v1.5| NO (vision-only)    | NO                |
| Titan     | NO (contrastive)    | YES (CLIP, slides)|
| PRISM2    | YES (Phi-3-mini)    | YES               |

For CONCH-based zero-shot labelling see 02_conch_zero_shot_labels.py.

Output per slide (prism2_conch_patch_descriptions/<slide>.txt):
    Free-form PRISM2 description of the top-K high-attention tiles.

Also writes an aggregate JSONL (results root / prism2_patch_descriptions.jsonl)
and a wide CSV (results root / prism2_patch_descriptions.csv).

Usage:
    python 03_prism2_patch_describe.py --slide 10407210HE1
    python 03_prism2_patch_describe.py --all
    python 03_prism2_patch_describe.py --all --n_patches 16 --prompt "Describe the key pathological findings"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

PRISM2_MODEL_PATH = "/home/jovyan/shared-data/users/kexin/models/VLM/prism2"
FEAT_DIR    = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/features_virchow2")
HEATMAP_DIR = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/prism2_attention_heatmap")
OUT_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/prism2_patch_descriptions")
RESULTS_ROOT = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2_manual_knn")

N_TOP_PATCHES   = 8
DEFAULT_PROMPT  = (
    "These are the most diagnostically important patches from this biopsy, "
    "selected by attention weighting. Describe the key pathological findings "
    "visible in these regions."
)
VIRCHOW2_DIM    = 2560
CLASS_TOKEN_DIM = 1280


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_top_k_features(
    feat_h5: Path,
    heatmap_h5: Path,
    k: int,
) -> tuple[torch.Tensor, list[int], np.ndarray]:
    """Return (features (1, k, 1280), top_indices, attn_scores (k,))."""
    with h5py.File(feat_h5) as f:
        feats = f["features"][:]   # (N, 2560)
    if feats.shape[1] == VIRCHOW2_DIM:
        feats = feats[:, :CLASS_TOKEN_DIM]

    with h5py.File(heatmap_h5) as f:
        heatmap = f["heatmap"][:]  # (N,) normalised [0,1]

    top_idx = np.argsort(heatmap)[::-1][:k].tolist()
    selected = feats[top_idx]                            # (k, 1280)
    scores   = heatmap[top_idx].astype(np.float32)
    return torch.from_numpy(selected).unsqueeze(0), top_idx, scores


def rebuild_csv(jsonl_path: Path, csv_path: Path):
    data: dict = {}
    prompts: list = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            slide, prompt, text = r["slide"], r["prompt"], r["description"]
            if prompt not in prompts:
                prompts.append(prompt)
            data.setdefault(slide, {})[prompt] = text
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["slide"] + prompts, extrasaction="ignore")
        writer.writeheader()
        for slide in sorted(data):
            writer.writerow({"slide": slide, **data[slide]})


def append_result(slide: str, prompt: str, description: str, n_patches: int):
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    jsonl = RESULTS_ROOT / "prism2_patch_descriptions.jsonl"
    with open(jsonl, "a") as f:
        f.write(json.dumps({
            "slide": slide,
            "prompt": prompt,
            "n_patches": n_patches,
            "description": description,
        }) + "\n")
    rebuild_csv(jsonl, RESULTS_ROOT / "prism2_patch_descriptions.csv")


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------

def process_slide(
    slide: str,
    model,
    processor,
    device: torch.device,
    prompt: str,
    n_patches: int,
    max_new_tokens: int,
    skip_existing: bool,
):
    out_txt = OUT_DIR / f"{slide}.txt"
    if skip_existing and out_txt.exists():
        print(f"  [skip] {slide}")
        return

    feat_h5    = FEAT_DIR    / f"{slide}.h5"
    heatmap_h5 = HEATMAP_DIR / f"{slide}.h5"

    for path in (feat_h5, heatmap_h5):
        if not path.exists():
            print(f"  [warn] not found: {path.name}")
            return

    k = n_patches
    tile_emb, top_idx, scores = load_top_k_features(feat_h5, heatmap_h5, k)
    k_actual = tile_emb.shape[1]
    print(f"  {slide}: top-{k_actual} patches  attn=[{scores.min():.2f},{scores.max():.2f}]",
          flush=True)

    batch = processor(tile_embeddings=[tile_emb.squeeze(0)]).to(device)

    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
        responses = model.get_response(
            **batch,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
    description = responses[0].strip()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(description)
    append_result(slide, prompt, description, k_actual)
    print(f"  → {out_txt.name}")
    print(f"     {description[:120]}{'...' if len(description) > 120 else ''}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slide",          help="Slide stem (e.g. 10407210HE1)")
    p.add_argument("--all",            action="store_true")
    p.add_argument("--prompt",         default=DEFAULT_PROMPT)
    p.add_argument("--n_patches",      type=int, default=N_TOP_PATCHES,
                   help="Number of top-attention patches to describe")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--gpu",            type=int, default=0)
    p.add_argument("--skip_existing",  action="store_true", default=True)
    args = p.parse_args()

    if not args.slide and not args.all:
        p.error("Specify --slide STEM or --all")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Loading PRISM2 from {PRISM2_MODEL_PATH} ...")
    model = AutoModel.from_pretrained(
        PRISM2_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        PRISM2_MODEL_PATH, trust_remote_code=True, local_files_only=True,
    )
    print(f"Model ready on {device}.\n")

    if args.slide:
        slides = [args.slide]
    else:
        slides = [p.stem for p in sorted(HEATMAP_DIR.glob("*.h5"))]
        print(f"Found {len(slides)} slides.\n")

    for i, slide in enumerate(slides, 1):
        print(f"[{i}/{len(slides)}]", end=" ")
        try:
            process_slide(
                slide, model, processor, device,
                prompt=args.prompt,
                n_patches=args.n_patches,
                max_new_tokens=args.max_new_tokens,
                skip_existing=args.skip_existing,
            )
        except Exception as e:
            print(f"  [ERROR] {slide}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
