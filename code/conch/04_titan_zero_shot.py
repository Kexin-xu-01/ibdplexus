"""
Titan slide-level zero-shot classification for IBD UAMP terms.

Follows the method in the Titan paper (CLIP-style zero-shot, §Methods):
  1. Build a (D, C) classifier matrix: one column per UAMP class.
     Each column = L2-normalised mean of all (classname × template)
     text embeddings, using titan.zero_shot_classifier().
  2. Project and L2-normalise the slide embedding (via titan.zero_shot()).
  3. Score = cosine similarity between slide and each class embedding.
     Raw cosine similarities are output (not softmax) so that co-occurring
     findings can each score highly independently.

Classnames: exact UAMP term labels (underscores → spaces), matching the
CONCH zero-shot script.

Output per slide (titan_zero_shot/per_slide/<slide>.json):
    {"slide": "...", "inflammation_involvement": 0.42, ...}

Aggregated: titan_zero_shot_scores.csv / .json

Usage:
    python 04_titan_zero_shot.py
    python 04_titan_zero_shot.py --slide 10407210HE1
    python 04_titan_zero_shot.py --gpu 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

TITAN_DIR = Path("/home/jovyan/shared-data/users/kexin/models/histology/titan")
FEAT_DIR  = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/"
                 "20x_512px_0px_overlap/slide_features_titan")
OUT_DIR   = Path("/home/jovyan/kgbk271-ibd-volume/results/titan_zero_shot")

UAMP_TERMS = [
    "inflammation_involvement",
    "crypt_architectural_distortion",
    "neutrophil_granulocytic_infiltration",
    "crypt_abscesses",
    "lymphoid_aggregates",
    "histiocytic_granulomas",
    "mucin_depletion",
    "pyloric_gland_metaplasia",
    "paneth_cell_metaplasia",
    "neuronal_hyperplasia",
    "muscular_hypertrophy",
    "normal_mucosa",
]

# One classname per UAMP term — exact label with underscores replaced by spaces.
# Passed as a list-of-lists to zero_shot_classifier() (one inner list per class).
# Multiple synonyms can be added here to extend the ensemble.
CLASSNAMES = [[term.replace("_", " ")] for term in UAMP_TERMS]

# Slide-level templates appropriate for Titan (WSI encoder).
TEMPLATES = [
    "A whole-slide image of a colonic biopsy showing CLASSNAME.",
    "Colorectal biopsy histology demonstrating CLASSNAME.",
    "Hematoxylin and eosin stained colonic biopsy with CLASSNAME.",
    "A whole-slide image showing CLASSNAME.",
    "CLASSNAME.",
    "CLASSNAME, H&E.",
]


# ---------------------------------------------------------------------------
# Titan loader (bypasses AutoModel to avoid transformers 5.x API mismatch)
# ---------------------------------------------------------------------------

def load_titan(device: torch.device):
    td = str(TITAN_DIR)
    pkg = types.ModuleType("titan_local")
    pkg.__path__ = [td]
    pkg.__package__ = "titan_local"
    sys.modules["titan_local"] = pkg

    def load_as(name):
        spec = importlib.util.spec_from_file_location(
            f"titan_local.{name}", f"{td}/{name}.py", submodule_search_locations=[]
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "titan_local"
        sys.modules[f"titan_local.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    for m in ["conch_tokenizer", "conch_v1_5", "configuration_titan",
              "text_transformer", "vision_transformer", "modeling_titan"]:
        load_as(m)

    from safetensors.torch import load_file
    with open(TITAN_DIR / "config.json") as f:
        cfg = json.load(f)

    cfg_mod = sys.modules["titan_local.configuration_titan"]
    config = cfg_mod.TitanConfig(**{k: v for k, v in cfg.items()
                                    if k not in ("auto_map", "architectures")})
    model_mod = sys.modules["titan_local.modeling_titan"]
    titan = model_mod.Titan(config)
    sd = load_file(str(TITAN_DIR / "model.safetensors"))
    missing, unexpected = titan.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [warn] missing={len(missing)} unexpected={len(unexpected)}")
    titan.to(device).eval()
    return titan


# ---------------------------------------------------------------------------
# Build C-class classifier (paper §Methods, CLIP zero-shot)
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_classifier(titan, device: torch.device) -> torch.Tensor:
    """
    Build the (768, 12) classifier matrix following the Titan paper.

    Uses titan.zero_shot_classifier(classnames, templates):
      - classnames : list of C lists, one per UAMP class
      - templates  : list of T template strings with CLASSNAME placeholder
      Each class embedding = L2-normalised mean over all (classname × template)
      text embeddings.

    Returns float32 tensor on device, shape (768, 12).
    """
    clf = titan.zero_shot_classifier(
        classnames=CLASSNAMES,
        templates=TEMPLATES,
        device=device,
    )
    return clf  # (768, C)


# ---------------------------------------------------------------------------
# Per-slide scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_slide(titan, slide_emb: torch.Tensor,
                classifier: torch.Tensor, device: torch.device) -> dict:
    """
    Return {term: cosine_similarity} for one slide.

    Applies titan.vision_encoder.proj + L2-normalisation to the raw slide
    embedding (same projection used during contrastive pre-training), then
    computes dot products with the classifier columns — i.e. cosine
    similarities since both sides are L2-normalised.

    Raw cosine similarities are returned (not softmax) so that co-occurring
    findings are scored independently.
    """
    emb = slide_emb.float().to(device)           # (768,)
    emb = emb @ titan.vision_encoder.proj        # project
    emb = F.normalize(emb, dim=-1).unsqueeze(0)  # (1, 768)
    logits = emb @ classifier                    # (1, C)  — cosine similarities
    scores = logits.squeeze(0).cpu().float().tolist()
    return {term: round(float(s), 6) for term, s in zip(UAMP_TERMS, scores)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slide",         help="Single slide stem (e.g. 10407210HE1)")
    p.add_argument("--gpu",           type=int, default=0)
    p.add_argument("--skip_existing", action="store_true", default=False)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Loading Titan from {TITAN_DIR} ...")
    titan = load_titan(device)
    print(f"Model ready on {device}.\n")

    print(f"Building zero-shot classifier ({len(UAMP_TERMS)} classes × "
          f"{len(TEMPLATES)} templates × {len(CLASSNAMES[0])} classname(s)) ...")
    classifier = build_classifier(titan, device)
    print(f"  Classifier shape: {classifier.shape}\n")

    if args.slide:
        slides = [args.slide]
    else:
        slides = [p.stem for p in sorted(FEAT_DIR.glob("*.h5"))]
        print(f"Found {len(slides)} slides.\n")

    PER_SLIDE_DIR = OUT_DIR / "per_slide"
    PER_SLIDE_DIR.mkdir(parents=True, exist_ok=True)

    ok, skipped, failed = 0, 0, 0
    for i, slide in enumerate(slides, 1):
        out_json = PER_SLIDE_DIR / f"{slide}.json"
        if args.skip_existing and out_json.exists():
            skipped += 1
            if skipped % 500 == 0:
                print(f"[{i}/{len(slides)}] ... (skipped {skipped} so far)")
            continue

        feat_h5 = FEAT_DIR / f"{slide}.h5"
        if not feat_h5.exists():
            print(f"[{i}/{len(slides)}] {slide} [missing feat]")
            failed += 1
            continue

        try:
            with h5py.File(feat_h5) as f:
                emb = torch.from_numpy(f["features"][:].astype(np.float32))

            scores = score_slide(titan, emb, classifier, device)
            row = {"slide": slide, **scores}
            out_json.write_text(json.dumps(row))
            ok += 1
            top = max(scores, key=scores.get)
            if ok % 100 == 0 or ok <= 5:
                print(f"[{i}/{len(slides)}] {slide}  top={top}({scores[top]:.3f})  done={ok}")

        except Exception as e:
            print(f"[{i}/{len(slides)}] {slide} [ERROR] {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\nScoring done: ok={ok} skipped={skipped} failed={failed}")
    print("Aggregating results ...")
    aggregate(PER_SLIDE_DIR, OUT_DIR)
    print(f"Done. Results → {OUT_DIR}")


def aggregate(per_slide_dir: Path, out_dir: Path):
    import csv
    results = []
    for f in sorted(per_slide_dir.glob("*.json")):
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            pass
    if not results:
        print("No results to aggregate.")
        return
    with open(out_dir / "titan_zero_shot_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    fieldnames = ["slide"] + UAMP_TERMS
    with open(out_dir / "titan_zero_shot_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  Aggregated {len(results)} slides → {out_dir}")


if __name__ == "__main__":
    main()
