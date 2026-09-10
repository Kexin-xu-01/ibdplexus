"""
Test linguistic robustness of PRISM2 yes/no scores.

For each of the 11 UAMP concepts, 4 differently-phrased yes/no questions are scored.
P(Yes) is recorded for each phrasing. The SD across phrasings measures how sensitive
the model is to wording — high SD means the score depends heavily on how the question
is framed, which is a form of instability independent of image content.

Image embeddings are encoded once per batch and reused across all 44 questions.

Outputs:
  <OUT_DIR>/prism2_robustness_scores.csv

  Columns per concept (e.g. inflammation_involvement_*):
    _q1 … _q4    P(Yes) for each phrasing
    _mean         mean P(Yes) across phrasings
    _sd           SD across phrasings  ← robustness metric (lower = more robust)

Usage:
  python 22_run_prism2_question_robustness.py
  python 22_run_prism2_question_robustness.py --batch_size 4 --gpu 0
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

VIRCHOW2_DIM    = 2560
CLASS_TOKEN_DIM = 1280

FEAT_DIR   = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/"
                  "tissue_threshold_15_filtered_no_darkspot_manual_knn/"
                  "20x_224px_0px_overlap/features_virchow2")
MODEL_PATH = "/home/jovyan/shared-data/users/kexin/models/VLM/prism2"
OUT_DIR    = Path("/home/jovyan/kgbk271-ibd-volume/results/"
                  "prism2_manual_knn/uncertainty_estimation/question_robustness")

# 4 reframings per concept — same underlying meaning, different wording
QUESTION_VARIANTS = {
    "Inflammation involvement": [
        "Is inflammation involvement present?",
        "Is the tissue inflamed?",
        "Is there mucosal inflammation?",
        "Is inflammation present?",
    ],
    "Crypt architectural distortion": [
        "Is crypt architectural distortion present?",
        "Is crypt distorted?",
        "Is crypt deformed?",
        "Is crypt deformation present?",
    ],
    "Neutrophil granulocytic infiltration": [
        "Is neutrophil granulocytic infiltration present?",
        "Is there neutrophil infiltration?",
        "Are neutrophils infiltrating the tissue?",
        "Is neutrophilic inflammation present?",
    ],
    "Crypt abscesses": [
        "Is crypt abscesses present?",
        "Are there crypt abscesses?",
        "Is cryptitis with abscess formation present?",
        "Are crypt lumens filled with neutrophils?",
    ],
    "Lymphoid aggregates": [
        "Is lymphoid aggregates present?",
        "Are there lymphoid aggregates?",
        "Is lymphocytic aggregation present?",
        "Are lymphoid follicles present?",
    ],
    "Histiocytic granulomas": [
        "Is histiocytic granulomas present?",
        "Are there granulomas?",
        "Is granulomatous inflammation present?",
        "Are epithelioid granulomas present?",
    ],
    "Mucin depletion": [
        "Is mucin depletion present?",
        "Is there mucin depletion?",
        "Is goblet cell depletion present?",
        "Is mucin loss present?",
    ],
    "Pyloric gland metaplasia": [
        "Is pyloric gland metaplasia present?",
        "Is there pyloric metaplasia?",
        "Is pseudopyloric metaplasia present?",
        "Is Brunner gland-like metaplasia present?",
    ],
    "Paneth cell metaplasia": [
        "Is Paneth cell metaplasia present?",
        "Are Paneth cells present?",
        "Is Paneth cell differentiation present?",
        "Is there ectopic Paneth cell metaplasia?",
    ],
    "Neuronal hyperplasia": [
        "Is neuronal hyperplasia present?",
        "Is there neural hyperplasia?",
        "Is nerve fiber hyperplasia present?",
        "Is neural proliferation present?",
    ],
    "Muscular hypertrophy": [
        "Is muscular hypertrophy present?",
        "Is the muscularis propria thickened?",
        "Is muscle hypertrophy present?",
        "Is muscular thickening present?",
    ],
}

N_VARIANTS = 4  # same for every concept


def col_name(term: str) -> str:
    return term.lower().replace(" ", "_")


SCORE_COLS = [col_name(t) for t in QUESTION_VARIANTS]
SUFFIXES   = [f"q{i}" for i in range(1, N_VARIANTS + 1)] + ["mean", "sd"]
FIELDNAMES = ["slide"] + [f"{c}_{s}" for c in SCORE_COLS for s in SUFFIXES]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--feat_dir",    type=str, default=str(FEAT_DIR))
    p.add_argument("--out_dir",     type=str, default=str(OUT_DIR))
    p.add_argument("--skip_errors", action="store_true", default=True)
    return p.parse_args()


def load_h5(path: Path) -> torch.Tensor:
    with h5py.File(path, "r") as f:
        feats = torch.from_numpy(f["features"][:])
    if feats.shape[1] == VIRCHOW2_DIM:
        feats = feats[:, :CLASS_TOKEN_DIM]
    elif feats.shape[1] != CLASS_TOKEN_DIM:
        raise ValueError(f"Unexpected feature dim {feats.shape[1]} in {path.name}")
    return feats


def score_slides(model, processor, device, h5_paths, batch_size, skip_errors):
    try:
        from transformers_modules.prism2.processing_prism2 import Prism2Processor
    except ImportError:
        try:
            sys.path.insert(0, MODEL_PATH)
            from processing_prism2 import Prism2Processor  # type: ignore
        except ImportError:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            Prism2Processor = get_class_from_dynamic_module(
                "prism2.processing_prism2.Prism2Processor", MODEL_PATH
            )

    yes_id = model.tokenizer("Yes", add_special_tokens=False).input_ids[-1]
    no_id  = model.tokenizer("No",  add_special_tokens=False).input_ids[-1]

    table = {}
    n = len(h5_paths)
    done = 0

    for start in range(0, n, batch_size):
        batch_paths = h5_paths[start : start + batch_size]
        slides, stems = [], []
        for path in batch_paths:
            try:
                slides.append(load_h5(path))
                stems.append(path.stem)
            except Exception as e:
                if skip_errors:
                    print(f"  [SKIP load] {path.name}: {e}", file=sys.stderr)
                else:
                    raise
        if not slides:
            continue

        try:
            batch = processor(tile_embeddings=slides).to(device)
            model_dtype = next(model.parameters()).dtype
            batch["tile_embeddings"] = batch["tile_embeddings"].to(model_dtype)
            B = batch["tile_embeddings"].shape[0]

            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                # Image encoding — once per batch
                resampler_out    = model._encode_images(
                    batch["tile_embeddings"], batch["attention_mask"]
                )
                image_embeddings = model._project(resampler_out["latents"])

                prism_proc = Prism2Processor(
                    tokenizer=model.tokenizer,
                    num_img_tokens=model.config.num_img_tokens,
                )

                # scores_by_concept[term][variant_idx] = (B,) tensor of P(Yes)
                scores_by_concept = {term: [] for term in QUESTION_VARIANTS}

                for term, questions in QUESTION_VARIANTS.items():
                    for question in questions:
                        messages = [
                            [{"role": "user", "content": "<|image_1|>" + question}]
                            for _ in range(B)
                        ]
                        tok = prism_proc.tokenize(messages, generation=True)
                        input_ids = tok["input_ids"].to(device)

                        out = model.text_decoder(
                            input_ids, image_embeddings, output_hidden_states=False
                        )
                        last_logits = out["logits"][:, -1, :].float()  # (B, vocab)
                        pair = torch.stack(
                            [last_logits[:, yes_id], last_logits[:, no_id]], dim=-1
                        )
                        p_yes = pair.softmax(-1)[:, 0]  # (B,)
                        scores_by_concept[term].append(p_yes.cpu())

            # Assemble rows
            for term, variant_scores in scores_by_concept.items():
                col = col_name(term)
                stacked = torch.stack(variant_scores, dim=1)  # (B, 4)
                means   = stacked.mean(dim=1)
                sds     = stacked.std(dim=1, unbiased=True)
                for stem, qs, mean_v, sd_v in zip(
                    stems, stacked.tolist(), means.tolist(), sds.tolist()
                ):
                    row = table.setdefault(stem, {"slide": stem})
                    for i, q_score in enumerate(qs, 1):
                        row[f"{col}_q{i}"] = round(q_score, 6)
                    row[f"{col}_mean"] = round(mean_v, 6)
                    row[f"{col}_sd"]   = round(sd_v,   6)

            done += len(stems)
            if done % 100 == 0 or done == n:
                print(f"    {done}/{n}")

        except Exception as e:
            if skip_errors:
                print(f"  [SKIP batch] {[p.name for p in batch_paths]}: {e}",
                      file=sys.stderr)
            else:
                raise

    return table


def main():
    args     = parse_args()
    feat_dir = Path(args.feat_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_h5 = sorted(feat_dir.glob("*.h5"))
    print(f"Found {len(all_h5)} slides in {feat_dir}")
    print(f"Questions per concept: {N_VARIANTS}  |  Total questions per slide: "
          f"{len(QUESTION_VARIANTS) * N_VARIANTS}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading PRISM2 from {MODEL_PATH} ...")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    print(f"Model loaded on {device}. Scoring {len(all_h5)} slides.\n")

    # Print question plan
    for term, qs in QUESTION_VARIANTS.items():
        print(f"  {term}")
        for i, q in enumerate(qs, 1):
            print(f"    q{i}: {q}")
    print()

    table = score_slides(model, processor, device, all_h5,
                         args.batch_size, args.skip_errors)

    out_path = out_dir / "prism2_robustness_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table.values())
    print(f"\nSaved {len(table)} slides → {out_path}")

    # Summary: mean SD per concept (lower = more robust to wording)
    rows = list(table.values())
    print("\nMean SD across phrasings per concept (lower = more robust):")
    header = f"{'concept':<45} {'mean_P(Yes)':>11} {'mean_SD':>8}"
    print(header)
    print("-" * len(header))
    concept_sds = {}
    for col, term in zip(SCORE_COLS, QUESTION_VARIANTS):
        mean_p = np.mean([r[f"{col}_mean"] for r in rows if f"{col}_mean" in r])
        mean_sd = np.mean([r[f"{col}_sd"]  for r in rows if f"{col}_sd"   in r])
        concept_sds[term] = mean_sd
        print(f"{term:<45} {mean_p:>11.4f} {mean_sd:>8.4f}")

    most_robust   = min(concept_sds, key=concept_sds.get)
    least_robust  = max(concept_sds, key=concept_sds.get)
    print(f"\nMost robust to wording:  {most_robust} (SD={concept_sds[most_robust]:.4f})")
    print(f"Least robust to wording: {least_robust} (SD={concept_sds[least_robust]:.4f})")


if __name__ == "__main__":
    main()
