"""
Score all slides with a single 10-way multiple-choice prompt listing each
histological concept (excluding muscular hypertrophy) as a lettered option:

    What can you observe?
    Options:
    A. inflammation involvement
    B. crypt architectural distortion
    C. neutrophil granulocytic infiltration
    D. crypt abscesses
    E. lymphoid aggregates
    F. histiocytic granulomas
    G. mucin depletion
    H. pyloric gland metaplasia
    I. paneth cell metaplasia
    J. neuronal hyperplasia

Logits for tokens A-J are extracted and softmax'd to give P(A)..P(J).
Each probability represents how strongly the model identifies that concept
as the dominant observation in the slide.

One forward pass per slide (no repetition per concept).

Outputs:
  <OUT_DIR>/prism2_multi_feature_scores.csv

  Columns: slide, p_inflammation_involvement, p_crypt_architectural_distortion, ...

Usage:
  python 20_run_prism2_multi_feature.py
  python 20_run_prism2_multi_feature.py --batch_size 4 --gpu 0
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
                  "prism2_manual_knn/uncertainty_estimation/multi_feature")

# 10 options — muscular hypertrophy excluded
OPTIONS = [
    ("A", "inflammation involvement"),
    ("B", "crypt architectural distortion"),
    ("C", "neutrophil granulocytic infiltration"),
    ("D", "crypt abscesses"),
    ("E", "lymphoid aggregates"),
    ("F", "histiocytic granulomas"),
    ("G", "mucin depletion"),
    ("H", "pyloric gland metaplasia"),
    ("I", "paneth cell metaplasia"),
    ("J", "neuronal hyperplasia"),
]

PROMPT = (
    "What can you observe?\n"
    "Options:\n"
    + "".join(f"{letter}. {label}\n" for letter, label in OPTIONS)
)

SCORE_COLS = [f"p_{label.replace(' ', '_')}" for _, label in OPTIONS]
FIELDNAMES = ["slide"] + SCORE_COLS


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


def get_option_ids(tokenizer):
    return [
        tokenizer(letter, add_special_tokens=False).input_ids[-1]
        for letter, _ in OPTIONS
    ]


def score_slides(model, processor, device, h5_paths, batch_size, skip_errors):
    """Return dict: slide_stem -> row dict of P(A)..P(J)."""
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

    option_ids = get_option_ids(model.tokenizer)
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

                # Tokenize prompt (same for every slide in batch)
                prism_proc = Prism2Processor(
                    tokenizer=model.tokenizer,
                    num_img_tokens=model.config.num_img_tokens,
                )
                messages = [
                    [{"role": "user", "content": "<|image_1|>" + PROMPT}]
                    for _ in range(B)
                ]
                tok = prism_proc.tokenize(messages, generation=True)
                input_ids = tok["input_ids"].to(device)

                out = model.text_decoder(
                    input_ids, image_embeddings, output_hidden_states=False
                )
                last_logits = out["logits"][:, -1, :].float()  # (B, vocab)

                # Extract logits for A-J and softmax over those 10
                option_logits = torch.stack(
                    [last_logits[:, tid] for tid in option_ids], dim=-1
                )  # (B, 10)
                probs = F.softmax(option_logits, dim=-1)  # (B, 10)

            for stem, row_probs in zip(stems, probs.cpu().tolist()):
                row = {"slide": stem}
                for col, prob in zip(SCORE_COLS, row_probs):
                    row[col] = round(prob, 6)
                table[stem] = row

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
    args = parse_args()
    feat_dir = Path(args.feat_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_h5 = sorted(feat_dir.glob("*.h5"))
    print(f"Found {len(all_h5)} slides in {feat_dir}")
    print(f"\nPrompt:\n{PROMPT}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Loading PRISM2 from {MODEL_PATH} ...")
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

    table = score_slides(model, processor, device, all_h5,
                         args.batch_size, args.skip_errors)

    out_path = out_dir / "prism2_multi_feature_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table.values())
    print(f"\nSaved {len(table)} slides → {out_path}")

    rows = list(table.values())
    print("\nMean P per option:")
    header = f"{'option':<45} {'mean P':>8}"
    print(header)
    print("-" * len(header))
    for (letter, label), col in zip(OPTIONS, SCORE_COLS):
        mean_p = np.mean([r[col] for r in rows if col in r])
        print(f"{letter}. {label:<43} {mean_p:>8.4f}")


if __name__ == "__main__":
    main()
