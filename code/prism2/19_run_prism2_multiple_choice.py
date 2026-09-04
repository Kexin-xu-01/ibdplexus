"""
Score 11 UAMP histological concepts with a 3-way multiple-choice prompt:

    What observation can you make?
    Options:
    A. <term>
    B. normal
    C. not sure

Logits for tokens A / B / C are extracted and softmax'd to give P(A), P(B), P(C).
P(A) = prevalence score; P(C) = direct model uncertainty.

Image embeddings are computed once per batch and reused across all 11 terms,
so the expensive resampler/projector forward pass runs only once per slide.

Outputs:
  <OUT_DIR>/prism2_mc_scores.csv

  Columns per concept (e.g. inflammation_involvement_*):
    _p_term       P(A) — model says the finding is present
    _p_normal     P(B) — model says normal
    _p_not_sure   P(C) — model says it cannot tell (uncertainty)

Usage:
  python 19_run_prism2_multiple_choice.py
  python 19_run_prism2_multiple_choice.py --batch_size 4 --gpu 0
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
                  "prism2_manual_knn/uncertainty_estimation/multiple_choice")

UAMP_TERMS = [
    "Inflammation involvement",
    "Crypt architectural distortion",
    "Neutrophil granulocytic infiltration",
    "Crypt abscesses",
    "Lymphoid aggregates",
    "Histiocytic granulomas",
    "Mucin depletion",
    "Pyloric gland metaplasia",
    "Paneth cell metaplasia",
    "Neuronal hyperplasia",
    "Muscular hypertrophy",
]

PROMPT_TEMPLATE = (
    "What observation can you make?\n"
    "Options:\n"
    "A. {term}\n"
    "B. normal\n"
    "C. not sure\n"
)


def col_name(term: str) -> str:
    return term.lower().replace(" ", "_")


SCORE_COLS = [col_name(t) for t in UAMP_TERMS]
SUFFIXES   = ["p_term", "p_normal", "p_not_sure"]
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


def get_abc_ids(tokenizer):
    """Token IDs for the literal characters A, B, C (last subword)."""
    a_id = tokenizer("A", add_special_tokens=False).input_ids[-1]
    b_id = tokenizer("B", add_special_tokens=False).input_ids[-1]
    c_id = tokenizer("C", add_special_tokens=False).input_ids[-1]
    return a_id, b_id, c_id


def score_batch(model, processor_cls, tile_emb, attn_mask, abc_ids, device):
    """
    Run all 11 terms against one batch.

    Returns dict: term -> (B,3) tensor of [P(A), P(B), P(C)].
    Image encoding is done once and reused.
    """
    B = tile_emb.shape[0]
    a_id, b_id, c_id = abc_ids

    # Image encoding — done once per batch (expensive)
    resampler_out   = model._encode_images(tile_emb, attn_mask)
    image_embeddings = model._project(resampler_out["latents"])

    results = {}
    prism_proc = processor_cls(
        tokenizer=model.tokenizer,
        num_img_tokens=model.config.num_img_tokens,
    )

    for term in UAMP_TERMS:
        prompt = PROMPT_TEMPLATE.format(term=term)
        messages = [
            [{"role": "user", "content": "<|image_1|>" + prompt}]
            for _ in range(B)
        ]
        tok = prism_proc.tokenize(messages, generation=True)
        input_ids = tok["input_ids"].to(device)

        out = model.text_decoder(
            input_ids, image_embeddings, output_hidden_states=False
        )
        last_logits = out["logits"][:, -1, :].float()  # (B, vocab)

        # Extract A / B / C logits and softmax over just those three
        triple = torch.stack(
            [last_logits[:, a_id],
             last_logits[:, b_id],
             last_logits[:, c_id]], dim=-1
        )  # (B, 3)
        probs = F.softmax(triple, dim=-1)  # (B, 3)
        results[term] = probs

    return results


def score_slides(model, processor, device, h5_paths, batch_size, skip_errors):
    """Return dict: slide_stem -> flat row dict."""
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

    abc_ids = get_abc_ids(model.tokenizer)
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

            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                term_probs = score_batch(
                    model, Prism2Processor,
                    batch["tile_embeddings"], batch["attention_mask"],
                    abc_ids, device,
                )

            for term, probs in term_probs.items():
                col = col_name(term)
                for stem, (p_a, p_b, p_c) in zip(stems, probs.cpu().tolist()):
                    row = table.setdefault(stem, {"slide": stem})
                    row[f"{col}_p_term"]     = round(p_a, 6)
                    row[f"{col}_p_normal"]   = round(p_b, 6)
                    row[f"{col}_p_not_sure"] = round(p_c, 6)

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
    print(f"Model loaded on {device}. Scoring {len(UAMP_TERMS)} concepts × {len(all_h5)} slides.\n")

    table = score_slides(model, processor, device, all_h5,
                         args.batch_size, args.skip_errors)

    out_path = out_dir / "prism2_mc_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table.values())
    print(f"\nSaved {len(table)} slides → {out_path}")

    # Summary
    rows = list(table.values())
    print("\nMean scores per concept:")
    header = f"{'concept':<45} {'P(term)':>8} {'P(normal)':>10} {'P(not sure)':>12}"
    print(header)
    print("-" * len(header))
    for col, term in zip(SCORE_COLS, UAMP_TERMS):
        p_term     = np.mean([r[f"{col}_p_term"]     for r in rows if f"{col}_p_term"     in r])
        p_normal   = np.mean([r[f"{col}_p_normal"]   for r in rows if f"{col}_p_normal"   in r])
        p_not_sure = np.mean([r[f"{col}_p_not_sure"] for r in rows if f"{col}_p_not_sure" in r])
        print(f"{term:<45} {p_term:>8.4f} {p_normal:>10.4f} {p_not_sure:>12.4f}")


if __name__ == "__main__":
    main()
