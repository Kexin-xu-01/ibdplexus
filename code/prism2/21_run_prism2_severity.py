"""
Score severity of each UAMP histological concept with a 5-way prompt:

    How much <term> do you see?
    Options:
    A. Mild
    B. Moderate
    C. Severe
    D. Not sure
    E. None

Logits for A-E are softmax'd to give P(mild), P(moderate), P(severe), P(not_sure), P(none).
A weighted severity score is also computed: mild×1 + moderate×2 + severe×3 (0–3 scale,
higher = more severe; not_sure and none contribute 0).

Image embeddings are computed once per batch and reused across all 11 terms.

Outputs:
  <OUT_DIR>/prism2_severity_scores.csv

  Columns per concept (e.g. inflammation_involvement_*):
    _p_mild       P(A)
    _p_moderate   P(B)
    _p_severe     P(C)
    _p_none       P(D)
    _severity     weighted score = p_mild×1 + p_moderate×2 + p_severe×3  (range 0–3)

Usage:
  python 21_run_prism2_severity.py
  python 21_run_prism2_severity.py --batch_size 4 --gpu 0
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
                  "prism2_manual_knn/uncertainty_estimation/severity")

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
    "How much {term} do you see?\n"
    "Options:\n"
    "A. Mild\n"
    "B. Moderate\n"
    "C. Severe\n"
    "D. Not sure\n"
    "E. None\n"
)

SEVERITY_WEIGHTS = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0])  # A=mild, B=mod, C=severe, D=not_sure, E=none


def col_name(term: str) -> str:
    return term.lower().replace(" ", "_")


SCORE_COLS = [col_name(t) for t in UAMP_TERMS]
SUFFIXES   = ["p_mild", "p_moderate", "p_severe", "p_not_sure", "p_none", "severity"]
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


def get_option_ids(tokenizer):
    return [
        tokenizer(letter, add_special_tokens=False).input_ids[-1]
        for letter in ("A", "B", "C", "D", "E")
    ]


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

    abcd_ids = get_option_ids(model.tokenizer)
    weights  = SEVERITY_WEIGHTS.to(device)
    table    = {}
    n        = len(h5_paths)
    done     = 0

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

                for term in UAMP_TERMS:
                    prompt   = PROMPT_TEMPLATE.format(term=term)
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

                    quad = torch.stack(
                        [last_logits[:, tid] for tid in abcd_ids], dim=-1
                    )  # (B, 4)
                    probs    = F.softmax(quad, dim=-1)             # (B, 4)
                    severity = (probs * weights).sum(dim=-1)       # (B,)

                    col = col_name(term)
                    for stem, p, sev in zip(stems, probs.cpu().tolist(), severity.cpu().tolist()):
                        row = table.setdefault(stem, {"slide": stem})
                        row[f"{col}_p_mild"]     = round(p[0], 6)
                        row[f"{col}_p_moderate"] = round(p[1], 6)
                        row[f"{col}_p_severe"]   = round(p[2], 6)
                        row[f"{col}_p_not_sure"] = round(p[3], 6)
                        row[f"{col}_p_none"]     = round(p[4], 6)
                        row[f"{col}_severity"]   = round(sev,  6)

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

    out_path = out_dir / "prism2_severity_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table.values())
    print(f"\nSaved {len(table)} slides → {out_path}")

    rows = list(table.values())
    print("\nMean scores per concept:")
    header = f"{'concept':<45} {'P(mild)':>8} {'P(mod)':>8} {'P(sev)':>8} {'P(n/s)':>8} {'P(none)':>8} {'severity':>9}"
    print(header)
    print("-" * len(header))
    for col, term in zip(SCORE_COLS, UAMP_TERMS):
        pm  = np.mean([r[f"{col}_p_mild"]     for r in rows if f"{col}_p_mild"     in r])
        pmo = np.mean([r[f"{col}_p_moderate"] for r in rows if f"{col}_p_moderate" in r])
        ps  = np.mean([r[f"{col}_p_severe"]   for r in rows if f"{col}_p_severe"   in r])
        pns = np.mean([r[f"{col}_p_not_sure"] for r in rows if f"{col}_p_not_sure" in r])
        pn  = np.mean([r[f"{col}_p_none"]     for r in rows if f"{col}_p_none"     in r])
        sv  = np.mean([r[f"{col}_severity"]   for r in rows if f"{col}_severity"   in r])
        print(f"{term:<45} {pm:>8.4f} {pmo:>8.4f} {ps:>8.4f} {pns:>8.4f} {pn:>8.4f} {sv:>9.4f}")


if __name__ == "__main__":
    main()
