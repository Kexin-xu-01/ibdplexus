"""
Uncertainty estimation for PRISM2 yes/no histological scores via
temperature-scaled stochastic sampling.

For each slide × concept the model's log-softmax values over the {Yes, No}
token pair are extracted (yes_no_score with return_log_probs=True).  A
temperature T is applied before re-normalising over that pair:

    p_yes = exp(z_yes / T) / (exp(z_yes / T) + exp(z_no / T))

N_SAMPLES Bernoulli(p_yes) draws are then taken.  Outputs per slide × concept:

  p_yes_T      -- temperature-scaled P(Yes)  [primary score]
  frac_yes     -- fraction of samples that were Yes  [Monte-Carlo estimate]
  entropy      -- binary entropy H(p_yes_T) in bits  [uncertainty; max 1.0 at p=0.5]

Outputs:
  <OUT_DIR>/prism2_temperature_scores.csv

Usage:
  python 17_run_prism2_temperature_sampling.py
  python 17_run_prism2_temperature_sampling.py --temperature 2.0 --n_samples 50 --gpu 0
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
                  "prism2_manual_knn/uncertainty_estimation/"
                  "temperature_sampling")

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


def col_name(term: str) -> str:
    return term.lower().replace(" ", "_")


SCORE_COLS = [col_name(t) for t in UAMP_TERMS]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Softmax temperature T (>1 flattens, <1 sharpens)")
    p.add_argument("--n_samples",   type=int,   default=50,
                   help="Number of Bernoulli samples per slide × concept")
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--gpu",         type=int,   default=0)
    p.add_argument("--feat_dir",    type=str,   default=str(FEAT_DIR))
    p.add_argument("--out_dir",     type=str,   default=str(OUT_DIR))
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


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Binary entropy H(p) in bits; clamp to avoid log(0)."""
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(p * p.log2() + (1 - p) * (1 - p).log2())


def score_slides(model, processor, device, h5_paths, batch_size, temperature,
                 n_samples, skip_errors):
    """Return dict: slide_stem -> {col: {p_yes_T, frac_yes, entropy}}."""
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
                for term in UAMP_TERMS:
                    # log_pair: (B, 2) — [log P(Yes), log P(No)] from full-vocab log-softmax
                    log_pair = model.yes_no_score(
                        tile_embeddings=batch["tile_embeddings"],
                        attention_mask=batch["attention_mask"],
                        question=f"Is {term} present?",
                        return_log_probs=True,
                    ).float()  # ensure float32 for numerical stability

                    # Temperature scaling then re-normalise over the Yes/No pair
                    p_yes = F.softmax(log_pair / temperature, dim=-1)[:, 0]  # (B,)

                    # Stochastic sampling: (B, n_samples) binary tensor
                    samples = torch.bernoulli(
                        p_yes.unsqueeze(1).expand(-1, n_samples)
                    )  # (B, n_samples)

                    frac_yes = samples.mean(dim=1)   # (B,)
                    H        = binary_entropy(p_yes)  # (B,)

                    col = col_name(term)
                    for stem, py, fy, h in zip(
                        stems,
                        p_yes.cpu().tolist(),
                        frac_yes.cpu().tolist(),
                        H.cpu().tolist(),
                    ):
                        row = table.setdefault(stem, {"slide": stem})
                        row[f"{col}_p_yes_T"] = round(py, 6)
                        row[f"{col}_frac_yes"] = round(fy, 6)
                        row[f"{col}_entropy"]  = round(h,  6)

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


def build_fieldnames():
    suffixes = ["p_yes_T", "frac_yes", "entropy"]
    return ["slide"] + [
        f"{col}_{s}" for col in SCORE_COLS for s in suffixes
    ]


def main():
    args = parse_args()
    feat_dir = Path(args.feat_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_h5 = sorted(feat_dir.glob("*.h5"))
    print(f"Found {len(all_h5)} slides in {feat_dir}")
    print(f"Temperature: {args.temperature}  |  Samples per concept: {args.n_samples}")

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

    table = score_slides(
        model, processor, device, all_h5,
        args.batch_size, args.temperature, args.n_samples, args.skip_errors,
    )

    fieldnames = build_fieldnames()
    out_path = out_dir / "prism2_temperature_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table.values())

    print(f"\nSaved {len(table)} slides → {out_path}")

    # Quick summary: mean entropy per concept (higher = more uncertain)
    rows = list(table.values())
    print("\nMean entropy per concept (bits, max=1.0):")
    header = f"{'concept':<45} {'mean_entropy':>12} {'mean_p_yes_T':>13}"
    print(header)
    print("-" * len(header))
    for col, term in zip(SCORE_COLS, UAMP_TERMS):
        ents  = [r[f"{col}_entropy"]  for r in rows if f"{col}_entropy"  in r]
        pyes  = [r[f"{col}_p_yes_T"]  for r in rows if f"{col}_p_yes_T"  in r]
        print(f"{term:<45} {np.mean(ents):>12.4f} {np.mean(pyes):>13.4f}")


if __name__ == "__main__":
    main()
