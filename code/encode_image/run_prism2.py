"""
Extract PRISM2 base and diagnostic embeddings from pre-computed Virchow2 features.

PRISM2 requires (N, 1280) class-token-only Virchow2 embeddings.
Trident stores (N, 2560) = [class_token (1280) | mean_token (1280)] concatenated,
so we slice [:, :1280] to get the class token.

Output format mirrors trident slide encoder .h5 files for downstream compatibility
(UMAP, linear probes, etc.):
  dataset "features"  shape (dim,)  float32
  attr    "encoder"   = "prism2_base" or "prism2_diagnostic"
  attr    "name"      = slide stem

Outputs written to:
  <JOB_DIR>/20x_224px_0px_overlap/prism2_base/<slide>.h5        shape (2560,)
  <JOB_DIR>/20x_224px_0px_overlap/prism2_diagnostic/<slide>.h5  shape (3072,)

For report generation (offline, no HF connection needed), use generate_reports.py
which re-loads tile embeddings and calls model.get_response().

Usage:
  python run_prism2.py [--batch_size 8] [--gpu 0] [--skip_errors]

NOTE: model shard model-00004-of-00004.safetensors must be present at MODEL_PATH.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

VIRCHOW2_DIM   = 2560
CLASS_TOKEN_DIM = 1280   # first half of Virchow2's 2560-dim output

FEAT_DIR   = Path("/home/jovyan/kgbk271-ibd-datavol-1/data/processed/trident_processed/20x_224px_0px_overlap/features_virchow2")
OUT_BASE   = Path("/home/jovyan/kgbk271-ibd-datavol-1/data/processed/trident_processed/20x_224px_0px_overlap/prism2_base")
OUT_DIAG   = Path("/home/jovyan/kgbk271-ibd-datavol-1/data/processed/trident_processed/20x_224px_0px_overlap/prism2_diagnostic")
MODEL_PATH = "/home/jovyan/shared-data/users/kexin/models/VLM/prism2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=8,
                   help="Slides per forward pass. Reduce if OOM.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip_errors", action="store_true", default=True)
    return p.parse_args()


def load_virchow2_h5(path: Path) -> torch.Tensor:
    """Return (N, 1280) class-token embeddings from a trident virchow2 .h5 file."""
    with h5py.File(path, "r") as f:
        feats = torch.from_numpy(f["features"][:])   # (N, 2560) float32
    if feats.shape[1] == VIRCHOW2_DIM:
        feats = feats[:, :CLASS_TOKEN_DIM]            # (N, 1280) — class token only
    elif feats.shape[1] != CLASS_TOKEN_DIM:
        raise ValueError(f"Unexpected feature dim {feats.shape[1]} in {path.name}")
    return feats


def save_h5(path: Path, embedding: np.ndarray, encoder_name: str, slide_name: str):
    """Save a 1D slide embedding as .h5, matching trident slide encoder format."""
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("features", data=embedding)
        ds.attrs["encoder"] = encoder_name
        ds.attrs["name"] = slide_name


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    OUT_DIAG.mkdir(parents=True, exist_ok=True)

    # Verify all model shards present before loading
    index_path = Path(MODEL_PATH) / "model.safetensors.index.json"
    if index_path.exists():
        import json
        shards = set(json.loads(index_path.read_text())["weight_map"].values())
        missing = [s for s in shards if not (Path(MODEL_PATH) / s).exists()]
        if missing:
            print(f"ERROR: Missing model shards: {missing}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading PRISM2 from {MODEL_PATH} ...")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device=device, dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    print("Model loaded.")

    # Collect pending slides (skip already done)
    all_h5 = sorted(FEAT_DIR.glob("*.h5"))
    pending = [
        p for p in all_h5
        if not (OUT_BASE / f"{p.stem}.h5").exists()
        or not (OUT_DIAG / f"{p.stem}.h5").exists()
    ]

    print(f"Total virchow2 slides : {len(all_h5)}")
    print(f"Already done          : {len(all_h5) - len(pending)}")
    print(f"To process            : {len(pending)}")

    done_count = 0
    for batch_start in range(0, len(pending), args.batch_size):
        batch_paths = pending[batch_start : batch_start + args.batch_size]
        slides, stems = [], []

        for h5_path in batch_paths:
            try:
                feats = load_virchow2_h5(h5_path)
                slides.append(feats)
                stems.append(h5_path.stem)
            except Exception as e:
                if args.skip_errors:
                    print(f"[SKIP load] {h5_path.name}: {e}", file=sys.stderr)
                else:
                    raise

        if not slides:
            continue

        try:
            batch = processor(tile_embeddings=slides)
            batch = {k: v.to(device=device, dtype=torch.bfloat16) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                base_embs = model.get_base_embedding(**batch)        # (B, 2560)
                diag_embs = model.get_diagnostic_embedding(**batch)  # (B, 3072)

            for i, stem in enumerate(stems):
                save_h5(OUT_BASE / f"{stem}.h5",
                        base_embs[i].cpu().float().numpy(),
                        "prism2_base", stem)
                save_h5(OUT_DIAG / f"{stem}.h5",
                        diag_embs[i].cpu().float().numpy(),
                        "prism2_diagnostic", stem)

            done_count += len(stems)
            print(f"[{done_count}/{len(pending)}] {stems[0]} ... {stems[-1]}")

        except Exception as e:
            if args.skip_errors:
                print(f"[SKIP batch] {[p.name for p in batch_paths]}: {e}", file=sys.stderr)
            else:
                raise

    n_base = len(list(OUT_BASE.glob("*.h5")))
    n_diag = len(list(OUT_DIAG.glob("*.h5")))
    print(f"\nDone. Base: {n_base}  Diagnostic: {n_diag}")


if __name__ == "__main__":
    main()
