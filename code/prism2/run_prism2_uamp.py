"""
Score 11 UAMP histological terms per slide using PRISM2 yes/no scoring.

For each slide, asks "Is <term> present?" for every term in UAMP_TERMS and
saves P(Yes) scores to a wide CSV (one row per slide, one column per term).

Outputs:
  <RESULTS_ROOT>/uamp_report.csv  — slide + one p_yes column per term

Usage:
  python run_prism2_uamp.py
  python run_prism2_uamp.py --batch_size 2 --gpu 1
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import torch
from transformers import AutoModel, AutoProcessor

VIRCHOW2_DIM    = 2560
CLASS_TOKEN_DIM = 1280

FEAT_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/20x_224px_0px_overlap/features_virchow2")
MODEL_PATH   = "/home/jovyan/shared-data/users/kexin/models/VLM/prism2"
RESULTS_ROOT = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2")

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

# Column name: snake_case version of each term
def col_name(term: str) -> str:
    return term.lower().replace(" ", "_")

COLUMNS = ["slide"] + [col_name(t) for t in UAMP_TERMS]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=4,
                   help="Slides per forward pass. Reduce if OOM.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip_errors", action="store_true", default=True)
    return p.parse_args()


def load_virchow2_h5(path: Path) -> torch.Tensor:
    with h5py.File(path, "r") as f:
        feats = torch.from_numpy(f["features"][:])
    if feats.shape[1] == VIRCHOW2_DIM:
        feats = feats[:, :CLASS_TOKEN_DIM]
    elif feats.shape[1] != CLASS_TOKEN_DIM:
        raise ValueError(f"Unexpected feature dim {feats.shape[1]} in {path.name}")
    return feats


def load_table(csv_path: Path) -> tuple[list, list[dict]]:
    """Load existing CSV, return (fieldnames, rows)."""
    if not csv_path.exists():
        return [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def save_table(csv_path: Path, fieldnames: list, rows: list[dict]):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_ROOT / "prism2_histological_score.csv"

    # Load existing table and index by slide
    existing_cols, existing_rows = load_table(csv_path)
    table: dict[str, dict] = {row["slide"]: row for row in existing_rows}

    # Extend fieldnames with any new UAMP columns not already present
    new_uamp_cols = [c for c in COLUMNS[1:] if c not in existing_cols]
    fieldnames = existing_cols if existing_cols else COLUMNS[:]
    for c in new_uamp_cols:
        if c not in fieldnames:
            fieldnames.append(c)

    # Slides that already have all UAMP columns scored can be skipped
    uamp_cols = COLUMNS[1:]
    done_slides = {
        slide for slide, row in table.items()
        if all(row.get(c) not in (None, "") for c in uamp_cols)
    }

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
    print(f"Model loaded. Scoring {len(UAMP_TERMS)} UAMP terms.")

    all_h5 = sorted(FEAT_DIR.glob("*.h5"))
    pending = [p for p in all_h5 if p.stem not in done_slides]
    print(f"Total slides: {len(all_h5)}  |  Already done: {len(done_slides)}  |  To process: {len(pending)}")

    done_count = 0
    for batch_start in range(0, len(pending), args.batch_size):
        batch_paths = pending[batch_start : batch_start + args.batch_size]
        slides, stems = [], []

        for h5_path in batch_paths:
            try:
                slides.append(load_virchow2_h5(h5_path))
                stems.append(h5_path.stem)
            except Exception as e:
                if args.skip_errors:
                    print(f"[SKIP load] {h5_path.name}: {e}", file=sys.stderr)
                else:
                    raise

        if not slides:
            continue

        try:
            batch = processor(tile_embeddings=slides).to(device)

            # Score all terms; merge into existing table rows
            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                for term in UAMP_TERMS:
                    question = f"Is {term} present?"
                    scores = model.yes_no_score(
                        tile_embeddings=batch["tile_embeddings"],
                        attention_mask=batch["attention_mask"],
                        question=question,
                    )  # (B,) P(Yes)
                    for stem, score in zip(stems, scores.cpu().tolist()):
                        row = table.setdefault(stem, {"slide": stem})
                        row[col_name(term)] = round(score, 6)

            # Persist after each batch so progress isn't lost on failure
            save_table(csv_path, fieldnames, list(table.values()))

            done_count += len(stems)
            print(f"[{done_count}/{len(pending)}] {stems[0]} ... {stems[-1]}")

        except Exception as e:
            if args.skip_errors:
                print(f"[SKIP batch] {[p.name for p in batch_paths]}: {e}", file=sys.stderr)
            else:
                raise

    print(f"\nDone. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
