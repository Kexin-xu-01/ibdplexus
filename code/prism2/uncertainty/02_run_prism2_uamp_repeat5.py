"""
Score 11 UAMP histological concepts 5 times on the same slides
(tissue_threshold_15_filtered_no_darkspot) to assess model reproducibility.

Loads the model once, runs all 5 scoring passes, then computes per-slide
mean and SD across runs for each concept.

Outputs:
  <OUT_DIR>/run_1/prism2_histological_score.csv  … run_5/…
  <OUT_DIR>/reproducibility_stats.csv   — mean_<term>, sd_<term> per slide
  <OUT_DIR>/reproducibility_summary.csv — per-concept mean SD (± across slides)

Usage:
  python run_prism2_umap_repeat5.py
  python run_prism2_umap_repeat5.py --n_runs 5 --batch_size 4 --gpu 0
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

VIRCHOW2_DIM    = 2560
CLASS_TOKEN_DIM = 1280

FEAT_DIR   = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/"
                  "tissue_threshold_15_filtered_no_darkspot/"
                  "20x_224px_0px_overlap/features_virchow2")
MODEL_PATH = "/home/jovyan/shared-data/users/kexin/models/VLM/prism2"
OUT_DIR    = Path("/home/jovyan/kgbk271-ibd-volume/results/"
                  "prism2_tissue_threshold_15_filtered_no_darkspot/repeat_five_times")

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
FIELDNAMES = ["slide"] + SCORE_COLS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs",      type=int, default=5)
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
    """Return dict: slide_stem -> {col: score}."""
    table = {}
    n = len(h5_paths)
    done = 0
    for start in range(0, n, batch_size):
        batch_paths = h5_paths[start : start + batch_size]
        slides, stems = [], []
        for p in batch_paths:
            try:
                slides.append(load_h5(p))
                stems.append(p.stem)
            except Exception as e:
                if skip_errors:
                    print(f"  [SKIP load] {p.name}: {e}", file=sys.stderr)
                else:
                    raise
        if not slides:
            continue
        try:
            batch = processor(tile_embeddings=slides).to(device)
            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                for term in UAMP_TERMS:
                    scores = model.yes_no_score(
                        tile_embeddings=batch["tile_embeddings"],
                        attention_mask=batch["attention_mask"],
                        question=f"Is {term} present?",
                    )
                    for stem, score in zip(stems, scores.cpu().tolist()):
                        row = table.setdefault(stem, {"slide": stem})
                        row[col_name(term)] = round(score, 6)
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


def save_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> dict:
    """Return dict: slide -> {col: float}."""
    with open(path, newline="") as f:
        return {row["slide"]: row for row in csv.DictReader(f)}


def already_done(csv_path: Path, n_slides: int) -> bool:
    if not csv_path.exists():
        return False
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < n_slides:
        return False
    return all(row.get(c) not in (None, "") for row in rows for c in SCORE_COLS)


def compute_stats(run_tables: list, out_dir: Path):
    """Compute per-slide mean and SD across runs; save two CSV files."""
    # Collect all slides seen across any run
    all_slides = sorted({slide for t in run_tables for slide in t})
    n_runs = len(run_tables)

    # Per-slide stats
    stats_fieldnames = ["slide"] + [f"mean_{c}" for c in SCORE_COLS] + [f"sd_{c}" for c in SCORE_COLS]
    stats_rows = []
    # Accumulate per-concept SD values for the summary
    concept_sds = {c: [] for c in SCORE_COLS}
    concept_means = {c: [] for c in SCORE_COLS}

    for slide in all_slides:
        row = {"slide": slide}
        for col in SCORE_COLS:
            vals = []
            for t in run_tables:
                v = t.get(slide, {}).get(col)
                if v not in (None, ""):
                    vals.append(float(v))
            mean_v = round(float(np.mean(vals)), 6) if vals else None
            sd_v   = round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else None
            row[f"mean_{col}"] = mean_v
            row[f"sd_{col}"]   = sd_v
            if mean_v is not None:
                concept_means[col].append(mean_v)
            if sd_v is not None:
                concept_sds[col].append(sd_v)
        stats_rows.append(row)

    stats_path = out_dir / "reproducibility_stats.csv"
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f"  Per-slide stats → {stats_path}")

    # Per-concept summary
    summary_fieldnames = ["concept", "mean_score_across_runs", "mean_sd", "median_sd", "max_sd", "n_slides"]
    summary_rows = []
    for col in SCORE_COLS:
        sds   = concept_sds[col]
        means = concept_means[col]
        summary_rows.append({
            "concept":               col,
            "mean_score_across_runs": round(float(np.mean(means)), 6) if means else None,
            "mean_sd":               round(float(np.mean(sds)),   6) if sds else None,
            "median_sd":             round(float(np.median(sds)), 6) if sds else None,
            "max_sd":                round(float(np.max(sds)),    6) if sds else None,
            "n_slides":              len(sds),
        })

    summary_path = out_dir / "reproducibility_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Per-concept summary → {summary_path}")

    print("\nReproducibility summary:")
    header = f"{'concept':<45} {'mean_score':>10} {'mean_sd':>8} {'median_sd':>9} {'max_sd':>8} {'n_slides':>8}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(f"{r['concept']:<45} {r['mean_score_across_runs']:>10.4f} "
              f"{r['mean_sd']:>8.4f} {r['median_sd']:>9.4f} {r['max_sd']:>8.4f} {r['n_slides']:>8}")


def main():
    args = parse_args()
    feat_dir = Path(args.feat_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_h5 = sorted(feat_dir.glob("*.h5"))
    print(f"Found {len(all_h5)} slides in {feat_dir}")

    # Check which runs still need to be done
    pending_runs = []
    for run_i in range(1, args.n_runs + 1):
        csv_path = out_dir / f"run_{run_i}" / "prism2_histological_score.csv"
        if already_done(csv_path, len(all_h5)):
            print(f"Run {run_i}: already complete, skipping.")
        else:
            pending_runs.append(run_i)

    if pending_runs:
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
        print(f"Model loaded on {device}. Scoring {len(UAMP_TERMS)} concepts × "
              f"{len(all_h5)} slides × {len(pending_runs)} runs.\n")

        for run_i in pending_runs:
            print(f"── Run {run_i}/{args.n_runs} ──────────────────────────────")
            table = score_slides(model, processor, device,
                                 all_h5, args.batch_size, args.skip_errors)
            csv_path = out_dir / f"run_{run_i}" / "prism2_histological_score.csv"
            save_csv(csv_path, list(table.values()))
            print(f"  Saved {len(table)} slides → {csv_path}\n")

    # Load all completed runs and compute stats
    print("Computing reproducibility statistics ...")
    run_tables = []
    for run_i in range(1, args.n_runs + 1):
        csv_path = out_dir / f"run_{run_i}" / "prism2_histological_score.csv"
        if csv_path.exists():
            run_tables.append(load_csv(csv_path))
        else:
            print(f"  [WARN] Missing {csv_path}, skipping from stats.", file=sys.stderr)

    if len(run_tables) >= 2:
        compute_stats(run_tables, out_dir)
    else:
        print("Need at least 2 completed runs to compute stats.", file=sys.stderr)

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
