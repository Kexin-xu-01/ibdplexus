"""
Generate PRISM2 histology reports from pre-computed Virchow2 features.

Calls model.get_response() with a configurable prompt per slide.
Unlike run_prism2.py (which saves embeddings), this requires re-loading
the tile embeddings each time since get_response() needs the raw tiles
to condition the Phi-3 decoder.

Outputs per slide:
  <OUT_DIR>/<slide>.txt             — generated report text (text generation)
  <OUT_DIR>/scores.csv              — P(Yes) per slide (yes/no mode)

Aggregate outputs (appended incrementally as slides complete):
  <RESULTS_ROOT>/prism2_reports.jsonl  — {slide, prompt, report} one record per line
  <RESULTS_ROOT>/prism2_reports.csv    — slide, prompt, report rows

Usage:
  # Report generation
  python run_prism2_reports.py --prompt "Write a report"

  # Free-text QA
  python run_prism2_reports.py --prompt "What tissue compartments are present?"

  # Yes/No scoring (saves CSV with P(Yes) per slide)
  python run_prism2_reports.py --yes_no --prompt "Is there evidence of inflammation?"
"""

import argparse
import csv
import json
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="Write a report",
                   help="Prompt passed to every slide.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory. Defaults to <job_dir>/prism2_reports/<prompt_slug>/")
    p.add_argument("--yes_no", action="store_true",
                   help="Run yes/no scoring instead of text generation. Saves a CSV.")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Slides per forward pass. Reduce if OOM.")
    p.add_argument("--max_new_tokens", type=int, default=200)
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


def prompt_slug(prompt: str) -> str:
    return prompt.lower().replace(" ", "_")[:40]


def rebuild_csv(jsonl_path: Path, csv_path: Path):
    """Rebuild wide CSV from JSONL: one column per unique prompt."""
    data: dict[str, dict[str, str]] = {}
    prompts: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            slide, prompt, report = r["slide"], r["prompt"], r["report"]
            if prompt not in prompts:
                prompts.append(prompt)
            data.setdefault(slide, {})[prompt] = report
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["slide"] + prompts, extrasaction="ignore")
        writer.writeheader()
        for slide in sorted(data):
            writer.writerow({"slide": slide, **data[slide]})


def append_results(prompt: str, stems: list, texts: list):
    """Append records to the shared JSONL then rebuild the wide CSV."""
    jsonl_path = RESULTS_ROOT / "prism2_reports.jsonl"
    csv_path   = RESULTS_ROOT / "prism2_reports.csv"

    with open(jsonl_path, "a") as jf:
        for slide, report in zip(stems, texts):
            jf.write(json.dumps({"slide": slide, "prompt": prompt, "report": report}) + "\n")

    rebuild_csv(jsonl_path, csv_path)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        tag = "yesno" if args.yes_no else "reports"
        out_dir = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2") / tag / prompt_slug(args.prompt)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Model loaded. Prompt: '{args.prompt}'")

    all_h5 = sorted(FEAT_DIR.glob("*.h5"))
    ext = ".csv" if args.yes_no else ".txt"
    pending = [p for p in all_h5 if not (out_dir / f"{p.stem}{ext}").exists()]

    # for yes/no, write all scores to a single CSV
    yesno_rows = []

    print(f"Total slides: {len(all_h5)}  |  To process: {len(pending)}")

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

            with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
                if args.yes_no:
                    scores = model.yes_no_score(
                        tile_embeddings=batch["tile_embeddings"],
                        attention_mask=batch["attention_mask"],
                        question=args.prompt,
                    )  # (B,) P(Yes)
                    for stem, score in zip(stems, scores.cpu().tolist()):
                        yesno_rows.append({"slide": stem, "p_yes": round(score, 6)})
                else:
                    responses = model.get_response(
                        **batch,
                        prompt=args.prompt,
                        max_new_tokens=args.max_new_tokens,
                    )
                    for stem, text in zip(stems, responses):
                        (out_dir / f"{stem}.txt").write_text(text)
                    append_results(args.prompt, stems, responses)

            done_count += len(stems)
            print(f"[{done_count}/{len(pending)}] {stems[0]} ... {stems[-1]}")

        except Exception as e:
            if args.skip_errors:
                print(f"[SKIP batch] {[p.name for p in batch_paths]}: {e}", file=sys.stderr)
            else:
                raise

    if args.yes_no and yesno_rows:
        csv_path = out_dir / "scores.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["slide", "p_yes"])
            writer.writeheader()
            writer.writerows(yesno_rows)
        print(f"Yes/No scores saved to {csv_path}")

    print(f"\nDone. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
