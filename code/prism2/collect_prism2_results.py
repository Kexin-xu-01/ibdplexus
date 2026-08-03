"""
Backfill prism2_reports.jsonl and prism2_reports.csv from existing per-slide
.txt files.  Safe to re-run: skips (slide, prompt) pairs already in the JSONL.

Usage:
  python collect_prism2_results.py [--reports_dir ...] [--out_dir ...]
"""

import argparse
import csv
import json
from pathlib import Path

REPORTS_DIR = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2/reports")
OUT_DIR     = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2")


def prompt_slug(prompt: str) -> str:
    return prompt.lower().replace(" ", "_")[:40]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reports_dir", type=Path, default=REPORTS_DIR)
    p.add_argument("--out_dir",     type=Path, default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = args.out_dir / "prism2_reports.jsonl"
    csv_path   = args.out_dir / "prism2_reports.csv"

    # Load already-written records keyed by (slide, slug) to avoid duplicates
    # regardless of whether prompt text vs slug was stored
    seen: set[tuple[str, str]] = set()
    slug_to_full: dict[str, str] = {}  # slug -> first full prompt text seen
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    slug = prompt_slug(r["prompt"])
                    seen.add((r["slide"], slug))
                    slug_to_full.setdefault(slug, r["prompt"])

    new_records: list[dict] = []

    for prompt_dir in sorted(args.reports_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        dir_slug = prompt_dir.name
        # Use the full prompt text already in JSONL if available, else use dir name
        prompt = slug_to_full.get(dir_slug, dir_slug.replace("_", " ").strip())
        for txt_file in sorted(prompt_dir.glob("*.txt")):
            slide = txt_file.stem
            if (slide, dir_slug) in seen:
                continue
            report = txt_file.read_text().strip()
            new_records.append({"slide": slide, "prompt": prompt, "report": report})
            seen.add((slide, dir_slug))

    if not new_records:
        print("Nothing new to add.")
        return

    with open(jsonl_path, "a") as jf:
        for r in new_records:
            jf.write(json.dumps(r) + "\n")

    # Rebuild wide CSV: one column per prompt
    all_records: dict[str, dict[str, str]] = {}
    prompts: list[str] = []
    with open(jsonl_path) as jf:
        for line in jf:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["prompt"] not in prompts:
                prompts.append(r["prompt"])
            all_records.setdefault(r["slide"], {})[r["prompt"]] = r["report"]

    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["slide"] + prompts, extrasaction="ignore")
        writer.writeheader()
        for slide in sorted(all_records):
            writer.writerow({"slide": slide, **all_records[slide]})

    print(f"Added {len(new_records)} new records → {jsonl_path.name}, {csv_path.name}")


if __name__ == "__main__":
    main()
