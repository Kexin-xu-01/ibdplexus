"""
Attention heatmap for PRISM2, displayed alongside pre-computed UAMP scores.

P(Yes) scores for the 11 UAMP terms are read from the pre-computed
prism2_histological_score.csv (produced by 04_run_prism2_umap.py) — the
text decoder is NOT re-run.

The attention heatmap is computed from the Perceiver's cross-attention using
a pure PyTorch (math_gqa) kernel that returns attention weights:

    heatmap[k] = (Σ_{h,q} attn[h,q,k]) × ‖v[k]‖₂

Usage:
    python 18_attention_heatmap.py --slide 10407210HE1
    python 18_attention_heatmap.py --all
    python 18_attention_heatmap.py --slide 10407210HE1 --verify

Outputs (per slide, in OUT_ROOT):
    <slide>.h5    heatmap (N,), heatmap_raw (N,), coords (N,2)
                  attrs: p_yes_<term> for each UAMP term
    <slide>.png   heatmap raster + P(Yes) bar chart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor

# Utility functions live in utils/attention_heatmap.py
sys.path.insert(0, str(Path(__file__).parent))
from utils.attention_heatmap import (
    compute_attention_heatmap,
    verify_math_gqa_vs_flash,
)

VIRCHOW2_DIM    = 2560
CLASS_TOKEN_DIM = 1280
MODEL_PATH      = "/home/jovyan/shared-data/users/kexin/models/VLM/prism2"
DEFAULT_FEAT_DIR   = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/features_virchow2")
DEFAULT_SCORES_CSV = Path("/home/jovyan/kgbk271-ibd-volume/results/prism2_manual_knn/prism2_histological_score.csv")
DEFAULT_OUT_ROOT   = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/prism2_attention_heatmap")

UAMP_COLS = [
    "inflammation_involvement",
    "crypt_architectural_distortion",
    "neutrophil_granulocytic_infiltration",
    "crypt_abscesses",
    "lymphoid_aggregates",
    "histiocytic_granulomas",
    "mucin_depletion",
    "pyloric_gland_metaplasia",
    "paneth_cell_metaplasia",
    "neuronal_hyperplasia",
    "muscular_hypertrophy",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_slide(h5_path: Path):
    with h5py.File(h5_path) as f:
        feats  = f["features"][:]
        coords = f["coords"][:]
        meta   = dict(f["coords"].attrs)
    if feats.shape[1] == VIRCHOW2_DIM:
        feats = feats[:, :CLASS_TOKEN_DIM]
    return feats.astype(np.float32), coords, meta


def load_scores(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "slide" in df.columns:
        df = df.set_index("slide")
    return df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_figure(
    out_path: Path,
    heatmap_norm: np.ndarray,
    coords: np.ndarray,
    meta: dict,
    slide: str,
    p_yes: dict[str, float],
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  [skip PNG] matplotlib not available")
        return

    patch_px = int(float(meta.get("patch_size_level0", 672)))
    grid_col = (coords[:, 0] // patch_px).astype(int)
    grid_row = (coords[:, 1] // patch_px).astype(int)
    grid = np.full((int(grid_row.max()) + 1, int(grid_col.max()) + 1), np.nan, np.float32)
    grid[grid_row, grid_col] = heatmap_norm

    bg = "#141414"
    fig = plt.figure(figsize=(14, 6), dpi=130, facecolor=bg)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.06,
                            left=0.02, right=0.98, top=0.92, bottom=0.08,
                            width_ratios=[1, 1])

    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.set_facecolor("#000000")
    im = ax_map.imshow(grid, cmap="inferno", vmin=0, vmax=1,
                       interpolation="nearest", aspect="equal")
    ax_map.set_title("Attention heatmap", color="white", fontsize=9)
    ax_map.axis("off")
    cb = plt.colorbar(im, ax=ax_map, fraction=0.04, pad=0.02)
    cb.set_label("score (norm.)", color="white", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_facecolor(bg)
    labels = [c.replace("_", " ") for c in UAMP_COLS]
    vals   = [p_yes.get(c, float("nan")) for c in UAMP_COLS]
    colors = ["#e05c5c" if (v == v and v > 0.5) else "#5c9ee0" for v in vals]
    y_pos  = range(len(labels))
    ax_bar.barh(list(y_pos), vals, color=colors, height=0.6)
    ax_bar.axvline(0.5, color="#888888", lw=0.8, ls="--")
    ax_bar.set_xlim(0, 1)
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(labels, color="white", fontsize=7)
    ax_bar.set_xlabel("P(Yes)", color="white", fontsize=8)
    ax_bar.tick_params(axis="x", colors="white", labelsize=7)
    ax_bar.spines[:].set_color("#444444")
    ax_bar.set_facecolor(bg)
    ax_bar.set_title("UAMP histology scores", color="white", fontsize=9)

    fig.suptitle(f"{slide} — PRISM2 attention heatmap",
                 color="white", fontsize=10, y=0.98)
    fig.savefig(out_path, bbox_inches="tight", facecolor=bg, dpi=130)
    plt.close(fig)
    print(f"  saved PNG : {out_path.name}")


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------

def process_slide(
    slide_path: Path,
    model,
    processor,
    device: torch.device,
    model_dtype,
    scores_df: pd.DataFrame,
    out_dir: Path,
    skip_existing: bool,
    do_verify: bool,
):
    stem   = slide_path.stem
    out_h5 = out_dir / f"{stem}.h5"

    if skip_existing and out_h5.exists():
        print(f"  [skip] {stem}")
        return

    feats_np, coords, meta = load_slide(slide_path)
    print(f"  {stem}: {len(feats_np)} tiles", flush=True)

    batch     = processor(tile_embeddings=[torch.from_numpy(feats_np)])
    tile_emb  = batch["tile_embeddings"].to(device=device, dtype=model_dtype)
    attn_mask = batch["attention_mask"].to(device=device)

    if do_verify:
        print("  Verification (flash vs math_gqa):")
        verify_math_gqa_vs_flash(model, tile_emb, attn_mask)

    heatmap_norm, heatmap_raw = compute_attention_heatmap(model, tile_emb, attn_mask)
    print(f"  heatmap raw [{heatmap_raw.min():.3g}, {heatmap_raw.max():.3g}]")

    # P(Yes) scores from pre-computed CSV
    p_yes: dict[str, float] = {}
    if stem in scores_df.index:
        row = scores_df.loc[stem]
        p_yes = {c: float(row[c]) for c in UAMP_COLS if c in row.index}
    else:
        print(f"  [warn] {stem} not in scores CSV")

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("heatmap",     data=heatmap_norm)
        f.create_dataset("heatmap_raw", data=heatmap_raw)
        f.create_dataset("coords",      data=coords)
        for k, v in meta.items():
            f["coords"].attrs[k] = v
        f.attrs["slide"] = stem
        for col, val in p_yes.items():
            f.attrs[f"p_yes_{col}"] = val
    print(f"  saved h5  : {out_h5.name}")

    save_figure(out_dir / f"{stem}.png", heatmap_norm, coords, meta, stem, p_yes)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slide",         help="Slide stem (or path to .h5 feature file)")
    p.add_argument("--all",           action="store_true", help="Process all slides")
    p.add_argument("--feat_dir",      default=str(DEFAULT_FEAT_DIR),
                   help="Directory of Virchow2 .h5 feature files  (default: %(default)s)")
    p.add_argument("--scores_csv",    default=str(DEFAULT_SCORES_CSV),
                   help="prism2_histological_score.csv  (default: %(default)s)")
    p.add_argument("--out_dir",       default=str(DEFAULT_OUT_ROOT),
                   help="Output directory  (default: %(default)s)")
    p.add_argument("--gpu",           type=int, default=0)
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--verify",        action="store_true",
                   help="Print flash vs math_gqa diff stats on the first slide")
    args = p.parse_args()

    if not args.slide and not args.all:
        p.error("Specify --slide STEM or --all")

    feat_dir = Path(args.feat_dir)
    out_root = Path(args.out_dir)

    device      = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading PRISM2 from {MODEL_PATH} ...")
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
    ).to(device=device, dtype=model_dtype).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True,
    )
    print("Model ready.")

    print(f"Loading UAMP scores from {args.scores_csv} ...")
    scores_df = load_scores(Path(args.scores_csv))
    print(f"  {len(scores_df)} slides scored.\n")

    out_root.mkdir(parents=True, exist_ok=True)

    if args.slide:
        path = Path(args.slide)
        if not path.exists():
            path = feat_dir / f"{args.slide}.h5"
        if not path.exists():
            sys.exit(f"Slide not found: {args.slide}")
        slides = [path]
    else:
        slides = sorted(feat_dir.glob("*.h5"))
        print(f"Found {len(slides)} slides.\n")

    first = True
    for i, slide_path in enumerate(slides, 1):
        print(f"[{i}/{len(slides)}]", end=" ")
        try:
            process_slide(
                slide_path, model, processor, device, model_dtype,
                scores_df=scores_df,
                out_dir=out_root,
                skip_existing=args.skip_existing,
                do_verify=(args.verify and first),
            )
            first = False
        except Exception as e:
            print(f"  [ERROR] {slide_path.name}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nDone. Output: {out_root}")


if __name__ == "__main__":
    main()
