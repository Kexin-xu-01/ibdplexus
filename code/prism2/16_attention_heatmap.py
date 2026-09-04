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
THUMB_DIR          = Path("/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/20x_224px_0px_overlap/visualization")
WSI_DIR            = Path("/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected")

N_TOP_PATCHES = 8

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
# Visualisation helpers
# ---------------------------------------------------------------------------

def _make_overlay(thumb_np, heatmap_norm, coords, sx, sy, pw, ph, alpha=0.45, cmap=None):
    """Blend attention colour squares onto H&E thumbnail.

    Each tile is drawn as a crisp square at a minimum of 10 px so it is
    visible even when the thumbnail is very small relative to the slide.
    """
    H, W = thumb_np.shape[:2]
    bg = thumb_np.astype(np.float32) / 255.0
    result = bg.copy()
    min_px = max(10, int(round(pw)))
    colors = cmap(heatmap_norm)  # (N, 4) RGBA

    for i, (x, y) in enumerate(coords):
        if heatmap_norm[i] == 0:
            continue
        x0 = int(round(x * sx))
        y0 = int(round(y * sy))
        x1 = min(W, x0 + min_px)
        y1 = min(H, y0 + min_px)
        tile_rgb = colors[i, :3]
        result[y0:y1, x0:x1] = (1.0 - alpha) * bg[y0:y1, x0:x1] + alpha * tile_rgb
    return np.clip(result, 0, 1)


def _top_patches(wsi_path: Path, coords, heatmap_norm, n, patch_px):
    """Return list of (RGB uint8 array, score) for the n highest-scoring tiles."""
    try:
        import openslide
    except ImportError:
        return []
    top_idx = np.argsort(heatmap_norm)[::-1][:n]
    sl = openslide.OpenSlide(str(wsi_path))
    patches = []
    for i in top_idx:
        x, y = int(coords[i, 0]), int(coords[i, 1])
        r = sl.read_region((x, y), 0, (patch_px, patch_px))
        patches.append((np.array(r.convert("RGB")), float(heatmap_norm[i])))
    sl.close()
    return patches


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def save_figure(
    out_path: Path,
    heatmap_norm: np.ndarray,   # (N,) [0,1]
    coords: np.ndarray,         # (N,2) level-0 pixel coords
    meta: dict,
    slide: str,
    p_yes: dict[str, float],
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import FancyBboxPatch
        from PIL import Image
    except ImportError:
        print("  [skip PNG] matplotlib / PIL not available")
        return

    bg       = "#141414"
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "attn_he", ["#000033", "#00FFCC", "#FFE000"], N=256
    )
    patch_px = int(float(meta.get("patch_size_level0", 672)))

    # --- get true slide dimensions from OpenSlide (metadata often missing) ---
    wsi_path = WSI_DIR / f"{slide}.tiff"
    l0w_slide, l0h_slide = None, None
    if wsi_path.exists():
        try:
            import openslide as _osl
            _sl = _osl.OpenSlide(str(wsi_path))
            l0w_slide, l0h_slide = _sl.dimensions
            _sl.close()
        except Exception:
            pass

    # --- load thumbnail ---
    thumb_path = THUMB_DIR / f"{slide}.jpg"
    if thumb_path.exists():
        thumb_np = np.array(Image.open(thumb_path))
        H, W = thumb_np.shape[:2]
        l0w = l0w_slide or int(meta.get("level0_width",  0)) or (int(coords[:,0].max()) + 2 * patch_px)
        l0h = l0h_slide or int(meta.get("level0_height", 0)) or (int(coords[:,1].max()) + 2 * patch_px)
        sx, sy = W / l0w, H / l0h
        pw, ph = patch_px * sx, patch_px * sy
        have_thumb = True
    else:
        have_thumb = False

    # --- build overlay ---
    if have_thumb:
        overlay = _make_overlay(thumb_np, heatmap_norm, coords, sx, sy, pw, ph,
                                alpha=0.45, cmap=cmap)

    # --- top patches ---
    patches  = _top_patches(wsi_path, coords, heatmap_norm, N_TOP_PATCHES, patch_px) \
               if wsi_path.exists() else []
    n_patches = len(patches)

    # --- layout ---
    # Two rows: top (3 panels) + bottom (patches).  Bottom row only if patches exist.
    top_h   = 5.5
    bot_h   = 2.2 if n_patches else 0
    fig_h   = top_h + bot_h + 0.4
    fig     = plt.figure(figsize=(18, fig_h), dpi=130, facecolor=bg,
                         layout="constrained")

    if n_patches:
        outer = gridspec.GridSpec(
            2, 1, figure=fig, hspace=0.08,
            height_ratios=[top_h, bot_h],
        )
        gs_top = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[0], wspace=0.04,
            width_ratios=[2, 2, 1.6],
        )
        gs_bot = gridspec.GridSpecFromSubplotSpec(
            1, n_patches, subplot_spec=outer[1], wspace=0.03
        )
    else:
        outer  = gridspec.GridSpec(1, 1, figure=fig)
        gs_top = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[0], wspace=0.04,
            width_ratios=[2, 2, 1.6],
        )
        gs_bot = None

    def _dark_ax(ax):
        ax.set_facecolor(bg)
        ax.axis("off")

    # Panel 0 — raw H&E
    ax_he = fig.add_subplot(gs_top[0, 0])
    _dark_ax(ax_he)
    if have_thumb:
        ax_he.imshow(thumb_np)
    ax_he.set_title("H&E", color="white", fontsize=9, pad=3)

    # Panel 1 — attention overlay
    ax_ov = fig.add_subplot(gs_top[0, 1])
    _dark_ax(ax_ov)
    if have_thumb:
        ax_ov.imshow(overlay)
        # Colorbar inset — anchored inside the overlay, lower-left corner
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax_ov, width="3%", height="35%",
                         loc="lower left", borderpad=0.8)
        sm  = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cb  = plt.colorbar(sm, cax=cax)
        cb.set_label("attn.", color="white", fontsize=5, labelpad=2)
        cb.ax.yaxis.set_tick_params(color="white", labelsize=5)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        cb.outline.set_edgecolor("#555555")
    else:
        # fallback: grid raster
        grid_col = (coords[:, 0] // patch_px).astype(int)
        grid_row = (coords[:, 1] // patch_px).astype(int)
        grid = np.full((int(grid_row.max()) + 1, int(grid_col.max()) + 1), np.nan, np.float32)
        grid[grid_row, grid_col] = heatmap_norm
        im = ax_ov.imshow(grid, cmap=cmap, vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
        plt.colorbar(im, ax=ax_ov, fraction=0.04, pad=0.02)
    ax_ov.set_title("Attention heatmap", color="white", fontsize=9, pad=3)

    # Panel 2 — UAMP scores bar chart
    ax_bar = fig.add_subplot(gs_top[0, 2])
    ax_bar.set_facecolor(bg)
    # Shortened display labels so they fit in one column
    _LABEL_MAP = {
        "inflammation_involvement":            "Inflammation",
        "crypt_architectural_distortion":      "Crypt distortion",
        "neutrophil_granulocytic_infiltration":"Neutrophils",
        "crypt_abscesses":                     "Crypt abscesses",
        "lymphoid_aggregates":                 "Lymphoid aggregates",
        "histiocytic_granulomas":              "Granulomas",
        "mucin_depletion":                     "Mucin depletion",
        "pyloric_gland_metaplasia":            "Pyloric metaplasia",
        "paneth_cell_metaplasia":              "Paneth metaplasia",
        "neuronal_hyperplasia":                "Neuronal hyperpl.",
        "muscular_hypertrophy":                "Muscular hypertrophy",
    }
    labels = [_LABEL_MAP.get(c, c.replace("_", " ")) for c in UAMP_COLS]
    vals   = [p_yes.get(c, float("nan")) for c in UAMP_COLS]
    bar_colors = ["#e05c5c" if (v == v and v > 0.5) else "#5c9ee0" for v in vals]
    y_pos  = list(range(len(labels)))
    ax_bar.barh(y_pos, vals, color=bar_colors, height=0.6)
    ax_bar.axvline(0.5, color="#888888", lw=0.8, ls="--")
    ax_bar.set_xlim(0, 1)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, color="white", fontsize=7)
    ax_bar.set_xlabel("P(Yes)", color="white", fontsize=8)
    ax_bar.tick_params(axis="x", colors="white", labelsize=7)
    ax_bar.spines[:].set_color("#444444")
    ax_bar.set_facecolor(bg)
    ax_bar.set_title("UMAP histological scores", color="white", fontsize=9, pad=3)
    ax_bar.margins(y=0.02)

    # Bottom row — top-N patches
    if n_patches and gs_bot is not None:
        for i, (patch, score) in enumerate(patches):
            ax_p = fig.add_subplot(gs_bot[0, i])
            ax_p.imshow(patch)
            ax_p.axis("off")
            ax_p.text(0.04, 0.96, f"#{i+1}", transform=ax_p.transAxes,
                      color="white", fontsize=7, fontweight="bold", va="top",
                      bbox=dict(boxstyle="round,pad=0.15", fc="#00000099", ec="none"))
            bar_w = max(0.02, min(score, 0.98))
            ax_p.add_patch(FancyBboxPatch(
                (0, 0), bar_w, 0.06, transform=ax_p.transAxes, clip_on=True,
                boxstyle="square,pad=0", fc=cmap(score), ec="none", zorder=5,
            ))
            ax_p.text(0.5, 0.02, f"{score:.2f}", transform=ax_p.transAxes,
                      color="white", fontsize=6, ha="center", va="bottom",
                      fontweight="bold", zorder=6)

    fig.suptitle(f"{slide} — PRISM2 attention heatmap",
                 color="white", fontsize=10)
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
