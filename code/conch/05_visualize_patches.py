"""
Patch visualisation: CONCH zero-shot scores + PRISM2 attention.

Produces a two-panel figure per slide:
  Top row  — actual H&E patch images, each annotated with its PRISM2 attention
             score (normalised to the top patch in the slide).
  Bottom   — heatmap of CONCH CLIP cosine-similarity scores (12 IBD UAMP labels
             × N patches).  Cells are annotated with the numeric score.

How CONCH scores each concept
------------------------------
CONCH v1 is a CoCa-style contrastive vision-language model (ViT-B/16 image
encoder + 12-layer text encoder) trained on pathology image-caption pairs via
the CLIP objective.  Both encoders project to a shared 512-dim L2-normalised
space.

For zero-shot scoring we:
  1. Encode the patch image  → v  ∈ ℝ⁵¹² (L2 normalised).
  2. Encode each text prompt  → t_k ∈ ℝ⁵¹² (L2 normalised).
     Prompt template: "a histology patch showing <finding description>"
  3. Score  s_k = v · t_k  ∈ [-1, 1]  (cosine similarity, since both normalised).

These are NOT probabilities — they are raw cosine similarities.  A score >~0.35
is generally meaningful for CONCH.  Ranking labels by score gives the most
likely finding for that patch region.

Usage
-----
    # Interactive (single slide)
    from importlib import import_module
    viz = import_module('05_visualize_patches')
    fig = viz.plot_patch_scores('10407210HE1')
    fig.savefig('10407210HE1_patches.pdf', dpi=150, bbox_inches='tight')

    # CLI
    python 05_visualize_patches.py --slide 10407210HE1
    python 05_visualize_patches.py --slide 10407210HE1 --n_patches 16 --out figures/
    python 05_visualize_patches.py --all --out figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openslide
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONCH_DIR = Path(
    "/home/jovyan/kgbk271-ibd-volume/data/processed/"
    "tissue_threshold_15_filtered_no_darkspot_manual_knn/"
    "20x_224px_0px_overlap/prism2_conch_zero_shot"
)
WSI_DIR = Path(
    "/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected"
)
FIG_DIR = Path(
    "/home/jovyan/kgbk271-ibd-volume/results/conch_patch_figures"
)
PRISM2_CSV = Path(
    "/home/jovyan/kgbk271-ibd-volume/results/prism2_manual_knn/prism2_histological_score.csv"
)

# Lazy-loaded lookup: slide → {label: prism2_score}
_prism2_cache: dict | None = None


def _load_prism2_scores() -> dict:
    global _prism2_cache
    if _prism2_cache is None:
        import pandas as pd
        df = pd.read_csv(PRISM2_CSV)
        _prism2_cache = {r["slide"]: r.to_dict() for _, r in df.iterrows()}
    return _prism2_cache

PATCH_SIZE_LEVEL0 = 1311  # px at level-0 — CONCH native 20x scale (448px @ 0.5MPP, WSI at 0.171MPP)
                          # Must match PRISM2_PATCH_SIZE used in 02_conch_zero_shot_labels.py

# Clinical display order for the heatmap rows (most common active IBD findings first)
LABEL_ORDER = [
    "inflammation_involvement",
    "neutrophil_granulocytic_infiltration",
    "crypt_abscesses",
    "crypt_architectural_distortion",
    "mucin_depletion",
    "lymphoid_aggregates",
    "histiocytic_granulomas",
    "pyloric_gland_metaplasia",
    "paneth_cell_metaplasia",
    "neuronal_hyperplasia",
    "muscular_hypertrophy",
    "normal_mucosa",
]

LABEL_DISPLAY = {
    "inflammation_involvement":            "Inflammation",
    "neutrophil_granulocytic_infiltration": "Neutrophilic infiltration",
    "crypt_abscesses":                     "Crypt abscesses",
    "crypt_architectural_distortion":      "Crypt distortion",
    "mucin_depletion":                     "Mucin depletion",
    "lymphoid_aggregates":                 "Lymphoid aggregates",
    "histiocytic_granulomas":              "Histiocytic granulomas",
    "pyloric_gland_metaplasia":            "Pyloric gland metaplasia",
    "paneth_cell_metaplasia":              "Paneth cell metaplasia",
    "neuronal_hyperplasia":               "Neuronal hyperplasia",
    "muscular_hypertrophy":               "Muscular hypertrophy",
    "normal_mucosa":                       "Normal mucosa",
}


# ---------------------------------------------------------------------------
# Core plotting function
# ---------------------------------------------------------------------------

def plot_patch_scores(
    slide: str,
    n_patches: int = 8,
    sort_labels: str = "clinical",   # "clinical" | "mean_score" | "max_score"
    patch_display_px: int = 336,     # resize patches to this for display
    cmap_patches: str = "YlOrRd",    # colormap for PRISM2 attention border
    cmap_heat: str = "RdYlBu_r",     # colormap for CONCH score heatmap
    vmin: float = 0.0,
    vmax: float = 0.60,
    annot_fontsize: int = 10,
    figsize_w_per_patch: float = 1.4,   # narrower for PowerPoint
    save_path: Path | None = None,
    dpi: int = 200,
) -> plt.Figure:
    """
    Visualise top-N PRISM2-attention patches for *slide* with CONCH scores.

    Parameters
    ----------
    slide            : slide stem, e.g. '10407210HE1'
    n_patches        : how many top-attention patches to show (≤ stored in JSON)
    sort_labels      : row order in the heatmap
                       'clinical'   — fixed IBD clinical order
                       'mean_score' — sort by mean CONCH score across patches (desc)
                       'max_score'  — sort by max  CONCH score across patches (desc)
    patch_display_px : side length (px) to resize patch thumbnail for display
    cmap_patches     : colormap for the PRISM2 attention border around each image
    cmap_heat        : colormap for the CONCH heatmap cells
    vmin / vmax      : cosine-similarity range for the heatmap colour scale
    annot_fontsize   : font size of numeric annotations in the heatmap
    figsize_w_per_patch : inches per patch column
    save_path        : if set, save figure here (PNG / PDF / SVG)
    dpi              : dots-per-inch for rasterised output

    Returns
    -------
    matplotlib Figure
    """
    # ------------------------------------------------------------------
    # 1. Load CONCH zero-shot results
    # ------------------------------------------------------------------
    json_path = CONCH_DIR / f"{slide}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"CONCH results not found: {json_path}\n"
            "Run 02_conch_zero_shot_labels.py first."
        )
    with open(json_path) as f:
        patches_data = json.load(f)

    n_patches = min(n_patches, len(patches_data))
    patches_data = patches_data[:n_patches]

    # ------------------------------------------------------------------
    # 2. Build score matrix  (n_labels × n_patches)
    # ------------------------------------------------------------------
    score_matrix = np.full((len(LABEL_ORDER), n_patches), np.nan)
    for j, p in enumerate(patches_data):
        scores_dict = {lbl["label"]: lbl["score"] for lbl in p["labels"]}
        for i, label in enumerate(LABEL_ORDER):
            score_matrix[i, j] = scores_dict.get(label, np.nan)

    attn_scores = np.array([p["attn_score"] for p in patches_data])

    # ------------------------------------------------------------------
    # 3. Optionally reorder labels
    # ------------------------------------------------------------------
    if sort_labels == "mean_score":
        order = np.argsort(-np.nanmean(score_matrix, axis=1))
        score_matrix = score_matrix[order]
        label_keys = [LABEL_ORDER[i] for i in order]
    elif sort_labels == "max_score":
        order = np.argsort(-np.nanmax(score_matrix, axis=1))
        score_matrix = score_matrix[order]
        label_keys = [LABEL_ORDER[i] for i in order]
    else:
        label_keys = LABEL_ORDER[:]

    label_display_names = [LABEL_DISPLAY[k] for k in label_keys]

    # ------------------------------------------------------------------
    # 4. Read patch images from WSI
    # ------------------------------------------------------------------
    wsi_path = WSI_DIR / f"{slide}.tiff"
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    images = []
    sl = openslide.OpenSlide(str(wsi_path))
    try:
        for p in patches_data:
            region = sl.read_region(
                (int(p["coord_x"]), int(p["coord_y"])),
                0,
                (PATCH_SIZE_LEVEL0, PATCH_SIZE_LEVEL0),
            )
            img = region.convert("RGB").resize(
                (patch_display_px, patch_display_px),
                resample=PILImage.LANCZOS,
            )
            images.append(np.array(img))
    finally:
        sl.close()

    # ------------------------------------------------------------------
    # 5. Build figure
    # ------------------------------------------------------------------
    n_label_rows = len(label_keys)
    fig_w = figsize_w_per_patch * n_patches + 2.2   # +2.2 for label column
    top_h = figsize_w_per_patch * 1.15              # ~square patch cells + 2-line title
    heat_h = n_label_rows * 0.42
    fig_h = top_h + heat_h + 0.5

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: 2 rows — top=images, bottom=heatmap
    gs = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[top_h, heat_h],
        hspace=0.02,
    )

    # --- Top row: patch images ----------------------------------------
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, n_patches, subplot_spec=gs[0], wspace=0.04
    )

    for j in range(n_patches):
        ax = fig.add_subplot(gs_top[0, j])
        ax.imshow(images[j])
        ax.axis("off")

        # Rank + PRISM2 attention as a two-line title above the image
        ax.set_title(
            f"Rank {j}\nattn={attn_scores[j]:.2f}",
            fontsize=11, fontweight="bold", pad=3,
        )

    # --- Bottom: CONCH heatmap + PRISM2 column ------------------------
    # Look up PRISM2 slide-level scores (may be missing for some slides)
    prism2_scores = _load_prism2_scores().get(slide)
    prism2_col = None
    if prism2_scores is not None:
        prism2_col = np.array(
            [prism2_scores.get(k, np.nan) for k in label_keys], dtype=float
        )

    # Widths: CONCH heatmap = n_patches columns, PRISM2 column = 1
    prism_width = 1.0 if prism2_col is not None else 0.0
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs[1],
        width_ratios=[n_patches, prism_width] if prism2_col is not None else [1, 0.001],
        wspace=0.04,
    )
    ax_heat = fig.add_subplot(gs_bot[0, 0])

    im = ax_heat.imshow(
        score_matrix,
        aspect="auto",
        cmap=cmap_heat,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Annotate CONCH cells
    for i in range(n_label_rows):
        for j in range(n_patches):
            val = score_matrix[i, j]
            if not np.isnan(val):
                brightness = (val - vmin) / (vmax - vmin)
                text_col = "white" if brightness > 0.65 else "black"
                ax_heat.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=annot_fontsize,
                    color=text_col,
                    fontweight="bold" if val == np.nanmax(score_matrix[:, j]) else "normal",
                )

    ax_heat.set_xticks(range(n_patches))
    ax_heat.set_xticklabels(
        [f"Rank {j}" for j in range(n_patches)],
        fontsize=11,
    )
    ax_heat.set_yticks(range(n_label_rows))
    ax_heat.set_yticklabels(label_display_names, fontsize=12)
    ax_heat.xaxis.set_ticks_position("bottom")
    ax_heat.tick_params(axis="both", length=0)

    # Highlight the top-scoring label per patch (red outline)
    for j in range(n_patches):
        top_i = int(np.nanargmax(score_matrix[:, j]))
        rect = mpatches.FancyBboxPatch(
            (j - 0.5, top_i - 0.5), 1.0, 1.0,
            boxstyle="round,pad=0.05",
            linewidth=2, edgecolor="crimson", facecolor="none",
            transform=ax_heat.transData, clip_on=True,
        )
        ax_heat.add_patch(rect)

    # --- PRISM2 column (slide-level probability, 0-1) -----------------
    if prism2_col is not None:
        ax_p = fig.add_subplot(gs_bot[0, 1], sharey=ax_heat)
        # Use a distinct colormap (PRISM2 is a probability, different scale)
        prism_data = prism2_col.reshape(-1, 1)
        im_p = ax_p.imshow(
            prism_data,
            aspect="auto",
            cmap="Purples",
            vmin=0.0, vmax=1.0,
            interpolation="nearest",
        )
        for i, v in enumerate(prism2_col):
            if not np.isnan(v):
                col = "white" if v > 0.55 else "black"
                ax_p.text(0, i, f"{v:.2f}",
                          ha="center", va="center",
                          fontsize=annot_fontsize,
                          color=col,
                          fontweight="bold" if v >= 0.5 else "normal")
        ax_p.set_xticks([0])
        ax_p.set_xticklabels(["PRISM2\nslide"], fontsize=10, fontweight="bold")
        ax_p.tick_params(axis="both", length=0)
        # Hide y-tick labels on PRISM2 column (shared with heatmap)
        plt.setp(ax_p.get_yticklabels(), visible=False)
        for spine in ax_p.spines.values():
            spine.set_edgecolor("#666")

        # Two colorbars in dedicated inset axes on the right of the figure
        # Positions are figure-relative (left, bottom, width, height)
        cbar_ax_conch = fig.add_axes([0.945, 0.08, 0.010, 0.30])
        cbar_conch = fig.colorbar(im, cax=cbar_ax_conch, orientation="vertical")
        cbar_conch.ax.tick_params(labelsize=9)
        cbar_conch.set_label("CONCH cosine", fontsize=10)

        cbar_ax_p = fig.add_axes([0.983, 0.08, 0.010, 0.30])
        cbar_p = fig.colorbar(im_p, cax=cbar_ax_p, orientation="vertical")
        cbar_p.ax.tick_params(labelsize=9)
        cbar_p.set_label("PRISM2", fontsize=10)
    else:
        cbar_conch = fig.colorbar(
            im, ax=ax_heat,
            orientation="vertical",
            fraction=0.015, pad=0.01,
        )
        cbar_conch.ax.tick_params(labelsize=10)
        cbar_conch.set_label("CONCH cosine similarity", fontsize=11)

    # ------------------------------------------------------------------
    # 6. Figure title
    # ------------------------------------------------------------------
    fig.suptitle(
        f"Slide: {slide}",
        fontsize=14, fontweight="bold", y=1.005,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slide",     help="Slide stem (e.g. 10407210HE1)")
    p.add_argument("--all",       action="store_true",
                   help="Plot every slide that has CONCH results")
    p.add_argument("--n_patches", type=int, default=8)
    p.add_argument("--out",       type=Path, default=FIG_DIR,
                   help="Output directory for figures")
    p.add_argument("--fmt",       default="png",
                   help="Output format: png | pdf | svg  (default: png)")
    p.add_argument("--sort",      default="mean_score",
                   choices=["clinical", "mean_score", "max_score"],
                   help="Label row order in heatmap (default: mean_score)")
    args = p.parse_args()

    if not args.slide and not args.all:
        p.error("Specify --slide STEM or --all")

    if args.slide:
        slides = [args.slide]
    else:
        slides = [f.stem for f in sorted(CONCH_DIR.glob("*.json"))]
        print(f"Found {len(slides)} slides with CONCH results.\n")

    for i, slide in enumerate(slides, 1):
        out_path = args.out / f"{slide}_patches.{args.fmt}"
        if out_path.exists():
            print(f"[{i}/{len(slides)}] {slide} [skip — exists]")
            continue
        print(f"[{i}/{len(slides)}] {slide} ...", end=" ", flush=True)
        try:
            fig = plot_patch_scores(
                slide,
                n_patches=args.n_patches,
                sort_labels=args.sort,
                save_path=out_path,
            )
            plt.close(fig)
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone. Figures → {args.out}")


if __name__ == "__main__":
    main()
