"""
QC plots for trident patch extraction.

Produces two figures saved to training/qc/:
  1. patch_counts_<config>.png        — histogram of patches per slide
  2. low_patch_count_slides_<config>.png — thumbnail grid of slides below LOW_THRESH

Usage:
    python qc_patch_counts.py [patch_dir] [viz_dir] [qc_out_dir]

Defaults to the 20x_224px_0px_overlap patch set.
"""

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image

PATCH_DIR = Path(
    "/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed"
    "/20x_224px_0px_overlap/patches"
)
VIZ_DIR = Path(
    "/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed"
    "/20x_224px_0px_overlap/visualization"
)
QC_OUT_DIR = Path("/home/jovyan/kgbk271-ibd-volume/training/qc")
LOW_THRESH = 50   # patches; slides below this appear in the thumbnail grid

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def collect_patch_counts(patch_dir: Path) -> dict:
    counts = {}
    for f in sorted(patch_dir.glob("*_patches.h5")):
        try:
            with h5py.File(f, "r") as h:
                counts[f.stem.replace("_patches", "")] = h["coords"].shape[0]
        except Exception as e:
            print(f"Warning: could not read {f.name}: {e}")
    return counts


def plot_histogram(vals: np.ndarray, output_path: Path) -> None:
    mean = vals.mean()
    med = float(np.median(vals))
    stdev = vals.std()

    plt.rcParams.update(NATURE_RC)

    bar_color  = "#4472C4"
    bar_edge   = "#2E5090"
    mean_color = "#C0392B"
    med_color  = "#E67E22"

    # Nature double-column width: 7.08 in
    fig, ax = plt.subplots(figsize=(7.08, 3.5), facecolor="white")
    ax.set_facecolor("white")

    bins = np.arange(0, vals.max() + 51, 50)
    counts_hist, bin_edges = np.histogram(vals, bins=bins)
    ax.bar(
        bin_edges[:-1], counts_hist,
        width=np.diff(bin_edges),
        align="edge",
        color=bar_color,
        edgecolor=bar_edge,
        linewidth=0.5,
        zorder=3,
    )

    ax.axvline(mean, color=mean_color, linewidth=1.2, linestyle="--", zorder=4,
               label=f"Mean ({mean:.0f})")
    ax.axvline(med,  color=med_color,  linewidth=1.2, linestyle=":",  zorder=4,
               label=f"Median ({med:.0f})")

    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.5, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    ax.tick_params(axis="both", colors="#333333", direction="out")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    ax.set_xlabel("Patches per slide", labelpad=4)
    ax.set_ylabel("Number of slides", labelpad=4)
    ax.set_title(
        f"Patch count distribution  (n = {len(vals):,} slides, "
        f"range {vals.min()}–{vals.max()}, SD {stdev:.0f})",
        pad=6, loc="left",
    )

    ax.legend(frameon=False, loc="upper right", handlelength=1.6,
              handletextpad=0.5, borderpad=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", transparent=False)
    print(f"Saved → {output_path}")


def plot_low_patch_slides(
    counts: dict,
    viz_dir: Path,
    output_path: Path,
    thresh: int = LOW_THRESH,
    ncols: int = 6,
) -> None:
    plt.rcParams.update(NATURE_RC)

    low = sorted(
        [(slide, n) for slide, n in counts.items() if n < thresh],
        key=lambda x: x[1],
    )
    if not low:
        print(f"No slides below {thresh} patches — skipping thumbnail grid.")
        return

    nrows = (len(low) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 1.4, nrows * 1.6 + 0.5),
        facecolor="white",
    )
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, (slide, n) in enumerate(low):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        img_path = viz_dir / f"{slide}.jpg"
        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img)
        else:
            ax.set_facecolor("#EEEEEE")
            ax.text(0.5, 0.5, "no image", ha="center", va="center",
                    transform=ax.transAxes, fontsize=6, color="#999999")
        ax.set_title(f"{slide}\n{n} patches", fontsize=5.5, pad=2,
                     color="#C0392B" if n < 25 else "#333333")
        ax.axis("off")

    # hide unused axes
    for idx in range(len(low), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Slides with < {thresh} patches  (n = {len(low)})",
        fontsize=9, fontweight="bold", y=1.01, x=0.02, ha="left",
    )
    fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", transparent=False)
    print(f"Saved → {output_path}")


def main():
    patch_dir  = Path(sys.argv[1]) if len(sys.argv) > 1 else PATCH_DIR
    viz_dir    = Path(sys.argv[2]) if len(sys.argv) > 2 else VIZ_DIR
    qc_out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else QC_OUT_DIR

    config = patch_dir.parents[0].name   # e.g. 20x_224px_0px_overlap

    counts = collect_patch_counts(patch_dir)
    if not counts:
        print("No patch files found.")
        sys.exit(1)

    vals = np.array(list(counts.values()))
    print(f"Slides: {len(vals):,}  min={vals.min()}  max={vals.max()}  "
          f"mean={vals.mean():.0f}  median={np.median(vals):.0f}")

    plot_histogram(vals, qc_out_dir / f"patch_counts_{config}.png")
    plot_low_patch_slides(
        counts, viz_dir,
        qc_out_dir / f"low_patch_count_slides_{config}.png",
    )


if __name__ == "__main__":
    main()
