"""
SHAP beeswarm plots for the concept_learning histoscore series.

For each arm the script loads per-sample SHAP values and feature values saved by
10c (beeswarm_*.npz), then draws a Nature-style beeswarm summary plot:

  • Y-axis  — features ordered by mean|SHAP| (most important on top)
  • X-axis  — SHAP value (positive → UC, negative → CD)
  • Dot colour — feature value (blue = low, red = high, normalized per feature)
  • Dot jitter  — beeswarm kernel (no overlap along Y)

Arms produced
-------------
  beeswarm_img_histoscore.png   — 11 histological scores
  beeswarm_rna_histoscore.png   — top 50 RNA genes
  beeswarm_concat_histoscore.png — top 50 concat features

All outputs go to PLOTS_DIR.

Usage
-----
    python 11d_beeswarm_histoscore.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/shap/data')
PLOTS_DIR = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/shap/plots')

# symbol lookup for ENSG IDs
VST_GCT   = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/'
             'transcriptomics/GSF1491805_CombatSeq_vst_mtx_batch_corrected_'
             'alltissues_all3releases_header.gct')

# ── style constants ────────────────────────────────────────────────────────────
UC_COLOR  = '#c94040'   # red  — positive SHAP → UC
CD_COLOR  = '#2a78d6'   # blue — negative SHAP → CD
INK       = '#0b0b0b'
INK2      = '#52514e'
MUTED     = '#898781'
GRID      = '#e1e0d9'
SURF      = '#fcfcfb'

# colormap: blue (low feature value) → red (high feature value)
FEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'feat', ['#2166ac', '#f7f7f7', '#d6604d'])


# ── helpers ────────────────────────────────────────────────────────────────────

def load_ensg_symbol_map():
    try:
        gmap = pd.read_csv(VST_GCT, sep='\t', skiprows=2, usecols=['Name', 'Description'])
        return dict(zip(gmap['Name'], gmap['Description']))
    except Exception:
        return {}


def display_name(feature, ensg_map):
    """Return gene symbol if available, else strip 'histo_' prefix."""
    if feature in ensg_map:
        return ensg_map[feature]
    if feature.startswith('histo_'):
        return feature[6:].replace('_', ' ')
    return feature


def beeswarm_jitter(values, dot_size_pts=3.5, ax_height_pts=None,
                    y_center=0.0, max_iter=50):
    """
    1-D beeswarm: place dots along the y-axis so they do not overlap.

    Parameters
    ----------
    values : 1-D array, the SHAP values (x-positions in data space)
    dot_size_pts : diameter of each dot in points
    ax_height_pts : height of the full axes in points (used to convert y extent)
    y_center : center row position in data coordinates

    Returns
    -------
    y_pos : 1-D array of y-positions (same length as values)
    """
    if ax_height_pts is None:
        ax_height_pts = 72.0   # 1-inch default
    n = len(values)
    if n == 0:
        return np.array([])

    order = np.argsort(values)
    inv   = np.argsort(order)

    # work in sorted order
    sv = values[order]
    yp = np.zeros(n)

    # radius in data-y units: half dot size
    dot_radius_pts = dot_size_pts / 2.0

    # for each dot, pack it as close to y=0 as possible without overlap
    placed_x = [sv[0]]
    placed_y = [0.0]
    yp[0] = 0.0

    for i in range(1, n):
        xi = sv[i]
        candidate_ys = [0.0]
        # try above and below existing dots
        for j in range(len(placed_x)):
            dx = xi - placed_x[j]
            # minimum y separation given x overlap
            if abs(dx) < dot_size_pts * 1.1:   # rough unit — we work in indices
                candidate_ys += [placed_y[j] + dot_size_pts * 1.1,
                                  placed_y[j] - dot_size_pts * 1.1]

        best_y = None
        best_dist = np.inf
        for cy in sorted(set(candidate_ys), key=abs):
            # check no overlap with already placed
            ok = True
            for j in range(len(placed_x)):
                dx = xi - placed_x[j]
                dy = cy - placed_y[j]
                if (dx**2 + dy**2) < (dot_size_pts * 1.0)**2:
                    ok = False
                    break
            if ok and abs(cy) < best_dist:
                best_dist = abs(cy)
                best_y = cy
        if best_y is None:
            best_y = cy  # fallback: use last candidate

        placed_x.append(xi)
        placed_y.append(best_y)
        yp[i] = best_y

    # normalise y so the spread is ± 0.4 units in data space
    scale = placed_y
    if max(abs(np.array(scale) + 1e-9)) > 0:
        spread = max(abs(np.array(scale)))
        if spread > 0:
            yp = yp / spread * 0.42

    return yp[inv] + y_center


def beeswarm_simple(values, y_center=0.0, bandwidth=0.38):
    """
    Fast approximate beeswarm using histogram-style binning.
    Sufficient for 945 samples.
    """
    n = len(values)
    if n == 0:
        return np.full(n, y_center)

    # bin values
    n_bins = max(20, n // 8)
    counts, edges = np.histogram(values, bins=n_bins)
    bin_idx = np.digitize(values, edges[:-1]) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    yp = np.zeros(n)
    for b in range(n_bins):
        mask = bin_idx == b
        k = mask.sum()
        if k == 0:
            continue
        if k == 1:
            yp[mask] = 0.0
        else:
            # evenly space the k dots, centred at 0
            spread = bandwidth * min(1.0, k / 8.0)
            positions = np.linspace(-spread, spread, k)
            np.random.shuffle(positions)
            yp[mask] = positions

    return yp + y_center


# ── main plot function ─────────────────────────────────────────────────────────

def beeswarm_plot(npz_path, out_path, title, ensg_map,
                  top_n=None, dot_size=4.0, fig_width=5.2):
    """
    Draw a beeswarm SHAP summary plot from an .npz file.

    npz keys expected: shap_values, feature_values, feature_names, labels
    """
    data = np.load(npz_path, allow_pickle=True)
    shap_vals   = data['shap_values']     # (n_samples, n_features)
    feat_vals   = data['feature_values']  # (n_samples, n_features)
    feat_names  = data['feature_names'].tolist()
    labels      = data['labels']          # (n_samples,)

    n_samples, n_features = shap_vals.shape
    if top_n is not None:
        n_features = min(top_n, n_features)
        shap_vals  = shap_vals[:, :n_features]
        feat_vals  = feat_vals[:, :n_features]
        feat_names = feat_names[:n_features]

    # features already in descending mean|SHAP| order (from 10c)
    # — plot bottom-to-top so most important is at top
    n = n_features
    fig_height = max(2.8, n * 0.38 + 1.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)

    np.random.seed(42)

    for row_idx in range(n):
        # y position: 0 = least important (bottom), n-1 = most important (top)
        y_center = row_idx   # bottom=0 → top=n-1

        sv = shap_vals[:, (n - 1 - row_idx)]   # reverse: col 0 = most important → top row
        fv = feat_vals[:, (n - 1 - row_idx)]

        # normalise feature values to [0, 1] for coloring
        fmin, fmax = np.nanpercentile(fv, 1), np.nanpercentile(fv, 99)
        if fmax > fmin:
            fv_norm = np.clip((fv - fmin) / (fmax - fmin), 0.0, 1.0)
        else:
            fv_norm = np.full_like(fv, 0.5)

        y_jitter = beeswarm_simple(sv, y_center=y_center, bandwidth=0.38)

        colors = FEAT_CMAP(fv_norm)
        ax.scatter(sv, y_jitter, s=dot_size ** 2 * 0.8, c=colors,
                   alpha=0.65, linewidths=0.0, zorder=3, rasterized=True)

    # zero line
    ax.axvline(0, color=MUTED, lw=0.7, ls='--', zorder=2)

    # y-axis: feature names (most important on top = n-1, least = 0)
    y_labels = [display_name(feat_names[n - 1 - i], ensg_map) for i in range(n)]
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=7.0, color=INK, fontfamily='sans-serif')
    ax.set_ylim(-0.6, n - 0.4)

    ax.set_xlabel('SHAP value  (← CD  |  UC →)',
                  fontsize=7.5, color=INK2, fontfamily='sans-serif')
    ax.tick_params(axis='x', labelsize=7, colors=INK2, width=0.6, length=3)
    ax.tick_params(axis='y', length=0)
    ax.spines['bottom'].set_linewidth(0.6); ax.spines['bottom'].set_color(MUTED)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(f'{title}\nn = {n_samples} visits',
                 fontsize=8.5, fontweight='bold', color=INK,
                 fontfamily='sans-serif', loc='left', pad=6)

    # colorbar for feature value
    sm = plt.cm.ScalarMappable(cmap=FEAT_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.03, pad=0.02, aspect=30,
                        ticks=[0, 0.5, 1])
    cbar.set_ticklabels(['low', 'mid', 'high'], fontsize=6, color=INK2)
    cbar.set_label('Feature value', fontsize=6.5, color=INK2,
                   fontfamily='sans-serif')
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=6)


    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=SURF, transparent=False)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    np.random.seed(42)

    print('Loading gene symbol map ...')
    ensg_map = load_ensg_symbol_map()
    print(f'  {len(ensg_map)} ENSG → symbol entries loaded')

    arms = [
        (
            os.path.join(DATA_DIR, 'beeswarm_img_histoscore.npz'),
            os.path.join(PLOTS_DIR, 'beeswarm_img_histoscore.png'),
            'Histological scores — SHAP beeswarm\nimg_histoscore_visit  ·  AUC 0.733',
            None,
        ),
        (
            os.path.join(DATA_DIR, 'beeswarm_rna_histoscore.npz'),
            os.path.join(PLOTS_DIR, 'beeswarm_rna_histoscore.png'),
            'RNA-seq — SHAP beeswarm (top 30 genes)\nrna_visit  ·  AUC 0.816',
            30,
        ),
        (
            os.path.join(DATA_DIR, 'beeswarm_concat_histoscore.npz'),
            os.path.join(PLOTS_DIR, 'beeswarm_concat_histoscore.png'),
            'RNA + histological scores — SHAP beeswarm (top 30 features)\nconcat_histoscore_visit  ·  AUC 0.821',
            30,
        ),
    ]

    for npz_path, out_path, title, top_n in arms:
        if not os.path.exists(npz_path):
            print(f'  MISSING: {npz_path}  (re-run 10c first)')
            continue
        print(f'\nPlotting {os.path.basename(npz_path)} ...')
        beeswarm_plot(npz_path, out_path, title, ensg_map, top_n=top_n)

    print('\nDone.')


if __name__ == '__main__':
    main()
