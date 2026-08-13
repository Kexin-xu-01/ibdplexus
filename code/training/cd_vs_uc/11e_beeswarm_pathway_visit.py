"""
SHAP beeswarm plots for pathway ssGSEA models — visit-level cohort.

Arms
----
  pathway_kegg_visit      — top 25 of 320 KEGG pathways
  pathway_combined_visit  — top 25 of 370 (Hallmark + KEGG) pathways

Reads beeswarm_pathway_*.npz produced by 10d_shap_analysis_pathway_visit.py.

Output
------
  <PLOTS_DIR>/beeswarm_pathway_kegg_visit.png
  <PLOTS_DIR>/beeswarm_pathway_combined_visit.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DATA_DIR  = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/shap/plots')
PLOTS_DIR = DATA_DIR
NPZ_DIR   = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/shap/data')

INK   = '#0b0b0b'
INK2  = '#52514e'
MUTED = '#898781'
GRID  = '#e1e0d9'
SURF  = '#fcfcfb'

FEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'feat', ['#2166ac', '#f7f7f7', '#d6604d'])


def clean_name(name, max_len=42):
    """Strip library tag, truncate long names."""
    name = name.removeprefix('[H] ').removeprefix('[K] ')
    if len(name) > max_len:
        name = name[:max_len - 1] + '…'
    return name


def tag_color(name):
    """Return a subtle tag color for [H] vs [K] prefix, or None."""
    if name.startswith('[H]'):
        return '#2a78d6'   # blue — Hallmark
    if name.startswith('[K]'):
        return '#eb6834'   # orange — KEGG
    return None


def beeswarm_simple(values, y_center=0.0, bandwidth=0.38):
    n = len(values)
    if n == 0:
        return np.full(n, y_center)
    n_bins = max(20, n // 8)
    counts, edges = np.histogram(values, bins=n_bins)
    bin_idx = np.clip(np.digitize(values, edges[:-1]) - 1, 0, n_bins - 1)
    yp = np.zeros(n)
    for b in range(n_bins):
        mask = bin_idx == b
        k = mask.sum()
        if k == 0:
            continue
        spread = bandwidth * min(1.0, k / 8.0)
        positions = np.linspace(-spread, spread, k)
        np.random.shuffle(positions)
        yp[mask] = positions
    return yp + y_center


def beeswarm_plot(npz_path, out_path, title, top_n=25,
                  dot_size=4.0, fig_width=6.0, show_source_tag=False):
    data = np.load(npz_path, allow_pickle=True)
    shap_vals  = data['shap_values']
    feat_vals  = data['X']            # saved as 'X' in 10d
    feat_names = data['feature_names'].tolist()
    labels     = data['labels']

    # sort features by mean |SHAP| descending, then take top_n
    order      = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1]
    shap_vals  = shap_vals[:, order]
    feat_vals  = feat_vals[:, order]
    feat_names = [feat_names[i] for i in order]
    n = min(top_n, shap_vals.shape[1])
    shap_vals  = shap_vals[:, :n]
    feat_vals  = feat_vals[:, :n]
    feat_names = feat_names[:n]
    n_samples  = shap_vals.shape[0]

    fig_height = max(3.0, n * 0.40 + 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)

    np.random.seed(42)

    for row_idx in range(n):
        y_center = row_idx
        col = n - 1 - row_idx      # col 0 = most important → top row
        sv = shap_vals[:, col]
        fv = feat_vals[:, col]

        fmin, fmax = np.nanpercentile(fv, 1), np.nanpercentile(fv, 99)
        fv_norm = np.clip((fv - fmin) / (fmax - fmin), 0.0, 1.0) if fmax > fmin \
                  else np.full_like(fv, 0.5)

        y_jitter = beeswarm_simple(sv, y_center=y_center, bandwidth=0.38)
        colors   = FEAT_CMAP(fv_norm)
        ax.scatter(sv, y_jitter, s=dot_size ** 2 * 0.8, c=colors,
                   alpha=0.65, linewidths=0.0, zorder=3, rasterized=True)

    ax.axvline(0, color=MUTED, lw=0.7, ls='--', zorder=2)

    # y-labels: most important on top
    raw_labels = [feat_names[n - 1 - i] for i in range(n)]
    y_labels   = [clean_name(lbl) for lbl in raw_labels]

    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=7.0, color=INK, fontfamily='sans-serif')

    # colour the y-tick labels by source ([H] vs [K]) for combined plot
    if show_source_tag:
        for tick, raw_lbl in zip(ax.get_yticklabels(), raw_labels):
            tc = tag_color(raw_lbl)
            if tc:
                tick.set_color(tc)

    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel('SHAP value  (← CD  |  UC →)',
                  fontsize=7.5, color=INK2, fontfamily='sans-serif')
    ax.tick_params(axis='x', labelsize=7, colors=INK2, width=0.6, length=3)
    ax.tick_params(axis='y', length=0)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['bottom'].set_color(MUTED)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(f'{title}\nn = {n_samples} visits',
                 fontsize=8.5, fontweight='bold', color=INK,
                 fontfamily='sans-serif', loc='left', pad=6)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=FEAT_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.025, pad=0.02, aspect=35,
                        ticks=[0, 0.5, 1])
    cbar.set_ticklabels(['low', 'mid', 'high'], fontsize=6, color=INK2)
    cbar.set_label('NES (feature value)', fontsize=6.5, color=INK2,
                   fontfamily='sans-serif')
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=6)

    # source legend for combined plot
    if show_source_tag:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2a78d6',
                   markersize=5, label='Hallmark'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#eb6834',
                   markersize=5, label='KEGG'),
        ]
        ax.legend(handles=handles, title='Gene set library',
                  fontsize=6, title_fontsize=6.5,
                  frameon=True, framealpha=0.9, edgecolor=GRID,
                  loc='lower right', handletextpad=0.4, borderpad=0.6)

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=SURF, transparent=False)
    plt.close(fig)
    print(f'  Saved: {out_path}')


def beeswarm_plot_2col(npz_path, out_path, title, top_n=20,
                       dot_size=3.5, fig_width=7.08):
    """Two-column beeswarm layout — wide and short for PowerPoint."""
    data = np.load(npz_path, allow_pickle=True)
    shap_vals  = data['shap_values']
    feat_vals  = data['X']
    feat_names = data['feature_names'].tolist()
    labels     = data['labels']

    order      = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1]
    shap_vals  = shap_vals[:, order]
    feat_vals  = feat_vals[:, order]
    feat_names = [feat_names[i] for i in order]
    n          = min(top_n, shap_vals.shape[1])
    shap_vals  = shap_vals[:, :n]
    feat_vals  = feat_vals[:, :n]
    feat_names = feat_names[:n]
    n_samples  = shap_vals.shape[0]

    n_per_col = (n + 1) // 2  # left col gets extra if odd
    fig_height = max(2.5, n_per_col * 0.36 + 1.2)
    fig, axes  = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(SURF)

    # shared x range
    p1, p99 = np.nanpercentile(shap_vals, 1), np.nanpercentile(shap_vals, 99)
    pad     = (p99 - p1) * 0.12
    x_min, x_max = p1 - pad, p99 + pad

    np.random.seed(42)

    for col_idx, ax in enumerate(axes):
        ax.set_facecolor(SURF)
        start = col_idx * n_per_col
        end   = min(start + n_per_col, n)
        sv_block = shap_vals[:, start:end]
        fv_block = feat_vals[:, start:end]
        names    = feat_names[start:end]
        nc       = end - start

        for row_idx in range(nc):
            y_center = row_idx
            col      = nc - 1 - row_idx   # most important → top row
            sv = sv_block[:, col]
            fv = fv_block[:, col]

            fmin, fmax = np.nanpercentile(fv, 1), np.nanpercentile(fv, 99)
            fv_norm = np.clip((fv - fmin) / (fmax - fmin), 0, 1) \
                      if fmax > fmin else np.full_like(fv, 0.5)

            y_jitter = beeswarm_simple(sv, y_center=y_center, bandwidth=0.38)
            ax.scatter(sv, y_jitter, s=dot_size**2 * 0.8, c=FEAT_CMAP(fv_norm),
                       alpha=0.65, linewidths=0.0, zorder=3, rasterized=True)

        ax.axvline(0, color=MUTED, lw=0.7, ls='--', zorder=2)

        raw_labels = [names[nc - 1 - i] for i in range(nc)]
        ax.set_yticks(range(nc))
        ax.set_yticklabels([clean_name(l) for l in raw_labels],
                           fontsize=7.0, color=INK, fontfamily='sans-serif')
        ax.set_ylim(-0.6, nc - 0.4)
        ax.set_xlim(x_min, x_max)

        ax.set_xlabel('SHAP value  (← CD  |  UC →)',
                      fontsize=6.5, color=INK2, fontfamily='sans-serif')
        ax.tick_params(axis='x', labelsize=6, colors=INK2, width=0.6, length=3)
        ax.tick_params(axis='y', length=0)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.spines['bottom'].set_color(MUTED)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_title(f'{title}\nn = {n_samples} visits',
                      fontsize=8.0, fontweight='bold', color=INK,
                      fontfamily='sans-serif', loc='left', pad=5)

    sm   = plt.cm.ScalarMappable(cmap=FEAT_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], orientation='vertical',
                        fraction=0.06, pad=0.04, aspect=30, ticks=[0, 0.5, 1])
    cbar.set_ticklabels(['low', 'mid', 'high'], fontsize=6, color=INK2)
    cbar.set_label('NES (feature value)', fontsize=6.5, color=INK2,
                   fontfamily='sans-serif')
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=6)

    plt.tight_layout(pad=0.5, w_pad=1.5)
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=SURF, transparent=False)
    plt.close(fig)
    print(f'  Saved: {out_path}')


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    np.random.seed(42)

    arms = [
        (
            os.path.join(NPZ_DIR,   'beeswarm_pathway_hallmark_visit.npz'),
            os.path.join(PLOTS_DIR, 'beeswarm_pathway_hallmark_visit.png'),
            'Hallmark pathway scores — SHAP beeswarm (all 50)\npathway_hallmark_visit  ·  AUC 0.729',
            50, False,
        ),
        (
            os.path.join(NPZ_DIR,   'beeswarm_pathway_kegg_visit.npz'),
            os.path.join(PLOTS_DIR, 'beeswarm_pathway_kegg_visit.png'),
            'KEGG pathway scores — SHAP beeswarm (top 25)\npathway_kegg_visit  ·  AUC 0.759',
            25, False,
        ),
        (
            os.path.join(NPZ_DIR,   'beeswarm_pathway_combined_visit.npz'),
            os.path.join(PLOTS_DIR, 'beeswarm_pathway_combined_visit.png'),
            'Hallmark + KEGG pathway scores — SHAP beeswarm (top 25)\npathway_combined_visit  ·  AUC 0.763',
            25, True,
        ),
    ]

    for npz_path, out_path, title, top_n, tag in arms:
        if not os.path.exists(npz_path):
            print(f'  MISSING: {npz_path}  (re-run 10d first)')
            continue
        print(f'\nPlotting {os.path.basename(npz_path)} ...')
        beeswarm_plot(npz_path, out_path, title, top_n=top_n, show_source_tag=tag)

    # PowerPoint-friendly 2-column version: Hallmark top 20
    hall_npz = os.path.join(NPZ_DIR, 'beeswarm_pathway_hallmark_visit.npz')
    if os.path.exists(hall_npz):
        print('\nPlotting 2-column Hallmark (top 20, PPT layout) ...')
        beeswarm_plot_2col(
            hall_npz,
            os.path.join(PLOTS_DIR, 'beeswarm_pathway_hallmark_visit_ppt.png'),
            'Hallmark pathway scores — SHAP beeswarm (top 20)\npathway_hallmark_visit  ·  AUC 0.729',
            top_n=20,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
