"""
Comparison bar charts: AUC and F1 across all at-20-cm RF arms + proteomics.

Sources
-------
  at20cm_visit_manual_knn/results/rf/at20cm_rf_fold_metrics.csv    (10 RF arms)
  at20cm_proteomics/results/at20cm_proteomics_fold_metrics.csv     (proteomics arm)

Output
------
  at20cm_proteomics/plots/comparison_auc_f1_proteomics.png
  at20cm_proteomics/plots/comparison_auc_f1_proteomics.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ──────────────────────────────────────────────────────────────────────
RF_METRICS    = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                 'at20cm_visit_manual_knn/results/rf/at20cm_rf_fold_metrics.csv')
PROT_METRICS  = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                 'at20cm_proteomics/results/at20cm_proteomics_fold_metrics.csv')
OUT_DIR       = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                 'at20cm_proteomics/plots')

# ── style ──────────────────────────────────────────────────────────────────────
SURF     = '#fcfcfb'
INK      = '#0b0b0b'
INK2     = '#52514e'
MUTED    = '#898781'
GRID     = '#e1e0d9'
UC_COLOR = '#c94040'
CD_COLOR = '#2a78d6'

# arm colours: imaging=teal, RNA=purple, fusion=dark-green, proteomics=amber
ARM_PALETTE = {
    'img_base_visit':          '#2a9d8f',
    'img_histoscore_visit':    '#57b4a8',
    'bulkformer_visit':        '#8b6fbf',
    'rna_visit':               '#5e4fa2',
    'pca640_rna_visit':        '#7b6db5',
    'concat_raw_visit':        '#264653',
    'concat_histoscore_visit': '#386641',
    'concat_bf_base_visit':    '#1b4332',
    'concat_bf_histo_visit':   '#40916c',
    'concat_pca640_base':      '#1a3a2a',
    'proteomics_visit':        '#e07b39',   # amber — highlighted
}

DISPLAY = {
    'img_base_visit':          'Imaging\n(prism2_base)',
    'img_histoscore_visit':    'Histological\nscores',
    'bulkformer_visit':        'BulkFormer',
    'rna_visit':               'RNA-seq',
    'pca640_rna_visit':        'RNA-seq\nPCA-640',
    'concat_raw_visit':        'RNA +\nImaging',
    'concat_histoscore_visit': 'RNA +\nHistoscore',
    'concat_bf_base_visit':    'BulkFormer\n+ Imaging',
    'concat_bf_histo_visit':   'BulkFormer\n+ Histoscore',
    'concat_pca640_base':      'PCA-RNA\n+ Imaging',
    'proteomics_visit':        'Proteomics\n(Olink)',
}

# ── data ───────────────────────────────────────────────────────────────────────

def load_metrics():
    rf   = pd.read_csv(RF_METRICS)
    prot = pd.read_csv(PROT_METRICS)
    all_df = pd.concat([rf, prot], ignore_index=True)

    rows = []
    for strategy, grp in all_df.groupby('strategy'):
        rows.append({
            'strategy':  strategy,
            'mean_auc':  grp['auc'].mean(),
            'std_auc':   grp['auc'].std(ddof=1),
            'mean_f1cd': grp['cd_f1'].mean(),
            'std_f1cd':  grp['cd_f1'].std(ddof=1),
            'mean_f1uc': grp['uc_f1'].mean(),
            'std_f1uc':  grp['uc_f1'].std(ddof=1),
            'mean_acc':  grp['accuracy'].mean(),
            'std_acc':   grp['accuracy'].std(ddof=1),
            'folds':     grp[['fold','auc','cd_f1','uc_f1']].to_dict('records'),
        })
    return pd.DataFrame(rows).sort_values('mean_auc', ascending=True).reset_index(drop=True)


# ── plot ───────────────────────────────────────────────────────────────────────

def make_comparison_figure(df):
    n = len(df)
    # PPT-friendly: wide 16:9 aspect, height fits max ~7.0" in a slide
    fig_h = min(7.0, max(4.5, n * 0.45 + 1.6))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, fig_h))
    fig.patch.set_facecolor(SURF)

    for ax in axes:
        ax.set_facecolor(SURF)

    y_pos   = np.arange(n)
    labels  = [DISPLAY.get(s, s) for s in df['strategy']]
    colors  = [ARM_PALETTE.get(s, MUTED) for s in df['strategy']]
    is_prot = [s == 'proteomics_visit' for s in df['strategy']]

    # ── panel 1: AUC ──────────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.barh(y_pos, df['mean_auc'], xerr=df['std_auc'],
                   color=colors, alpha=0.88, height=0.62,
                   error_kw=dict(elinewidth=0.8, capsize=2.5,
                                 ecolor=MUTED, capthick=0.8),
                   zorder=3)
    # outline proteomics bar
    for i, (bar, is_p) in enumerate(zip(bars, is_prot)):
        if is_p:
            bar.set_linewidth(1.8)
            bar.set_edgecolor(INK)
    # value labels
    for i, row in df.iterrows():
        ax.text(row['mean_auc'] + row['std_auc'] + 0.004,
                y_pos[i], f"{row['mean_auc']:.3f}",
                va='center', ha='left', fontsize=6.5, color=INK2,
                fontfamily='sans-serif')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5, color=INK, fontfamily='sans-serif')
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel('Mean AUC (5-fold CV)', fontsize=8, color=INK2,
                  fontfamily='sans-serif')
    ax.axvline(0.5, color=MUTED, lw=0.5, ls=':', zorder=1)
    ax.set_title('AUC', fontsize=9, fontweight='bold', color=INK,
                 fontfamily='sans-serif', loc='left', pad=5)
    _style_ax(ax)

    # ── panel 2: F1-CD ────────────────────────────────────────────────────────
    ax = axes[1]
    bars = ax.barh(y_pos, df['mean_f1cd'], xerr=df['std_f1cd'],
                   color=colors, alpha=0.88, height=0.62,
                   error_kw=dict(elinewidth=0.8, capsize=2.5,
                                 ecolor=MUTED, capthick=0.8),
                   zorder=3)
    for i, (bar, is_p) in enumerate(zip(bars, is_prot)):
        if is_p:
            bar.set_linewidth(1.8)
            bar.set_edgecolor(INK)
    for i, row in df.iterrows():
        ax.text(row['mean_f1cd'] + row['std_f1cd'] + 0.004,
                y_pos[i], f"{row['mean_f1cd']:.3f}",
                va='center', ha='left', fontsize=6.5, color=INK2,
                fontfamily='sans-serif')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel('Mean F1-CD (5-fold CV)', fontsize=8, color=INK2,
                  fontfamily='sans-serif')
    ax.axvline(0.5, color=MUTED, lw=0.5, ls=':', zorder=1)
    ax.set_title('F1 — CD', fontsize=9, fontweight='bold', color=CD_COLOR,
                 fontfamily='sans-serif', loc='left', pad=5)
    _style_ax(ax)

    # ── panel 3: F1-UC ────────────────────────────────────────────────────────
    ax = axes[2]
    bars = ax.barh(y_pos, df['mean_f1uc'], xerr=df['std_f1uc'],
                   color=colors, alpha=0.88, height=0.62,
                   error_kw=dict(elinewidth=0.8, capsize=2.5,
                                 ecolor=MUTED, capthick=0.8),
                   zorder=3)
    for i, (bar, is_p) in enumerate(zip(bars, is_prot)):
        if is_p:
            bar.set_linewidth(1.8)
            bar.set_edgecolor(INK)
    for i, row in df.iterrows():
        ax.text(row['mean_f1uc'] + row['std_f1uc'] + 0.004,
                y_pos[i], f"{row['mean_f1uc']:.3f}",
                va='center', ha='left', fontsize=6.5, color=INK2,
                fontfamily='sans-serif')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlim(-0.02, 1.05)
    ax.set_xlabel('Mean F1-UC (5-fold CV)', fontsize=8, color=INK2,
                  fontfamily='sans-serif')
    ax.axvline(0.0, color=MUTED, lw=0.5, ls=':', zorder=1)
    ax.set_title('F1 — UC', fontsize=9, fontweight='bold', color=UC_COLOR,
                 fontfamily='sans-serif', loc='left', pad=5)
    _style_ax(ax)

    # ── legend ─────────────────────────────────────────────────────────────────
    groups = [
        ('Imaging',          '#2a9d8f'),
        ('RNA / BulkFormer', '#5e4fa2'),
        ('Fusion',           '#264653'),
        ('Proteomics (Olink)','#e07b39'),
    ]
    patches = [mpatches.Patch(facecolor=c, label=l, alpha=0.88)
               for l, c in groups]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=7.5, frameon=False, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        'CD vs UC — RF classifier: AUC and F1 across modalities  (at-20-cm, visit-level)',
        fontsize=10, fontweight='bold', color=INK, fontfamily='sans-serif', y=1.01)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.90,
                        bottom=0.14, wspace=0.06)
    return fig


def make_fold_scatter(df):
    """Strip-plot showing per-fold AUC for each arm, with mean ± SD band."""
    n   = len(df)
    fig, ax = plt.subplots(figsize=(13.0, min(7.0, max(4.5, n * 0.45 + 1.6))))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)

    y_pos  = np.arange(n)
    labels = [DISPLAY.get(s, s) for s in df['strategy']]
    colors = [ARM_PALETTE.get(s, MUTED) for s in df['strategy']]
    np.random.seed(42)

    for i, (row, col) in enumerate(zip(df.itertuples(), colors)):
        fold_aucs = [f['auc'] for f in row.folds]
        jitter = np.random.uniform(-0.18, 0.18, len(fold_aucs))
        # SD band
        ax.barh(i, row.std_auc * 2, left=row.mean_auc - row.std_auc,
                color=col, alpha=0.18, height=0.55, zorder=2)
        # mean line
        ax.plot([row.mean_auc, row.mean_auc], [i - 0.28, i + 0.28],
                color=col, lw=1.8, zorder=4)
        # fold dots
        ax.scatter(fold_aucs, i + jitter,
                   s=22, color=col, alpha=0.8, zorder=5,
                   edgecolors='white', linewidths=0.4)
        # label
        ax.text(max(fold_aucs) + 0.005, i,
                f"{row.mean_auc:.3f}±{row.std_auc:.3f}",
                va='center', ha='left', fontsize=6.2, color=INK2,
                fontfamily='sans-serif')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8, color=INK, fontfamily='sans-serif')
    ax.set_xlim(0.45, 1.06)
    ax.axvline(0.5, color=MUTED, lw=0.5, ls=':', zorder=1)
    ax.set_xlabel('AUC (per fold + mean ± SD)', fontsize=8.5, color=INK2,
                  fontfamily='sans-serif')
    ax.set_title(
        'Per-fold AUC — all arms  (RF, at-20-cm, visit-level)',
        fontsize=9.5, fontweight='bold', color=INK,
        fontfamily='sans-serif', loc='left', pad=6)
    _style_ax(ax)
    plt.tight_layout()
    return fig


def _style_ax(ax):
    ax.tick_params(axis='x', labelsize=7, colors=INK2, width=0.6, length=3)
    ax.tick_params(axis='y', length=0)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['bottom'].set_color(MUTED)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['left'].set_color(MUTED)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading metrics ...')
    df = load_metrics()

    print(f'\n{"Arm":<30} {"AUC":>10} {"F1-CD":>10} {"F1-UC":>10}')
    print('-' * 62)
    for _, row in df.sort_values('mean_auc', ascending=False).iterrows():
        marker = '  ◀' if row['strategy'] == 'proteomics_visit' else ''
        print(f"  {row['strategy']:<28} "
              f"{row['mean_auc']:.4f}±{row['std_auc']:.4f}  "
              f"{row['mean_f1cd']:.4f}±{row['std_f1cd']:.4f}  "
              f"{row['mean_f1uc']:.4f}±{row['std_f1uc']:.4f}{marker}")

    # main comparison figure (3 panels)
    out_png = os.path.join(OUT_DIR, 'comparison_auc_f1_proteomics.png')
    out_pdf = os.path.join(OUT_DIR, 'comparison_auc_f1_proteomics.pdf')
    fig = make_comparison_figure(df)
    fig.savefig(out_png, dpi=200, bbox_inches='tight', facecolor=SURF)
    fig.savefig(out_pdf,           bbox_inches='tight', facecolor=SURF)
    plt.close(fig)
    print(f'\n  Saved: {out_png}')

    # fold scatter
    out_sc_png = os.path.join(OUT_DIR, 'fold_scatter_auc_proteomics.png')
    out_sc_pdf = os.path.join(OUT_DIR, 'fold_scatter_auc_proteomics.pdf')
    fig2 = make_fold_scatter(df)
    fig2.savefig(out_sc_png, dpi=200, bbox_inches='tight', facecolor=SURF)
    fig2.savefig(out_sc_pdf,           bbox_inches='tight', facecolor=SURF)
    plt.close(fig2)
    print(f'  Saved: {out_sc_png}')
    print(f'\nAll outputs in {OUT_DIR}/')


if __name__ == '__main__':
    main()
