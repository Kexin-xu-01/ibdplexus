"""
Single-modality comparison: Clinical · Proteomics · RNA · Prism2 (imaging).

All arms are RF classifiers, visit-level, at-20-cm patient splits.

Sources
-------
  clinical     multimodal/at20cm_multimodal_fold_metrics.csv  → clinical_visit_rf
  rna          rf/at20cm_rf_fold_metrics.csv                  → rna_visit
  img          rf/at20cm_rf_fold_metrics.csv                  → img_base_visit
  proteomics   at20cm_proteomics/results/…fold_metrics.csv    → proteomics_visit

Output
------
  at20cm_proteomics/plots/single_modality_comparison.png  (.pdf)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc'
# All four arms use the same 605 patients and same fold assignments (matched cohort)
MATCHED = f'{BASE}/at20cm_proteomics/results/matched_cohort_fold_metrics.csv'
SOURCES = {
    'clinical':   (MATCHED, 'clinical_matched'),
    'rna':        (MATCHED, 'rna_matched'),
    'imaging':    (MATCHED, 'imaging_matched'),
    'proteomics': (MATCHED, 'proteomics_matched'),
}
OUT_DIR = f'{BASE}/at20cm_proteomics/plots'

# ── style ──────────────────────────────────────────────────────────────────────
SURF  = '#fcfcfb'
INK   = '#0b0b0b'
INK2  = '#52514e'
MUTED = '#898781'
GRID  = '#e1e0d9'

MODALITY_COLOR = {
    'clinical':   '#b07d3f',   # warm brown
    'proteomics': '#e07b39',   # amber
    'rna':        '#5e4fa2',   # purple
    'imaging':    '#2a9d8f',   # teal
}
MODALITY_LABEL = {
    'clinical':   'Clinical',
    'proteomics': 'Proteomics\n(Olink plasma)',
    'rna':        'RNA-seq\n(VST)',
    'imaging':    'Prism2\n(whole-slide)',
}

# ── load ───────────────────────────────────────────────────────────────────────

def load():
    rows = []
    for modality, (path, strategy) in SOURCES.items():
        df = pd.read_csv(path)
        df = df[df['strategy'] == strategy].copy()
        rows.append({
            'modality':  modality,
            'strategy':  strategy,
            'fold_auc':  df['auc'].tolist(),
            'fold_f1cd': df['cd_f1'].tolist(),
            'fold_f1uc': df['uc_f1'].tolist(),
            'fold_acc':  df['accuracy'].tolist(),
            'mean_auc':  df['auc'].mean(),
            'std_auc':   df['auc'].std(ddof=1),
            'mean_f1cd': df['cd_f1'].mean(),
            'std_f1cd':  df['cd_f1'].std(ddof=1),
            'mean_f1uc': df['uc_f1'].mean(),
            'std_f1uc':  df['uc_f1'].std(ddof=1),
            'mean_acc':  df['accuracy'].mean(),
            'std_acc':   df['accuracy'].std(ddof=1),
        })
    return pd.DataFrame(rows).sort_values('mean_auc').reset_index(drop=True)


# ── figure ─────────────────────────────────────────────────────────────────────

def plot(df):
    np.random.seed(42)
    n = len(df)
    y = np.arange(n)

    # PPT-friendly wide (16:9)
    fig = plt.figure(figsize=(13.0, 6.0))
    fig.patch.set_facecolor(SURF)

    # 3 metric panels + 1 fold-scatter panel (top) + table (bottom)
    gs = fig.add_gridspec(2, 4, wspace=0.08, hspace=0.55,
                          left=0.13, right=0.985, top=0.88, bottom=0.06,
                          height_ratios=[3.2, 1.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax in axes:
        ax.set_facecolor(SURF)

    metrics = [
        ('mean_auc',  'std_auc',  'fold_auc',  'AUC',    (0.40, 1.02)),
        ('mean_f1cd', 'std_f1cd', 'fold_f1cd', 'F1 — CD', (0.40, 1.02)),
        ('mean_f1uc', 'std_f1uc', 'fold_f1uc', 'F1 — UC', (-0.02, 1.02)),
    ]

    for col, (mean_k, std_k, fold_k, title, xlim) in enumerate(metrics):
        ax = axes[col]
        for i, row in df.iterrows():
            color = MODALITY_COLOR[row['modality']]
            mean  = row[mean_k]
            std   = row[std_k]

            # bar
            ax.barh(i, mean, height=0.55, color=color, alpha=0.82,
                    zorder=2, left=0)
            # error whisker
            ax.errorbar(mean, i, xerr=std, fmt='none',
                        elinewidth=1.1, capsize=3.5, capthick=1.1,
                        ecolor=INK2, zorder=4)
            # fold dots
            jitter = np.random.uniform(-0.16, 0.16, 5)
            ax.scatter(row[fold_k], i + jitter,
                       s=18, color='white', edgecolors=color,
                       linewidths=0.9, zorder=5, alpha=0.95)
            # value label on last panel only to avoid clutter
            if col == 0:
                ax.text(-0.01, i, f'{mean:.3f}',
                        va='center', ha='right', fontsize=7.5,
                        color=color, fontweight='bold',
                        fontfamily='sans-serif')

        ax.set_xlim(xlim)
        ax.set_ylim(-0.55, n - 0.45)
        ax.axvline(xlim[0] + 0.01, color=MUTED, lw=0.4, zorder=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        ax.tick_params(axis='x', labelsize=7, colors=INK2, width=0.5, length=2.5)
        ax.tick_params(axis='y', length=0)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['bottom'].set_color(MUTED)
        ax.spines[['top', 'right']].set_visible(False)

        if col == 0:
            ax.set_yticklabels(
                [MODALITY_LABEL[row['modality']] for _, row in df.iterrows()],
                fontsize=8.5, color=INK, fontfamily='sans-serif')
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['left'].set_color(MUTED)
        else:
            ax.set_yticklabels([])
            ax.spines['left'].set_visible(False)

        # title coloured by direction
        title_color = {'AUC': INK, 'F1 — CD': '#2a78d6', 'F1 — UC': '#c94040'}
        ax.set_title(title, fontsize=9, fontweight='bold', color=title_color[title],
                     fontfamily='sans-serif', loc='left', pad=4)
        ax.set_xlabel('Mean ± SD  (5-fold CV)', fontsize=6.8, color=INK2,
                      fontfamily='sans-serif', labelpad=3)

    # ── panel 4: per-fold AUC strip ────────────────────────────────────────────
    ax4 = axes[3]
    for i, row in df.iterrows():
        color = MODALITY_COLOR[row['modality']]
        fold_aucs = row['fold_auc']

        # SD band
        ax4.barh(i, row['std_auc'] * 2,
                 left=row['mean_auc'] - row['std_auc'],
                 height=0.55, color=color, alpha=0.18, zorder=2)
        # mean tick
        ax4.plot([row['mean_auc']] * 2, [i - 0.28, i + 0.28],
                 color=color, lw=2.2, zorder=4, solid_capstyle='round')
        # fold dots, labeled
        jitter = np.random.uniform(-0.16, 0.16, 5)
        ax4.scatter(fold_aucs, i + jitter,
                    s=22, color=color, alpha=0.9, zorder=5,
                    edgecolors='white', linewidths=0.5)
        for f_idx, fv in enumerate(fold_aucs):
            ax4.text(fv, i + jitter[f_idx] + 0.19,
                     str(f_idx), fontsize=4.8, ha='center', va='bottom',
                     color=color, alpha=0.7, fontfamily='sans-serif')

    ax4.set_xlim(0.40, 1.02)
    ax4.set_ylim(-0.55, n - 0.45)
    ax4.axvline(0.5, color=MUTED, lw=0.4, ls=':', zorder=1)
    ax4.set_yticks(y)
    ax4.set_yticklabels([])
    ax4.tick_params(axis='x', labelsize=7, colors=INK2, width=0.5, length=2.5)
    ax4.tick_params(axis='y', length=0)
    ax4.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax4.set_axisbelow(True)
    ax4.spines['bottom'].set_linewidth(0.5)
    ax4.spines['bottom'].set_color(MUTED)
    ax4.spines[['top', 'right', 'left']].set_visible(False)
    ax4.set_title('Per-fold AUC', fontsize=9, fontweight='bold', color=INK,
                  fontfamily='sans-serif', loc='left', pad=4)
    ax4.set_xlabel('AUC (fold 0–4)', fontsize=6.8, color=INK2,
                   fontfamily='sans-serif', labelpad=3)

    # ── figure title ───────────────────────────────────────────────────────────
    fig.suptitle(
        'Single-modality RF classifiers — CD vs UC  '
        '(matched cohort: 605 patients, same splits, at-20-cm, 5-fold CV)',
        fontsize=11, fontweight='bold', color=INK,
        fontfamily='sans-serif', x=0.55, y=0.955)

    # ── summary stats table (inside figure, below the panels) ─────────────────
    col_labels = ['Modality', 'N visits / patients', 'AUC', 'F1-CD', 'F1-UC', 'Accuracy']
    n_visits = {'clinical': '605 / 605 (patient-level)', 'rna': '798 visits / 605',
                'imaging': '751 visits / 605', 'proteomics': '1,031 visits / 605'}
    ordered_mods = df.sort_values('mean_auc', ascending=False)['modality'].tolist()
    table_rows = []
    for _, row in df.sort_values('mean_auc', ascending=False).iterrows():
        table_rows.append([
            MODALITY_LABEL[row['modality']].replace('\n', ' '),
            n_visits[row['modality']],
            f"{row['mean_auc']:.3f} ± {row['std_auc']:.3f}",
            f"{row['mean_f1cd']:.3f} ± {row['std_f1cd']:.3f}",
            f"{row['mean_f1uc']:.3f} ± {row['std_f1uc']:.3f}",
            f"{row['mean_acc']:.3f} ± {row['std_acc']:.3f}",
        ])
    tbl = fig.add_subplot(gs[1, :])
    tbl.axis('off')
    table = tbl.table(
        cellText=table_rows, colLabels=col_labels,
        loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#e8e7e0')
            cell.set_text_props(fontweight='bold', color=INK,
                                fontfamily='sans-serif')
        else:
            mod = ordered_mods[r - 1]
            cell.set_facecolor(SURF)
            if c == 0:
                cell.set_text_props(color=MODALITY_COLOR[mod],
                                    fontweight='bold',
                                    fontfamily='sans-serif')
            else:
                cell.set_text_props(color=INK2, fontfamily='sans-serif')

    return fig


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    df = load()

    print(f'\n{"Modality":<14} {"AUC":>20} {"F1-CD":>20} {"F1-UC":>20}')
    print('-' * 76)
    for _, row in df.sort_values('mean_auc', ascending=False).iterrows():
        print(f"  {row['modality']:<12}  "
              f"AUC={row['mean_auc']:.4f}±{row['std_auc']:.4f}  "
              f"F1-CD={row['mean_f1cd']:.4f}±{row['std_f1cd']:.4f}  "
              f"F1-UC={row['mean_f1uc']:.4f}±{row['std_f1uc']:.4f}")

    fig = plot(df)
    out_png = os.path.join(OUT_DIR, 'single_modality_comparison.png')
    out_pdf = os.path.join(OUT_DIR, 'single_modality_comparison.pdf')
    fig.savefig(out_png, dpi=200, bbox_inches='tight', facecolor=SURF)
    fig.savefig(out_pdf,           bbox_inches='tight', facecolor=SURF)
    plt.close(fig)
    print(f'\n  Saved: {out_png}')
    print(f'  Saved: {out_pdf}')


if __name__ == '__main__':
    main()
