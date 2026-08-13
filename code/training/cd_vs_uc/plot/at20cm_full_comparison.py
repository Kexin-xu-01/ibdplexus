"""
Full AUC comparison across all visit-level at-20-cm arms.

Arms (6 total)
--------------
  img_base_visit         prism2 base embeddings (imaging)
  hist_visit             11 histological scores (imaging)
  rna_visit              RNA-seq raw (17,963 VST genes)
  pathway_hallmark_visit Hallmark ssGSEA (50 pathways)
  concat_raw_visit       prism2 + RNA-seq (raw concat)
  hallmark_histo_visit   Hallmark pathways + histo scores

All evaluated on the same matched cohort: 945 visits, 817 patients.

Output
------
  <REPORTS_DIR>/at20cm_full_comparison.png
  <CONCEPT_DIR>/at20cm_full_comparison.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc'
RESULTS = os.path.join(BASE, '08_09_at20cm_site_controlled/results')

VISIT_CSV          = os.path.join(RESULTS, 'at20cm_visit_fold_metrics.csv')
HIST_CSV           = os.path.join(RESULTS, 'at20cm_hist_visit_fold_metrics.csv')
PATHWAY_CSV        = os.path.join(RESULTS, 'at20cm_pathway_visit_fold_metrics.csv')
HALLMARK_HISTO_CSV = os.path.join(RESULTS, 'at20cm_hallmark_histo_visit_fold_metrics.csv')

REPORTS_DIR = os.path.join(BASE, '08_09_at20cm_site_controlled/reports')
CONCEPT_DIR = os.path.join(BASE, 'concept_learning/plots')

# palette slots 1-6 (adjacent-pair validated order)
BLUE    = '#2a78d6'
ORANGE  = '#eb6834'
AQUA    = '#1baf7a'
YELLOW  = '#eda100'
MAGENTA = '#e87ba4'
GREEN   = '#008300'

INK   = '#0b0b0b'
MUTED = '#898781'
GRID  = '#e1e0d9'

ARMS = [
    # (strategy_key, label, color, csv_path)
    ('img_base_visit',
     'prism2 base\n(image, 2,560 dim)',
     BLUE,   VISIT_CSV),
    ('hist_visit',
     'Histological scores\n(11 features)',
     ORANGE, HIST_CSV),
    ('rna_visit',
     'RNA-seq raw\n(VST, 17,963 genes)',
     AQUA,   VISIT_CSV),
    ('pathway_hallmark_visit',
     'Pathway: Hallmark\n(ssGSEA, 50 sets)',
     YELLOW, PATHWAY_CSV),
    ('concat_raw_visit',
     'prism2 + RNA-seq\n(20,523 dim)',
     MAGENTA, VISIT_CSV),
    ('hallmark_histo_visit',
     'Hallmark + Histo\n(50 + 11 features)',
     GREEN,  HALLMARK_HISTO_CSV),
]


def plot(out_dir: str = REPORTS_DIR, extra_dirs: list = None) -> str:
    dfs = {}
    for _, _, _, csv in ARMS:
        if csv not in dfs:
            dfs[csv] = pd.read_csv(csv)

    os.makedirs(out_dir, exist_ok=True)
    if extra_dirs:
        for d in extra_dirs:
            os.makedirs(d, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    n = len(ARMS)
    for i, (key, label, color, csv) in enumerate(ARMS):
        sub  = dfs[csv][dfs[csv]['strategy'] == key]['auc'].values
        mean = sub.mean()
        std  = sub.std(ddof=1)

        ax.barh(i, mean - 0.5, height=0.52, left=0.5,
                color=color, linewidth=0, zorder=2)
        ax.errorbar(mean, i, xerr=std, fmt='none', color=INK,
                    capsize=2.5, capthick=0.8, elinewidth=0.8, zorder=4)
        ax.text(mean + std + 0.004, i, f'{mean:.3f}±{std:.3f}',
                va='center', ha='left', fontsize=5.8, color=INK,
                fontfamily='sans-serif')

    # separator line between unimodal and multimodal groups
    ax.axhline(3.5, color=MUTED, lw=0.5, ls=':', zorder=1)

    ax.axvline(0.5, color=MUTED, lw=0.6, ls='--', zorder=1)
    ax.text(0.501, -0.75, 'chance', va='top', ha='left',
            fontsize=5.5, color=MUTED, fontfamily='sans-serif')

    ax.set_yticks(range(n))
    ax.set_yticklabels([a[1] for a in ARMS], fontsize=6.2, color=INK,
                       fontfamily='sans-serif')
    ax.set_xlabel('AUC (5-fold CV, patient-level split)', fontsize=7,
                  color=INK, fontfamily='sans-serif')
    ax.set_xlim(0.5, 1.00)
    ax.set_ylim(-0.75, n - 0.25)

    ax.xaxis.set_tick_params(width=0.6, length=3, labelsize=6.5, colors=INK)
    ax.tick_params(axis='y', length=0, labelsize=6.2)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['bottom'].set_color(INK)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # group annotations
    ax.text(0.502, 1.5, 'Unimodal', fontsize=5.5, color=MUTED,
            fontfamily='sans-serif', va='center', style='italic')
    ax.text(0.502, 4.5, 'Multimodal', fontsize=5.5, color=MUTED,
            fontfamily='sans-serif', va='center', style='italic')

    ax.set_title('Model comparison · 817 patients · 945 visits',
                 fontsize=7.5, fontweight='bold', color=INK, pad=6,
                 fontfamily='sans-serif', loc='left')

    plt.tight_layout(pad=0.4)
    fname = 'at20cm_full_comparison.png'
    out   = os.path.join(out_dir, fname)
    plt.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
    if extra_dirs:
        for d in extra_dirs:
            plt.savefig(os.path.join(d, fname), dpi=300,
                        bbox_inches='tight', transparent=True)
    plt.close()
    return out


if __name__ == '__main__':
    path = plot(extra_dirs=[CONCEPT_DIR])
    print(f'Saved: {path}')
    print(f'Saved: {os.path.join(CONCEPT_DIR, "at20cm_full_comparison.png")}')
