"""
AUC comparison: RNA raw (17,963 genes) vs three ssGSEA pathway arms.

All arms evaluated on the same matched cohort: 945 visits, 817 patients.
Reads fold metrics from at20cm_visit_fold_metrics.csv and
at20cm_pathway_visit_fold_metrics.csv.

Output
------
    <REPORTS_DIR>/at20cm_pathway_comparison.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VISIT_CSV   = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
               '08_09_at20cm_site_controlled/results/at20cm_visit_fold_metrics.csv')
PATHWAY_CSV = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
               '08_09_at20cm_site_controlled/results/at20cm_pathway_visit_fold_metrics.csv')
REPORTS_DIR   = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                 '08_09_at20cm_site_controlled/reports')
CONCEPT_DIR   = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                 'concept_learning/plots')

# palette — adjacent ΔE: aqua-blue 24.0, blue-orange 33.6, orange-yellow 13.7
AQUA   = '#1baf7a'
BLUE   = '#2a78d6'
ORANGE = '#eb6834'
YELLOW = '#eda100'
INK    = '#0b0b0b'
MUTED  = '#898781'
GRID   = '#e1e0d9'

ARMS = [
    ('rna_visit',              'RNA-seq raw\n(VST, 17,963 genes)',       AQUA,   VISIT_CSV),
    ('pathway_hallmark_visit', 'Pathway: Hallmark\n(ssGSEA, 50 sets)',   BLUE,   PATHWAY_CSV),
    ('pathway_kegg_visit',     'Pathway: KEGG\n(ssGSEA, 320 sets)',      ORANGE, PATHWAY_CSV),
    ('pathway_combined_visit', 'Pathway: Hallmark + KEGG\n(370 sets)',   YELLOW, PATHWAY_CSV),
]


def plot(out_dir: str = REPORTS_DIR, extra_dirs: list = None) -> str:
    dfs = {VISIT_CSV: pd.read_csv(VISIT_CSV), PATHWAY_CSV: pd.read_csv(PATHWAY_CSV)}
    os.makedirs(out_dir, exist_ok=True)
    if extra_dirs:
        for d in extra_dirs:
            os.makedirs(d, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    for i, (key, label, color, csv) in enumerate(ARMS):
        sub  = dfs[csv][dfs[csv]['strategy'] == key]['auc'].values
        mean = sub.mean()
        std  = sub.std(ddof=1)

        ax.barh(i, mean - 0.5, height=0.52, left=0.5,
                color=color, linewidth=0, zorder=2)
        ax.errorbar(mean, i, xerr=std, fmt='none', color=INK,
                    capsize=2.5, capthick=0.8, elinewidth=0.8, zorder=4)
        ax.text(mean + std + 0.004, i, f'{mean:.3f} ± {std:.3f}',
                va='center', ha='left', fontsize=6, color=INK,
                fontfamily='sans-serif')

    ax.axvline(0.5, color=MUTED, lw=0.6, ls='--', zorder=1)
    ax.text(0.501, -0.65, 'chance', va='top', ha='left',
            fontsize=5.5, color=MUTED, fontfamily='sans-serif')

    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([a[1] for a in ARMS], fontsize=6.5, color=INK,
                       fontfamily='sans-serif')
    ax.set_xlabel('AUC (5-fold CV, patient-level split)', fontsize=7,
                  color=INK, fontfamily='sans-serif')
    ax.set_xlim(0.5, 0.97)
    ax.set_ylim(-0.75, len(ARMS) - 0.25)

    ax.xaxis.set_tick_params(width=0.6, length=3, labelsize=6.5, colors=INK)
    ax.tick_params(axis='y', length=0, labelsize=6.5)

    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['bottom'].set_color(INK)
    ax.spines[['top', 'right', 'left']].set_visible(False)

    ax.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title('RNA-seq raw vs pathway scores · 817 patients · 945 visits',
                 fontsize=7.5, fontweight='bold', color=INK, pad=6,
                 fontfamily='sans-serif', loc='left')

    plt.tight_layout(pad=0.4)
    fname = 'at20cm_pathway_comparison.png'
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
    print(f'Saved: {os.path.join(CONCEPT_DIR, "at20cm_pathway_comparison.png")}')
