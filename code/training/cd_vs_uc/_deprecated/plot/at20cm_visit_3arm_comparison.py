"""
Plot 3-arm AUC comparison for the at-20-cm cohort — visit-level (945 visits, 817 patients).

Arms: imaging only (prism2_base), RNA-seq only (VST), RNA + imaging (concat raw).
Reads fold metrics from at20cm_visit_fold_metrics.csv; saves a Nature-style
horizontal bar chart with ±1 SD error bars.

Usage
-----
    python at20cm_visit_3arm_comparison.py

Output
------
    <REPORTS_DIR>/at20cm_visit_3arm_comparison.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS_CSV = (
    '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
    '08_09_at20cm_site_controlled/results/at20cm_visit_fold_metrics.csv'
)
REPORTS_DIR = (
    '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
    '08_09_at20cm_site_controlled/reports'
)

BLUE   = '#2a78d6'
AQUA   = '#1baf7a'
YELLOW = '#eda100'
INK    = '#000000'
MUTED  = '#555555'

ARMS = [
    ('img_base_visit',   'Imaging only\n(prism2_base, 2,560-d)', BLUE),
    ('rna_visit',        'RNA-seq only\n(VST, 17,963-d)',         AQUA),
    ('concat_raw_visit', 'RNA + Imaging\n(concat raw)',           YELLOW),
]


def plot(metrics_csv: str = METRICS_CSV, out_dir: str = REPORTS_DIR) -> str:
    df = pd.read_csv(metrics_csv)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    for i, (key, label, color) in enumerate(ARMS):
        sub = df[df['strategy'] == key]['auc'].values
        mean = sub.mean()
        std  = sub.std(ddof=1)
        ax.barh(i, mean - 0.5, height=0.52, color=color, zorder=2, left=0.5,
                linewidth=0)
        ax.errorbar(mean, i, xerr=std, fmt='none', color=INK,
                    capsize=2.5, capthick=0.8, elinewidth=0.8, zorder=4)
        ax.text(mean + std + 0.004, i, f'{mean:.3f} ± {std:.3f}',
                va='center', ha='left', fontsize=6, color=INK,
                fontfamily='sans-serif')

    ax.axvline(0.5, color=MUTED, lw=0.6, ls='--', zorder=1)
    ax.text(0.501, -0.62, 'chance', va='top', ha='left', fontsize=5.5,
            color=MUTED, fontfamily='sans-serif')

    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([a[1] for a in ARMS], fontsize=6.5, color=INK,
                       fontfamily='sans-serif')
    ax.set_xlabel('AUC (5-fold CV, patient-level split)', fontsize=7, color=INK,
                  fontfamily='sans-serif')
    ax.set_xlim(0.5, 0.97)
    ax.set_ylim(-0.75, len(ARMS) - 0.25)

    ax.xaxis.set_tick_params(width=0.6, length=3, labelsize=6.5, colors=INK)
    ax.tick_params(axis='y', length=0, labelsize=6.5)

    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['bottom'].set_color(INK)
    ax.spines[['top', 'right', 'left']].set_visible(False)

    ax.xaxis.grid(True, color='#cccccc', linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title('At-20-cm · 817 patients · 945 visits\n(H&E + RNA-seq, visit-level)',
                 fontsize=7.5, fontweight='bold', color=INK, pad=6,
                 fontfamily='sans-serif', loc='left')

    plt.tight_layout(pad=0.4)
    out = os.path.join(out_dir, 'at20cm_visit_3arm_comparison.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    return out


if __name__ == '__main__':
    path = plot()
    print(f'Saved: {path}')
