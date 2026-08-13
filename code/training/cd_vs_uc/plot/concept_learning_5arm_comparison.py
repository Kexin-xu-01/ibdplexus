"""
5-arm AUC comparison: prism2_base vs prism2_histoscore vs RNA vs two multimodal arms.

Arms (all on the same 945-visit / 817-patient matched cohort):
  img_base_visit          — prism2_base (2,560-d Virchow2 perceiver)
  img_histoscore_visit    — prism2 histological scores (11-d)
  rna_visit               — RNA-seq VST (17,963 genes)
  concat_raw_visit        — RNA + prism2_base
  concat_histoscore_visit — RNA + histological scores

Reads from two fold-metrics CSVs and saves a Nature-style horizontal bar chart.

Usage
-----
    python concept_learning_5arm_comparison.py

Output
------
    <PLOTS_DIR>/concept_learning_5arm_comparison.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_CSV  = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             '08_09_at20cm_site_controlled/results/at20cm_visit_fold_metrics.csv')
HISTO_CSV = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/results/at20cm_histoscore_fold_metrics.csv')
PLOTS_DIR = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/plots')

# ── color scheme ─────────────────────────────────────────────────────────────
#  imaging           blues
#  RNA               green
#  multimodal        amber / orange

C_BASE    = '#4e79a7'   # prism2_base  (steel blue)
C_HISTO   = '#76b7b2'   # histoscore   (teal)
C_RNA     = '#1baf7a'   # RNA          (green)
C_MM_BASE = '#eda100'   # RNA+base     (amber)
C_MM_HIST = '#eb6834'   # RNA+histo    (orange)
INK       = '#000000'
MUTED     = '#555555'

ARMS = [
    # (strategy_key,          csv,       label,                                color)
    ('img_base_visit',        BASE_CSV,  'Imaging — prism2 base\n(2,560-d)',   C_BASE),
    ('img_histoscore_visit',  HISTO_CSV, 'Imaging — histo scores\n(11-d)',     C_HISTO),
    ('rna_visit',             BASE_CSV,  'RNA-seq only\n(17,963 genes)',        C_RNA),
    ('concat_raw_visit',      BASE_CSV,  'RNA + prism2 base\n(multimodal)',     C_MM_BASE),
    ('concat_histoscore_visit', HISTO_CSV, 'RNA + histo scores\n(multimodal)', C_MM_HIST),
]

# section dividers: (after index, label)
DIVIDERS = [
    (1, 'Imaging'),
    (2, 'Transcriptomics'),
    (4, 'Multimodal'),
]


def plot(base_csv=BASE_CSV, histo_csv=HISTO_CSV, out_dir=PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)

    csv_cache = {}
    def load(path):
        if path not in csv_cache:
            csv_cache[path] = pd.read_csv(path)
        return csv_cache[path]

    stats = []
    for key, csv_path, label, color in ARMS:
        df  = load(csv_path)
        sub = df[df['strategy'] == key]['auc'].values
        stats.append({'key': key, 'label': label, 'color': color,
                      'mean': sub.mean(), 'std': sub.std(ddof=1),
                      'folds': sub})

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    y_pos = list(range(len(stats)))[::-1]   # top = first arm

    for i, (s, yp) in enumerate(zip(stats, y_pos)):
        mean, std, color = s['mean'], s['std'], s['color']
        ax.barh(yp, mean - 0.5, height=0.54, color=color, zorder=2,
                left=0.5, linewidth=0)
        ax.errorbar(mean, yp, xerr=std, fmt='none', color=INK,
                    capsize=2.5, capthick=0.8, elinewidth=0.8, zorder=4)
        ax.text(mean + std + 0.005, yp,
                f'{mean:.3f} ± {std:.3f}',
                va='center', ha='left', fontsize=6, color=INK,
                fontfamily='sans-serif')

    # section bracket lines
    section_boundaries = {
        'Imaging':        (y_pos[0] + 0.45, y_pos[1] - 0.45),
        'Transcriptomics':(y_pos[2] + 0.45, y_pos[2] - 0.45),
        'Multimodal':     (y_pos[3] + 0.45, y_pos[4] - 0.45),
    }
    x_bracket = 0.497
    for section, (ytop, ybot) in section_boundaries.items():
        ax.plot([x_bracket, x_bracket], [ybot, ytop],
                color=MUTED, lw=0.7, solid_capstyle='round', zorder=1)
        ax.text(x_bracket - 0.001, (ytop + ybot) / 2, section,
                va='center', ha='right', fontsize=5.5, color=MUTED,
                fontfamily='sans-serif', rotation=90)

    # horizontal separator between sections
    for sep_y in [y_pos[1] - 0.55, y_pos[2] - 0.55]:
        ax.axhline(sep_y, color='#cccccc', lw=0.4, ls=':', zorder=0)

    # chance line
    ax.axvline(0.5, color=MUTED, lw=0.6, ls='--', zorder=1)
    ax.text(0.501, y_pos[-1] - 0.7, 'chance', va='top', ha='left',
            fontsize=5.5, color=MUTED, fontfamily='sans-serif')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([s['label'] for s in stats], fontsize=6.5, color=INK,
                       fontfamily='sans-serif')
    ax.set_xlabel('AUC (5-fold CV, patient-level split)', fontsize=7,
                  color=INK, fontfamily='sans-serif')
    ax.set_xlim(0.488, 0.975)
    ax.set_ylim(y_pos[-1] - 0.85, y_pos[0] + 0.65)

    ax.xaxis.set_tick_params(width=0.6, length=3, labelsize=6.5, colors=INK)
    ax.tick_params(axis='y', length=0)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['bottom'].set_color(INK)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.xaxis.grid(True, color='#cccccc', linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title('At-20-cm · 817 patients · 945 visits\n'
                 'visit-level predictions · matched cohort',
                 fontsize=7.5, fontweight='bold', color=INK, pad=6,
                 fontfamily='sans-serif', loc='left')

    plt.tight_layout(pad=0.4)
    out = os.path.join(out_dir, 'concept_learning_5arm_comparison.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f'Saved: {out}')
    return out


if __name__ == '__main__':
    plot()
