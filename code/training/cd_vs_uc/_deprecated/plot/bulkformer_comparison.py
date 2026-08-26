"""
10-arm AUC comparison: imaging × transcriptomics modality panel with BulkFormer + PCA control.

Arms (all on the same 945-visit / 817-patient matched cohort):
  img_base_visit          — prism2 base (2,560-d)
  img_histoscore_visit    — histo scores (11-d)
  bulkformer_visit        — BulkFormer embeddings (640-d)
  rna_visit               — RNA-seq VST (17,963 genes)
  pca640_rna_visit        — PCA(640) of RNA VST [dimensionality control]
  concat_raw_visit        — RNA + prism2 base
  concat_histoscore_visit — RNA + histo scores
  concat_bf_base_visit    — BulkFormer + prism2 base
  concat_bf_histo_visit   — BulkFormer + histo scores
  concat_pca640_base      — PCA-640 RNA + prism2 base [control: same dim as BulkFormer arm]

Arms whose CSV is missing are silently skipped (so the 8-arm plot renders before 08g finishes).

Usage
-----
    python plot/bulkformer_comparison.py

Output
------
    <PLOTS_DIR>/bulkformer_comparison.png/.pdf
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_CSV       = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                  '08_09_at20cm_site_controlled/results/at20cm_visit_fold_metrics.csv')
HISTO_CSV      = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                  'concept_learning/results/at20cm_histoscore_fold_metrics.csv')
BULKFORMER_CSV = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                  'concept_learning/results/at20cm_bulkformer_fold_metrics.csv')
PCA_CSV        = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                  'concept_learning/results/at20cm_pca_rna_fold_metrics.csv')
PLOTS_DIR      = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                  '08_09_at20cm_site_controlled/plots')

# ── colour scheme ─────────────────────────────────────────────────────────────
C_BASE       = '#4e79a7'   # prism2_base               (steel blue)
C_HISTO      = '#76b7b2'   # histoscores               (teal)
C_BF         = '#b07aa1'   # BulkFormer                (mauve)
C_RNA        = '#1baf7a'   # RNA VST                   (green)
C_PCA        = '#59a14f'   # PCA-640 RNA               (dark green — dimensionality control)
C_MM_BASE    = '#eda100'   # RNA + prism2              (amber)
C_MM_HIST    = '#eb6834'   # RNA + histoscores         (orange)
C_MM_BF_BASE = '#c47ac0'   # BulkFormer + prism2       (violet)
C_MM_BF_HIST = '#9c5fa8'   # BulkFormer + histoscores  (deep purple)
C_MM_PCA     = '#8cd17d'   # PCA-640 + prism2          (light green — control)
INK          = '#000000'
MUTED        = '#555555'

ALL_ARMS = [
    # (strategy_key,              csv,            label,                                         color)
    ('img_base_visit',          BASE_CSV,       'Imaging — prism2 base\n(2,560-d)',             C_BASE),
    ('img_histoscore_visit',    HISTO_CSV,      'Imaging — histo scores\n(11-d)',               C_HISTO),
    ('bulkformer_visit',        BULKFORMER_CSV, 'BulkFormer embeddings\n(640-d)',                C_BF),
    ('rna_visit',               BASE_CSV,       'RNA-seq VST\n(17,963 genes)',                   C_RNA),
    ('pca640_rna_visit',        PCA_CSV,        'PCA-640 RNA\n(dim. control)',                   C_PCA),
    ('concat_raw_visit',        BASE_CSV,       'RNA + prism2 base\n(multimodal)',               C_MM_BASE),
    ('concat_histoscore_visit', HISTO_CSV,      'RNA + histo scores\n(multimodal)',              C_MM_HIST),
    ('concat_bf_base_visit',    BULKFORMER_CSV, 'BulkFormer + prism2 base\n(multimodal)',        C_MM_BF_BASE),
    ('concat_bf_histo_visit',   BULKFORMER_CSV, 'BulkFormer + histo scores\n(multimodal)',       C_MM_BF_HIST),
    ('concat_pca640_base',      PCA_CSV,        'PCA-640 RNA + prism2 base\n(dim. control)',     C_MM_PCA),
]

HISTO_KEYS = {'img_histoscore_visit', 'concat_histoscore_visit', 'concat_bf_histo_visit'}

# section bracket spans are computed dynamically from present arms
SECTION_KEYS = {
    'Imaging':        {'img_base_visit', 'img_histoscore_visit'},
    'Transcriptomics':{'bulkformer_visit', 'rna_visit', 'pca640_rna_visit'},
    'Multimodal':     {'concat_raw_visit', 'concat_histoscore_visit',
                       'concat_bf_base_visit', 'concat_bf_histo_visit', 'concat_pca640_base'},
}


def plot(out_dir=PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)

    csv_cache = {}
    def load(path):
        if path not in csv_cache:
            if not os.path.exists(path):
                return None
            csv_cache[path] = pd.read_csv(path)
        return csv_cache[path]

    stats = []
    for key, csv_path, label, color in ALL_ARMS:
        if key in HISTO_KEYS:
            continue
        df = load(csv_path)
        if df is None:
            continue
        sub = df[df['strategy'] == key]['auc'].values
        if len(sub) == 0:
            continue
        stats.append({'key': key, 'label': label, 'color': color,
                      'mean': sub.mean(), 'std': sub.std(ddof=1)})

    if not stats:
        raise RuntimeError('No arms found — check CSV paths')

    present_keys = {s['key'] for s in stats}
    n = len(stats)
    fig_h = max(3.0, 0.55 * n + 1.8)
    fig, ax = plt.subplots(figsize=(4.8, fig_h))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    y_pos = list(range(n))[::-1]   # top = first arm

    for s, yp in zip(stats, y_pos):
        mean, std, color = s['mean'], s['std'], s['color']
        ax.barh(yp, mean - 0.5, height=0.54, color=color, zorder=2,
                left=0.5, linewidth=0)
        ax.errorbar(mean, yp, xerr=std, fmt='none', color=INK,
                    capsize=2.5, capthick=0.8, elinewidth=0.8, zorder=4)
        ax.text(mean + std + 0.005, yp,
                f'{mean:.3f} ± {std:.3f}',
                va='center', ha='left', fontsize=6, color=INK,
                fontfamily='sans-serif')

    # build section brackets dynamically
    x_bracket = 0.497
    key_to_idx = {s['key']: i for i, s in enumerate(stats)}
    section_order = ['Imaging', 'Transcriptomics', 'Multimodal']
    section_sep_after = []   # y_pos indices after which to draw a separator
    for sec_label in section_order:
        sec_keys = SECTION_KEYS[sec_label]
        idxs = sorted([key_to_idx[k] for k in sec_keys if k in key_to_idx])
        if not idxs:
            continue
        ytop = y_pos[idxs[0]]  + 0.45
        ybot = y_pos[idxs[-1]] - 0.45
        ax.plot([x_bracket, x_bracket], [ybot, ytop],
                color=MUTED, lw=0.7, solid_capstyle='round', zorder=1)
        ax.text(x_bracket - 0.001, (ytop + ybot) / 2, sec_label,
                va='center', ha='right', fontsize=5.5, color=MUTED,
                fontfamily='sans-serif', rotation=90)
        section_sep_after.append(idxs[-1])

    # horizontal separators between sections
    for idx in section_sep_after[:-1]:
        ax.axhline(y_pos[idx] - 0.55, color='#cccccc', lw=0.4, ls=':', zorder=0)

    # chance line
    ax.axvline(0.5, color=MUTED, lw=0.6, ls='--', zorder=1)
    ax.text(0.501, y_pos[-1] - 0.7, 'chance', va='top', ha='left',
            fontsize=5.5, color=MUTED, fontfamily='sans-serif')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([s['label'] for s in stats], fontsize=6.5, color=INK,
                       fontfamily='sans-serif')
    ax.set_xlabel('AUC (5-fold CV, patient-level split)', fontsize=7,
                  color=INK, fontfamily='sans-serif')
    ax.set_xlim(0.488, 0.985)
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

    n_arms = len(stats)
    stem = f'bulkformer_{n_arms}arm_comparison'
    plt.tight_layout(pad=0.4)
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'{stem}.{ext}')
        plt.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
        print(f'Saved: {out}')
    plt.close()


if __name__ == '__main__':
    plot()