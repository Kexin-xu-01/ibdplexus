"""
10-arm AUC comparison — MLP classifier, same cohort as RF experiments.

Reads at20cm_mlp_fold_metrics.csv and overlays MLP vs RF for each arm.

Usage
-----
    python plot/mlp_comparison.py

Outputs
-------
    concept_learning/plots/mlp_10arm_comparison.png/.pdf
    concept_learning/plots/rf_vs_mlp_comparison.png/.pdf   (side-by-side)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MLP_CSV = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
           'concept_learning/results/at20cm_mlp_fold_metrics.csv')

# RF sources (pre-existing)
RF_BASE_CSV       = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                     '08_09_at20cm_site_controlled/results/at20cm_visit_fold_metrics.csv')
RF_HISTO_CSV      = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                     'concept_learning/results/at20cm_histoscore_fold_metrics.csv')
RF_BULKFORMER_CSV = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                     'concept_learning/results/at20cm_bulkformer_fold_metrics.csv')
RF_PCA_CSV        = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                     'concept_learning/results/at20cm_pca_rna_fold_metrics.csv')

PLOTS_DIR = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/plots')

C_BASE       = '#4e79a7'
C_HISTO      = '#76b7b2'
C_BF         = '#b07aa1'
C_RNA        = '#1baf7a'
C_PCA        = '#59a14f'
C_MM_BASE    = '#eda100'
C_MM_HIST    = '#eb6834'
C_MM_BF_BASE = '#c47ac0'
C_MM_BF_HIST = '#9c5fa8'
C_MM_PCA     = '#8cd17d'
INK          = '#000000'
MUTED        = '#555555'

ARM_DEFS = [
    ('img_base_visit',          RF_BASE_CSV,       'Imaging — prism2 base\n(2,560-d)',          C_BASE),
    ('img_histoscore_visit',    RF_HISTO_CSV,      'Imaging — histo scores\n(11-d)',             C_HISTO),
    ('bulkformer_visit',        RF_BULKFORMER_CSV, 'BulkFormer embeddings\n(640-d)',              C_BF),
    ('rna_visit',               RF_BASE_CSV,       'RNA-seq VST\n(17,963 genes)',                 C_RNA),
    ('pca640_rna_visit',        RF_PCA_CSV,        'PCA-640 RNA\n(dim. control)',                 C_PCA),
    ('concat_raw_visit',        RF_BASE_CSV,       'RNA + prism2 base\n(multimodal)',             C_MM_BASE),
    ('concat_histoscore_visit', RF_HISTO_CSV,      'RNA + histo scores\n(multimodal)',            C_MM_HIST),
    ('concat_bf_base_visit',    RF_BULKFORMER_CSV, 'BulkFormer + prism2 base\n(multimodal)',      C_MM_BF_BASE),
    ('concat_bf_histo_visit',   RF_BULKFORMER_CSV, 'BulkFormer + histo scores\n(multimodal)',     C_MM_BF_HIST),
    ('concat_pca640_base',      RF_PCA_CSV,        'PCA-640 RNA + prism2 base\n(dim. control)',   C_MM_PCA),
]

SECTION_KEYS = {
    'Imaging':        {'img_base_visit', 'img_histoscore_visit'},
    'Transcriptomics':{'bulkformer_visit', 'rna_visit', 'pca640_rna_visit'},
    'Multimodal':     {'concat_raw_visit', 'concat_histoscore_visit',
                       'concat_bf_base_visit', 'concat_bf_histo_visit', 'concat_pca640_base'},
}


def load_stats(arm_defs, csv_override=None):
    csv_cache = {}
    def load(path):
        if path not in csv_cache:
            if not os.path.exists(path):
                return None
            csv_cache[path] = pd.read_csv(path)
        return csv_cache[path]

    stats = []
    for key, csv_path, label, color in arm_defs:
        src = csv_override if csv_override else csv_path
        df  = load(src)
        if df is None:
            continue
        sub = df[df['strategy'] == key]['auc'].values
        if len(sub) == 0:
            continue
        stats.append({'key': key, 'label': label, 'color': color,
                      'mean': sub.mean(), 'std': sub.std(ddof=1)})
    return stats


def draw_bars(ax, stats, x_bracket=0.497):
    n = len(stats)
    y_pos = list(range(n))[::-1]

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

    key_to_idx = {s['key']: i for i, s in enumerate(stats)}
    section_sep_after = []
    for sec_label in ['Imaging', 'Transcriptomics', 'Multimodal']:
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

    for idx in section_sep_after[:-1]:
        ax.axhline(y_pos[idx] - 0.55, color='#cccccc', lw=0.4, ls=':', zorder=0)

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
    ax.spines['bottom'].set_linewidth(0.6); ax.spines['bottom'].set_color(INK)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.xaxis.grid(True, color='#cccccc', linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    return y_pos


def plot_single(stats, title_suffix, stem, out_dir):
    n = len(stats)
    fig_h = max(3.0, 0.55 * n + 1.8)
    fig, ax = plt.subplots(figsize=(4.8, fig_h))
    fig.patch.set_alpha(0); ax.set_facecolor('none')
    draw_bars(ax, stats)
    ax.set_title(f'At-20-cm · 817 patients · 945 visits\n{title_suffix}',
                 fontsize=7.5, fontweight='bold', color=INK, pad=6,
                 fontfamily='sans-serif', loc='left')
    plt.tight_layout(pad=0.4)
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'{stem}.{ext}')
        plt.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
        print(f'Saved: {out}')
    plt.close()


def plot_side_by_side(stats_rf, stats_mlp, out_dir):
    """Two-panel figure: RF (left) | MLP (right), shared arm order."""
    # align to the same arms in same order
    mlp_dict = {s['key']: s for s in stats_mlp}
    rf_dict  = {s['key']: s for s in stats_rf}
    keys = [s['key'] for s in stats_rf if s['key'] in mlp_dict]
    if not keys:
        print('No overlapping arms for side-by-side — skipping')
        return

    stats_rf_al  = [rf_dict[k]  for k in keys]
    stats_mlp_al = [mlp_dict[k] for k in keys]

    n = len(keys)
    fig_h = max(3.0, 0.55 * n + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, fig_h), sharey=False)
    fig.patch.set_alpha(0)

    for ax in axes:
        ax.set_facecolor('none')

    draw_bars(axes[0], stats_rf_al)
    axes[0].set_title('Random Forest', fontsize=8, fontweight='bold',
                       color=INK, pad=5, fontfamily='sans-serif')

    draw_bars(axes[1], stats_mlp_al)
    axes[1].set_title('MLP (256-128)', fontsize=8, fontweight='bold',
                       color=INK, pad=5, fontfamily='sans-serif')
    # remove y-tick labels from right panel (same arms, readable from left)
    axes[1].set_yticklabels([])
    axes[1].set_ylabel('')

    fig.suptitle('At-20-cm · 817 patients · 945 visits · matched cohort',
                 fontsize=8, fontweight='bold', color=INK,
                 fontfamily='sans-serif', y=1.01)

    plt.tight_layout(pad=0.4, w_pad=1.0)
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'rf_vs_mlp_comparison.{ext}')
        plt.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
        print(f'Saved: {out}')
    plt.close()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    stats_mlp = load_stats(ARM_DEFS, csv_override=MLP_CSV)
    stats_rf  = load_stats(ARM_DEFS)

    if not stats_mlp:
        raise RuntimeError(f'MLP results not found: {MLP_CSV}')

    n_arms = len(stats_mlp)
    plot_single(stats_mlp, 'MLP (256-128) · visit-level predictions',
                f'mlp_{n_arms}arm_comparison', PLOTS_DIR)

    if stats_rf:
        plot_side_by_side(stats_rf, stats_mlp, PLOTS_DIR)


if __name__ == '__main__':
    main()
