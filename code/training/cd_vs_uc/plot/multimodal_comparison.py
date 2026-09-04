"""
Multimodal combination comparison — all fusion arms, RF vs MLP.

Groups:
  RNA + Imaging       — concat_raw_visit, concat_pca640_base, concat_bf_base_visit
  Histoscore fusion   — concat_histoscore_visit, concat_bf_histo_visit
  Imaging + Clinical  — img_clinical_visit
  RNA + Clinical      — rna_clinical_visit
  Trimodal            — trimodal_visit  (RNA + Imaging + Clinical)

Each group shows RF (solid) and MLP (hatched) variants.
Panels: AUC  ·  F1-CD  ·  F1-UC

Output
------
  at20cm_proteomics/plots/multimodal_comparison.png  (.pdf)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc'
RF_METRICS  = f'{BASE}/at20cm_visit_manual_knn/results/rf/at20cm_rf_fold_metrics.csv'
MLP_METRICS = f'{BASE}/at20cm_visit_manual_knn/results/mlp/at20cm_mlp_fold_metrics.csv'
MM_METRICS  = f'{BASE}/at20cm_visit_manual_knn/results/multimodal/at20cm_multimodal_fold_metrics.csv'
OUT_DIR     = f'{BASE}/at20cm_proteomics/plots'

# ── style ──────────────────────────────────────────────────────────────────────
SURF  = '#fcfcfb'
INK   = '#0b0b0b'
INK2  = '#52514e'
MUTED = '#898781'
GRID  = '#e1e0d9'

# one colour per combination group
GROUP_COLOR = {
    'rna_imaging':  '#1b6ca8',   # deep blue
    'img_clinical': '#e76f51',   # orange-red
    'rna_clinical': '#8338ec',   # violet
    'trimodal':     '#073b4c',   # near-black
}

# ARM CATALOGUE — (strategy_key, source_file_tag, group, classifier, display_label)
ARMS = [
    # ── RNA + Imaging ──────────────────────────────────────────────────────────
    ('concat_raw_visit', 'rf',  'rna_imaging', 'RF',  'RNA + Imaging'),
    ('concat_raw_visit', 'mlp', 'rna_imaging', 'MLP', 'RNA + Imaging'),
    # ── Imaging + Clinical ─────────────────────────────────────────────────────
    ('img_clinical_visit_rf',  'mm', 'img_clinical', 'RF',  'Imaging + Clinical'),
    ('img_clinical_visit_mlp', 'mm', 'img_clinical', 'MLP', 'Imaging + Clinical'),
    # ── RNA + Clinical ─────────────────────────────────────────────────────────
    ('rna_clinical_visit_rf',  'mm', 'rna_clinical', 'RF',  'RNA + Clinical'),
    ('rna_clinical_visit_mlp', 'mm', 'rna_clinical', 'MLP', 'RNA + Clinical'),
    # ── Trimodal ───────────────────────────────────────────────────────────────
    ('trimodal_visit_rf',  'mm', 'trimodal', 'RF',  'Trimodal\n(RNA + Imaging + Clinical)'),
    ('trimodal_visit_mlp', 'mm', 'trimodal', 'MLP', 'Trimodal\n(RNA + Imaging + Clinical)'),
]

GROUP_ORDER = ['rna_imaging', 'img_clinical', 'rna_clinical', 'trimodal']
GROUP_TITLE = {
    'rna_imaging':  'RNA + Imaging',
    'img_clinical': 'Imaging + Clinical',
    'rna_clinical': 'RNA + Clinical',
    'trimodal':     'Trimodal',
}


# ── load ───────────────────────────────────────────────────────────────────────

def load():
    src = {
        'rf':  pd.read_csv(RF_METRICS),
        'mlp': pd.read_csv(MLP_METRICS),
        'mm':  pd.read_csv(MM_METRICS),
    }
    rows = []
    for strategy, file_tag, group, clf, label in ARMS:
        df = src[file_tag]
        sub = df[df['strategy'] == strategy]
        if sub.empty:
            print(f'  WARNING: {strategy} not found in {file_tag}')
            continue
        rows.append({
            'strategy': strategy,
            'file_tag': file_tag,
            'group':    group,
            'clf':      clf,
            'label':    label,
            'arm_id':   f'{strategy}__{clf}',
            'fold_auc':  sub['auc'].tolist(),
            'fold_f1cd': sub['cd_f1'].tolist(),
            'fold_f1uc': sub['uc_f1'].tolist(),
            'mean_auc':  sub['auc'].mean(),
            'std_auc':   sub['auc'].std(ddof=1),
            'mean_f1cd': sub['cd_f1'].mean(),
            'std_f1cd':  sub['cd_f1'].std(ddof=1),
            'mean_f1uc': sub['uc_f1'].mean(),
            'std_f1uc':  sub['uc_f1'].std(ddof=1),
            'mean_acc':  sub['accuracy'].mean(),
            'std_acc':   sub['accuracy'].std(ddof=1),
        })
    return pd.DataFrame(rows)


def ordered_rows(df):
    """Rows sorted within each group by AUC asc; small gap between groups."""
    ordered = []
    for g in GROUP_ORDER:
        grp = df[df['group'] == g].sort_values('mean_auc', ascending=True)
        for _, row in grp.iterrows():
            ordered.append(row)
        ordered.append(None)
    while ordered and ordered[-1] is None:
        ordered.pop()
    return ordered


# ── helpers ────────────────────────────────────────────────────────────────────

def _style_ax(ax, xlim, ylabel=True):
    ax.set_xlim(xlim)
    ax.tick_params(axis='x', labelsize=7, colors=INK2, width=0.5, length=2.5)
    ax.tick_params(axis='y', length=0, pad=2)
    ax.spines['bottom'].set_linewidth(0.5);  ax.spines['bottom'].set_color(MUTED)
    ax.spines[['top', 'right']].set_visible(False)
    if ylabel:
        ax.spines['left'].set_linewidth(0.5);  ax.spines['left'].set_color(MUTED)
    else:
        ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


# ── figure ─────────────────────────────────────────────────────────────────────

def plot(df):
    np.random.seed(42)
    ordered = ordered_rows(df)

    # y positions — data rows only, small gap between groups
    y_positions = {}
    y_group_spans = {g: [np.inf, -np.inf] for g in GROUP_ORDER}
    y = 0.0
    for item in ordered:
        if item is None:
            y += 0.5
        else:
            y_positions[item['arm_id']] = y
            g = item['group']
            y_group_spans[g][0] = min(y_group_spans[g][0], y)
            y_group_spans[g][1] = max(y_group_spans[g][1], y)
            y += 1.0
    y_max = y - 1.0

    n_rows = sum(1 for i in ordered if i is not None)

    # PowerPoint-friendly: wide, not tall.  16:9 slide ≈ 13.33 × 7.5 inches
    fig_w = 13.0
    fig_h = max(4.8, n_rows * 0.42 + 1.5)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(SURF)
    for ax in axes:
        ax.set_facecolor(SURF)

    metrics = [
        ('mean_auc',  'std_auc',  'fold_auc',  'AUC',     (0.55, 1.03),  INK),
        ('mean_f1cd', 'std_f1cd', 'fold_f1cd', 'F1 — CD', (0.55, 1.03),  '#2a78d6'),
        ('mean_f1uc', 'std_f1uc', 'fold_f1uc', 'F1 — UC', (-0.02, 1.03), '#c94040'),
    ]

    ytick_pos    = []
    ytick_labels = []
    ytick_colors = []

    for col, (mean_k, std_k, fold_k, title, xlim, tcol) in enumerate(metrics):
        ax = axes[col]

        for item in ordered:
            if item is None:
                continue
            yp    = y_positions[item['arm_id']]
            color = GROUP_COLOR[item['group']]
            mean  = item[mean_k]
            std   = item[std_k]
            is_mlp = item['clf'] == 'MLP'

            hatch = '///' if is_mlp else None
            ax.barh(yp, mean, height=0.58, color=color,
                    alpha=0.52 if is_mlp else 0.88,
                    hatch=hatch, edgecolor=color if is_mlp else 'none',
                    linewidth=0.6, zorder=2)
            ax.errorbar(mean, yp, xerr=std, fmt='none',
                        elinewidth=0.9, capsize=2.5, capthick=0.9,
                        ecolor=INK2, zorder=4)
            jitter = np.random.uniform(-0.14, 0.14, 5)
            ax.scatter(item[fold_k], yp + jitter,
                       s=14, color='white', edgecolors=color,
                       linewidths=0.8, zorder=5, alpha=0.9)
            ax.text(mean + std + 0.007, yp,
                    f'{mean:.3f}', va='center', ha='left',
                    fontsize=6.2, color=color,
                    fontfamily='sans-serif')

            if col == 0:
                clf_label = 'MLP' if is_mlp else 'RF'
                ytick_pos.append(yp)
                ytick_labels.append(clf_label)
                ytick_colors.append(color)

        ax.set_ylim(-0.5, y_max + 0.5)
        _style_ax(ax, xlim, ylabel=(col == 0))

        if col == 0:
            # Combine group name + RF/MLP into a single 2-part tick label so
            # matplotlib manages spacing — no manual overlays, no overlap.
            # Only the top row of each group carries the group name; other rows
            # in the same group get a blank prefix aligned in the same column.
            combined_labels  = []
            combined_colors  = []
            # find top row per group
            top_per_group = {}
            for item in ordered:
                if item is None:
                    continue
                g = item['group']
                yp = y_positions[item['arm_id']]
                if g not in top_per_group or yp > top_per_group[g][0]:
                    top_per_group[g] = (yp, item['arm_id'])

            for item in ordered:
                if item is None:
                    continue
                is_top = (top_per_group[item['group']][1] == item['arm_id'])
                g_name = GROUP_TITLE[item['group']] if is_top else ''
                clf    = 'MLP' if item['clf'] == 'MLP' else 'RF'
                combined_labels.append(f'{g_name:>20}   {clf}')
                combined_colors.append(GROUP_COLOR[item['group']])

            ax.set_yticks(ytick_pos)
            ax.set_yticklabels(combined_labels, fontsize=8.5,
                               fontfamily='monospace', fontweight='bold')
            for tick, col_c in zip(ax.get_yticklabels(), combined_colors):
                tick.set_color(col_c)
        else:
            ax.set_yticks([])

        ax.set_title(title, fontsize=10, fontweight='bold', color=tcol,
                     fontfamily='sans-serif', loc='left', pad=5)
        ax.set_xlabel('Mean ± SD  (5-fold CV)', fontsize=7.5, color=INK2,
                      fontfamily='sans-serif', labelpad=3)

    # ── legend ──────────────────────────────────────────────────────────────────
    legend_elements = []
    for g in GROUP_ORDER:
        legend_elements.append(
            mpatches.Patch(facecolor=GROUP_COLOR[g], alpha=0.85,
                           label=GROUP_TITLE[g]))
    legend_elements += [
        mpatches.Patch(facecolor='#888', alpha=0.85, label='RF (solid)'),
        mpatches.Patch(facecolor='#888', alpha=0.55, hatch='///',
                       edgecolor='#888', linewidth=0.6, label='MLP (hatched)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=7,
               fontsize=7.5, frameon=False, labelcolor=INK2,
               bbox_to_anchor=(0.55, -0.02))

    fig.suptitle(
        'Multimodal RF vs MLP — CD vs UC  (at-20-cm, visit-level, 5-fold CV)',
        fontsize=10.5, fontweight='bold', color=INK,
        fontfamily='sans-serif', y=1.01)

    fig.subplots_adjust(left=0.18, right=0.985, top=0.88,
                        bottom=0.14, wspace=0.06)
    return fig


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    df = load()

    print(f'\n{"Arm":<44} {"Clf":>4} {"AUC":>20} {"F1-CD":>20} {"F1-UC":>20}')
    print('-' * 110)
    for g in GROUP_ORDER:
        print(f'\n  ── {GROUP_TITLE[g]} ──')
        grp = df[df['group'] == g].sort_values('mean_auc', ascending=False)
        for _, row in grp.iterrows():
            print(f"    {row['label'].replace(chr(10),' '):<42} {row['clf']:>4}  "
                  f"AUC={row['mean_auc']:.4f}±{row['std_auc']:.4f}  "
                  f"F1-CD={row['mean_f1cd']:.4f}±{row['std_f1cd']:.4f}  "
                  f"F1-UC={row['mean_f1uc']:.4f}±{row['std_f1uc']:.4f}")

    fig = plot(df)
    out_png = os.path.join(OUT_DIR, 'multimodal_comparison.png')
    out_pdf = os.path.join(OUT_DIR, 'multimodal_comparison.pdf')
    fig.savefig(out_png, dpi=200, bbox_inches='tight', facecolor=SURF)
    fig.savefig(out_pdf,           bbox_inches='tight', facecolor=SURF)
    plt.close(fig)
    print(f'\n  Saved: {out_png}')
    print(f'  Saved: {out_pdf}')


if __name__ == '__main__':
    main()
