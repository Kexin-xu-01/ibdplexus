"""
Plots for the clinical CD vs UC classifier.

Produces a 3-panel figure:
  Panel A — AUC per fold + mean ± SD
  Panel B — Confusion matrix (aggregated across all folds)
  Panel C — SHAP horizontal bar (top features, coloured by direction)

Run after train_clinical.py and shap_clinical.py.

Outputs  /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/plots/
--------
  clinical_results_panel.pdf
  clinical_results_panel.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix

RESULTS_DIR = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/results'
SHAP_DIR    = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/shap'
PLOTS_DIR   = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/plots'

# ── Style (matches project palette) ───────────────────────────────────────────
SURF  = '#fcfcfb'
INK   = '#0b0b0b'
INK2  = '#52514e'
MUTED = '#898781'
GRID  = '#e1e0d9'
BASE  = '#c3c2b7'
UC_COLOR = '#c94040'
CD_COLOR = '#2a78d6'
NEUTRAL  = '#6baed6'

FEATURE_LABELS = {
    'rectal_bleed':                   'Rectal bleeding',
    'first_abdopain':                 'Abdominal pain',
    'first_daily_bm':                 'Daily BM (delta)',
    'first_well_being':               'Well-being score',
    'stool_freq':                     'Stool frequency',
    'fecal_urgency':                  'Fecal urgency',
    'da6m':                           'Disease activity (6 m)',
    'smk':                            'Smoking',
    'cardiovascular_disease':         'Cardiovascular disease',
    'hypertension':                   'Hypertension',
    'diabetes_type_i':                'Diabetes type I',
    'diabetes_type_ii':               'Diabetes type II',
    'ckd':                            'CKD',
    'hiv':                            'HIV',
    'tuberculosis':                   'Tuberculosis',
}


def setup_style():
    plt.rcParams.update({
        'figure.facecolor':   SURF,
        'axes.facecolor':     SURF,
        'axes.edgecolor':     BASE,
        'axes.linewidth':     0.6,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.grid':          True,
        'grid.color':         GRID,
        'grid.linewidth':     0.4,
        'axes.axisbelow':     True,
        'xtick.color':        MUTED,
        'ytick.color':        INK2,
        'xtick.labelsize':    8,
        'ytick.labelsize':    9,
        'font.family':        'DejaVu Sans',
        'font.size':          9,
        'text.color':         INK,
    })


# ── Panel A: AUC per fold ──────────────────────────────────────────────────────

def panel_auc(ax, fold_df):
    folds      = fold_df['fold'].values
    aucs       = fold_df['auc'].values
    mean_auc   = aucs.mean()
    std_auc    = aucs.std(ddof=1)

    x = np.arange(len(folds))
    bars = ax.bar(x, aucs, width=0.55, color=NEUTRAL,
                  edgecolor=SURF, linewidth=0.8, zorder=3)

    # Value labels on bars
    for xi, auc in zip(x, aucs):
        ax.text(xi, auc + 0.005, f'{auc:.3f}',
                ha='center', va='bottom', fontsize=8, color=INK2)

    # Mean ± SD line
    ax.axhline(mean_auc, color=CD_COLOR, linewidth=1.4, linestyle='--', zorder=4)
    ax.axhspan(mean_auc - std_auc, mean_auc + std_auc,
               color=CD_COLOR, alpha=0.10, zorder=2)
    ax.text(len(folds) - 0.45, mean_auc + std_auc + 0.004,
            f'Mean {mean_auc:.3f} ± {std_auc:.3f}',
            ha='right', fontsize=8, color=CD_COLOR, fontweight='bold')

    # Chance line
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=':', zorder=2)
    ax.text(len(folds) - 0.45, 0.502, 'Chance', ha='right',
            fontsize=7, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in folds], fontsize=8)
    ax.set_ylim(0.3, 0.85)
    ax.set_ylabel('AUC-ROC', fontsize=9, color=INK2)
    ax.set_title('A   AUC per fold', fontsize=10, fontweight='bold',
                 color=INK, loc='left', pad=6)
    ax.yaxis.grid(True, color=GRID, linewidth=0.4)
    ax.xaxis.grid(False)
    ax.spines['left'].set_color(BASE)
    ax.spines['bottom'].set_color(BASE)


# ── Panel B: Confusion matrix ──────────────────────────────────────────────────

def panel_cm(ax, preds_df):
    # Aggregate predictions: one row per patient (mode of pred_label across folds)
    pat = preds_df.groupby('patient_id').agg(
        true_label=('true_label', 'first'),
        pred_label=('pred_label', lambda x: x.mode()[0]),
    ).reset_index()

    cm = confusion_matrix(pat['true_label'], pat['pred_label'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels = ['CD', 'UC']
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm_norm[i, j] > 0.6 else INK
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.1%})',
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color=color)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=9, color=INK2)
    ax.set_ylabel('True', fontsize=9, color=INK2)
    ax.set_title('B   Confusion matrix\n(patient-level, mode across folds)',
                 fontsize=10, fontweight='bold', color=INK, loc='left', pad=6)
    ax.grid(False)

    n = len(pat)
    cd_n = (pat['true_label'] == 0).sum()
    uc_n = (pat['true_label'] == 1).sum()
    ax.text(0.5, -0.22, f'N={n}  (CD={cd_n}, UC={uc_n})',
            transform=ax.transAxes, ha='center', fontsize=8, color=MUTED)


# ── Panel C: SHAP bar ──────────────────────────────────────────────────────────

def panel_shap(ax, shap_df):
    top = shap_df.head(15).copy()
    top = top.iloc[::-1].reset_index(drop=True)   # flip for horizontal bar

    labels = [FEATURE_LABELS.get(f, f) for f in top['feature']]
    values = top['mean_abs_shap'].values
    colors = [UC_COLOR if d == 'UC' else CD_COLOR for d in top['direction']]

    y = np.arange(len(top))
    ax.barh(y, values, height=0.6, color=colors, edgecolor=SURF, linewidth=0.8)

    max_v = values.max()
    for i, (val, row) in enumerate(zip(values, top.itertuples())):
        ax.text(val + max_v * 0.01, i, f'{val:.4f}',
                va='center', ha='left', fontsize=7.5, color=INK2)
        tag = f'↑{row.direction}'
        tag_color = UC_COLOR if row.direction == 'UC' else CD_COLOR
        ax.text(val + max_v * 0.01, i - 0.3, tag,
                va='center', ha='left', fontsize=6.5,
                color=tag_color, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel('Mean |SHAP value|', fontsize=9, color=INK2)
    ax.set_xlim(0, max_v * 1.4)
    ax.set_title('C   SHAP feature importance\n(mean |SHAP| across 5 folds)',
                 fontsize=10, fontweight='bold', color=INK, loc='left', pad=6)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4)
    ax.yaxis.grid(False)
    ax.spines['left'].set_color(BASE)
    ax.spines['bottom'].set_color(BASE)

    handles = [
        mpatches.Patch(color=UC_COLOR, label='↑ UC  (higher in UC)'),
        mpatches.Patch(color=CD_COLOR, label='↑ CD  (higher in CD)'),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=True, framealpha=0.9,
              edgecolor=GRID, loc='lower right')


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    setup_style()

    fold_df  = pd.read_csv(os.path.join(RESULTS_DIR, 'clinical_fold_metrics.csv'))
    preds_df = pd.read_csv(os.path.join(RESULTS_DIR, 'clinical_patient_predictions.csv'))
    shap_df  = pd.read_csv(os.path.join(SHAP_DIR, 'shap_clinical.csv'))

    with open(os.path.join(SHAP_DIR, 'shap_clinical_summary.json')) as f:
        summary = json.load(f)

    n_patients = summary['n_patients']
    mean_auc   = summary['auc']
    std_auc    = summary['std_auc']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(SURF)
    fig.suptitle(
        f'Clinical variables — CD vs UC  ·  At-20-cm matched cohort  ·  '
        f'N={n_patients}  ·  AUC {mean_auc:.3f} ± {std_auc:.3f}',
        fontsize=12, fontweight='bold', color=INK, y=1.02,
    )

    panel_auc(axes[0], fold_df[fold_df['strategy'] == 'clinical'])
    panel_cm(axes[1], preds_df[preds_df['strategy'] == 'clinical'])
    panel_shap(axes[2], shap_df)

    fig.tight_layout(w_pad=3)

    for ext in ('pdf', 'png'):
        path = os.path.join(PLOTS_DIR, f'clinical_results_panel.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=SURF)
        print(f'Saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
