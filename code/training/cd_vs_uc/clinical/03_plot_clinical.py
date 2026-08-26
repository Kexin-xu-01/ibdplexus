"""
Plots for the clinical CD vs UC classifier.

Produces a 3-panel figure:
  Panel A — AUC per fold + mean ± SD
  Panel B — Confusion matrix (aggregated across all folds)
  Panel C — SHAP beeswarm (dot colour = feature value, x = SHAP value)

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
import matplotlib.colors as mcolors
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

# colormap: blue (low feature value) → red (high feature value)
FEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'feat', ['#2166ac', '#f7f7f7', '#d6604d'])

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

    for xi, auc in zip(x, aucs):
        ax.text(xi, auc + 0.005, f'{auc:.3f}',
                ha='center', va='bottom', fontsize=8, color=INK2)

    ax.axhline(mean_auc, color=CD_COLOR, linewidth=1.4, linestyle='--', zorder=4)
    ax.axhspan(mean_auc - std_auc, mean_auc + std_auc,
               color=CD_COLOR, alpha=0.10, zorder=2)
    ax.text(len(folds) - 0.45, mean_auc + std_auc + 0.004,
            f'Mean {mean_auc:.3f} ± {std_auc:.3f}',
            ha='right', fontsize=8, color=CD_COLOR, fontweight='bold')

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


# ── Panel C: SHAP beeswarm ─────────────────────────────────────────────────────

def beeswarm_simple(values, y_center=0.0, bandwidth=0.38):
    """Fast approximate beeswarm using histogram-style binning."""
    n = len(values)
    if n == 0:
        return np.full(n, y_center)

    n_bins = max(20, n // 8)
    counts, edges = np.histogram(values, bins=n_bins)
    bin_idx = np.digitize(values, edges[:-1]) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    yp = np.zeros(n)
    for b in range(n_bins):
        mask = bin_idx == b
        k = mask.sum()
        if k == 0:
            continue
        if k == 1:
            yp[mask] = 0.0
        else:
            spread = bandwidth * min(1.0, k / 8.0)
            positions = np.linspace(-spread, spread, k)
            np.random.shuffle(positions)
            yp[mask] = positions

    return yp + y_center


def panel_beeswarm(ax, npz_path, n_patients):
    data = np.load(npz_path, allow_pickle=True)
    shap_vals  = data['shap_values']    # (n_samples, n_features), sorted descending
    feat_vals  = data['feature_values'] # (n_samples, n_features)
    feat_names = data['feature_names'].tolist()

    n_samples, n_features = shap_vals.shape
    n = n_features

    np.random.seed(42)

    for row_idx in range(n):
        y_center = row_idx  # bottom=0 → top=n-1

        # col 0 = most important → top row (row_idx = n-1)
        sv = shap_vals[:, (n - 1 - row_idx)]
        fv = feat_vals[:, (n - 1 - row_idx)]

        fmin, fmax = np.nanpercentile(fv, 1), np.nanpercentile(fv, 99)
        if fmax > fmin:
            fv_norm = np.clip((fv - fmin) / (fmax - fmin), 0.0, 1.0)
        else:
            fv_norm = np.full_like(fv, 0.5)

        y_jitter = beeswarm_simple(sv, y_center=y_center, bandwidth=0.38)
        colors = FEAT_CMAP(fv_norm)
        ax.scatter(sv, y_jitter, s=4.0 ** 2 * 0.8, c=colors,
                   alpha=0.65, linewidths=0.0, zorder=3, rasterized=True)

    ax.axvline(0, color=MUTED, lw=0.7, ls='--', zorder=2)

    y_labels = [FEATURE_LABELS.get(feat_names[n - 1 - i], feat_names[n - 1 - i])
                for i in range(n)]
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=8, color=INK)
    ax.set_ylim(-0.6, n - 0.4)

    ax.set_xlabel('SHAP value  (← CD  |  UC →)', fontsize=9, color=INK2)
    ax.tick_params(axis='x', labelsize=7, colors=INK2, width=0.6, length=3)
    ax.tick_params(axis='y', length=0)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['bottom'].set_color(MUTED)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(f'C   SHAP feature importance\nn = {n_patients} patients',
                 fontsize=10, fontweight='bold', color=INK, loc='left', pad=6)

    sm = plt.cm.ScalarMappable(cmap=FEAT_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, orientation='vertical',
                               fraction=0.03, pad=0.02, aspect=30,
                               ticks=[0, 0.5, 1])
    cbar.set_ticklabels(['low', 'mid', 'high'], fontsize=6, color=INK2)
    cbar.set_label('Feature value', fontsize=6.5, color=INK2)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=6)


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    setup_style()

    fold_df  = pd.read_csv(os.path.join(RESULTS_DIR, 'clinical_fold_metrics.csv'))
    preds_df = pd.read_csv(os.path.join(RESULTS_DIR, 'clinical_patient_predictions.csv'))
    npz_path = os.path.join(SHAP_DIR, 'shap_clinical_values.npz')

    with open(os.path.join(SHAP_DIR, 'shap_clinical_summary.json')) as f:
        summary = json.load(f)

    n_patients = summary['n_patients']
    mean_auc   = summary['auc']
    std_auc    = summary['std_auc']

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor(SURF)
    fig.suptitle(
        f'Clinical variables — CD vs UC  ·  At-20-cm matched cohort  ·  '
        f'N={n_patients}  ·  AUC {mean_auc:.3f} ± {std_auc:.3f}',
        fontsize=12, fontweight='bold', color=INK, y=1.02,
    )

    panel_auc(axes[0], fold_df[fold_df['strategy'] == 'clinical'])
    panel_cm(axes[1], preds_df[preds_df['strategy'] == 'clinical'])
    panel_beeswarm(axes[2], npz_path, n_patients)

    fig.tight_layout(w_pad=3)

    for ext in ('pdf', 'png'):
        path = os.path.join(PLOTS_DIR, f'clinical_results_panel.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=SURF)
        print(f'Saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
