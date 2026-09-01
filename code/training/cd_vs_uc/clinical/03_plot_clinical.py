"""
Separate panel figures for the clinical CD vs UC classifier.

Produces three independent figures:
  Panel A — Confusion matrix (patient-level, mode across folds)
  Panel B — F1 scores: CD F1 + UC F1 per fold with means
  Panel C — SHAP beeswarm (dot colour = feature value, x = SHAP value)

Run after 01_train_clinical.py and 02_shap_clinical.py.

Outputs  /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/plots/
--------
  clinical_panel_cm.{pdf,png}
  clinical_panel_f1.{pdf,png}
  clinical_panel_beeswarm.{pdf,png}
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix

RESULTS_DIR = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/results'
SHAP_DIR    = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/shap'
PLOTS_DIR   = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/plots'

# ── Style ─────────────────────────────────────────────────────────────────────
SURF     = '#fcfcfb'
INK      = '#0b0b0b'
INK2     = '#52514e'
MUTED    = '#898781'
GRID     = '#e1e0d9'
BASE     = '#c3c2b7'
UC_COLOR = '#c94040'
CD_COLOR = '#2a78d6'

FEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'feat', ['#2166ac', '#f7f7f7', '#d6604d'])

FEATURE_LABELS = {
    'rectal_bleed':            'Rectal bleeding',
    'first_abdopain':          'Abdominal pain',
    'first_daily_bm':          'Daily BM (delta)',
    'first_well_being':        'Well-being score',
    'stool_freq':              'Stool frequency',
    'fecal_urgency':           'Fecal urgency',
    'da6m':                    'Disease activity (6 m)',
    'smk':                     'Smoking',
    'cardiovascular_disease':  'Cardiovascular disease',
    'hypertension':            'Hypertension',
    'diabetes_type_i':         'Diabetes type I',
    'diabetes_type_ii':        'Diabetes type II',
    'ckd':                     'CKD',
    'hiv':                     'HIV',
    'tuberculosis':            'Tuberculosis',
}


def _rc():
    plt.rcParams.update({
        'figure.facecolor':  SURF,
        'axes.facecolor':    SURF,
        'axes.edgecolor':    BASE,
        'axes.linewidth':    0.6,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.color':        GRID,
        'grid.linewidth':    0.4,
        'axes.axisbelow':    True,
        'xtick.color':       MUTED,
        'ytick.color':       INK2,
        'xtick.labelsize':   8,
        'ytick.labelsize':   9,
        'font.family':       'DejaVu Sans',
        'font.size':         9,
        'text.color':        INK,
    })


def _save(fig, stem):
    for ext in ('pdf', 'png'):
        path = os.path.join(PLOTS_DIR, f'{stem}.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=SURF)
        print(f'  Saved {path}')
    plt.close(fig)


# ── Panel A: Confusion matrix ─────────────────────────────────────────────────

def plot_cm(preds_df, n_patients, mean_auc, std_auc):
    clin = preds_df[preds_df['strategy'] == 'clinical']
    pat  = clin.groupby('patient_id').agg(
        true_label=('true_label', 'first'),
        pred_label=('pred_label', lambda x: x.mode()[0]),
    ).reset_index()

    cm      = confusion_matrix(pat['true_label'], pat['pred_label'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cd_n    = (pat['true_label'] == 0).sum()
    uc_n    = (pat['true_label'] == 1).sum()

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    fig.patch.set_facecolor(SURF)

    ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm_norm[i, j] > 0.6 else INK
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.1%})',
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color=color)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['CD', 'UC'], fontsize=11)
    ax.set_yticklabels(['CD', 'UC'], fontsize=11)
    ax.set_xlabel('Predicted', fontsize=9, color=INK2)
    ax.set_ylabel('True', fontsize=9, color=INK2)
    ax.grid(False)
    ax.set_title(
        f'A   Confusion matrix\n'
        f'N={n_patients}  (CD={cd_n}, UC={uc_n})  ·  AUC {mean_auc:.3f} ± {std_auc:.3f}',
        fontsize=10, fontweight='bold', color=INK, loc='left', pad=6)

    fig.tight_layout()
    _save(fig, 'clinical_panel_cm')


# ── Panel B: F1 scores ────────────────────────────────────────────────────────

def plot_f1(fold_df):
    df = fold_df[fold_df['strategy'] == 'clinical'].copy()

    folds   = df['fold'].values
    cd_f1   = df['cd_f1'].values
    uc_f1   = df['uc_f1'].values
    acc     = df['accuracy'].values
    n_folds = len(folds)

    mean_cd  = cd_f1.mean();  std_cd  = cd_f1.std(ddof=1)
    mean_uc  = uc_f1.mean();  std_uc  = uc_f1.std(ddof=1)
    mean_acc = acc.mean();    std_acc = acc.std(ddof=1)

    x = np.arange(n_folds)
    w = 0.28

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    fig.patch.set_facecolor(SURF)

    # Grouped bars: CD F1, UC F1, Accuracy
    bars_cd  = ax.bar(x - w, cd_f1,  width=w, color=CD_COLOR,
                      alpha=0.80, edgecolor=SURF, linewidth=0.6, label='CD F1', zorder=3)
    bars_uc  = ax.bar(x,     uc_f1,  width=w, color=UC_COLOR,
                      alpha=0.80, edgecolor=SURF, linewidth=0.6, label='UC F1', zorder=3)
    bars_acc = ax.bar(x + w, acc,    width=w, color=MUTED,
                      alpha=0.70, edgecolor=SURF, linewidth=0.6, label='Accuracy', zorder=3)

    # Mean lines
    ax.axhline(mean_cd,  color=CD_COLOR, lw=1.2, ls='--', zorder=4)
    ax.axhline(mean_uc,  color=UC_COLOR, lw=1.2, ls='--', zorder=4)
    ax.axhline(mean_acc, color=MUTED,    lw=1.0, ls=':',  zorder=4)

    # Mean ± SD annotation on right margin
    ax.text(n_folds - 0.5 + w + 0.08, mean_cd,
            f'{mean_cd:.3f}±{std_cd:.3f}',
            va='center', ha='left', fontsize=7, color=CD_COLOR, fontweight='bold')
    ax.text(n_folds - 0.5 + w + 0.08, mean_uc,
            f'{mean_uc:.3f}±{std_uc:.3f}',
            va='center', ha='left', fontsize=7, color=UC_COLOR, fontweight='bold')
    ax.text(n_folds - 0.5 + w + 0.08, mean_acc,
            f'{mean_acc:.3f}±{std_acc:.3f}',
            va='center', ha='left', fontsize=7, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in folds], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=9, color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.4)
    ax.xaxis.grid(False)
    ax.spines['left'].set_color(BASE)
    ax.spines['bottom'].set_color(BASE)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9,
              edgecolor=GRID, loc='lower right')
    ax.set_title('B   F1 scores per fold\n(CD F1, UC F1, Accuracy)',
                 fontsize=10, fontweight='bold', color=INK, loc='left', pad=6)

    fig.tight_layout()
    _save(fig, 'clinical_panel_f1')


# ── Panel C: SHAP beeswarm ────────────────────────────────────────────────────

def beeswarm_simple(values, y_center=0.0, bandwidth=0.38):
    n = len(values)
    if n == 0:
        return np.full(n, y_center)
    n_bins  = max(20, n // 8)
    _, edges = np.histogram(values, bins=n_bins)
    bin_idx  = np.clip(np.digitize(values, edges[:-1]) - 1, 0, n_bins - 1)
    yp = np.zeros(n)
    for b in range(n_bins):
        mask = bin_idx == b
        k = mask.sum()
        if k <= 1:
            continue
        spread = bandwidth * min(1.0, k / 8.0)
        pos = np.linspace(-spread, spread, k)
        np.random.shuffle(pos)
        yp[mask] = pos
    return yp + y_center


def plot_beeswarm(npz_path, n_patients):
    data       = np.load(npz_path, allow_pickle=True)
    shap_vals  = data['shap_values']
    feat_vals  = data['feature_values']
    feat_names = data['feature_names'].tolist()

    n_samples, n = shap_vals.shape
    fig_h = max(3.5, n * 0.40 + 1.6)

    fig, ax = plt.subplots(figsize=(5.4, fig_h))
    fig.patch.set_facecolor(SURF)
    np.random.seed(42)

    for row_idx in range(n):
        y_center = row_idx
        col      = n - 1 - row_idx        # col 0 = most important → top row
        sv = shap_vals[:, col]
        fv = feat_vals[:, col]

        fmin, fmax = np.nanpercentile(fv, 1), np.nanpercentile(fv, 99)
        fv_norm = np.clip((fv - fmin) / (fmax - fmin), 0.0, 1.0) if fmax > fmin \
                  else np.full_like(fv, 0.5)

        y_jitter = beeswarm_simple(sv, y_center=y_center)
        ax.scatter(sv, y_jitter, s=4.0 ** 2 * 0.8, c=FEAT_CMAP(fv_norm),
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
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.03, pad=0.02, aspect=30, ticks=[0, 0.5, 1])
    cbar.set_ticklabels(['low', 'mid', 'high'], fontsize=6, color=INK2)
    cbar.set_label('Feature value', fontsize=6.5, color=INK2)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=6)

    fig.tight_layout(pad=0.6)
    _save(fig, 'clinical_panel_beeswarm')


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    _rc()

    fold_df  = pd.read_csv(os.path.join(RESULTS_DIR, 'clinical_fold_metrics.csv'))
    preds_df = pd.read_csv(os.path.join(RESULTS_DIR, 'clinical_patient_predictions.csv'))
    npz_path = os.path.join(SHAP_DIR, 'shap_clinical_values.npz')

    with open(os.path.join(SHAP_DIR, 'shap_clinical_summary.json')) as f:
        summary = json.load(f)

    n_patients = summary['n_patients']
    mean_auc   = summary['auc']
    std_auc    = summary['std_auc']

    print('Plotting confusion matrix ...')
    plot_cm(preds_df, n_patients, mean_auc, std_auc)

    print('Plotting F1 scores ...')
    plot_f1(fold_df)

    print('Plotting SHAP beeswarm ...')
    plot_beeswarm(npz_path, n_patients)

    print('Done.')


if __name__ == '__main__':
    main()
