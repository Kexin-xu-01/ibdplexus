"""
Plots for the multimodal CD vs UC classifiers (clinical + RNA + imaging).

Produces four separate panel figures:
  panel_auc          — AUC & AP comparison (RF vs MLP, all arms)
  panel_cm_rf        — Confusion matrices for all four RF arms
  panel_beeswarm_rna — SHAP beeswarm for RNA RF arm (top 30 genes)
  panel_beeswarm_concat — SHAP beeswarm for concat RF arm (top 30 features)

Run after 01_train.py and 02_shap.py.

Outputs  /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/multimodal/clinical_rna_image/plots/
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

RESULTS_DIR = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/multimodal/'
               'clinical_rna_image/results')
SHAP_DIR    = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/multimodal/'
               'clinical_rna_image/shap')
PLOTS_DIR   = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/multimodal/'
               'clinical_rna_image/plots')
TPM_GCT     = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/'
               'transcriptomics/GSF2048892_combined_TPM_matrix_with_header.gct')

# ── Style ─────────────────────────────────────────────────────────────────────
SURF     = '#fcfcfb'
INK      = '#0b0b0b'
INK2     = '#52514e'
MUTED    = '#898781'
GRID     = '#e1e0d9'
BASE     = '#c3c2b7'
UC_COLOR = '#c94040'
CD_COLOR = '#2a78d6'
RF_COLOR = '#4a7fb5'
MLP_COLOR = '#e07b3a'

FEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'feat', ['#2166ac', '#f7f7f7', '#d6604d'])

CLINICAL_LABELS = {
    'rectal_bleed': 'Rectal bleeding', 'first_abdopain': 'Abdominal pain',
    'first_daily_bm': 'Daily BM (delta)', 'first_well_being': 'Well-being score',
    'stool_freq': 'Stool frequency', 'fecal_urgency': 'Fecal urgency',
    'da6m': 'Disease activity (6 m)', 'smk': 'Smoking',
    'cardiovascular_disease': 'Cardiovascular disease',
    'hypertension': 'Hypertension', 'diabetes_type_i': 'Diabetes type I',
    'diabetes_type_ii': 'Diabetes type II', 'ckd': 'CKD',
    'hiv': 'HIV', 'tuberculosis': 'Tuberculosis',
}

ARM_LABELS = {
    'clinical': 'Clinical', 'img': 'Imaging\n(prism2_base)',
    'rna': 'RNA\n(log-TPM)', 'concat': 'Concat\n(all three)',
}


def _rc():
    plt.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF,
        'axes.edgecolor': BASE, 'axes.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.4,
        'axes.axisbelow': True, 'xtick.color': MUTED, 'ytick.color': INK2,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK,
    })


def _save(fig, stem):
    for ext in ('pdf', 'png'):
        path = os.path.join(PLOTS_DIR, f'{stem}.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=SURF)
        print(f'  Saved {path}')
    plt.close(fig)


# ── Gene symbol lookup ────────────────────────────────────────────────────────

def load_gene_map():
    """Load ENSG→symbol from GCT Name/Description columns only (fast)."""
    print('Loading gene symbol map ...')
    try:
        gct = pd.read_csv(TPM_GCT, sep='\t', skiprows=2, header=0,
                          usecols=['Name', 'Description'])
        gene_names = gct['Name'].tolist()          # position → ENSG ID
        ensg_sym   = dict(zip(gct['Name'], gct['Description']))
        print(f'  {len(gene_names)} genes, {len(ensg_sym)} ENSG→symbol entries')
        return gene_names, ensg_sym
    except Exception as e:
        print(f'  Warning: could not load gene map ({e})')
        return [], {}


def display_name(feat, gene_names, ensg_sym):
    """Resolve g{idx} → symbol, clinical col → label, f{idx} → Img dim {idx}."""
    if feat in CLINICAL_LABELS:
        return CLINICAL_LABELS[feat]
    if feat.startswith('g') and feat[1:].isdigit():
        idx  = int(feat[1:])
        ensg = gene_names[idx] if idx < len(gene_names) else feat
        return ensg_sym.get(ensg, ensg)
    if feat.startswith('f') and feat[1:].isdigit():
        return f'Img {feat[1:]}'
    return feat


# ── Panel 1: AUC + AP comparison ─────────────────────────────────────────────

def plot_auc(fold_df):
    arms    = ['clinical', 'img', 'rna', 'concat']
    metrics = ['auc', 'ap']
    titles  = ['AUC-ROC', 'Average Precision']
    n_arms  = len(arms)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(SURF)

    for ax, metric, title in zip(axes, metrics, titles):
        x   = np.arange(n_arms)
        w   = 0.32

        rf_means, rf_stds   = [], []
        mlp_means, mlp_stds = [], []

        for arm in arms:
            rf_vals  = fold_df[fold_df['strategy'] == f'{arm}_rf'][metric].values
            mlp_vals = fold_df[fold_df['strategy'] == f'{arm}_mlp'][metric].values
            rf_means.append(rf_vals.mean());  rf_stds.append(rf_vals.std(ddof=1))
            mlp_means.append(mlp_vals.mean()); mlp_stds.append(mlp_vals.std(ddof=1))

        rf_means  = np.array(rf_means);  rf_stds  = np.array(rf_stds)
        mlp_means = np.array(mlp_means); mlp_stds = np.array(mlp_stds)

        ax.bar(x - w/2, rf_means, width=w, color=RF_COLOR, alpha=0.85,
               edgecolor=SURF, linewidth=0.6, label='RF', zorder=3)
        ax.bar(x + w/2, mlp_means, width=w, color=MLP_COLOR, alpha=0.85,
               edgecolor=SURF, linewidth=0.6, label='MLP', zorder=3)

        ax.errorbar(x - w/2, rf_means, yerr=rf_stds, fmt='none',
                    ecolor=INK2, elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
        ax.errorbar(x + w/2, mlp_means, yerr=mlp_stds, fmt='none',
                    ecolor=INK2, elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)

        # value labels on bars
        for xi, (rm, mm) in enumerate(zip(rf_means, mlp_means)):
            ax.text(xi - w/2, rm + rf_stds[xi] + 0.005, f'{rm:.3f}',
                    ha='center', va='bottom', fontsize=6.5, color=RF_COLOR, fontweight='bold')
            ax.text(xi + w/2, mm + mlp_stds[xi] + 0.005, f'{mm:.3f}',
                    ha='center', va='bottom', fontsize=6.5, color=MLP_COLOR, fontweight='bold')

        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':', zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABELS[a] for a in arms], fontsize=9)
        ax.set_ylim(0.3, 1.05)
        ax.set_ylabel(title, fontsize=9, color=INK2)
        ax.yaxis.grid(True, color=GRID, linewidth=0.4)
        ax.xaxis.grid(False)
        ax.spines['left'].set_color(BASE); ax.spines['bottom'].set_color(BASE)
        ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor=GRID)
        ax.set_title(title, fontsize=10, fontweight='bold', color=INK, loc='left', pad=6)

    fig.suptitle('Multimodal CD vs UC — 817 patients · 5-fold CV',
                 fontsize=11, fontweight='bold', color=INK, y=1.02)
    fig.tight_layout(w_pad=3)
    _save(fig, 'panel_auc')


# ── Panel 2: Confusion matrices (RF arms) ────────────────────────────────────

def plot_cm_rf(preds_df, fold_df):
    arms = ['clinical', 'img', 'rna', 'concat']
    labels_display = ['CD', 'UC']

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    fig.patch.set_facecolor(SURF)

    for ax, arm in zip(axes, arms):
        df = preds_df[preds_df['strategy'] == f'{arm}_rf']
        pat = df.groupby('patient_id').agg(
            true_label=('true_label', 'first'),
            pred_label=('pred_label', lambda x: x.mode()[0]),
        ).reset_index()

        cm      = confusion_matrix(pat['true_label'], pat['pred_label'])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for i in range(2):
            for j in range(2):
                color = 'white' if cm_norm[i, j] > 0.6 else INK
                ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.1%})',
                        ha='center', va='center', fontsize=9,
                        fontweight='bold', color=color)

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels_display, fontsize=10)
        ax.set_yticklabels(labels_display, fontsize=10)
        ax.set_xlabel('Predicted', fontsize=8, color=INK2)
        ax.set_ylabel('True', fontsize=8, color=INK2)
        ax.grid(False)

        rf_aucs = fold_df[fold_df['strategy'] == f'{arm}_rf']['auc']
        auc_str = f'AUC {rf_aucs.mean():.3f}±{rf_aucs.std(ddof=1):.3f}'
        ax.set_title(f'{ARM_LABELS[arm].replace(chr(10), " ")} · RF\n{auc_str}',
                     fontsize=9, fontweight='bold', color=INK, loc='left', pad=6)

    fig.suptitle('Confusion matrices — RF models (patient-level, mode across folds)',
                 fontsize=10, fontweight='bold', color=INK, y=1.04)
    fig.tight_layout(w_pad=2)
    _save(fig, 'panel_cm_rf')


# ── Panel 3 & 4: SHAP beeswarms ──────────────────────────────────────────────

def beeswarm_simple(values, y_center=0.0, bandwidth=0.38):
    n = len(values)
    if n == 0:
        return np.full(n, y_center)
    n_bins   = max(20, n // 8)
    _, edges = np.histogram(values, bins=n_bins)
    bin_idx  = np.clip(np.digitize(values, edges[:-1]) - 1, 0, n_bins - 1)
    yp = np.zeros(n)
    for b in range(n_bins):
        mask = bin_idx == b
        k    = mask.sum()
        if k <= 1:
            continue
        spread = bandwidth * min(1.0, k / 8.0)
        pos    = np.linspace(-spread, spread, k)
        np.random.shuffle(pos)
        yp[mask] = pos
    return yp + y_center


def plot_beeswarm(arm, gene_names, ensg_sym, top_n=30, stem=None):
    npz_path = os.path.join(SHAP_DIR, f'shap_{arm}_values.npz')
    data      = np.load(npz_path, allow_pickle=True)
    shap_vals = data['shap_values']    # (n_patients, n_features) sorted desc
    feat_vals = data['feature_values']
    feat_names = data['feature_names'].tolist()
    labels     = data['labels']

    n_samples, n_total = shap_vals.shape
    n = min(top_n, n_total)

    shap_vals  = shap_vals[:, :n]
    feat_vals  = feat_vals[:, :n]
    feat_names = feat_names[:n]

    fig_h = max(3.5, n * 0.40 + 1.8)
    fig, ax = plt.subplots(figsize=(5.6, fig_h))
    fig.patch.set_facecolor(SURF)
    np.random.seed(42)

    for row_idx in range(n):
        y_center = row_idx
        col = n - 1 - row_idx    # col 0 = most important → top row
        sv  = shap_vals[:, col]
        fv  = feat_vals[:, col]

        fmin, fmax = np.nanpercentile(fv, 1), np.nanpercentile(fv, 99)
        fv_norm = np.clip((fv - fmin) / (fmax - fmin), 0.0, 1.0) \
                  if fmax > fmin else np.full_like(fv, 0.5)

        y_jitter = beeswarm_simple(sv, y_center=y_center)
        ax.scatter(sv, y_jitter, s=4.0 ** 2 * 0.8,
                   c=FEAT_CMAP(fv_norm), alpha=0.65,
                   linewidths=0.0, zorder=3, rasterized=True)

    ax.axvline(0, color=MUTED, lw=0.7, ls='--', zorder=2)

    y_labels = [display_name(feat_names[n - 1 - i], gene_names, ensg_sym)
                for i in range(n)]
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=7.5, color=INK)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel('SHAP value  (← CD  |  UC →)', fontsize=9, color=INK2)
    ax.tick_params(axis='x', labelsize=7, colors=INK2, width=0.6, length=3)
    ax.tick_params(axis='y', length=0)
    ax.spines['bottom'].set_linewidth(0.6); ax.spines['bottom'].set_color(MUTED)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    with open(os.path.join(SHAP_DIR, 'shap_multimodal_summary.json')) as f:
        summ = {s['arm']: s for s in json.load(f)}
    auc = summ[arm]['mean_auc']; std = summ[arm]['std_auc']
    arm_label = ARM_LABELS.get(arm, arm).replace('\n', ' ')

    ax.set_title(
        f'SHAP beeswarm — {arm_label} RF\n'
        f'n = {n_samples} patients · AUC {auc:.3f} ± {std:.3f} · top {n} features',
        fontsize=9, fontweight='bold', color=INK, loc='left', pad=6)

    sm = plt.cm.ScalarMappable(cmap=FEAT_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.03, pad=0.02, aspect=30, ticks=[0, 0.5, 1])
    cbar.set_ticklabels(['low', 'mid', 'high'], fontsize=6, color=INK2)
    cbar.set_label('Feature value', fontsize=6.5, color=INK2)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=6)

    fig.tight_layout(pad=0.6)
    out_stem = stem or f'panel_beeswarm_{arm}'
    _save(fig, out_stem)


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    _rc()

    fold_df  = pd.read_csv(os.path.join(RESULTS_DIR, 'multimodal_fold_metrics.csv'))
    preds_df = pd.read_csv(os.path.join(RESULTS_DIR, 'multimodal_patient_predictions.csv'))

    print('Plotting AUC/AP comparison ...')
    plot_auc(fold_df)

    print('Plotting RF confusion matrices ...')
    plot_cm_rf(preds_df, fold_df)

    print('Loading gene symbol map ...')
    gene_names, ensg_sym = load_gene_map()

    print('Plotting SHAP beeswarm — clinical ...')
    plot_beeswarm('clinical', gene_names, ensg_sym, top_n=15)

    print('Plotting SHAP beeswarm — RNA ...')
    plot_beeswarm('rna', gene_names, ensg_sym, top_n=30)

    print('Plotting SHAP beeswarm — concat ...')
    plot_beeswarm('concat', gene_names, ensg_sym, top_n=30)

    print('Done.')


if __name__ == '__main__':
    main()
