"""
SHAP beeswarm, confusion matrix, and F1 scores for the proteomics RF classifier.

Replicates the same 5-fold CV pipeline as train_rf_proteomics.py (per-fold
median imputation + StandardScaler fitted on train only) so SHAP values are
computed without data leakage.

Outputs
-------
  at20cm_proteomics/shap/data/shap_proteomics_top500.csv
  at20cm_proteomics/shap/data/beeswarm_proteomics.npz
  at20cm_proteomics/shap/data/shap_proteomics_summary.json
  at20cm_proteomics/shap/plots/beeswarm_proteomics.png
  at20cm_proteomics/shap/plots/confusion_matrix_proteomics.png
  at20cm_proteomics/shap/plots/f1_scores_proteomics.png
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, f1_score,
)

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────────
PROTEOMICS_GCT = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/'
                  'proteomics/GSF1618983_NPX_below_lod_included.gct')
PROT_MAPPING   = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/'
                  'proteomics/ibd_21183_omics_patient_mapping_proteomics_genestack.csv')
CV_PATIENTS    = ('/home/jovyan/kgbk271-ibd-volume/training/'
                  'cv_splits_patients_at20cm_matched.csv')
OUT_SHAP_DATA  = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                  'at20cm_proteomics/shap/data')
OUT_SHAP_PLOTS = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                  'at20cm_proteomics/shap/plots')

RF_PARAMS = dict(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)

# ── plot style (project standard) ─────────────────────────────────────────────
SURF      = '#fcfcfb'
INK       = '#0b0b0b'
INK2      = '#52514e'
MUTED     = '#898781'
GRID      = '#e1e0d9'
UC_COLOR  = '#c94040'
CD_COLOR  = '#2a78d6'
FEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'feat', ['#2166ac', '#f7f7f7', '#d6604d'])


# ── helpers ────────────────────────────────────────────────────────────────────

def gene_symbol(full_name):
    """'GZMB.P10144.OID20604.Inflammation.v1' → 'GZMB'"""
    return full_name.split('.')[0]


def load_proteomics_visits(cv_patients):
    pat_df  = cv_patients.set_index('patient_id')
    cv_pids = set(pat_df.index)

    pmap = pd.read_csv(PROT_MAPPING)
    pmap = pmap[pmap['deidentified_master_patient_id'].isin(cv_pids)].copy()
    sample_ids = set(pmap['SampleID'].astype(str))

    with open(PROTEOMICS_GCT) as f:
        f.readline(); f.readline()
        header = f.readline().strip().split('\t')
    keep_cols = ['NAME'] + [c for c in header if c in sample_ids]
    gct = pd.read_csv(
        PROTEOMICS_GCT, sep='\t', skiprows=2, header=0, usecols=keep_cols,
        dtype={c: (str if c == 'NAME' else np.float32) for c in keep_cols},
    ).set_index('NAME')
    protein_names = gct.index.tolist()
    print(f'  GCT: {len(protein_names)} proteins x {gct.shape[1]} samples')

    X_df = gct.T.astype(np.float32).reset_index()
    X_df.columns = ['SampleID'] + protein_names
    X_df['SampleID'] = X_df['SampleID'].astype(str)

    pmap['SampleID'] = pmap['SampleID'].astype(str)
    merged = X_df.merge(
        pmap[['SampleID', 'deidentified_master_patient_id', 'visit_encounter_id']],
        on='SampleID', how='inner',
    ).merge(
        cv_patients[['patient_id', 'diagnosis', 'fold']],
        left_on='deidentified_master_patient_id', right_on='patient_id', how='inner',
    )
    merged['label'] = (merged['diagnosis'] == 'Ulcerative colitis').astype(int)
    print(f'  Cohort: {len(merged)} samples / {merged["patient_id"].nunique()} patients '
          f"(CD {(merged['label']==0).sum()}, UC {(merged['label']==1).sum()})")
    return merged, protein_names


def prep_fold(X_raw, y, folds, fold):
    """Per-fold median imputation + StandardScaler — no leakage."""
    tr = folds != fold;  va = folds == fold
    X_tr = X_raw[tr].copy();  X_va = X_raw[va].copy()

    train_med = np.nanmedian(X_tr, axis=0)
    for j in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, j]), j] = train_med[j]
        X_va[np.isnan(X_va[:, j]), j] = train_med[j]

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_va = sc.transform(X_va)
    return X_tr, X_va, tr, va


def beeswarm_simple(values, y_center=0.0, bandwidth=0.38):
    n = len(values)
    if n == 0:
        return np.full(n, y_center)
    n_bins = max(20, n // 8)
    _, edges = np.histogram(values, bins=n_bins)
    bin_idx  = np.clip(np.digitize(values, edges[:-1]) - 1, 0, n_bins - 1)
    yp = np.zeros(n)
    for b in range(n_bins):
        mask = bin_idx == b;  k = mask.sum()
        if k <= 1:
            continue
        spread = bandwidth * min(1.0, k / 8.0)
        pos = np.linspace(-spread, spread, k)
        np.random.shuffle(pos)
        yp[mask] = pos
    return yp + y_center


# ── SHAP computation ───────────────────────────────────────────────────────────

def run_shap(X_raw, y, folds, protein_names, n_background=100):
    """5-fold SHAP with TreeExplainer — interventional, 100 background samples."""
    n_samples   = len(y)
    n_features  = len(protein_names)
    shap_matrix = np.zeros((n_samples, n_features), dtype=np.float32)
    feat_matrix = np.zeros((n_samples, n_features), dtype=np.float32)
    aucs, fold_f1_cd, fold_f1_uc = [], [], []
    fold_cms = []

    for fold in range(5):
        X_tr, X_va, tr, va = prep_fold(X_raw, y, folds, fold)

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y[tr])
        uc_col = list(clf.classes_).index(1)

        proba  = clf.predict_proba(X_va)
        y_pred = clf.predict(X_va)
        y_val  = y[va]

        auc = roc_auc_score(y_val, proba[:, uc_col])
        aucs.append(auc)
        cr  = classification_report(y_val, y_pred,
                                    target_names=['CD', 'UC'], output_dict=True)
        fold_f1_cd.append(cr['CD']['f1-score'])
        fold_f1_uc.append(cr['UC']['f1-score'])
        fold_cms.append(confusion_matrix(y_val, y_pred))

        print(f'  fold {fold}: AUC={auc:.4f}  '
              f"F1-CD={cr['CD']['f1-score']:.4f}  F1-UC={cr['UC']['f1-score']:.4f}  "
              f'n_val={va.sum()}', flush=True)

        # SHAP
        bg_idx = np.random.RandomState(42 + fold).choice(
            X_tr.shape[0], min(n_background, X_tr.shape[0]), replace=False)
        explainer = shap.TreeExplainer(
            clf, data=X_tr[bg_idx], feature_perturbation='interventional')
        sv = explainer.shap_values(X_va, check_additivity=False)
        sv_uc = sv[uc_col] if isinstance(sv, list) else sv[:, :, uc_col]

        shap_matrix[va] = sv_uc.astype(np.float32)
        feat_matrix[va] = X_va.astype(np.float32)

    return shap_matrix, feat_matrix, aucs, fold_f1_cd, fold_f1_uc, fold_cms


def build_shap_df(shap_matrix, protein_names):
    mean_abs    = np.abs(shap_matrix).mean(axis=0)
    mean_signed = shap_matrix.mean(axis=0)
    order       = np.argsort(mean_abs)[::-1]
    return pd.DataFrame({
        'protein':       [protein_names[i] for i in order],
        'gene_symbol':   [gene_symbol(protein_names[i]) for i in order],
        'feature_idx':   order,
        'mean_abs_shap': mean_abs[order],
        'mean_shap':     mean_signed[order],
        'direction':     ['UC' if mean_signed[i] > 0 else 'CD' for i in order],
        'rank':          np.arange(1, len(order) + 1),
    })


# ── plot: SHAP beeswarm ────────────────────────────────────────────────────────

def plot_beeswarm(shap_df, shap_matrix, feat_matrix, labels,
                  out_path, top_n=30, mean_auc=None):
    top_idx   = shap_df['feature_idx'].values[:top_n]
    top_names = shap_df['gene_symbol'].values[:top_n]
    sv  = shap_matrix[:, top_idx]   # (n_samples, top_n)
    fv  = feat_matrix[:, top_idx]
    n   = top_n

    fig_height = max(3.5, n * 0.38 + 1.6)
    fig, ax = plt.subplots(figsize=(5.6, fig_height))
    fig.patch.set_facecolor(SURF);  ax.set_facecolor(SURF)
    np.random.seed(42)

    for row_idx in range(n):
        y_center = row_idx
        col = n - 1 - row_idx   # col 0 = most important → top row (y = n-1)
        sv_col = sv[:, col]
        fv_col = fv[:, col]

        fmin, fmax = np.nanpercentile(fv_col, 1), np.nanpercentile(fv_col, 99)
        fv_norm = (np.clip((fv_col - fmin) / (fmax - fmin + 1e-9), 0., 1.)
                   if fmax > fmin else np.full_like(fv_col, 0.5))

        yj = beeswarm_simple(sv_col, y_center=y_center, bandwidth=0.38)
        ax.scatter(sv_col, yj, s=3.5 ** 2 * 0.8, c=FEAT_CMAP(fv_norm),
                   alpha=0.65, linewidths=0.0, zorder=3, rasterized=True)

    ax.axvline(0, color=MUTED, lw=0.7, ls='--', zorder=2)

    y_labels = list(reversed(top_names))   # top of plot = most important
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=7.0, color=INK, fontfamily='sans-serif')
    ax.set_ylim(-0.6, n - 0.4)

    ax.set_xlabel('SHAP value  (← CD  |  UC →)',
                  fontsize=7.5, color=INK2, fontfamily='sans-serif')
    ax.tick_params(axis='x', labelsize=7, colors=INK2, width=0.6, length=3)
    ax.tick_params(axis='y', length=0)
    ax.spines['bottom'].set_linewidth(0.6);  ax.spines['bottom'].set_color(MUTED)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    title = f'Proteomics (Olink) — SHAP beeswarm (top {top_n} proteins)\n' \
            f'proteomics_visit  ·  n={len(labels)} samples'
    if mean_auc is not None:
        title += f'  ·  AUC {mean_auc:.3f}'
    ax.set_title(title, fontsize=8.5, fontweight='bold', color=INK,
                 fontfamily='sans-serif', loc='left', pad=6)

    sm = plt.cm.ScalarMappable(cmap=FEAT_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.03, pad=0.02, aspect=30, ticks=[0, 0.5, 1])
    cbar.set_ticklabels(['low', 'mid', 'high'], fontsize=6, color=INK2)
    cbar.set_label('Feature value (normalised)', fontsize=6.5, color=INK2,
                   fontfamily='sans-serif')
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=6)

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=SURF, transparent=False)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ── plot: confusion matrix ─────────────────────────────────────────────────────

def plot_confusion_matrix(fold_cms, out_path, aucs, fold_f1_cd, fold_f1_uc):
    """Aggregated confusion matrix + per-fold F1 bar chart side by side."""
    cm_agg = np.sum(fold_cms, axis=0)   # 2x2 aggregated
    cm_pct = cm_agg.astype(float) / cm_agg.sum(axis=1, keepdims=True) * 100

    fig = plt.figure(figsize=(10, 4.2))
    fig.patch.set_facecolor(SURF)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.35)

    # --- left: confusion matrix ---
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor(SURF)
    im = ax0.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    for i in range(2):
        for j in range(2):
            pct  = cm_pct[i, j]
            cnt  = cm_agg[i, j]
            txt_color = 'white' if pct > 55 else INK
            ax0.text(j, i, f'{pct:.1f}%\n({cnt})',
                     ha='center', va='center', fontsize=10,
                     color=txt_color, fontweight='bold',
                     fontfamily='sans-serif')

    ax0.set_xticks([0, 1]);  ax0.set_yticks([0, 1])
    ax0.set_xticklabels(['Pred CD', 'Pred UC'], fontsize=9, color=INK,
                         fontfamily='sans-serif')
    ax0.set_yticklabels(['True CD', 'True UC'], fontsize=9, color=INK,
                         fontfamily='sans-serif', rotation=90, va='center')
    ax0.tick_params(length=0)
    ax0.spines[['top', 'right', 'bottom', 'left']].set_linewidth(0.5)
    ax0.spines[['top', 'right', 'bottom', 'left']].set_color(MUTED)
    ax0.set_title('Confusion matrix (aggregated, 5 folds)',
                  fontsize=8.5, fontweight='bold', color=INK,
                  fontfamily='sans-serif', loc='left', pad=6)
    fig.colorbar(im, ax=ax0, fraction=0.05, pad=0.04,
                 label='Row %').ax.tick_params(labelsize=6)

    # overall metrics text below matrix
    tn, fp, fn, tp = cm_agg.ravel()
    overall_f1_cd = 2 * tn / (2 * tn + fp + fn)
    overall_f1_uc = 2 * tp / (2 * tp + fp + fn)
    overall_acc   = (tn + tp) / cm_agg.sum()
    macro_f1      = (overall_f1_cd + overall_f1_uc) / 2
    ax0.set_xlabel(
        f'Accuracy={overall_acc:.3f}  Macro-F1={macro_f1:.3f}\n'
        f'F1-CD={overall_f1_cd:.3f}  F1-UC={overall_f1_uc:.3f}',
        fontsize=7.5, color=INK2, fontfamily='sans-serif', labelpad=6)

    # --- right: per-fold F1 + AUC bar chart ---
    ax1 = fig.add_subplot(gs[1])
    ax1.set_facecolor(SURF)
    folds_x = np.arange(5)
    bar_w   = 0.25

    bars_cd  = ax1.bar(folds_x - bar_w, fold_f1_cd, bar_w,
                       color=CD_COLOR, alpha=0.85, label='F1-CD')
    bars_uc  = ax1.bar(folds_x,         fold_f1_uc, bar_w,
                       color=UC_COLOR,  alpha=0.85, label='F1-UC')
    bars_auc = ax1.bar(folds_x + bar_w, aucs,       bar_w,
                       color='#4a9d77', alpha=0.85, label='AUC')

    for bars in [bars_cd, bars_uc, bars_auc]:
        for b in bars:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                     f'{b.get_height():.3f}', ha='center', va='bottom',
                     fontsize=5.5, color=INK2, fontfamily='sans-serif')

    ax1.set_xticks(folds_x)
    ax1.set_xticklabels([f'Fold {i}' for i in range(5)], fontsize=8, color=INK2)
    ax1.set_ylabel('Score', fontsize=8, color=INK2, fontfamily='sans-serif')
    ax1.set_ylim(0, 1.08)
    ax1.tick_params(axis='y', labelsize=7, colors=INK2, width=0.6, length=3)
    ax1.tick_params(axis='x', length=0)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.spines['bottom'].set_linewidth(0.6);  ax1.spines['bottom'].set_color(MUTED)
    ax1.spines['left'].set_linewidth(0.6);    ax1.spines['left'].set_color(MUTED)
    ax1.yaxis.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=7, frameon=False, loc='lower right',
               labelcolor=INK2)
    ax1.set_title(
        f'Per-fold F1 and AUC\n'
        f'Mean AUC={np.mean(aucs):.3f}±{np.std(aucs, ddof=1):.3f}  '
        f'Mean F1-CD={np.mean(fold_f1_cd):.3f}  Mean F1-UC={np.mean(fold_f1_uc):.3f}',
        fontsize=8.5, fontweight='bold', color=INK,
        fontfamily='sans-serif', loc='left', pad=6)

    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor=SURF, transparent=False)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    for d in [OUT_SHAP_DATA, OUT_SHAP_PLOTS]:
        os.makedirs(d, exist_ok=True)
    np.random.seed(42)

    print('=== Loading proteomics visits ===')
    cv_patients = pd.read_csv(CV_PATIENTS)
    visit_df, protein_names = load_proteomics_visits(cv_patients)

    X_raw = visit_df[protein_names].values.astype(np.float32)
    y     = visit_df['label'].values
    folds = visit_df['fold'].values
    labels = y

    print('\n=== Running 5-fold SHAP (TreeExplainer, interventional) ===')
    shap_matrix, feat_matrix, aucs, fold_f1_cd, fold_f1_uc, fold_cms = run_shap(
        X_raw, y, folds, protein_names)

    mean_auc = float(np.mean(aucs))
    print(f'\n  Mean AUC: {mean_auc:.4f} ± {np.std(aucs, ddof=1):.4f}')
    print(f'  Mean F1-CD: {np.mean(fold_f1_cd):.4f} ± {np.std(fold_f1_cd, ddof=1):.4f}')
    print(f'  Mean F1-UC: {np.mean(fold_f1_uc):.4f} ± {np.std(fold_f1_uc, ddof=1):.4f}')

    # ── SHAP summary dataframe ─────────────────────────────────────────────────
    shap_df = build_shap_df(shap_matrix, protein_names)
    shap_df.to_csv(
        os.path.join(OUT_SHAP_DATA, 'shap_proteomics_top500.csv'),
        index=False, float_format='%.6f')
    print(f'\n  Top 15 proteins by mean|SHAP|:')
    print(shap_df[['gene_symbol', 'mean_abs_shap', 'direction']].head(15).to_string(index=False))

    # ── save beeswarm .npz (top 50) ────────────────────────────────────────────
    top50_idx  = shap_df['feature_idx'].values[:50]
    np.savez_compressed(
        os.path.join(OUT_SHAP_DATA, 'beeswarm_proteomics.npz'),
        shap_values   = shap_matrix[:, top50_idx],
        feature_values= feat_matrix[:, top50_idx],
        feature_names = np.array(shap_df['protein'].values[:50]),
        labels        = labels,
    )

    # ── summary JSON ───────────────────────────────────────────────────────────
    cm_agg = np.sum(fold_cms, axis=0)
    tn, fp, fn, tp = cm_agg.ravel()
    overall_f1_cd = 2 * tn / (2 * tn + fp + fn)
    overall_f1_uc = 2 * tp / (2 * tp + fp + fn)
    summary = {
        'strategy':  'proteomics_visit',
        'n_samples': int(len(y)),
        'n_patients': int(visit_df['patient_id'].nunique()),
        'n_proteins': len(protein_names),
        'mean_auc':   round(mean_auc, 4),
        'std_auc':    round(float(np.std(aucs, ddof=1)), 4),
        'mean_f1_cd': round(float(np.mean(fold_f1_cd)), 4),
        'std_f1_cd':  round(float(np.std(fold_f1_cd, ddof=1)), 4),
        'mean_f1_uc': round(float(np.mean(fold_f1_uc)), 4),
        'std_f1_uc':  round(float(np.std(fold_f1_uc, ddof=1)), 4),
        'macro_f1_overall': round(float((overall_f1_cd + overall_f1_uc) / 2), 4),
        'accuracy_overall': round(float((tn + tp) / cm_agg.sum()), 4),
        'confusion_matrix_aggregated': cm_agg.tolist(),
        'per_fold': [
            {'fold': i, 'auc': round(aucs[i], 4),
             'f1_cd': round(fold_f1_cd[i], 4),
             'f1_uc': round(fold_f1_uc[i], 4),
             'confusion_matrix': fold_cms[i].tolist()}
            for i in range(5)
        ],
        'top20_proteins': shap_df[['protein', 'gene_symbol', 'mean_abs_shap',
                                    'direction']].head(20).to_dict('records'),
    }
    with open(os.path.join(OUT_SHAP_DATA, 'shap_proteomics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ── plots ──────────────────────────────────────────────────────────────────
    print('\n=== Generating plots ===')
    plot_beeswarm(
        shap_df, shap_matrix, feat_matrix, labels,
        out_path=os.path.join(OUT_SHAP_PLOTS, 'beeswarm_proteomics.png'),
        top_n=30, mean_auc=mean_auc,
    )
    plot_confusion_matrix(
        fold_cms,
        out_path=os.path.join(OUT_SHAP_PLOTS, 'confusion_matrix_proteomics.png'),
        aucs=aucs, fold_f1_cd=fold_f1_cd, fold_f1_uc=fold_f1_uc,
    )

    print('\n\n=== SUMMARY ===')
    print(f"  AUC:       {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    print(f"  F1-CD:     {summary['mean_f1_cd']:.4f} ± {summary['std_f1_cd']:.4f}")
    print(f"  F1-UC:     {summary['mean_f1_uc']:.4f} ± {summary['std_f1_uc']:.4f}")
    print(f"  Macro-F1:  {summary['macro_f1_overall']:.4f}  (aggregated across all folds)")
    print(f"  Accuracy:  {summary['accuracy_overall']:.4f}  (aggregated)")
    print(f'\nOutputs:')
    print(f'  {OUT_SHAP_DATA}/')
    print(f'  {OUT_SHAP_PLOTS}/')


if __name__ == '__main__':
    main()
