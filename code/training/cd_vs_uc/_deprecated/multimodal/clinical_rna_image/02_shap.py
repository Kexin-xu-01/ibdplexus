"""
SHAP feature importance for the multimodal classifiers (RF only).

Trains one RF per arm on all 817 patients (global gene filter),
computes SHAP TreeExplainer values, saves sorted npz + CSV per arm.

Arms: clinical, img, rna, concat

Outputs  /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/multimodal/clinical_rna_image/shap/
--------
  shap_<arm>.csv               — feature, mean_abs_shap, direction, rank
  shap_<arm>_values.npz        — shap_values, feature_values, feature_names, labels
  shap_multimodal_summary.json — top-20 per arm + AUC
"""

import os
import json
import warnings
import importlib.util
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
np.random.seed(42)

_DIR      = os.path.dirname(os.path.abspath(__file__))
_TRAIN_PY = os.path.join(_DIR, '01_train.py')
OUT_DIR   = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/multimodal/'
             'clinical_rna_image/shap')

_spec = importlib.util.spec_from_file_location('train_multi', _TRAIN_PY)
_m    = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

build_patient_table = _m.build_patient_table
CLINICAL_FEATURES   = _m.CLINICAL_FEATURES
MIN_MEAN_LOG2TPM    = _m.MIN_MEAN_LOG2TPM
CV_PATIENTS         = _m.CV_PATIENTS

RF_PARAMS = dict(n_estimators=300, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)


def run_arm_shap(df, arm_name, feature_cols, gene_cols, n_background=50):
    print(f'\n{"="*60}')
    print(f'  SHAP arm: {arm_name}')
    print(f'{"="*60}')

    X_raw      = df[feature_cols].values.astype(np.float32)
    y          = df['label'].values
    folds      = df['fold'].values
    feat_names = np.array(feature_cols)
    clin_set   = set(CLINICAL_FEATURES)
    gene_set   = set(gene_cols or [])

    # ── Global gene filter ────────────────────────────────────────────────────
    gene_idx = np.array([i for i, c in enumerate(feature_cols) if c in gene_set],
                        dtype=np.intp)
    if len(gene_idx):
        gene_means = X_raw[:, gene_idx].mean(axis=0)
        gene_keep  = np.where(gene_means > MIN_MEAN_LOG2TPM)[0]
        non_gene   = np.array([i for i in range(X_raw.shape[1])
                               if i not in set(gene_idx.tolist())], dtype=np.intp)
        keep_idx   = np.concatenate([non_gene, gene_idx[gene_keep]])
        X_filt     = X_raw[:, keep_idx]
        feat_names = feat_names[keep_idx]
        print(f'  Genes retained: {len(gene_keep)} / {len(gene_idx)}')
    else:
        X_filt   = X_raw
        keep_idx = np.arange(X_raw.shape[1])

    # ── Impute clinical NaN ───────────────────────────────────────────────────
    clin_in_keep = np.array([i for i, c in enumerate(feat_names) if c in clin_set],
                            dtype=np.intp)
    if len(clin_in_keep):
        imp = SimpleImputer(strategy='median')
        X_filt = X_filt.copy()
        X_filt[:, clin_in_keep] = imp.fit_transform(X_filt[:, clin_in_keep])

    print(f'  Feature matrix: {X_filt.shape[0]} patients × {X_filt.shape[1]} features')

    # ── 5-fold SHAP ───────────────────────────────────────────────────────────
    shap_vals_all = np.zeros((len(df), X_filt.shape[1]), dtype=np.float32)
    fold_aucs     = []

    for fold in range(5):
        tr = folds != fold
        va = folds == fold

        clf    = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_filt[tr], y[tr])
        uc_col = list(clf.classes_).index(1)

        auc = roc_auc_score(y[va], clf.predict_proba(X_filt[va])[:, uc_col])
        fold_aucs.append(auc)
        print(f'  fold {fold}: AUC={auc:.4f}  n_val={va.sum()}', flush=True)

        rng    = np.random.RandomState(42)
        bg_idx = rng.choice(X_filt[tr].shape[0],
                            min(n_background, X_filt[tr].shape[0]), replace=False)
        explainer = shap.TreeExplainer(clf, data=X_filt[tr][bg_idx],
                                       feature_perturbation='interventional')
        sv    = explainer.shap_values(X_filt[va], check_additivity=False)
        sv_uc = sv[uc_col] if isinstance(sv, list) else sv[:, :, uc_col]
        shap_vals_all[va] = sv_uc.astype(np.float32)

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs, ddof=1))
    print(f'  Mean AUC={mean_auc:.4f} ± {std_auc:.4f}')

    # ── Direction from data means ─────────────────────────────────────────────
    cd_means  = X_filt[y == 0].mean(axis=0)
    uc_means  = X_filt[y == 1].mean(axis=0)
    direction = np.where(uc_means > cd_means, 'UC', 'CD')

    mean_abs = np.abs(shap_vals_all).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]

    result_df = pd.DataFrame({
        'feature':       feat_names[order],
        'mean_abs_shap': mean_abs[order],
        'direction':     direction[order],
        'rank':          np.arange(1, len(order) + 1),
    })
    result_df.to_csv(os.path.join(OUT_DIR, f'shap_{arm_name}.csv'),
                     index=False, float_format='%.6f')

    np.savez(os.path.join(OUT_DIR, f'shap_{arm_name}_values.npz'),
             shap_values=shap_vals_all[:, order],
             feature_values=X_filt[:, order],
             feature_names=feat_names[order],
             labels=y)

    print(f'\n  Top 10:')
    for _, row in result_df.head(10).iterrows():
        print(f'    {int(row["rank"]):<4} {row["feature"]:<42} '
              f'{row["mean_abs_shap"]:.5f}  {row["direction"]}')

    return dict(arm=arm_name,
                mean_auc=round(mean_auc, 4), std_auc=round(std_auc, 4),
                fold_aucs=[round(a, 4) for a in fold_aucs],
                n_patients=len(df),
                n_features=int(X_filt.shape[1]),
                top20=result_df.head(20)[['feature', 'mean_abs_shap', 'direction']
                                         ].to_dict('records'))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cv_patients = pd.read_csv(CV_PATIENTS)
    print('=== Building patient table ===')
    df, gene_names = build_patient_table(cv_patients)

    gene_cols = [f'g{i}' for i in range(len(gene_names))]
    img_cols  = [f'f{i}' for i in range(2560)]
    clin_cols = list(CLINICAL_FEATURES)

    arms = [
        ('clinical', clin_cols,                        None),
        ('img',      img_cols,                         None),
        ('rna',      gene_cols,                        gene_cols),
        ('concat',   clin_cols + img_cols + gene_cols, gene_cols),
    ]

    summaries = []
    for arm_name, feat_cols, gcols in arms:
        n_bg = 50 if arm_name in ('rna', 'concat') else 100
        s    = run_arm_shap(df, arm_name, feat_cols, gcols, n_background=n_bg)
        summaries.append(s)

    with open(os.path.join(OUT_DIR, 'shap_multimodal_summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)

    print(f'\nAll SHAP outputs saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
