"""
SHAP feature importance for the clinical CD vs UC classifier.

Imports the cohort builder from train_clinical.py, runs 5-fold CV + SHAP,
and saves importances + raw SHAP matrix.

Outputs  /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/shap/
--------
  shap_clinical.csv          — feature, mean_abs_shap, mean_shap, direction, rank
  shap_clinical_values.npz   — raw SHAP matrix (n_patients × n_features)
  shap_clinical_summary.json — top-20 + AUC
"""

import os
import sys
import json
import importlib.util
import warnings
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ── import from train_clinical ────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    'train_clinical',
    os.path.join(os.path.dirname(__file__), 'train_clinical.py'))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

build_feature_matrix = _m.build_feature_matrix
FEATURE_NAMES        = _m.FEATURE_NAMES
CV_PATIENTS          = _m.CV_PATIENTS

OUT_DIR   = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/shap'
RF_PARAMS = dict(n_estimators=300, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)


def run_shap(df, n_background=100):
    X_raw = df[FEATURE_NAMES].values.astype(float)
    y     = df['label'].values
    folds = df['fold'].values

    imp_global = SimpleImputer(strategy='median').fit(X_raw)
    X_imp = imp_global.transform(X_raw)

    shap_vals = np.zeros((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    aucs = []

    for fold in range(5):
        tr = folds != fold
        va = folds == fold

        imp = SimpleImputer(strategy='median')
        X_tr = imp.fit_transform(X_raw[tr])
        X_va = imp.transform(X_raw[va])

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y[tr])
        uc_col = list(clf.classes_).index(1)

        auc = roc_auc_score(y[va], clf.predict_proba(X_va)[:, uc_col])
        aucs.append(auc)
        print(f'  fold {fold}: AUC={auc:.4f}  n_val={va.sum()}', flush=True)

        rng = np.random.RandomState(42)
        bg_idx = rng.choice(X_tr.shape[0], min(n_background, X_tr.shape[0]), replace=False)
        explainer = shap.TreeExplainer(
            clf, data=X_tr[bg_idx], feature_perturbation='interventional')
        sv = explainer.shap_values(X_va, check_additivity=False)
        sv_uc = sv[uc_col] if isinstance(sv, list) else sv[:, :, uc_col]
        shap_vals[va] = sv_uc.astype(np.float32)

    mean_auc = float(np.mean(aucs))
    print(f'  mean AUC={mean_auc:.4f}  std={float(np.std(aucs, ddof=1)):.4f}')

    mean_abs  = np.abs(shap_vals).mean(axis=0)
    mean_sign = shap_vals.mean(axis=0)
    order     = np.argsort(mean_abs)[::-1]

    # Compute direction from data means (UC mean > CD mean → UC direction).
    # This matches 11b_shap_plots_visit.py's compute_gene_directions() approach
    # and avoids sign artefacts from an imbalanced / low-AUC model.
    X_imp = SimpleImputer(strategy='median').fit_transform(
        df[FEATURE_NAMES].values.astype(float))
    cd_means = X_imp[df['label'].values == 0].mean(axis=0)
    uc_means = X_imp[df['label'].values == 1].mean(axis=0)
    data_direction = ['UC' if uc_means[i] > cd_means[i] else 'CD'
                      for i in range(len(FEATURE_NAMES))]

    result = pd.DataFrame({
        'feature':       [FEATURE_NAMES[i] for i in order],
        'feature_idx':   order,
        'mean_abs_shap': mean_abs[order],
        'mean_shap':     mean_sign[order],
        'direction':     [data_direction[i] for i in order],
        'rank':          np.arange(1, len(order) + 1),
    })
    return result, shap_vals, mean_auc, aucs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    cv_patients = pd.read_csv(CV_PATIENTS)
    matched_pids = set(cv_patients['patient_id'])

    print('=== Building feature matrix ===')
    df = build_feature_matrix(cv_patients, matched_pids)

    print(f'\n=== Running SHAP ({len(FEATURE_NAMES)} features, 5-fold) ===')
    result, shap_vals, mean_auc, aucs = run_shap(df)

    # Save CSV
    result.to_csv(os.path.join(OUT_DIR, 'shap_clinical.csv'),
                  index=False, float_format='%.6f')

    # Build imputed feature values matrix (same imputation as training)
    X_raw = df[FEATURE_NAMES].values.astype(float)
    feat_vals_imp = SimpleImputer(strategy='median').fit_transform(X_raw)

    # Sort both arrays by mean|SHAP| descending so beeswarm_plot works directly
    mean_abs_all = np.abs(shap_vals).mean(axis=0)
    sort_order   = np.argsort(mean_abs_all)[::-1]
    shap_sorted  = shap_vals[:, sort_order]
    feat_sorted  = feat_vals_imp[:, sort_order]
    names_sorted = np.array(FEATURE_NAMES)[sort_order]

    # Save npz for beeswarm (keys match 11d_beeswarm_histoscore.py convention)
    np.savez(os.path.join(OUT_DIR, 'shap_clinical_values.npz'),
             shap_values=shap_sorted,
             feature_values=feat_sorted,
             feature_names=names_sorted,
             labels=df['label'].values)

    # Summary JSON
    summary = {
        'strategy': 'clinical',
        'auc': round(mean_auc, 4),
        'std_auc': round(float(np.std(aucs, ddof=1)), 4),
        'fold_aucs': [round(a, 4) for a in aucs],
        'n_patients': len(df),
        'n_features': len(FEATURE_NAMES),
        'top20': result.head(20)[
            ['feature', 'mean_abs_shap', 'mean_shap', 'direction']
        ].to_dict('records'),
    }
    with open(os.path.join(OUT_DIR, 'shap_clinical_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n=== Top 15 features by mean |SHAP| ===')
    print(f'  {"Rank":<5} {"Feature":<38} {"Mean|SHAP|":>10}  {"Direction"}')
    print('  ' + '-' * 65)
    for _, row in result.head(15).iterrows():
        print(f'  {int(row["rank"]):<5} {row["feature"]:<38} '
              f'{row["mean_abs_shap"]:>10.5f}  {row["direction"]}')

    print(f'\nOutputs saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
