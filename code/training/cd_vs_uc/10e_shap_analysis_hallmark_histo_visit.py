"""
SHAP feature importance for hallmark_histo_visit model.

Features: 11 prism2 histological scores + 50 Hallmark ssGSEA NES scores (61 total).
Imports data loaders from 08e, which chains 08c (histo) and 08d (ssGSEA).

Outputs (under OUT_DIR)
------------------------
  shap_hallmark_histo_visit_top500.csv
  beeswarm_hallmark_histo_visit.npz     — raw arrays for 11f beeswarm
  shap_hallmark_histo_summary.json
"""

import os
import json
import importlib.util
import warnings
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ── import loaders via 08e ────────────────────────────────────────────────────
_dir  = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    'train08e', os.path.join(_dir, '08e_train_at20cm_hallmark_histo_visitlevel.py'))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

load_hist_visits   = _m.load_hist_visits
common_visits      = _m.common_visits
HIST_FEATURES      = _m.HIST_FEATURES
get_rna_visit_meta = _m.get_rna_visit_meta
load_vst_symbol    = _m.load_vst_symbol
run_ssgsea         = _m.run_ssgsea
build_visit_df     = _m.build_visit_df
CV_PATIENTS        = _m.CV_PATIENTS
LIBRARIES          = _m.LIBRARIES

OUT_DIR   = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/shap/data')
RF_PARAMS = dict(n_estimators=300, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)


def run_shap(name, X, y, folds, feat_names, n_background=100):
    print(f'\n  [{name}]  n={len(y)}  n_features={X.shape[1]}')
    shap_vals = np.zeros_like(X, dtype=np.float32)
    aucs = []

    for fold in range(5):
        tr = folds != fold
        va = folds == fold
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X[tr], y[tr])
        uc_col = list(clf.classes_).index(1)

        auc = roc_auc_score(y[va], clf.predict_proba(X[va])[:, uc_col])
        aucs.append(auc)
        print(f'    fold {fold}: AUC={auc:.4f}  n_val={va.sum()}', flush=True)

        bg_idx   = np.random.RandomState(42).choice(
            X[tr].shape[0], min(n_background, X[tr].shape[0]), replace=False)
        explainer = shap.TreeExplainer(
            clf, data=X[tr][bg_idx], feature_perturbation='interventional')
        sv     = explainer.shap_values(X[va])
        sv_uc  = sv[uc_col] if isinstance(sv, list) else sv[:, :, uc_col]
        shap_vals[va] = sv_uc.astype(np.float32)

    mean_auc = float(np.mean(aucs))
    print(f'  [{name}] mean AUC={mean_auc:.4f}')

    mean_abs  = np.abs(shap_vals).mean(axis=0)
    mean_shap = shap_vals.mean(axis=0)
    order     = np.argsort(mean_abs)[::-1]

    result = pd.DataFrame({
        'feature':       [feat_names[i] for i in order],
        'mean_abs_shap': mean_abs[order],
        'mean_shap':     mean_shap[order],
        'direction':     ['UC' if mean_shap[i] > 0 else 'CD' for i in order],
        'rank':          np.arange(1, len(order) + 1),
    })
    return result, shap_vals, mean_auc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    cv_patients = pd.read_csv(CV_PATIENTS)

    # ── Build feature matrices (same as 08e main) ─────────────────────────────
    print('=== Loading RNA visit metadata ===')
    rna_meta = get_rna_visit_meta(cv_patients)

    print('\n=== Loading VST expression ===')
    expr = load_vst_symbol(rna_meta['SampleID'].tolist())

    print('\n=== ssGSEA: Hallmark ===')
    scores_h    = run_ssgsea(expr, LIBRARIES['hallmark'])
    hall_visits = build_visit_df(rna_meta, scores_h, prefix='H')

    print('\n=== Loading histological score visits ===')
    hist_df = load_hist_visits(cv_patients)

    print('\n=== Matching ===')
    hist_m, hall_m = common_visits(hist_df, hall_visits)
    print(f'  {len(hist_m)} visits / {hist_m["patient_id"].nunique()} patients')

    meta_cols = ['visit_key', 'patient_id', 'visit_date', 'label', 'fold']
    h_cols    = [c for c in hall_m.columns if c.startswith('H_')]
    combined  = (hist_m[meta_cols + HIST_FEATURES]
                 .copy().reset_index(drop=True))
    combined  = pd.concat(
        [combined, hall_m[h_cols].reset_index(drop=True)], axis=1)

    # feature display names: tag source for beeswarm colouring
    histo_display = [f'[Histo] {f.replace("_", " ").title()}' for f in HIST_FEATURES]
    hall_display  = [f'[H] {c.removeprefix("H_")}' for c in h_cols]
    feat_names    = histo_display + hall_display

    all_feat_cols = HIST_FEATURES + h_cols
    X     = combined[all_feat_cols].values.astype(np.float32)
    y     = combined['label'].values
    folds = combined['fold'].values

    # ── SHAP ──────────────────────────────────────────────────────────────────
    print('\n=== SHAP: hallmark_histo_visit ===')
    df_sh, sv, auc = run_shap('hallmark_histo_visit', X, y, folds, feat_names)

    df_sh.to_csv(
        os.path.join(OUT_DIR, 'shap_hallmark_histo_visit_top500.csv'),
        index=False, float_format='%.6f')
    np.savez_compressed(
        os.path.join(OUT_DIR, 'beeswarm_hallmark_histo_visit.npz'),
        shap_values=sv,
        X=X,
        feature_names=np.array(feat_names),
        labels=y,
    )

    # modality split
    n_h = len(HIST_FEATURES)
    mean_abs = np.abs(sv).mean(axis=0)
    histo_frac = mean_abs[:n_h].sum() / mean_abs.sum()
    hall_frac  = mean_abs[n_h:].sum() / mean_abs.sum()
    print(f'  SHAP split: Histo={histo_frac*100:.1f}%  Hallmark={hall_frac*100:.1f}%')

    summary = {
        'hallmark_histo_visit': {
            'auc': round(auc, 4),
            'n_features': int(X.shape[1]),
            'histo_shap_fraction':   round(float(histo_frac), 4),
            'hallmark_shap_fraction': round(float(hall_frac), 4),
            'top20': df_sh.head(20)[['feature', 'mean_abs_shap', 'direction']
                                    ].to_dict('records'),
        }
    }
    with open(os.path.join(OUT_DIR, 'shap_hallmark_histo_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n=== TOP 15 FEATURES ===')
    for i, r in df_sh.head(15).iterrows():
        print(f'  {r["rank"]:>3}. {r["feature"]:<55}  '
              f'{r["mean_abs_shap"]:.5f}  ({r["direction"]})')

    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
