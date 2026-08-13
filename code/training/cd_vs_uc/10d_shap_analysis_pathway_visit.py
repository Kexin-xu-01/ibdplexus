"""
SHAP feature importance for pathway ssGSEA models — visit-level cohort.

Arms analysed
-------------
  pathway_kegg_visit      — KEGG 2021 NES scores (320 pathways)
  pathway_combined_visit  — Hallmark + KEGG NES scores (370 pathways)

Imports loaders from 08d_train_at20cm_pathway_visitlevel.py so the cohort
and feature construction are identical to the training run.

Outputs  (under OUT_DIR)
------------------------
  shap_pathway_kegg_visit_top500.csv
  shap_pathway_combined_visit_top500.csv
  beeswarm_pathway_kegg_visit.npz      — raw SHAP values for beeswarm plots
  beeswarm_pathway_combined_visit.npz
  shap_pathway_summary.json
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ── import loaders from 08d ───────────────────────────────────────────────────
_script = os.path.join(os.path.dirname(__file__),
                       '08d_train_at20cm_pathway_visitlevel.py')
_spec = importlib.util.spec_from_file_location('train08d', _script)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

get_rna_visit_meta = _m.get_rna_visit_meta
load_vst_symbol    = _m.load_vst_symbol
run_ssgsea         = _m.run_ssgsea
build_visit_df     = _m.build_visit_df
load_hist_visits   = _m.load_hist_visits
common_visits      = _m.common_visits
CV_PATIENTS        = _m.CV_PATIENTS
LIBRARIES          = _m.LIBRARIES

OUT_DIR = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
           'concept_learning/shap/data')

RF_PARAMS = dict(n_estimators=300, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)


def feat_cols(df):
    skip = {'visit_key', 'patient_id', 'visit_date', 'visit_encounter_id',
            'SampleID', 'label', 'fold'}
    return [c for c in df.columns if c not in skip]


def run_shap(name, X, y, folds, feat_names, n_background=100):
    print(f'\n  [{name}] 5-fold SHAP  n={len(y)}  n_features={X.shape[1]}')
    shap_vals = np.zeros_like(X, dtype=np.float32)
    aucs = []

    for fold in range(5):
        tr = folds != fold
        va = folds == fold
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_tr)
        uc_col = list(clf.classes_).index(1)

        auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, uc_col])
        aucs.append(auc)
        print(f'    fold {fold}: AUC={auc:.4f}  n_val={va.sum()}', flush=True)

        bg_idx = np.random.RandomState(42).choice(
            X_tr.shape[0], min(n_background, X_tr.shape[0]), replace=False)
        explainer = shap.TreeExplainer(
            clf, data=X_tr[bg_idx], feature_perturbation='interventional')
        sv = explainer.shap_values(X_va)
        sv_uc = sv[uc_col] if isinstance(sv, list) else sv[:, :, uc_col]
        shap_vals[va] = sv_uc.astype(np.float32)

    mean_auc = float(np.mean(aucs))
    print(f'  [{name}] mean AUC={mean_auc:.4f}')

    mean_abs  = np.abs(shap_vals).mean(axis=0)
    mean_shap = shap_vals.mean(axis=0)
    order     = np.argsort(mean_abs)[::-1]

    result = pd.DataFrame({
        'feature':       [feat_names[i] for i in order],
        'feature_idx':   order,
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

    # ── Build pathway visit DataFrames (same as 08d main()) ───────────────────
    print('=== Loading RNA visit metadata ===')
    rna_meta = get_rna_visit_meta(cv_patients)

    print('\n=== Loading VST expression ===')
    expr = load_vst_symbol(rna_meta['SampleID'].tolist())
    print(f'  {expr.shape[0]} genes × {expr.shape[1]} samples')

    print('\n=== ssGSEA: Hallmark ===')
    scores_h = run_ssgsea(expr, LIBRARIES['hallmark'])

    print('\n=== ssGSEA: KEGG ===')
    scores_k = run_ssgsea(expr, LIBRARIES['kegg'])

    hall_visits = build_visit_df(rna_meta, scores_h, prefix='H')
    kegg_visits = build_visit_df(rna_meta, scores_k, prefix='K')
    comb_visits = hall_visits.merge(
        kegg_visits[
            [c for c in kegg_visits.columns if c.startswith('K_')] + ['SampleID']
        ], on='SampleID', how='inner')

    print('\n=== Matching to hist cohort ===')
    hist_df = load_hist_visits(cv_patients)
    _, hall_m = common_visits(hist_df, hall_visits)
    _, kegg_m = common_visits(hist_df, kegg_visits)
    _, comb_m = common_visits(hist_df, comb_visits)
    print(f'Matched: {len(kegg_m)} visits / {kegg_m["patient_id"].nunique()} patients')

    summary = {}

    # ── ARM 0: pathway_hallmark_visit ─────────────────────────────────────────
    print('\n=== ARM: pathway_hallmark_visit ===')
    cols_h = feat_cols(hall_m)
    display_h = [c.removeprefix('H_') for c in cols_h]
    X = hall_m[cols_h].values.astype(np.float32)
    y = hall_m['label'].values
    folds = hall_m['fold'].values

    df_h, sv_h, auc_h = run_shap('pathway_hallmark_visit', X, y, folds, display_h)
    df_h.to_csv(os.path.join(OUT_DIR, 'shap_pathway_hallmark_visit_top500.csv'),
                index=False, float_format='%.6f')
    np.savez_compressed(
        os.path.join(OUT_DIR, 'beeswarm_pathway_hallmark_visit.npz'),
        shap_values=sv_h,
        X=X,
        feature_names=np.array(display_h),
        labels=y,
    )
    summary['pathway_hallmark_visit'] = {
        'auc': round(auc_h, 4),
        'n_pathways': len(cols_h),
        'top20': df_h.head(20)[['feature','mean_abs_shap','direction']].to_dict('records'),
    }
    print(f'  Saved shap_pathway_hallmark_visit_top500.csv  +  beeswarm .npz')

    # ── ARM 1: pathway_kegg_visit ─────────────────────────────────────────────
    print('\n=== ARM: pathway_kegg_visit ===')
    cols_k = feat_cols(kegg_m)
    # strip prefix for readability
    display_k = [c.removeprefix('K_') for c in cols_k]
    X = kegg_m[cols_k].values.astype(np.float32)
    y = kegg_m['label'].values
    folds = kegg_m['fold'].values

    df_k, sv_k, auc_k = run_shap('pathway_kegg_visit', X, y, folds, display_k)
    df_k.to_csv(os.path.join(OUT_DIR, 'shap_pathway_kegg_visit_top500.csv'),
                index=False, float_format='%.6f')
    np.savez_compressed(
        os.path.join(OUT_DIR, 'beeswarm_pathway_kegg_visit.npz'),
        shap_values=sv_k,
        X=X,
        feature_names=np.array(display_k),
        labels=y,
    )
    summary['pathway_kegg_visit'] = {
        'auc': round(auc_k, 4),
        'n_pathways': len(cols_k),
        'top20': df_k.head(20)[['feature','mean_abs_shap','direction']].to_dict('records'),
    }
    print(f'  Saved shap_pathway_kegg_visit_top500.csv  +  beeswarm .npz')

    # ── ARM 2: pathway_combined_visit ─────────────────────────────────────────
    print('\n=== ARM: pathway_combined_visit ===')
    cols_c = feat_cols(comb_m)
    display_c = [c.removeprefix('H_').removeprefix('K_') for c in cols_c]
    # keep source tag for combined so names stay unique (Hallmark + KEGG may share names)
    display_c_tagged = [
        f'[H] {c.removeprefix("H_")}' if c.startswith('H_')
        else f'[K] {c.removeprefix("K_")}' for c in cols_c
    ]
    X = comb_m[cols_c].values.astype(np.float32)
    y = comb_m['label'].values
    folds = comb_m['fold'].values

    df_c, sv_c, auc_c = run_shap('pathway_combined_visit', X, y, folds, display_c_tagged)
    df_c.to_csv(os.path.join(OUT_DIR, 'shap_pathway_combined_visit_top500.csv'),
                index=False, float_format='%.6f')
    np.savez_compressed(
        os.path.join(OUT_DIR, 'beeswarm_pathway_combined_visit.npz'),
        shap_values=sv_c,
        X=X,
        feature_names=np.array(display_c_tagged),
        labels=y,
    )

    # modality split: H_ vs K_ contribution
    mean_abs_full = np.abs(sv_c).mean(axis=0)
    n_h = sum(1 for c in cols_c if c.startswith('H_'))
    h_total = mean_abs_full[:n_h].sum()
    k_total = mean_abs_full[n_h:].sum()
    h_frac  = h_total / (h_total + k_total)
    k_frac  = k_total / (h_total + k_total)
    print(f'  Fusion SHAP split: Hallmark={h_frac*100:.1f}%  KEGG={k_frac*100:.1f}%')

    summary['pathway_combined_visit'] = {
        'auc': round(auc_c, 4),
        'n_pathways': len(cols_c),
        'hallmark_shap_fraction': round(float(h_frac), 4),
        'kegg_shap_fraction':     round(float(k_frac), 4),
        'top20': df_c.head(20)[['feature','mean_abs_shap','direction']].to_dict('records'),
    }
    print(f'  Saved shap_pathway_combined_visit_top500.csv  +  beeswarm .npz')

    with open(os.path.join(OUT_DIR, 'shap_pathway_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Print top 10 per arm ──────────────────────────────────────────────────
    print('\n\n=== PATHWAY SHAP SUMMARY ===')
    for arm, data in summary.items():
        print(f'\n── Top 10 pathways: {arm}  '
              f'(AUC={data["auc"]}, n={data["n_pathways"]}) ──')
        for i, r in enumerate(data['top20'][:10], 1):
            print(f'  {i:>3}. {r["feature"]:<55}  '
                  f'{r["mean_abs_shap"]:.5f}  ({r["direction"]})')

    print(f'\nAll outputs saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
