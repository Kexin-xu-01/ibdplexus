"""
SHAP feature importance for CD vs UC classifiers — visit-level cohort.

Mirrors 10_shap_analysis.py but uses the visit-level loaders from
08b_train_at20cm_visit_level.py so that each row is one visit, not one patient.
The patient-level CV split is preserved (all visits from a patient stay in the
same fold).

Arms analysed
-------------
  rna_visit        — RNA VST, At-20-cm, visit-level  (945 visits, 817 patients)
  img_base_visit   — prism2_base, At-20-cm, visit-level
  concat_raw_visit — raw concat fusion, visit-level

Outputs  (under OUT_DIR)
------------------------
  shap_rna_visit_top500.csv
  shap_img_base_visit_top500.csv
  shap_concat_raw_visit_top500.csv
  shap_visit_summary.json
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

# ── import loaders from 08b ───────────────────────────────────────────────────
_script = os.path.join(os.path.dirname(__file__),
                       '08b_train_at20cm_visit_level.py')
_spec = importlib.util.spec_from_file_location('train08b', _script)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

load_img_visits  = _m.load_img_visits
load_rna_visits  = _m.load_rna_visits
common_visits    = _m.common_visits
VST_GCT          = _m.VST_GCT
CV_PATIENTS      = _m.CV_PATIENTS

OUT_DIR   = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/10_11_shap_analysis/data_visit'
RF_PARAMS = dict(n_estimators=300, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)


def get_gene_names():
    """Read ordered gene IDs from GCT row index without loading the full matrix."""
    n_genes = int(open(VST_GCT).readlines()[1].split('\t')[0])
    names = pd.read_csv(VST_GCT, sep='\t', skiprows=2,
                        usecols=['Name'], nrows=n_genes)
    return list(names['Name'])


def feat(df):
    return df[[c for c in df.columns
               if c not in ('visit_key', 'patient_id', 'visit_date',
                            'visit_encounter_id', 'label', 'fold')]
              ].values.astype(np.float32)


# ── SHAP runner ───────────────────────────────────────────────────────────────

def run_shap(name, X, y, folds, feat_names, n_background=100):
    """
    5-fold CV + SHAP.  Returns (importance_df, raw_shap_array, mean_auc).
    Each row of the raw array corresponds to one visit (or patient) in the
    validation set of its fold.
    """
    print(f'\n  [{name}] Running 5-fold SHAP ... (n={len(y)}, n_features={X.shape[1]})')
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

    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    result   = pd.DataFrame({
        'feature':      [feat_names[i] for i in order],
        'feature_idx':  order,
        'mean_abs_shap': mean_abs[order],
        'rank':          np.arange(1, len(order) + 1),
    })
    return result, shap_vals, mean_auc


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    cv_patients = pd.read_csv(CV_PATIENTS)

    print('=== Loading At-20-cm visit-level data ===')
    img_df = load_img_visits(cv_patients)
    rna_df = load_rna_visits(cv_patients)
    img_m, rna_m = common_visits(img_df, rna_df)
    print(f'Matched cohort: {len(img_m)} visits from '
          f'{img_m["patient_id"].nunique()} patients  '
          f"(CD {(img_m['label']==0).sum()}, UC {(img_m['label']==1).sum()})")

    gene_names = get_gene_names()
    img_feat_names  = [c for c in img_m.columns
                       if c not in ('visit_key','patient_id','visit_date',
                                    'visit_encounter_id','label','fold')]
    rna_feat_names  = gene_names   # r0…r17962 → gene names in same order

    n_img = len(img_feat_names)
    n_rna = len(gene_names)

    summary = {}

    # ══ ARM 1: img_base_visit ════════════════════════════════════════════════
    print('\n=== ARM: img_base_visit ===')
    X = feat(img_m)
    y = img_m['label'].values
    folds = img_m['fold'].values
    df_shap_img, _, auc = run_shap('img_base_visit', X, y, folds, img_feat_names)
    df_shap_img.to_csv(os.path.join(OUT_DIR, 'shap_img_base_visit_top500.csv'),
                       index=False, float_format='%.6f')
    summary['img_base_visit'] = {
        'auc': round(auc, 4),
        'top20': df_shap_img.head(20)[['feature','mean_abs_shap']].to_dict('records'),
    }
    print(f'  Saved shap_img_base_visit_top500.csv')

    # ══ ARM 2: rna_visit ════════════════════════════════════════════════════
    print('\n=== ARM: rna_visit ===')
    X = feat(rna_m)
    y = rna_m['label'].values
    folds = rna_m['fold'].values
    df_shap_rna, _, auc = run_shap('rna_visit', X, y, folds, rna_feat_names)
    df_shap_rna.to_csv(os.path.join(OUT_DIR, 'shap_rna_visit_top500.csv'),
                       index=False, float_format='%.6f')
    summary['rna_visit'] = {
        'auc': round(auc, 4),
        'top20': df_shap_rna.head(20)[['feature','mean_abs_shap']].to_dict('records'),
    }
    print(f'  Saved shap_rna_visit_top500.csv')

    # ══ ARM 3: concat_raw_visit ══════════════════════════════════════════════
    print('\n=== ARM: concat_raw_visit ===')
    concat_feat_names = [f'img_{f}' for f in img_feat_names] + gene_names
    X_cat = np.hstack([feat(img_m), feat(rna_m)])
    y = img_m['label'].values
    folds = img_m['fold'].values
    df_shap_cat, shap_cat, auc = run_shap(
        'concat_raw_visit', X_cat, y, folds, concat_feat_names)
    df_shap_cat.to_csv(os.path.join(OUT_DIR, 'shap_concat_raw_visit_top500.csv'),
                       index=False, float_format='%.6f')

    mean_abs_full = np.abs(shap_cat).mean(axis=0)
    img_total = mean_abs_full[:n_img].sum()
    rna_total = mean_abs_full[n_img:].sum()
    img_frac  = img_total / (img_total + rna_total)
    rna_frac  = rna_total / (img_total + rna_total)
    print(f'  Fusion SHAP split: imaging={img_frac*100:.1f}%  RNA={rna_frac*100:.1f}%')

    summary['concat_raw_visit'] = {
        'auc': round(auc, 4),
        'imaging_shap_fraction': round(float(img_frac), 4),
        'rna_shap_fraction':     round(float(rna_frac), 4),
        'top20': df_shap_cat.head(20)[['feature','mean_abs_shap']].to_dict('records'),
    }
    print(f'  Saved shap_concat_raw_visit_top500.csv')

    with open(os.path.join(OUT_DIR, 'shap_visit_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n\n=== VISIT-LEVEL SHAP SUMMARY ===')
    for arm, data in summary.items():
        print(f'\n── Top 10 features: {arm} (AUC={data["auc"]}) ──')
        for i, r in enumerate(data['top20'][:10], 1):
            print(f'  {i:>3}. {r["feature"]:<25}  {r["mean_abs_shap"]:.5f}')

    print(f'\nAll outputs saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
