"""
SHAP feature importance for CD vs UC — prism2 histological scores (visit-level).

Uses the same patient-level CV split and matched visit cohort as 08c.
Only 11 features, so SHAP is near-instantaneous and we can store the full
signed values (mean_shap) alongside mean_abs_shap.

Arms
----
  img_histoscore_visit   — 11 histological scores
  rna_visit              — 17,963 VST genes (same matched cohort)
  concat_histoscore_visit — 11 scores + 17,963 genes

Outputs (under OUT_DIR)
-----------------------
  shap_img_histoscore_visit.csv   — all 11 features with mean_shap + mean_abs_shap
  shap_rna_histoscore_visit_top500.csv
  shap_concat_histoscore_visit_top500.csv
  shap_histoscore_summary.json
  beeswarm_img_histoscore.npz     — per-sample SHAP + feature values (all 11 features)
  beeswarm_rna_histoscore.npz     — per-sample SHAP + feature values (top 50 genes)
  beeswarm_concat_histoscore.npz  — per-sample SHAP + feature values (top 50 features)
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

# ── import loaders from 08c ───────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    'train08c',
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '08c_train_at20cm_histoscore.py'))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

load_histoscore_visits = _m.load_histoscore_visits
load_rna_visits        = _m.load_rna_visits
common_visits          = _m.common_visits
feat_img               = _m.feat_img
feat_rna               = _m.feat_rna
HISTO_COLS             = _m.HISTO_COLS
CV_PATIENTS            = _m.CV_PATIENTS
VST_GCT                = _m.VST_GCT

OUT_DIR   = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             'concept_learning/shap/data')
RF_PARAMS = dict(n_estimators=300, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)


def get_gene_names():
    """Read gene IDs from GCT without loading the full matrix."""
    n_genes = int(open(VST_GCT).readlines()[1].split('\t')[0])
    return list(pd.read_csv(VST_GCT, sep='\t', skiprows=2,
                             usecols=['Name'], nrows=n_genes)['Name'])


def run_shap(name, X, y, folds, feat_names, n_background=100):
    """5-fold CV + TreeExplainer SHAP. Returns (importance_df, mean_shap_array, mean_auc)."""
    print(f'\n  [{name}]  n={len(y)}  n_features={X.shape[1]}')
    shap_vals = np.zeros_like(X, dtype=np.float32)
    aucs = []

    for fold in range(5):
        tr = folds != fold; va = folds == fold
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X[tr], y[tr])
        uc_col = list(clf.classes_).index(1)

        auc = roc_auc_score(y[va], clf.predict_proba(X[va])[:, uc_col])
        aucs.append(auc)
        print(f'    fold {fold}: AUC={auc:.4f}  n_val={va.sum()}', flush=True)

        bg_idx = np.random.RandomState(42).choice(
            X[tr].shape[0], min(n_background, X[tr].shape[0]), replace=False)
        explainer = shap.TreeExplainer(
            clf, data=X[tr][bg_idx], feature_perturbation='interventional')
        sv = explainer.shap_values(X[va], check_additivity=False)
        sv_uc = sv[uc_col] if isinstance(sv, list) else sv[:, :, uc_col]
        shap_vals[va] = sv_uc.astype(np.float32)

    mean_auc     = float(np.mean(aucs))
    mean_abs     = np.abs(shap_vals).mean(axis=0)
    mean_signed  = shap_vals.mean(axis=0)
    order        = np.argsort(mean_abs)[::-1]

    result = pd.DataFrame({
        'feature':       [feat_names[i] for i in order],
        'feature_idx':   order,
        'mean_abs_shap': mean_abs[order],
        'mean_shap':     mean_signed[order],   # positive → UC, negative → CD
        'direction':     ['UC' if mean_signed[i] > 0 else 'CD' for i in order],
        'rank':          np.arange(1, len(order) + 1),
    })
    print(f'  [{name}] mean AUC={mean_auc:.4f}')
    return result, shap_vals, mean_auc


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    cv_patients = pd.read_csv(CV_PATIENTS)
    print('=== Loading At-20-cm visit-level data ===')
    img_df = load_histoscore_visits(cv_patients)
    rna_df = load_rna_visits(cv_patients)
    img_m, rna_m = common_visits(img_df, rna_df)
    n_vis = len(img_m); n_pat = img_m['patient_id'].nunique()
    print(f'Matched cohort: {n_vis} visits, {n_pat} patients  '
          f"(CD {(img_m['label']==0).sum()}, UC {(img_m['label']==1).sum()})")

    gene_names = get_gene_names()
    summary = {}

    # ══ ARM 1: img_histoscore_visit ══════════════════════════════════════════
    print('\n=== ARM: img_histoscore_visit ===')
    X_img  = feat_img(img_m)
    y      = img_m['label'].values
    folds  = img_m['fold'].values
    df_img, shap_img, auc = run_shap('img_histoscore_visit', X_img, y, folds, HISTO_COLS)
    df_img.to_csv(os.path.join(OUT_DIR, 'shap_img_histoscore_visit.csv'),
                  index=False, float_format='%.6f')
    # save full per-sample arrays for beeswarm (945 × 11 — tiny)
    np.savez_compressed(
        os.path.join(OUT_DIR, 'beeswarm_img_histoscore.npz'),
        shap_values=shap_img,
        feature_values=X_img,
        feature_names=np.array(HISTO_COLS),
        labels=y,
    )
    summary['img_histoscore_visit'] = {
        'auc': round(auc, 4),
        'top11': df_img[['feature', 'mean_abs_shap', 'mean_shap', 'direction']
                        ].to_dict('records'),
    }

    # ══ ARM 2: rna_visit ════════════════════════════════════════════════════
    print('\n=== ARM: rna_visit ===')
    X_rna  = feat_rna(rna_m)
    y      = rna_m['label'].values
    folds  = rna_m['fold'].values
    df_rna, shap_rna, auc = run_shap('rna_visit', X_rna, y, folds, gene_names)
    df_rna.to_csv(os.path.join(OUT_DIR, 'shap_rna_histoscore_visit_top500.csv'),
                  index=False, float_format='%.6f')
    # save top-50 features for beeswarm
    top50_rna_idx = df_rna['feature_idx'].values[:50]
    np.savez_compressed(
        os.path.join(OUT_DIR, 'beeswarm_rna_histoscore.npz'),
        shap_values=shap_rna[:, top50_rna_idx],
        feature_values=X_rna[:, top50_rna_idx],
        feature_names=np.array(df_rna['feature'].values[:50]),
        labels=y,
    )
    summary['rna_visit'] = {
        'auc': round(auc, 4),
        'top20': df_rna.head(20)[['feature', 'mean_abs_shap', 'direction']
                                 ].to_dict('records'),
    }

    # ══ ARM 3: concat_histoscore_visit ══════════════════════════════════════
    print('\n=== ARM: concat_histoscore_visit ===')
    n_img = len(HISTO_COLS)
    concat_names = [f'histo_{c}' for c in HISTO_COLS] + gene_names
    X_cat = np.hstack([X_img, X_rna])
    y     = img_m['label'].values
    folds = img_m['fold'].values
    df_cat, shap_cat, auc = run_shap('concat_histoscore_visit', X_cat, y, folds,
                                     concat_names)
    df_cat.to_csv(os.path.join(OUT_DIR, 'shap_concat_histoscore_visit_top500.csv'),
                  index=False, float_format='%.6f')

    mean_abs_full = np.abs(shap_cat).mean(axis=0)
    img_total = mean_abs_full[:n_img].sum()
    rna_total = mean_abs_full[n_img:].sum()
    img_frac  = img_total / (img_total + rna_total)
    rna_frac  = rna_total / (img_total + rna_total)
    print(f'  Fusion SHAP split: histoscore={img_frac*100:.1f}%  RNA={rna_frac*100:.1f}%')
    # save top-50 features for beeswarm
    top50_cat_idx = df_cat['feature_idx'].values[:50]
    np.savez_compressed(
        os.path.join(OUT_DIR, 'beeswarm_concat_histoscore.npz'),
        shap_values=shap_cat[:, top50_cat_idx],
        feature_values=X_cat[:, top50_cat_idx],
        feature_names=np.array(df_cat['feature'].values[:50]),
        labels=y,
    )

    summary['concat_histoscore_visit'] = {
        'auc': round(auc, 4),
        'histoscore_shap_fraction': round(float(img_frac), 4),
        'rna_shap_fraction':        round(float(rna_frac), 4),
        'top20': df_cat.head(20)[['feature', 'mean_abs_shap', 'direction']
                                 ].to_dict('records'),
    }

    with open(os.path.join(OUT_DIR, 'shap_histoscore_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n\n=== HISTOSCORE SHAP SUMMARY ===')
    for arm, d in summary.items():
        print(f'\n── {arm}  AUC={d["auc"]} ──')
        key = 'top11' if 'top11' in d else 'top20'
        for r in d[key]:
            print(f'  {r["feature"]:<45}  {r["mean_abs_shap"]:.5f}  {r["direction"]}')
    print(f'\nAll outputs saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
