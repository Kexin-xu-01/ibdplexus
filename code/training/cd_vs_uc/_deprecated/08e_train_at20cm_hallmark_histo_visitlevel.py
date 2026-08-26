"""
CD vs UC — At-20-cm, visit-level, combined Hallmark ssGSEA + histological scores.

Feature set = 50 Hallmark NES scores (H_*) + 11 prism2 histological scores
              = 61 features total.

Same matched cohort as other visit-level arms: proximity join ≤7 days between
a histological-score slide and an at-20-cm RNA sample.
Patient-level CV fold assignment (cv_splits_patients.csv).

Outputs (under OUT_DIR)
-----------------------
  at20cm_hallmark_histo_visit_fold_metrics.csv
  at20cm_hallmark_histo_visit_predictions.csv
  at20cm_hallmark_histo_visit_summary.json
"""

import os
import json
import importlib.util
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

# ── import loaders ────────────────────────────────────────────────────────────
_dir = os.path.dirname(__file__)

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_08c = _load_module('train08c', os.path.join(_dir, '08c_train_at20cm_hist_visitlevel.py'))
_08d = _load_module('train08d', os.path.join(_dir, '08d_train_at20cm_pathway_visitlevel.py'))

load_hist_visits   = _08c.load_hist_visits
common_visits      = _08c.common_visits
HIST_FEATURES      = _08c.HIST_FEATURES

get_rna_visit_meta = _08d.get_rna_visit_meta
load_vst_symbol    = _08d.load_vst_symbol
run_ssgsea         = _08d.run_ssgsea
build_visit_df     = _08d.build_visit_df
CV_PATIENTS        = _08d.CV_PATIENTS
LIBRARIES          = _08d.LIBRARIES

OUT_DIR   = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
             '08_09_at20cm_site_controlled/results')
RF_PARAMS = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)


def feat_cols(df):
    skip = {'visit_key', 'patient_id', 'visit_date', 'visit_encounter_id',
            'SampleID', 'label', 'fold'}
    return [c for c in df.columns if c not in skip]


def run_cv(name, df):
    cols  = feat_cols(df)
    X     = df[cols].values.astype(np.float32)
    y     = df['label'].values
    folds = df['fold'].values
    vkeys = df['visit_key'].values
    pids  = df['patient_id'].values

    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold
        va = folds == fold

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X[tr], y[tr])
        proba   = clf.predict_proba(X[va])
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]
        y_pred  = clf.predict(X[va])
        y_val   = y[va]

        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred, target_names=['CD', 'UC'],
                                    output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)
        fold_results.append(dict(
            strategy=name, fold=fold,
            n_train_visits=int(tr.sum()), n_val_visits=int(va.sum()),
            n_val_patients=int(pd.Series(pids[va]).nunique()),
            auc=round(auc, 4), ap=round(ap, 4),
            accuracy=round(cr['accuracy'], 4),
            cd_f1=round(cr['CD']['f1-score'], 4),
            uc_f1=round(cr['UC']['f1-score'], 4),
            tn=int(cm[0, 0]), fp=int(cm[0, 1]),
            fn=int(cm[1, 0]), tp=int(cm[1, 1]),
        ))
        for vk, pid, yt, yp, ys in zip(vkeys[va], pids[va], y_val, y_pred, y_score):
            all_preds.append(dict(
                visit_key=vk, patient_id=pid, strategy=name, fold=fold,
                true_label=int(yt), pred_label=int(yp),
                prob_uc=round(float(ys), 5),
            ))
        print(f'    {name}  fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  '
              f'({va.sum()} visits / {pd.Series(pids[va]).nunique()} patients)')
    return fold_results, all_preds


def summarise(fold_results):
    aucs = [r['auc'] for r in fold_results]
    aps  = [r['ap']  for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    return dict(
        mean_auc=round(np.mean(aucs), 4), std_auc=round(np.std(aucs, ddof=1), 4),
        mean_ap=round(np.mean(aps),   4), std_ap=round(np.std(aps,  ddof=1), 4),
        mean_acc=round(np.mean(accs),  4), std_acc=round(np.std(accs, ddof=1), 4),
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)

    # ── Hallmark ssGSEA ───────────────────────────────────────────────────────
    print('=== Loading RNA visit metadata ===')
    rna_meta = get_rna_visit_meta(cv_patients)

    print('\n=== Loading VST expression ===')
    expr = load_vst_symbol(rna_meta['SampleID'].tolist())
    print(f'  {expr.shape[0]} genes × {expr.shape[1]} samples')

    print('\n=== ssGSEA: Hallmark ===')
    scores_h    = run_ssgsea(expr, LIBRARIES['hallmark'])
    hall_visits = build_visit_df(rna_meta, scores_h, prefix='H')

    # ── Histological scores ───────────────────────────────────────────────────
    print('\n=== Loading histological score visits ===')
    hist_df = load_hist_visits(cv_patients)

    # ── Proximity-join ────────────────────────────────────────────────────────
    print('\n=== Matching hist ↔ Hallmark visits ===')
    hist_m, hall_m = common_visits(hist_df, hall_visits)
    n_vis = len(hist_m)
    n_pat = hist_m['patient_id'].nunique()
    print(f'Matched: {n_vis} visits / {n_pat} patients  '
          f"(CD {(hist_m['label']==0).sum()}, UC {(hist_m['label']==1).sum()})")

    # ── Merge feature sets ────────────────────────────────────────────────────
    meta_cols = ['visit_key', 'patient_id', 'visit_date', 'label', 'fold']
    h_cols    = [c for c in hall_m.columns if c.startswith('H_')]

    combined = (hist_m[meta_cols + HIST_FEATURES]
                .copy()
                .reset_index(drop=True))
    combined = pd.concat(
        [combined, hall_m[h_cols].reset_index(drop=True)], axis=1)

    n_hist = len(HIST_FEATURES)
    n_hall = len(h_cols)
    print(f'  Feature dim: {n_hist} histo + {n_hall} Hallmark = {n_hist + n_hall} total')

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    print('\n--- hallmark_histo_visit ---')
    fold_results, all_preds = run_cv('hallmark_histo_visit', combined)

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(fold_results).to_csv(
        os.path.join(OUT_DIR, 'at20cm_hallmark_histo_visit_fold_metrics.csv'),
        index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'at20cm_hallmark_histo_visit_predictions.csv'),
        index=False)

    s = summarise(fold_results)
    s.update(strategy='hallmark_histo_visit', n_visits=n_vis,
             n_patients=n_pat, n_features=n_hist + n_hall)
    with open(os.path.join(OUT_DIR,
                           'at20cm_hallmark_histo_visit_summary.json'), 'w') as f:
        json.dump([s], f, indent=2)

    print(f'\n=== RESULT ===')
    print(f"  hallmark_histo_visit  {n_vis} visits / {n_pat} patients  "
          f"dim={n_hist + n_hall}")
    print(f"  AUC = {s['mean_auc']:.4f} ± {s['std_auc']:.4f}")
    print(f"  AP  = {s['mean_ap']:.4f}  ± {s['std_ap']:.4f}")
    print(f"  Acc = {s['mean_acc']:.4f} ± {s['std_acc']:.4f}")
    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
