"""
Proteomics RF classifier — at-20-cm patient splits, visit-level, Olink plasma NPX.

Canonical pipeline:
  - Cohort:     1,031 plasma samples / 605 patients (CD 390, UC 215)
                (subset of 817 patients in cv_splits_patients_at20cm_matched.csv
                 that have Olink proteomics data)
  - Data:       Olink NPX (2,938 proteins, below-LOD included)
  - CV:         5-fold patient-level (cv_splits_patients_at20cm_matched.csv)
  - Normalise:  StandardScaler fitted on training fold only → applied to test fold
                (per-fold to prevent data leakage)
  - Imputation: per-fold median imputation for missing NPX values (fit on train)
  - Classifier: RandomForestClassifier, class_weight='balanced'

Arms
-----
  proteomics_visit  - all 2,938 Olink proteins

Outputs
-------
  at20cm_proteomics/results/at20cm_proteomics_fold_metrics.csv
  at20cm_proteomics/results/at20cm_proteomics_predictions.csv
  at20cm_proteomics/results/at20cm_proteomics_summary.json
  at20cm_proteomics/results/at20cm_proteomics_feature_importance.csv
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

PROTEOMICS_GCT = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/'
                  'proteomics/GSF1618983_NPX_below_lod_included.gct')
PROT_MAPPING   = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/'
                  'proteomics/ibd_21183_omics_patient_mapping_proteomics_genestack.csv')
CV_PATIENTS    = ('/home/jovyan/kgbk271-ibd-volume/training/'
                  'cv_splits_patients_at20cm_matched.csv')
OUT_DIR        = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                  'at20cm_proteomics/results')

RF_PARAMS = dict(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)


def load_proteomics_visits(cv_patients):
    """Load Olink NPX visits for patients in the splits file.

    Normalisation is intentionally deferred to the CV loop so that
    scaler statistics are computed from the training fold only.
    """
    pat_df  = cv_patients.set_index('patient_id')
    cv_pids = set(pat_df.index)

    pmap = pd.read_csv(PROT_MAPPING)
    pmap = pmap[pmap['deidentified_master_patient_id'].isin(cv_pids)].copy()
    sample_ids = set(pmap['SampleID'].astype(str))
    print(f'  Proteomics samples in cohort: {len(sample_ids)} '
          f'({pmap["deidentified_master_patient_id"].nunique()} patients)')

    # --- read GCT (proteins x samples), keep only relevant samples ---
    print('  Loading GCT ...')
    with open(PROTEOMICS_GCT) as f:
        f.readline()           # "#1.2"
        f.readline()           # dims line
        header = f.readline().strip().split('\t')

    keep_cols = ['NAME'] + [c for c in header if c in sample_ids]
    gct = pd.read_csv(
        PROTEOMICS_GCT, sep='\t', skiprows=2, header=0,
        usecols=keep_cols,
        dtype={c: (str if c == 'NAME' else np.float32) for c in keep_cols},
    )
    gct = gct.set_index('NAME')   # proteins x samples
    protein_names = gct.index.tolist()
    print(f'  GCT loaded: {len(protein_names)} proteins x {gct.shape[1]} samples')

    # --- transpose to samples x proteins ---
    X_df = gct.T.astype(np.float32)   # samples x proteins
    X_df.index.name = 'SampleID'
    X_df = X_df.reset_index()
    X_df['SampleID'] = X_df['SampleID'].astype(str)

    # --- merge with patient mapping then splits ---
    pmap['SampleID'] = pmap['SampleID'].astype(str)
    merged = X_df.merge(
        pmap[['SampleID', 'deidentified_master_patient_id', 'sample_collected_date',
              'visit_encounter_id']],
        on='SampleID', how='inner',
    )
    merged = merged.merge(
        cv_patients[['patient_id', 'diagnosis', 'fold']],
        left_on='deidentified_master_patient_id', right_on='patient_id', how='inner',
    )
    merged['label'] = (merged['diagnosis'] == 'Ulcerative colitis').astype(int)

    n_samp = len(merged)
    n_pat  = merged['patient_id'].nunique()
    n_cd   = (merged['label'] == 0).sum()
    n_uc   = (merged['label'] == 1).sum()
    print(f'  Visit cohort: {n_samp} samples / {n_pat} patients '
          f'(CD {n_cd}, UC {n_uc})')
    print(f'  Fold distribution:\n{merged["fold"].value_counts().sort_index().to_string()}')

    missing = merged[protein_names].isnull().sum()
    n_miss_cols = (missing > 0).sum()
    print(f'  Missing values: {missing.sum()} across {n_miss_cols} proteins '
          f'(will be imputed per fold)')

    return merged, protein_names


def run_arm(name, df, protein_cols):
    """5-fold CV with per-fold StandardScaler + median imputation (no leakage)."""
    y     = df['label'].values
    folds = df['fold'].values
    vkeys = df['visit_encounter_id'].values
    pids  = df['patient_id'].values
    X_raw = df[protein_cols].values.astype(np.float32)

    fold_results, all_preds, feat_imps = [], [], []

    for fold in range(5):
        tr_mask = folds != fold
        va_mask = folds == fold

        X_tr_raw = X_raw[tr_mask].copy()
        X_va_raw = X_raw[va_mask].copy()

        # --- per-fold median imputation (fit on train) ---
        train_medians = np.nanmedian(X_tr_raw, axis=0)
        nan_mask_tr = np.isnan(X_tr_raw)
        nan_mask_va = np.isnan(X_va_raw)
        for j in range(X_tr_raw.shape[1]):
            if nan_mask_tr[:, j].any():
                X_tr_raw[nan_mask_tr[:, j], j] = train_medians[j]
            if nan_mask_va[:, j].any():
                X_va_raw[nan_mask_va[:, j], j] = train_medians[j]

        # --- per-fold StandardScaler (fit on train only) ---
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y[tr_mask])

        proba   = clf.predict_proba(X_va)
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]
        y_pred  = clf.predict(X_va)
        y_val   = y[va_mask]

        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred,
                                    target_names=['CD', 'UC'], output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)

        fold_results.append(dict(
            strategy=name, fold=fold,
            n_train_samples=int(tr_mask.sum()),
            n_val_samples=int(va_mask.sum()),
            n_val_patients=int(pd.Series(pids[va_mask]).nunique()),
            n_train_patients=int(pd.Series(pids[tr_mask]).nunique()),
            auc=round(auc, 4), ap=round(ap, 4),
            accuracy=round(cr['accuracy'], 4),
            cd_f1=round(cr['CD']['f1-score'], 4),
            uc_f1=round(cr['UC']['f1-score'], 4),
            tn=int(cm[0, 0]), fp=int(cm[0, 1]),
            fn=int(cm[1, 0]), tp=int(cm[1, 1]),
        ))

        for vk, pid, yt, yp, ys in zip(vkeys[va_mask], pids[va_mask],
                                        y_val, y_pred, y_score):
            all_preds.append(dict(
                visit_encounter_id=vk, patient_id=pid, strategy=name, fold=fold,
                true_label=int(yt), pred_label=int(yp),
                prob_uc=round(float(ys), 5),
            ))

        feat_imps.append(clf.feature_importances_)
        print(f'    {name}  fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  '
              f'Acc={cr["accuracy"]:.4f}  '
              f'({va_mask.sum()} samples / {pd.Series(pids[va_mask]).nunique()} patients)')

    # --- mean feature importance across folds ---
    mean_imp = np.mean(feat_imps, axis=0)
    imp_df = pd.DataFrame({
        'protein': protein_cols,
        'mean_importance': mean_imp,
    }).sort_values('mean_importance', ascending=False).reset_index(drop=True)

    return fold_results, all_preds, imp_df


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
    global CV_PATIENTS, OUT_DIR
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cv_patients', type=str, default=CV_PATIENTS)
    p.add_argument('--out_dir',     type=str, default=OUT_DIR)
    a = p.parse_args()
    CV_PATIENTS = a.cv_patients
    OUT_DIR     = a.out_dir

    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)

    print('=== Loading proteomics visits ===')
    visit_df, protein_cols = load_proteomics_visits(cv_patients)

    print('\n=== Running RF: proteomics_visit ===')
    fold_results, all_preds, imp_df = run_arm('proteomics_visit', visit_df, protein_cols)

    # --- save outputs ---
    pd.DataFrame(fold_results).to_csv(
        os.path.join(OUT_DIR, 'at20cm_proteomics_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'at20cm_proteomics_predictions.csv'), index=False)
    imp_df.to_csv(
        os.path.join(OUT_DIR, 'at20cm_proteomics_feature_importance.csv'), index=False)

    summary = summarise(fold_results)
    summary['strategy']   = 'proteomics_visit'
    summary['n_samples']  = len(visit_df)
    summary['n_patients'] = visit_df['patient_id'].nunique()
    summary['n_proteins'] = len(protein_cols)
    with open(os.path.join(OUT_DIR, 'at20cm_proteomics_summary.json'), 'w') as f:
        json.dump([summary], f, indent=2)

    print('\n\n=== PROTEOMICS RF SUMMARY ===')
    print(f"  Strategy: proteomics_visit  ({len(protein_cols)} proteins)")
    print(f"  AUC  = {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    print(f"  AP   = {summary['mean_ap']:.4f} ± {summary['std_ap']:.4f}")
    print(f"  Acc  = {summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}")
    print(f'\n  Top 20 proteins by importance:')
    print(imp_df.head(20).to_string(index=False))
    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
