"""
CD vs UC — All colon sites, sample level, TPM normalization (no batch correction, no VST)
==========================================================================================

Leakage-free counterpart to 03_train_transcriptomics.py.

The original script uses GSF1491805 (CombatSeq + DESeq2 VST), which applies
both batch correction and variance stabilization across ALL samples before any
train/test split — test samples contribute to the normalisation of training
samples. See report_normalisation_audit.md for the full audit.

This script:
  - Uses GSF2048892 (TPM): purely per-sample, no cross-sample statistics.
  - Applies log2(TPM + 1) per sample (monotonic, no leakage).
  - Filters genes on training-fold samples only inside each CV loop.
  - One row per RNA sample (no patient-level averaging).
  - CV splits are patient-level: all samples from a patient stay in one fold.

Cohort: all colon biopsy locations (At 20 cm, Cecum, Rectum, Ascending/
        Descending/Sigmoid/Transverse Colon), CD and UC, QC-pass samples.

Outputs (under OUT_DIR)
-----------------------
  transcriptomics_tpm_fold_metrics.csv
  transcriptomics_tpm_sample_predictions.csv
  transcriptomics_tpm_summary.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
TRANSCRIPTOMICS_DIR = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/'
                       'genestack/transcriptomics')
TPM_GCT     = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF2048892_combined_TPM_matrix_with_header.gct')
MAPPING_CSV = os.path.join(TRANSCRIPTOMICS_DIR,
              'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
OUT_DIR     = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
               'tpm_analysis/allsites/results')

COLON_LOCS = {
    'at 20 cm', 'At 20 cm', 'Cecum', 'Rectum',
    'Ascending Colon', 'Descending Colon', 'Sigmoid Colon', 'Transverse Colon',
}

RF_PARAMS = dict(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)

# Gene filter: mean log2(TPM+1) > threshold computed from training fold only.
MIN_MEAN_LOG2TPM = 0.5


def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


def build_manifest(cv_patients):
    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(norm_dx)
    cv_pids = set(cv_patients['patient_id'])
    rna_colon = rna[
        rna['characteristics_bio_material'].isin(COLON_LOCS) &
        rna['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pids)
    ].copy()
    rna_colon = rna_colon.merge(
        cv_patients[['patient_id', 'fold']],
        left_on='deidentified_master_patient_id', right_on='patient_id', how='left')
    rna_colon['label'] = (rna_colon['diagnosis_norm'] == 'Ulcerative colitis').astype(int)
    print(f'Colon RNA samples in CV split: {len(rna_colon)}')
    print(f'Unique patients: {rna_colon["deidentified_master_patient_id"].nunique()}')
    print(rna_colon['diagnosis_norm'].value_counts().to_string())
    return rna_colon


def load_tpm_matrix(sample_ids_needed):
    print(f'\nLoading TPM GCT — {len(sample_ids_needed)} sample columns ...')
    with open(TPM_GCT) as f:
        f.readline(); f.readline()
        header_cols = f.readline().strip().split('\t')
    keep       = [0, 1] + [i for i, c in enumerate(header_cols) if c in sample_ids_needed]
    keep_names = [header_cols[i] for i in keep]
    gct_df = pd.read_csv(
        TPM_GCT, sep='\t', skiprows=2, header=0,
        usecols=keep_names, index_col=0,
        dtype={c: (str if c in ('Name', 'Description') else np.float32)
               for c in keep_names})
    gct_df = gct_df.drop(columns=['Description'])
    # log2(TPM+1) — per-sample transform, no cross-sample statistics
    log_tpm = np.log2(gct_df.values.T.astype(np.float32) + 1.0)
    sample_order = gct_df.columns.tolist()
    gene_names   = gct_df.index.tolist()
    print(f'Matrix: {log_tpm.shape[0]} samples × {log_tpm.shape[1]} genes (before filtering)')
    return sample_order, log_tpm, gene_names


def run_cv(manifest, sample_order, log_tpm):
    manifest_idx = manifest.set_index('SampleID')
    sid_to_row   = {sid: i for i, sid in enumerate(sample_order)}

    # align manifest to expression matrix order
    valid = [sid for sid in sample_order if sid in manifest_idx.index]
    row_idx = np.array([sid_to_row[s] for s in valid])
    X_all  = log_tpm[row_idx]
    y_all  = manifest_idx.loc[valid, 'label'].values
    folds  = manifest_idx.loc[valid, 'fold'].values
    pids   = manifest_idx.loc[valid, 'deidentified_master_patient_id'].values
    diags  = manifest_idx.loc[valid, 'diagnosis_norm'].values

    print(f'\nSamples in CV: {len(valid)}  '
          f"(CD {(y_all==0).sum()}, UC {(y_all==1).sum()})")
    print(f'Patients: {len(set(pids))}')

    fold_results, all_preds = [], []

    for fold in range(5):
        tr = folds != fold
        va = folds == fold

        # per-fold gene filtering — training samples only
        train_means = X_all[tr].mean(axis=0)
        gene_mask   = train_means > MIN_MEAN_LOG2TPM
        n_genes     = int(gene_mask.sum())

        X_tr = X_all[tr][:, gene_mask]
        X_va = X_all[va][:, gene_mask]

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_all[tr])

        proba   = clf.predict_proba(X_va)
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]
        y_pred  = clf.predict(X_va)
        y_val   = y_all[va]

        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred,
                                    target_names=['CD', 'UC'], output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)

        fold_results.append(dict(
            fold=fold,
            n_train=int(tr.sum()), n_val=int(va.sum()),
            n_train_patients=int(pd.Series(pids[tr]).nunique()),
            n_val_patients=int(pd.Series(pids[va]).nunique()),
            n_genes_used=n_genes,
            auc=round(auc, 4), ap=round(ap, 4),
            accuracy=round(cr['accuracy'], 4),
            cd_precision=round(cr['CD']['precision'], 4),
            cd_recall=round(cr['CD']['recall'], 4),
            cd_f1=round(cr['CD']['f1-score'], 4),
            uc_precision=round(cr['UC']['precision'], 4),
            uc_recall=round(cr['UC']['recall'], 4),
            uc_f1=round(cr['UC']['f1-score'], 4),
            tn=int(cm[0, 0]), fp=int(cm[0, 1]),
            fn=int(cm[1, 0]), tp=int(cm[1, 1]),
        ))

        val_sids = [valid[j] for j in np.where(va)[0]]
        for sid, pid, diag, yt, yp, ys in zip(
                val_sids, pids[va], diags[va], y_val, y_pred, y_score):
            all_preds.append(dict(
                sample_id=sid, patient_id=pid, fold=fold, diagnosis=diag,
                true_label=int(yt), pred_label=int(yp),
                prob_uc=round(float(ys), 5)))

        print(f'  Fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  '
              f'Acc={cr["accuracy"]:.4f}  '
              f'CD-F1={cr["CD"]["f1-score"]:.4f}  UC-F1={cr["UC"]["f1-score"]:.4f}  '
              f'{n_genes} genes')

    aucs = [r['auc'] for r in fold_results]
    aps  = [r['ap']  for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    print(f'\n  AUC:      {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}')
    print(f'  AP:       {np.mean(aps):.4f} ± {np.std(aps, ddof=1):.4f}')
    print(f'  Accuracy: {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}')
    return fold_results, all_preds, len(valid)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)
    manifest    = build_manifest(cv_patients)
    sample_order, log_tpm, gene_names = load_tpm_matrix(set(manifest['SampleID']))
    fold_results, all_preds, n_samples = run_cv(manifest, sample_order, log_tpm)

    aucs  = [r['auc'] for r in fold_results]
    aps   = [r['ap']  for r in fold_results]
    accs  = [r['accuracy'] for r in fold_results]
    genes = [r['n_genes_used'] for r in fold_results]

    pd.DataFrame(fold_results).to_csv(
        os.path.join(OUT_DIR, 'transcriptomics_tpm_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'transcriptomics_tpm_sample_predictions.csv'), index=False)

    summary = dict(
        model='transcriptomics_tpm',
        expression_matrix=os.path.basename(TPM_GCT),
        normalization='log2(TPM+1), per-sample, no batch correction, no VST',
        gene_filter=f'mean log2(TPM+1) > {MIN_MEAN_LOG2TPM} on training fold only',
        n_genes_total=len(gene_names),
        mean_n_genes_per_fold=round(float(np.mean(genes))),
        n_samples=n_samples,
        mean_auc=round(float(np.mean(aucs)), 4),
        std_auc=round(float(np.std(aucs, ddof=1)), 4),
        mean_ap=round(float(np.mean(aps)), 4),
        std_ap=round(float(np.std(aps, ddof=1)), 4),
        mean_acc=round(float(np.mean(accs)), 4),
        std_acc=round(float(np.std(accs, ddof=1)), 4),
        folds=fold_results,
        label_encoding={'CD': 0, 'UC': 1},
        positive_class='UC',
        rf_params=RF_PARAMS,
    )
    with open(os.path.join(OUT_DIR, 'transcriptomics_tpm_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
