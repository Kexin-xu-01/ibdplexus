"""
Train Random Forest classifier (CD vs UC) on CombatSeq + VST transcriptomics.

Data selection rationale
------------------------
File used: GSF1491805_CombatSeq_vst_mtx_batch_corrected_alltissues_all3releases_header.gct
- 17,963 genes (filtered protein-coding set) × 3,289 samples
- CombatSeq batch correction (first vs later sequencing run)
- DESeq2 VST: continuous floats ~3.5–12.5, approximately Gaussian → suitable for RF
- All 5 QC-failed samples excluded from this file
- All 2,186 colon CD/UC samples present in this file with no missing entries
- Other files considered but not used:
    GSF1491807 raw counts     — needs normalization + batch correction
    GSF2048892 TPM            — no batch correction
    GSF1485554 VST (no batch) — 3.3 GB, 62K genes, batch effects present
    log_tpm CSV               — derived from CombatSeq but not VST-transformed

Split strategy
--------------
Uses the existing patient-level 5-fold CV splits from 01_build_cv_splits.py.
501 / 997 patients have >1 colon RNAseq sample (from different biopsy sites).
All samples from one patient stay in the same fold (no leakage).
Each sample is treated independently at training and inference.

Inputs
------
- /home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/
      GSF1491805_CombatSeq_vst_mtx_batch_corrected_alltissues_all3releases_header.gct
      ibd_21183_omics_patient_mapping_genestack.csv
      GSF1478941_sample_combined_from1stRun.tsv__metadata.csv
- /home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv

Outputs  (all under /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/)
--------
- transcriptomics_vst_fold_metrics.csv
- transcriptomics_vst_sample_predictions.csv
- transcriptomics_vst_summary.json
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
VST_GCT      = os.path.join(TRANSCRIPTOMICS_DIR,
               'GSF1491805_CombatSeq_vst_mtx_batch_corrected_'
               'alltissues_all3releases_header.gct')
MAPPING_CSV  = os.path.join(TRANSCRIPTOMICS_DIR,
               'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META  = os.path.join(TRANSCRIPTOMICS_DIR,
               'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS  = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
OUT_DIR      = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/03_05_transcriptomics_allsites/results'

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


def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


def build_sample_manifest(cv_patients):
    """Return DataFrame of colon CD/UC samples that are in the CV split."""
    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})

    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'batch', 'Sample QC']],
        on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(norm_dx)

    cv_pats = set(cv_patients['patient_id'])
    rna_colon = rna[
        rna['characteristics_bio_material'].isin(COLON_LOCS) &
        rna['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pats)
    ].copy()

    rna_colon = rna_colon.merge(
        cv_patients[['patient_id', 'fold']],
        left_on='deidentified_master_patient_id', right_on='patient_id', how='left')
    rna_colon['label'] = (rna_colon['diagnosis_norm'] == 'Ulcerative colitis').astype(int)

    print(f"Colon RNAseq samples in CV split: {len(rna_colon)}")
    print(f"Unique patients: {rna_colon['deidentified_master_patient_id'].nunique()}")
    print(rna_colon['diagnosis_norm'].value_counts().to_string())
    return rna_colon


def load_vst_matrix(sample_ids_needed):
    """Parse GCT, load only required sample columns, transpose to samples × genes."""
    print(f"\nParsing VST GCT — loading {len(sample_ids_needed)} sample columns...")

    with open(VST_GCT) as f:
        f.readline(); f.readline()
        header_cols = f.readline().strip().split('\t')

    keep = [0, 1] + [i for i, c in enumerate(header_cols) if c in sample_ids_needed]
    keep_names = [header_cols[i] for i in keep]

    gct_df = pd.read_csv(
        VST_GCT, sep='\t', skiprows=2, header=0,
        usecols=keep_names, index_col=0,
        dtype={c: (str if c in ('Name', 'Description') else np.float32)
               for c in keep_names})
    gct_df = gct_df.drop(columns=['Description'])

    X_df = gct_df.T.astype(np.float32)   # samples × genes
    print(f"Matrix loaded: {X_df.shape[0]} samples × {X_df.shape[1]} genes")
    return X_df


def run_cv(X_df, manifest):
    """Run 5-fold CV and return (fold_results, all_preds)."""
    manifest = manifest.set_index('SampleID')
    common   = X_df.index.intersection(manifest.index)
    X_df     = X_df.loc[common]
    manifest = manifest.loc[common]

    X     = X_df.values
    y     = manifest['label'].values
    folds = manifest['fold'].values
    ids   = X_df.index.tolist()
    diags = manifest['diagnosis_norm'].tolist()

    print(f"\n{'='*60}")
    print(f"  transcriptomics_vst  (CD=0, UC=1, positive=UC)")
    print(f"{'='*60}")

    fold_results, all_preds = [], []

    for fold in range(5):
        train_mask = folds != fold
        val_mask   = folds == fold

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X[train_mask], y[train_mask])

        proba   = clf.predict_proba(X[val_mask])
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]
        y_pred  = clf.predict(X[val_mask])
        y_val   = y[val_mask]

        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred,
                                    target_names=['CD', 'UC'], output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)

        fold_res = dict(
            fold=fold,
            n_train=int(train_mask.sum()), n_val=int(val_mask.sum()),
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
        )
        fold_results.append(fold_res)
        print(f"  Fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  Acc={cr['accuracy']:.4f}  "
              f"CD-F1={cr['CD']['f1-score']:.4f}  UC-F1={cr['UC']['f1-score']:.4f}")

        val_idx  = np.where(val_mask)[0]
        for i, (sid, yt, yp, ys) in enumerate(
                zip([ids[j] for j in val_idx], y_val, y_pred, y_score)):
            all_preds.append(dict(
                sample_id=sid, fold=fold, diagnosis=diags[val_idx[i]],
                true_label=int(yt), pred_label=int(yp),
                prob_uc=round(float(ys), 5)))

    aucs = [r['auc'] for r in fold_results]
    aps  = [r['ap']  for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    print(f"\n  AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  AP:       {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    return fold_results, all_preds, X_df.shape[1], len(X_df)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cv_patients = pd.read_csv(CV_PATIENTS)
    manifest    = build_sample_manifest(cv_patients)
    X_df        = load_vst_matrix(set(manifest['SampleID']))

    fold_results, all_preds, n_genes, n_samples = run_cv(X_df, manifest)

    aucs = [r['auc'] for r in fold_results]
    aps  = [r['ap']  for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]

    pd.DataFrame(fold_results).to_csv(
        os.path.join(OUT_DIR, 'transcriptomics_vst_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'transcriptomics_vst_sample_predictions.csv'), index=False)

    summary = dict(
        model='transcriptomics_vst',
        expression_matrix=os.path.basename(VST_GCT),
        n_genes=n_genes,
        n_samples=n_samples,
        mean_auc=round(float(np.mean(aucs)), 4),
        std_auc=round(float(np.std(aucs)), 4),
        mean_ap=round(float(np.mean(aps)), 4),
        std_ap=round(float(np.std(aps)), 4),
        mean_acc=round(float(np.mean(accs)), 4),
        std_acc=round(float(np.std(accs)), 4),
        folds=fold_results,
        label_encoding={'CD': 0, 'UC': 1},
        positive_class='UC',
        score_column='prob_uc',
        rf_params=RF_PARAMS,
    )
    with open(os.path.join(OUT_DIR, 'transcriptomics_vst_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
