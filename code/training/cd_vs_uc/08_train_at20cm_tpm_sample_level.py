"""
CD vs UC — At-20-cm, sample level, TPM normalization (no batch correction, no VST)
===================================================================================

Why this script exists — data leakage audit
--------------------------------------------
All prior at-20-cm scripts (08_*, 08b_*) load from:

  GSF1491805_CombatSeq_vst_mtx_batch_corrected_*   <- CombatSeq + DESeq2 VST
  GSF1491803_CombatSeq_count_mtx_batch_corrected_* <- CombatSeq corrected counts

Both of these are computed across ALL samples BEFORE any train/test split:

  DESeq2 VST:
    Fits a mean-dispersion curve across the full sample set. Each sample's
    variance-stabilized value depends on the global dispersion trend, which is
    estimated from all samples including held-out test samples. Test-sample
    dispersion parameters leak into the training representation.

  CombatSeq batch correction:
    Estimates per-batch means and variances from the combined train+test pool,
    then applies the correction. Held-out (test) samples directly contribute
    to the batch offset that is subtracted from training samples.

This script uses the only available normalization that is free of cross-sample
contamination:

  GSF2048892_combined_TPM_matrix_with_header.gct

  TPM = (count_i / length_kb_i) / sum_j(count_j / length_kb_j) * 1e6

  Each sample's TPM values depend only on that sample's own read counts and
  gene lengths. No statistics from other samples are required. No batch
  correction is applied.

Additional safeguards against leakage
--------------------------------------
  1. Gene filtering (minimum expression) is computed from TRAINING fold samples
     only, inside each CV fold loop. Test-set expression values are never used
     to decide which genes to include as features.

  2. log2(TPM + 1) transform is per-sample (monotonic, no cross-sample params).

  3. CV splits are patient-level (cv_splits_patients.csv). All RNA samples
     from the same patient land in the same fold — no patient-level leakage.

  4. One row per RNA sample (sample level). Patients with multiple samples
     contribute multiple training/evaluation rows; the RF is fit on samples,
     not patient means.

Cohort
------
  At-20-cm biopsies (characteristics_bio_material in {'at 20 cm', 'At 20 cm'})
  from CV-split patients with CD or UC diagnosis and QC-pass RNA samples.

Output
------
  <OUT_DIR>/at20cm_tpm_sample_fold_metrics.csv
  <OUT_DIR>/at20cm_tpm_sample_predictions.csv
  <OUT_DIR>/at20cm_tpm_sample_summary.json
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

# Leakage-free: TPM is computed per-sample from counts + gene lengths only.
# No cross-sample statistics; no batch correction.
TPM_GCT = os.path.join(TRANSCRIPTOMICS_DIR,
           'GSF2048892_combined_TPM_matrix_with_header.gct')

MAPPING_CSV = os.path.join(TRANSCRIPTOMICS_DIR,
              'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
OUT_DIR     = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
               '08_at20cm_tpm_sample_level/results')

AT20 = {'at 20 cm', 'At 20 cm'}

RF_PARAMS = dict(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)

# Gene filter: keep genes with mean log2(TPM+1) > this threshold across
# training-fold samples. Computed per fold from training samples only.
MIN_MEAN_LOG2TPM = 0.5   # ~TPM > 0.4; typically retains ~12-20k genes


# ── helpers ───────────────────────────────────────────────────────────────────

def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


# ── data loading ──────────────────────────────────────────────────────────────

def load_tpm_samples(cv_patients):
    """
    Load at-20-cm RNA samples from the TPM GCT.

    Returns a DataFrame with columns:
      sample_id, patient_id, label (0=CD, 1=UC), fold, g0..gN (log2(TPM+1))

    One row per RNA sample. log2(TPM+1) is applied per-sample — no cross-sample
    statistics. Gene filtering is deferred to per-fold CV loop.
    """
    pat_df  = cv_patients.set_index('patient_id')
    cv_pids = set(pat_df.index)

    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})

    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(norm_dx)

    rna_20 = rna[
        rna['characteristics_bio_material'].isin(AT20) &
        rna['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pids)
    ].copy()

    sample_ids = set(rna_20['SampleID'])
    print(f'  At-20-cm RNA samples in CV cohort: {len(sample_ids)}')

    # Load only the relevant sample columns from the GCT to save memory.
    print(f'  Loading TPM GCT ...')
    with open(TPM_GCT) as f:
        f.readline()  # #1.2
        f.readline()  # dims
        header_cols = f.readline().strip().split('\t')

    keep_idx   = [0, 1] + [i for i, c in enumerate(header_cols) if c in sample_ids]
    keep_names = [header_cols[i] for i in keep_idx]
    n_samples_in_gct = len(keep_names) - 2  # subtract Name + Description
    print(f'  Columns to load: {n_samples_in_gct} samples (of {len(header_cols)-2} in GCT)')

    dtype_map = {c: (str if c in ('Name', 'Description') else np.float32)
                 for c in keep_names}
    gct_df = pd.read_csv(
        TPM_GCT, sep='\t', skiprows=2, header=0,
        usecols=keep_names, index_col=0,
        dtype=dtype_map,
    )
    gct_df = gct_df.drop(columns=['Description'])

    # Transpose: rows = samples, columns = genes
    # Apply log2(TPM+1) — per-sample, no cross-sample statistics.
    log_tpm = np.log2(gct_df.values.T.astype(np.float32) + 1.0)
    sample_order = gct_df.columns.tolist()
    gene_names   = gct_df.index.tolist()
    print(f'  Matrix shape after load: {log_tpm.shape[0]} samples × {log_tpm.shape[1]} genes')

    # Build per-sample rows using the mapping table.
    pid_map = rna_20.set_index('SampleID')['deidentified_master_patient_id']
    sid_to_idx = {sid: i for i, sid in enumerate(sample_order)}

    rows_meta = []
    rows_expr = []
    for sid in sample_order:
        if sid not in pid_map.index:
            continue
        pid = pid_map[sid]
        if pid not in pat_df.index:
            continue
        pat_row = pat_df.loc[pid]
        rows_meta.append({
            'sample_id':  sid,
            'patient_id': pid,
            'label': int(pat_row['diagnosis'] == 'Ulcerative colitis'),
            'fold':  int(pat_row['fold']),
        })
        rows_expr.append(log_tpm[sid_to_idx[sid]])

    meta_df = pd.DataFrame(rows_meta)
    expr_mat = np.stack(rows_expr).astype(np.float32)
    print(f'  Samples matched to CV cohort: {len(meta_df)}  '
          f"(CD {(meta_df['label']==0).sum()}, UC {(meta_df['label']==1).sum()})")
    print(f'  Patients represented: {meta_df["patient_id"].nunique()}')

    return meta_df, expr_mat, gene_names


# ── CV training ───────────────────────────────────────────────────────────────

def run_cv(name, meta_df, expr_mat, gene_names, min_mean_log2tpm=MIN_MEAN_LOG2TPM):
    """
    5-fold patient-level CV with per-fold gene filtering.

    Gene selection uses only training-fold samples (mean log2(TPM+1) threshold),
    so no test-set expression information influences feature selection.
    """
    y     = meta_df['label'].values
    folds = meta_df['fold'].values
    sids  = meta_df['sample_id'].values
    pids  = meta_df['patient_id'].values

    fold_results, all_preds = [], []
    gene_counts = []

    for fold in range(5):
        tr = folds != fold
        va = folds == fold

        # ── per-fold gene filtering (training samples only) ──────────────────
        train_means = expr_mat[tr].mean(axis=0)
        gene_mask   = train_means > min_mean_log2tpm
        n_genes     = int(gene_mask.sum())
        gene_counts.append(n_genes)

        X_tr = expr_mat[tr][:, gene_mask]
        X_va = expr_mat[va][:, gene_mask]
        y_tr = y[tr]
        y_va = y[va]

        # ── fit ───────────────────────────────────────────────────────────────
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_tr)

        # ── evaluate ──────────────────────────────────────────────────────────
        proba   = clf.predict_proba(X_va)
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]
        y_pred  = clf.predict(X_va)

        auc = roc_auc_score(y_va, y_score)
        ap  = average_precision_score(y_va, y_score, pos_label=1)
        cr  = classification_report(y_va, y_pred,
                                    target_names=['CD', 'UC'], output_dict=True)
        cm  = confusion_matrix(y_va, y_pred)

        fold_results.append(dict(
            strategy=name, fold=fold,
            n_train_samples=int(tr.sum()), n_val_samples=int(va.sum()),
            n_train_patients=int(pd.Series(pids[tr]).nunique()),
            n_val_patients=int(pd.Series(pids[va]).nunique()),
            n_genes_used=n_genes,
            auc=round(auc, 4), ap=round(ap, 4),
            accuracy=round(cr['accuracy'], 4),
            cd_f1=round(cr['CD']['f1-score'], 4),
            uc_f1=round(cr['UC']['f1-score'], 4),
            tn=int(cm[0, 0]), fp=int(cm[0, 1]),
            fn=int(cm[1, 0]), tp=int(cm[1, 1]),
        ))
        for sid, pid, yt, yp, ys in zip(sids[va], pids[va], y_va, y_pred, y_score):
            all_preds.append(dict(
                sample_id=sid, patient_id=pid, strategy=name, fold=fold,
                true_label=int(yt), pred_label=int(yp),
                prob_uc=round(float(ys), 5),
            ))
        print(f'    {name}  fold {fold}: '
              f'AUC={auc:.4f}  AP={ap:.4f}  Acc={cr["accuracy"]:.4f}  '
              f'({va.sum()} samples / {pd.Series(pids[va]).nunique()} patients  '
              f'{n_genes} genes)')

    return fold_results, all_preds


def summarise(fold_results):
    aucs = [r['auc'] for r in fold_results]
    aps  = [r['ap']  for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    genes = [r['n_genes_used'] for r in fold_results]
    return dict(
        mean_auc=round(np.mean(aucs), 4), std_auc=round(np.std(aucs, ddof=1), 4),
        mean_ap=round(np.mean(aps),   4), std_ap=round(np.std(aps,  ddof=1), 4),
        mean_acc=round(np.mean(accs),  4), std_acc=round(np.std(accs, ddof=1), 4),
        mean_n_genes=round(np.mean(genes)),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)

    print('=== Loading At-20-cm TPM data ===')
    meta_df, expr_mat, gene_names = load_tpm_samples(cv_patients)

    print(f'\nNormalization: log2(TPM+1), per-sample, no batch correction')
    print(f'Gene filter:   mean log2(TPM+1) > {MIN_MEAN_LOG2TPM} (training fold only)')
    print(f'CV:            5-fold patient-level (cv_splits_patients.csv)')
    print(f'Level:         sample (one row per RNA sample, no patient averaging)')

    all_fold_results, all_preds = [], []

    print('\n--- rna_tpm_sample (TPM, no batch correction, sample level) ---')
    fr, preds = run_cv('rna_tpm_sample', meta_df, expr_mat, gene_names)
    all_fold_results += fr
    all_preds        += preds

    # ── save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'at20cm_tpm_sample_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'at20cm_tpm_sample_predictions.csv'), index=False)

    summary_list = []
    for strat in list(dict.fromkeys(r['strategy'] for r in all_fold_results)):
        fr_s = [r for r in all_fold_results if r['strategy'] == strat]
        s = summarise(fr_s)
        s['strategy']   = strat
        s['n_samples']  = len(meta_df)
        s['n_patients'] = int(meta_df['patient_id'].nunique())
        s['normalization'] = 'log2(TPM+1), per-sample, no batch correction, no VST'
        s['gene_filter'] = f'mean log2(TPM+1) > {MIN_MEAN_LOG2TPM} on training fold only'
        summary_list.append(s)

    with open(os.path.join(OUT_DIR, 'at20cm_tpm_sample_summary.json'), 'w') as f:
        json.dump(summary_list, f, indent=2)

    print('\n\n=== RESULTS SUMMARY ===')
    print(f"  Normalization : log2(TPM+1) per sample — no batch correction, no VST")
    print(f"  Gene filter   : mean log2(TPM+1) > {MIN_MEAN_LOG2TPM} (training fold only)")
    print(f"  Level         : sample")
    print()
    print(f"{'Strategy':<26} {'Samples':<9} {'Patients':<10} "
          f"{'AUC':<22} {'AP':<22} {'Accuracy':<12} {'Genes (mean)'}")
    print('-' * 115)
    for s in summary_list:
        print(f"  {s['strategy']:<24} {s['n_samples']:<9} {s['n_patients']:<10} "
              f"AUC={s['mean_auc']:.4f}±{s['std_auc']:.4f}  "
              f"AP={s['mean_ap']:.4f}±{s['std_ap']:.4f}  "
              f"Acc={s['mean_acc']:.4f}      "
              f"{s['mean_n_genes']:.0f}")
    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
