"""
CD vs UC — At-20-cm, sample level, VST+CombatSeq normalization (matched cohort)
================================================================================

Matched counterpart to 08_train_at20cm_tpm_sample_level.py.

Same cohort, same granularity (sample level), same CV splits, same RF.
The only difference is the expression matrix:
  - TPM script : GSF2048892_combined_TPM_matrix_with_header.gct
                 log2(TPM+1), per-sample, no cross-sample statistics
  - This script: GSF1491805_CombatSeq_vst_mtx_batch_corrected_*
                 DESeq2 VST + CombatSeq batch correction, fitted on ALL samples
                   -> leaky: test-sample dispersion and batch parameters leak
                      into the training representation

Purpose
-------
Provide a fair head-to-head comparison with the TPM analysis.
The old rna_20cm arm in 08_train_at20cm_only.py used a different patient set
(828, restricted to imaging+RNA intersection) and averaged samples per patient.
This script uses the full RNA cohort (847 patients, 1068 samples) at sample level.

Cohort (At-20-cm, matched to TPM analysis)
-------------------------------------------
  RNA VST: 847 patients (564 CD, 283 UC)  ·  1,068 samples

Output
------
  <OUT_DIR>/at20cm_vst_sample_fold_metrics.csv
  <OUT_DIR>/at20cm_vst_sample_predictions.csv
  <OUT_DIR>/at20cm_vst_sample_summary.json
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

VST_GCT = os.path.join(TRANSCRIPTOMICS_DIR,
          'GSF1491805_CombatSeq_vst_mtx_batch_corrected_'
          'alltissues_all3releases_header.gct')

MAPPING_CSV = os.path.join(TRANSCRIPTOMICS_DIR,
              'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
OUT_DIR     = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
               '08_at20cm_vst_sample_level/results')

AT20 = {'at 20 cm', 'At 20 cm'}

RF_PARAMS = dict(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)

# Gene filter: per-fold, training samples only (mirrors TPM approach).
# VST values are on a roughly log2-count scale (~1-15 for expressed genes);
# threshold of 1.0 is analogous to the TPM min_mean_log2tpm=0.5 filter.
MIN_MEAN_VST = 1.0


# ── helpers ───────────────────────────────────────────────────────────────────

def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


# ── data loading ──────────────────────────────────────────────────────────────

def load_vst_samples(cv_patients):
    """
    Load at-20-cm RNA samples from the VST GCT.

    Returns a DataFrame with columns:
      sample_id, patient_id, label (0=CD, 1=UC), fold, g0..gN (VST values)

    One row per RNA sample. VST values are already variance-stabilized and
    batch-corrected across all samples (including test — leaky by design).
    Gene filtering is deferred to the per-fold CV loop.
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

    print('  Loading VST GCT ...')
    with open(VST_GCT) as f:
        f.readline()  # #1.2
        f.readline()  # dims
        header_cols = f.readline().strip().split('\t')

    keep_idx   = [0, 1] + [i for i, c in enumerate(header_cols) if c in sample_ids]
    keep_names = [header_cols[i] for i in keep_idx]
    n_samples_in_gct = len(keep_names) - 2
    print(f'  Columns to load: {n_samples_in_gct} samples (of {len(header_cols)-2} in GCT)')

    dtype_map = {c: (str if c in ('Name', 'Description') else np.float32)
                 for c in keep_names}
    gct_df = pd.read_csv(
        VST_GCT, sep='\t', skiprows=2, header=0,
        usecols=keep_names, index_col=0,
        dtype=dtype_map,
    )
    gct_df = gct_df.drop(columns=['Description'])

    # Transpose: rows = samples, columns = genes
    X = gct_df.values.T.astype(np.float32)
    sample_order = gct_df.columns.tolist()
    gene_names   = gct_df.index.tolist()
    print(f'  Matrix shape after load: {X.shape[0]} samples × {X.shape[1]} genes')

    pid_map   = rna_20.set_index('SampleID')['deidentified_master_patient_id']
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
        rows_expr.append(X[sid_to_idx[sid]])

    meta_df  = pd.DataFrame(rows_meta)
    expr_mat = np.stack(rows_expr).astype(np.float32)
    print(f'  Samples matched to CV cohort: {len(meta_df)}  '
          f"(CD {(meta_df['label']==0).sum()}, UC {(meta_df['label']==1).sum()})")
    print(f'  Patients represented: {meta_df["patient_id"].nunique()}')

    return meta_df, expr_mat, gene_names


# ── CV training ───────────────────────────────────────────────────────────────

def run_cv(name, meta_df, expr_mat, gene_names, min_mean_vst=MIN_MEAN_VST):
    """
    5-fold patient-level CV with per-fold gene filtering.

    Gene selection uses only training-fold samples (mean VST threshold),
    matching the leakage-free approach in the TPM analysis.
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

        # per-fold gene filtering (training samples only)
        train_means = expr_mat[tr].mean(axis=0)
        gene_mask   = train_means > min_mean_vst
        n_genes     = int(gene_mask.sum())
        gene_counts.append(n_genes)

        X_tr = expr_mat[tr][:, gene_mask]
        X_va = expr_mat[va][:, gene_mask]
        y_tr = y[tr]
        y_va = y[va]

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_tr)

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
    aucs  = [r['auc']  for r in fold_results]
    aps   = [r['ap']   for r in fold_results]
    accs  = [r['accuracy'] for r in fold_results]
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

    print('=== Loading At-20-cm VST data ===')
    meta_df, expr_mat, gene_names = load_vst_samples(cv_patients)

    print(f'\nNormalization: DESeq2 VST + CombatSeq batch correction (fitted on ALL samples — leaky)')
    print(f'Gene filter:   mean VST > {MIN_MEAN_VST} (training fold only)')
    print(f'CV:            5-fold patient-level (cv_splits_patients.csv)')
    print(f'Level:         sample (one row per RNA sample, no patient averaging)')

    all_fold_results, all_preds = [], []

    print('\n--- rna_vst_sample (VST+CombatSeq, sample level) ---')
    fr, preds = run_cv('rna_vst_sample', meta_df, expr_mat, gene_names)
    all_fold_results += fr
    all_preds        += preds

    # ── save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'at20cm_vst_sample_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'at20cm_vst_sample_predictions.csv'), index=False)

    summary_list = []
    for strat in list(dict.fromkeys(r['strategy'] for r in all_fold_results)):
        fr_s = [r for r in all_fold_results if r['strategy'] == strat]
        s = summarise(fr_s)
        s['strategy']      = strat
        s['n_samples']     = len(meta_df)
        s['n_patients']    = int(meta_df['patient_id'].nunique())
        s['normalization'] = 'DESeq2 VST + CombatSeq batch correction (all-sample fit — leaky)'
        s['gene_filter']   = f'mean VST > {MIN_MEAN_VST} on training fold only'
        summary_list.append(s)

    with open(os.path.join(OUT_DIR, 'at20cm_vst_sample_summary.json'), 'w') as f:
        json.dump(summary_list, f, indent=2)

    print('\n\n=== RESULTS SUMMARY ===')
    print(f"  Normalization : DESeq2 VST + CombatSeq (all-sample fit — leaky)")
    print(f"  Gene filter   : mean VST > {MIN_MEAN_VST} (training fold only)")
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

    # ── side-by-side comparison with TPM ──────────────────────────────────────
    tpm_summary_path = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                        '08_at20cm_tpm_sample_level/results/at20cm_tpm_sample_summary.json')
    if os.path.exists(tpm_summary_path):
        with open(tpm_summary_path) as f:
            tpm_s = json.load(f)[0]
        vst_s = summary_list[0]
        print('\n\n=== VST vs TPM COMPARISON (same 847 patients, same 1068 samples) ===')
        print(f"{'Method':<30} {'Samples':<9} {'Patients':<10} {'AUC':<22} {'AP':<22} {'Accuracy'}")
        print('-' * 100)
        print(f"  {'VST + CombatSeq (leaky)':<28} {vst_s['n_samples']:<9} {vst_s['n_patients']:<10} "
              f"AUC={vst_s['mean_auc']:.4f}±{vst_s['std_auc']:.4f}  "
              f"AP={vst_s['mean_ap']:.4f}±{vst_s['std_ap']:.4f}  "
              f"Acc={vst_s['mean_acc']:.4f}")
        print(f"  {'TPM log2(x+1) (clean)':<28} {tpm_s['n_samples']:<9} {tpm_s['n_patients']:<10} "
              f"AUC={tpm_s['mean_auc']:.4f}±{tpm_s['std_auc']:.4f}  "
              f"AP={tpm_s['mean_ap']:.4f}±{tpm_s['std_ap']:.4f}  "
              f"Acc={tpm_s['mean_acc']:.4f}")
        delta_auc = vst_s['mean_auc'] - tpm_s['mean_auc']
        print(f"\n  ΔAUC (VST − TPM) = {delta_auc:+.4f}")

    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
