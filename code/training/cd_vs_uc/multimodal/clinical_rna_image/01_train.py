"""
Multimodal CD vs UC — patient-level, at-20-cm matched cohort (817 patients).

Combines three feature modalities at PATIENT level (mean-pool multiple
samples/slides per patient):
  clinical  — 15 pre-diagnosis features (from clinical/01_train_clinical.py)
  img       — prism2_base 2,560-d embeddings (mean-pool at-20-cm slides)
  rna       — log2(TPM+1) (mean-pool at-20-cm QC-pass samples)
  concat    — all three concatenated

Models per arm:
  RF  — RandomForestClassifier (same params as all other arms)
  MLP — MLPClassifier with StandardScaler, early stopping

RNA gene filter: mean log2(TPM+1) > 0.5 on training-fold patients only (no leakage).

Outputs  /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/multimodal/clinical_rna_image/results/
--------
  multimodal_fold_metrics.csv
  multimodal_patient_predictions.csv
  multimodal_summary.json
"""

import os
import json
import warnings
import importlib.util
import numpy as np
import pandas as pd
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
_DIR         = os.path.dirname(os.path.abspath(__file__))
_CLINICAL_PY = os.path.join(_DIR, '../../clinical/01_train_clinical.py')

TRANSCRIPTOMICS_DIR = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/'
                       'genestack/transcriptomics')
TPM_GCT     = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF2048892_combined_TPM_matrix_with_header.gct')
MAPPING_CSV = os.path.join(TRANSCRIPTOMICS_DIR,
              'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
WSI_META    = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/'
               'IBD_meta_data_latest/wsi_metadata_raw.csv')
EMB_BASE    = ('/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
               '20x_224px_0px_overlap/prism2_base')
CV_PATIENTS = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients_at20cm_matched.csv'
OUT_DIR     = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/multimodal/'
               'clinical_rna_image/results')

AT20     = {'at 20 cm', 'At 20 cm'}
CDUC_RAW = ["Crohn's disease", 'Ulcerative colitis']
MIN_MEAN_LOG2TPM = 0.5

RF_PARAMS  = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
                  class_weight='balanced', n_jobs=-1, random_state=42)
MLP_PARAMS = dict(hidden_layer_sizes=(256, 128), activation='relu',
                  solver='adam', alpha=1e-3, batch_size=64,
                  max_iter=500, early_stopping=True,
                  validation_fraction=0.1, n_iter_no_change=20,
                  random_state=42)

# ── import clinical feature builder ──────────────────────────────────────────
_spec = importlib.util.spec_from_file_location('train_clinical', _CLINICAL_PY)
_m    = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
build_feature_matrix = _m.build_feature_matrix
CLINICAL_FEATURES    = _m.FEATURE_NAMES


def _norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


# ── data loaders ─────────────────────────────────────────────────────────────

def load_clinical(cv_patients):
    """Returns patient-level DataFrame: patient_id, label, fold, <15 features>."""
    matched_pids = set(cv_patients['patient_id'])
    return build_feature_matrix(cv_patients, matched_pids)


def load_rna_patient(cv_patients):
    """
    Mean-pool log2(TPM+1) across at-20-cm QC-pass samples per patient.
    Returns (pid_array, expr_mat, gene_names).
    """
    cv_pids = set(cv_patients['patient_id'])

    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(_norm_dx)

    rna_20 = rna[
        rna['characteristics_bio_material'].isin(AT20) &
        rna['diagnosis_norm'].isin(CDUC_RAW) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pids)
    ].copy()
    rna_20 = rna_20.drop_duplicates(subset='visit_encounter_id', keep='first')

    sample_ids = set(rna_20['SampleID'])
    print(f'  RNA: {len(sample_ids)} samples from '
          f'{rna_20["deidentified_master_patient_id"].nunique()} patients')

    print('  Loading TPM GCT ...')
    with open(TPM_GCT) as f:
        f.readline(); f.readline()
        header_cols = f.readline().strip().split('\t')
    keep       = [0, 1] + [i for i, c in enumerate(header_cols) if c in sample_ids]
    keep_names = [header_cols[i] for i in keep]
    gct_df = pd.read_csv(TPM_GCT, sep='\t', skiprows=2, header=0,
                         usecols=keep_names, index_col=0,
                         dtype={c: (str if c in ('Name', 'Description') else np.float32)
                                for c in keep_names})
    gct_df.drop(columns=['Description'], inplace=True)

    log_tpm      = np.log2(gct_df.values.T.astype(np.float32) + 1.0)
    sample_order = gct_df.columns.tolist()
    gene_names   = gct_df.index.tolist()

    pid_to_sid = rna_20.groupby('deidentified_master_patient_id')['SampleID'].apply(list).to_dict()
    sid_to_idx = {s: i for i, s in enumerate(sample_order)}

    pids_ordered, vecs = [], []
    for pid in sorted(cv_pids):
        sids = [s for s in pid_to_sid.get(pid, []) if s in sid_to_idx]
        if not sids:
            continue
        mat = np.stack([log_tpm[sid_to_idx[s]] for s in sids])
        vecs.append(mat.mean(axis=0))
        pids_ordered.append(pid)

    expr_mat  = np.stack(vecs).astype(np.float32)
    pid_array = np.array(pids_ordered)
    print(f'  RNA patient matrix: {expr_mat.shape[0]} patients × {expr_mat.shape[1]} genes')
    return pid_array, expr_mat, gene_names


def load_img_patient(cv_patients):
    """
    Mean-pool prism2_base embeddings across at-20-cm slides per patient.
    Returns (pid_array, img_mat).
    """
    cv_pids = set(cv_patients['patient_id'])
    wsi = pd.read_csv(WSI_META)
    wsi = wsi[
        wsi['BIOSAMPLE_LOCATION'].isin(AT20) &
        wsi['diagnosis'].isin(CDUC_RAW) &
        wsi['deidentified_master_patient_id'].isin(cv_pids)
    ].copy()
    wsi['slide_id'] = wsi['IMAGE_VSI'].str.replace('.vsi', '', regex=False)
    pid_to_slides = wsi.groupby('deidentified_master_patient_id')['slide_id'].apply(list).to_dict()

    pids_ordered, vecs = [], []
    missing = 0
    for pid in sorted(cv_pids):
        slide_vecs = []
        for sid in pid_to_slides.get(pid, []):
            h5p = os.path.join(EMB_BASE, f'{sid}.h5')
            if os.path.exists(h5p):
                with h5py.File(h5p, 'r') as h:
                    slide_vecs.append(h['features'][:].astype(np.float32))
        if not slide_vecs:
            missing += 1
            continue
        vecs.append(np.mean(slide_vecs, axis=0))
        pids_ordered.append(pid)

    img_mat   = np.stack(vecs).astype(np.float32)
    pid_array = np.array(pids_ordered)
    print(f'  IMG patient matrix: {img_mat.shape[0]} patients × {img_mat.shape[1]} dims'
          f'  (skipped {missing} with no h5)')
    return pid_array, img_mat


# ── build merged patient table ────────────────────────────────────────────────

def build_patient_table(cv_patients):
    print('=== Loading clinical features ===')
    clin_df = load_clinical(cv_patients)

    print('\n=== Loading RNA (log-TPM, patient mean) ===')
    rna_pids, rna_mat, gene_names = load_rna_patient(cv_patients)

    print('\n=== Loading imaging (prism2_base, patient mean) ===')
    img_pids, img_mat = load_img_patient(cv_patients)

    rna_df = pd.DataFrame(rna_mat,
                          index=pd.Index(rna_pids, name='patient_id'),
                          columns=[f'g{i}' for i in range(rna_mat.shape[1])])
    img_df = pd.DataFrame(img_mat,
                          index=pd.Index(img_pids, name='patient_id'),
                          columns=[f'f{i}' for i in range(img_mat.shape[1])])

    merged = (clin_df.set_index('patient_id')
              .join(rna_df, how='inner')
              .join(img_df, how='inner')
              .reset_index())

    print(f'\n=== Patient coverage ===')
    print(f'  Clinical: {len(clin_df)}  RNA: {len(rna_pids)}  IMG: {len(img_pids)}')
    print(f'  All three: {len(merged)}  '
          f"(CD {(merged['label']==0).sum()}, UC {(merged['label']==1).sum()})")
    return merged, gene_names


# ── CV ─────────────────────────────────────────────────────────────────────────

def _eval_metrics(strategy, fold, n_tr, n_va, y_va, y_pred, y_score, n_genes=None):
    auc = roc_auc_score(y_va, y_score)
    ap  = average_precision_score(y_va, y_score, pos_label=1)
    cr  = classification_report(y_va, y_pred, target_names=['CD', 'UC'], output_dict=True)
    cm  = confusion_matrix(y_va, y_pred)
    row = dict(strategy=strategy, fold=fold, n_train=int(n_tr), n_val=int(n_va),
               auc=round(auc, 4), ap=round(ap, 4),
               accuracy=round(cr['accuracy'], 4),
               cd_f1=round(cr['CD']['f1-score'], 4),
               uc_f1=round(cr['UC']['f1-score'], 4),
               tn=int(cm[0,0]), fp=int(cm[0,1]),
               fn=int(cm[1,0]), tp=int(cm[1,1]))
    if n_genes is not None:
        row['n_genes_used'] = n_genes
    return auc, row


def run_arm_cv(df, arm_name, feature_cols, gene_cols=None):
    """5-fold patient-level CV for RF and MLP."""
    X_all = df[feature_cols].values.astype(np.float32)
    y     = df['label'].values
    folds = df['fold'].values
    pids  = df['patient_id'].values

    clin_set  = set(CLINICAL_FEATURES)
    gene_set  = set(gene_cols or [])
    gene_idx  = np.array([i for i, c in enumerate(feature_cols) if c in gene_set],
                         dtype=np.intp)
    clin_idx  = np.array([i for i, c in enumerate(feature_cols) if c in clin_set],
                         dtype=np.intp)

    fold_results_rf, fold_results_mlp = [], []
    all_preds_rf, all_preds_mlp       = [], []

    for fold in range(5):
        tr = folds != fold
        va = folds == fold
        X_tr_raw, X_va_raw = X_all[tr].copy(), X_all[va].copy()

        # ── Per-fold gene filter (training patients only) ─────────────────────
        if len(gene_idx):
            train_gene_means = X_tr_raw[:, gene_idx].mean(axis=0)
            gene_keep        = np.where(train_gene_means > MIN_MEAN_LOG2TPM)[0]
            non_gene         = np.array([i for i in range(X_tr_raw.shape[1])
                                         if i not in set(gene_idx.tolist())],
                                        dtype=np.intp)
            keep_idx         = np.concatenate([non_gene, gene_idx[gene_keep]])
            X_tr_f  = X_tr_raw[:, keep_idx]
            X_va_f  = X_va_raw[:, keep_idx]
            n_genes = int(len(gene_keep))
            # re-derive clinical positions inside filtered array
            keep_set     = set(keep_idx.tolist())
            clin_in_keep = np.array([new_i
                                     for new_i, orig_i in enumerate(keep_idx)
                                     if int(orig_i) in set(clin_idx.tolist())],
                                    dtype=np.intp)
        else:
            X_tr_f, X_va_f = X_tr_raw, X_va_raw
            n_genes        = None
            clin_in_keep   = clin_idx

        # ── Impute clinical NaN (training stats only) ─────────────────────────
        if len(clin_in_keep):
            imp = SimpleImputer(strategy='median')
            X_tr_f[:, clin_in_keep] = imp.fit_transform(X_tr_f[:, clin_in_keep])
            X_va_f[:, clin_in_keep] = imp.transform(X_va_f[:, clin_in_keep])

        y_tr, y_va = y[tr], y[va]
        pids_va    = pids[va]
        n_tr_n, n_va_n = int(tr.sum()), int(va.sum())

        # ── RF ─────────────────────────────────────────────────────────────────
        clf_rf    = RandomForestClassifier(**RF_PARAMS)
        clf_rf.fit(X_tr_f, y_tr)
        uc_col    = list(clf_rf.classes_).index(1)
        y_pred_rf = clf_rf.predict(X_va_f)
        y_sc_rf   = clf_rf.predict_proba(X_va_f)[:, uc_col]
        auc_rf, row_rf = _eval_metrics(f'{arm_name}_rf', fold, n_tr_n, n_va_n,
                                       y_va, y_pred_rf, y_sc_rf, n_genes)
        fold_results_rf.append(row_rf)
        for pid, yt, yp, ys in zip(pids_va, y_va, y_pred_rf, y_sc_rf):
            all_preds_rf.append(dict(patient_id=pid, strategy=f'{arm_name}_rf',
                                     fold=fold, true_label=int(yt),
                                     pred_label=int(yp), prob_uc=round(float(ys), 5)))

        # ── MLP ────────────────────────────────────────────────────────────────
        scaler     = StandardScaler()
        X_tr_sc    = scaler.fit_transform(X_tr_f)
        X_va_sc    = scaler.transform(X_va_f)
        clf_mlp    = MLPClassifier(**MLP_PARAMS)
        clf_mlp.fit(X_tr_sc, y_tr)
        uc_col_mlp = list(clf_mlp.classes_).index(1)
        y_pred_mlp = clf_mlp.predict(X_va_sc)
        y_sc_mlp   = clf_mlp.predict_proba(X_va_sc)[:, uc_col_mlp]
        auc_mlp, row_mlp = _eval_metrics(f'{arm_name}_mlp', fold, n_tr_n, n_va_n,
                                         y_va, y_pred_mlp, y_sc_mlp, n_genes)
        fold_results_mlp.append(row_mlp)
        for pid, yt, yp, ys in zip(pids_va, y_va, y_pred_mlp, y_sc_mlp):
            all_preds_mlp.append(dict(patient_id=pid, strategy=f'{arm_name}_mlp',
                                      fold=fold, true_label=int(yt),
                                      pred_label=int(yp), prob_uc=round(float(ys), 5)))

        gene_info = f'  {n_genes} genes' if n_genes is not None else ''
        print(f'  {arm_name:<8}  fold {fold}:  '
              f'RF  AUC={auc_rf:.4f}   MLP AUC={auc_mlp:.4f}{gene_info}')

    return fold_results_rf + fold_results_mlp, all_preds_rf + all_preds_mlp


def summarise(fold_results, strategy):
    rows = [r for r in fold_results if r['strategy'] == strategy]
    aucs = [r['auc'] for r in rows]
    aps  = [r['ap']  for r in rows]
    accs = [r['accuracy'] for r in rows]
    return dict(strategy=strategy,
                mean_auc=round(np.mean(aucs), 4), std_auc=round(np.std(aucs, ddof=1), 4),
                mean_ap=round(np.mean(aps),   4), std_ap=round(np.std(aps,  ddof=1), 4),
                mean_acc=round(np.mean(accs),  4), std_acc=round(np.std(accs, ddof=1), 4))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    cv_patients = pd.read_csv(CV_PATIENTS)
    df, gene_names = build_patient_table(cv_patients)

    gene_cols = [f'g{i}' for i in range(len(gene_names))]
    img_cols  = [f'f{i}' for i in range(2560)]
    clin_cols = list(CLINICAL_FEATURES)

    arms = [
        ('clinical', clin_cols,                          None),
        ('img',      img_cols,                           None),
        ('rna',      gene_cols,                          gene_cols),
        ('concat',   clin_cols + img_cols + gene_cols,   gene_cols),
    ]

    all_fold_results, all_preds = [], []
    for arm_name, feat_cols, gcols in arms:
        print(f'\n{"="*60}')
        print(f'  ARM: {arm_name}  ({len(feat_cols)} raw features)')
        print(f'{"="*60}')
        fr, preds = run_arm_cv(df, arm_name, feat_cols, gcols)
        all_fold_results += fr
        all_preds        += preds

    # ── save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'multimodal_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'multimodal_patient_predictions.csv'), index=False)

    strategies = list(dict.fromkeys(r['strategy'] for r in all_fold_results))
    summary    = []
    for strat in strategies:
        s = summarise(all_fold_results, strat)
        s['n_patients'] = len(df)
        summary.append(s)

    with open(os.path.join(OUT_DIR, 'multimodal_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n\n=== RESULTS SUMMARY ===')
    print(f"{'Strategy':<22}  {'AUC':>22}  {'AP':>22}  Acc")
    print('-' * 80)
    for s in summary:
        print(f"  {s['strategy']:<20}  "
              f"AUC={s['mean_auc']:.4f}±{s['std_auc']:.4f}  "
              f"AP={s['mean_ap']:.4f}±{s['std_ap']:.4f}  "
              f"Acc={s['mean_acc']:.4f}")
    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
