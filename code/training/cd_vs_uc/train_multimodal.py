"""
Multimodal CD vs UC — visit-level, at-20-cm, manual-KNN-filtered patches.

Adds clinical features to the imaging+RNA combinations.
Same 945-visit / 817-patient cohort as train_rf.py (BulkFormer-filtered).

Clinical features (patient-level, joined to each visit):
  Symptoms:      rectal_bleed, first_abdopain, first_daily_bm, stool_freq,
                 first_well_being, fecal_urgency, da6m
  Lifestyle:     smk
  Comorbidities: cardiovascular_disease, hypertension, diabetes_type_i,
                 diabetes_type_ii, ckd, hiv, tuberculosis
  (pre-diagnosis only — no disease-specific scores, treatment, or surgery)

Missing values: median imputed on training fold only (no leakage).

Arms (each run with RF and MLP)
----------------------------------
  clinical_visit       — 15 clinical features (patient-level joined to visits)
  img_clinical_visit   — prism2_base + clinical (2,575-d)
  rna_clinical_visit   — RNA log2(TPM+1) + clinical
  trimodal_visit       — prism2_base + RNA log2(TPM+1) + clinical

Outputs
-------
  at20cm_visit_manual_knn/results/at20cm_multimodal_fold_metrics.csv
  at20cm_visit_manual_knn/results/at20cm_multimodal_predictions.csv
  at20cm_visit_manual_knn/results/at20cm_multimodal_summary.json
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import h5py
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

TRANSCRIPTOMICS_DIR = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/'
                       'genestack/transcriptomics')
WSI_META       = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/'
                  'IBD_meta_data_latest/wsi_metadata_raw.csv')
HISTOSCORE_CSV = '/home/jovyan/kgbk271-ibd-volume/results/prism2_manual_knn/prism2_histological_score.csv'
BULKFORMER_PT  = ('/home/jovyan/kgbk271-ibd-volume/data/processed/'
                  'bulk_rna_seq/bulkformer/transcriptomics_embeddings.pt')
LOG_TPM_CSV    = os.path.join(TRANSCRIPTOMICS_DIR,
                 'GSF1491803_CombatSeq_count_mtx_batch_corrected_'
                 'alltissues_all3releases_header_log_tpm_with_sampleID.csv')
TPM_GCT        = os.path.join(TRANSCRIPTOMICS_DIR,
                 'GSF2048892_combined_TPM_matrix_with_header.gct')
MAPPING_CSV    = os.path.join(TRANSCRIPTOMICS_DIR,
                 'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META    = os.path.join(TRANSCRIPTOMICS_DIR,
                 'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
METADATA_CSV   = '/home/jovyan/kgbk271-ibd-volume/metadata/merged_patient_metadata.csv'
CV_PATIENTS    = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
EMB_BASE       = ('/home/jovyan/kgbk271-ibd-volume/data/processed/'
                  'tissue_threshold_15_filtered_no_darkspot_manual_knn/'
                  '20x_224px_0px_overlap/prism2_base')
OUT_DIR        = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                  'at20cm_visit_manual_knn/results')

AT20      = {'at 20 cm', 'At 20 cm'}
CDUC_RAW  = ["Crohn's disease", 'Ulcerative colitis']
EMB_DIM   = 640
EMB_COLS  = [f'e{i}' for i in range(EMB_DIM)]
HISTO_COLS = [
    'inflammation_involvement', 'crypt_architectural_distortion',
    'neutrophil_granulocytic_infiltration', 'crypt_abscesses',
    'lymphoid_aggregates', 'histiocytic_granulomas', 'mucin_depletion',
    'pyloric_gland_metaplasia', 'paneth_cell_metaplasia',
    'neuronal_hyperplasia', 'muscular_hypertrophy',
]

# Clinical feature columns in the order they appear in X (first N_CLIN columns)
CLINICAL_NUMERIC = [
    'rectal_bleed', 'first_abdopain', 'first_daily_bm', 'first_well_being',
]
STOOL_FREQ_MAP = {
    'Normal': 0, '1-2 stools more than normal': 1,
    '3-4 stools more than normal': 2, '>4 stools more than normal': 3,
    'Not applicable, I have an ostomy': np.nan,
}
FECAL_URGENCY_MAP = {
    'Mild': 1, 'Moderate': 2, 'Moderately severe': 3, 'Severe': 4,
    'Not applicable, I have an ostomy': np.nan,
}
DA6M_MAP = {
    '1. Absence of symptoms': 0, '2. Rarely active': 1,
    '3. Sometimes active': 2, '4. Occasionally active': 3,
    '5. Often active': 4, '6. Constantly active': 5,
}
CLINICAL_BINARY = [
    'cardiovascular_disease', 'hypertension', 'diabetes_type_i',
    'diabetes_type_ii', 'ckd', 'hiv', 'tuberculosis',
]
CLINICAL_COLS = (
    CLINICAL_NUMERIC
    + ['stool_freq', 'fecal_urgency', 'da6m', 'smk']
    + CLINICAL_BINARY
)
N_CLIN = len(CLINICAL_COLS)

RF_PARAMS = dict(
    n_estimators=500, max_features='sqrt', min_samples_leaf=2,
    class_weight='balanced', n_jobs=-1, random_state=42,
)
MLP_PARAMS = dict(
    hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
    alpha=1e-3, batch_size=64, max_iter=500, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20, random_state=42,
)


def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


# ── Data loaders (same as train_rf.py) ────────────────────────────────────────

def load_histoscore_visits(cv_patients):
    pat_df  = cv_patients.set_index('patient_id')
    cv_pids = set(pat_df.index)
    wsi = pd.read_csv(WSI_META)
    wsi = wsi[
        wsi['BIOSAMPLE_LOCATION'].isin(AT20) &
        wsi['diagnosis'].isin(CDUC_RAW) &
        wsi['deidentified_master_patient_id'].isin(cv_pids)
    ].copy()
    wsi['date']      = pd.to_datetime(wsi['Date Sample Collected'],
                                      dayfirst=True, errors='coerce')
    wsi['slide_id']  = wsi['IMAGE_VSI'].str.replace('.vsi', '', regex=False)
    wsi['visit_key'] = list(zip(wsi['deidentified_master_patient_id'],
                                wsi['date'].dt.date))
    scores = pd.read_csv(HISTOSCORE_CSV).set_index('slide')
    rows = []
    for (pid, vdate), grp in wsi.groupby('visit_key'):
        if pid not in pat_df.index:
            continue
        vecs = [scores.loc[sid, HISTO_COLS].values.astype(np.float32)
                for sid in grp['slide_id'] if sid in scores.index]
        if not vecs:
            continue
        vec = np.mean(vecs, axis=0)
        row = pat_df.loc[pid]
        rows.append({'visit_key': str((pid, vdate)), 'patient_id': pid,
                     'visit_date': str(vdate),
                     'label': int(row['diagnosis'] == 'Ulcerative colitis'),
                     'fold': int(row['fold']),
                     **{c: float(v) for c, v in zip(HISTO_COLS, vec)}})
    df = pd.DataFrame(rows)
    print(f'  histoscore visits: {len(df)} / {df["patient_id"].nunique()} patients')
    return df


def load_rna_visits(cv_patients):
    pat_df  = cv_patients.set_index('patient_id')
    cv_pids = set(pat_df.index)
    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(norm_dx)
    rna['date'] = pd.to_datetime(rna['sample_collected_date'],
                                 dayfirst=True, errors='coerce')
    rna = rna[
        rna['characteristics_bio_material'].isin(AT20) &
        rna['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pids)
    ].copy()
    rna = rna.drop_duplicates(subset='visit_encounter_id', keep='first')
    sample_ids = set(rna['SampleID'])
    print(f'  Loading TPM GCT for {len(sample_ids)} RNA visits ...')
    with open(TPM_GCT) as f:
        f.readline(); f.readline()
        header_cols = f.readline().strip().split('\t')
    keep       = [0, 1] + [i for i, c in enumerate(header_cols) if c in sample_ids]
    keep_names = [header_cols[i] for i in keep]
    gct_df = pd.read_csv(TPM_GCT, sep='\t', skiprows=2, header=0,
                         usecols=keep_names, index_col=0,
                         dtype={c: (str if c in ('Name', 'Description') else np.float32)
                                for c in keep_names})
    gct_df = gct_df.drop(columns=['Description'])
    X_df = np.log2(gct_df.T.astype(np.float32) + 1.0)
    rows = []
    for _, enc_row in rna.iterrows():
        sid = enc_row['SampleID']
        pid = enc_row['deidentified_master_patient_id']
        if sid not in X_df.index or pid not in pat_df.index:
            continue
        vec   = X_df.loc[sid].values
        row   = pat_df.loc[pid]
        vdate = enc_row['date'].date() if not pd.isna(enc_row['date']) else None
        rows.append({
            'visit_key':  str((pid, vdate)), 'patient_id': pid,
            'visit_date': str(vdate), 'SampleID': sid,
            'label': int(row['diagnosis'] == 'Ulcerative colitis'),
            'fold':  int(row['fold']),
            **{f'r{i}': v for i, v in enumerate(vec)},
        })
    df = pd.DataFrame(rows)
    dim = len([c for c in df.columns if c.startswith('r')])
    print(f'  RNA visits: {len(df)} / {df["patient_id"].nunique()} patients  dim={dim}')
    return df


def common_visits(img_df, rna_df, max_gap_days=7):
    img_df = img_df.copy(); rna_df = rna_df.copy()
    img_df['_date'] = pd.to_datetime(img_df['visit_date'])
    rna_df['_date'] = pd.to_datetime(rna_df['visit_date'])
    rna_by_pid = {pid: grp for pid, grp in rna_df.groupby('patient_id')}
    img_rows, rna_rows = [], []
    for _, irow in img_df.iterrows():
        pid = irow['patient_id']
        if pid not in rna_by_pid:
            continue
        cands = rna_by_pid[pid].copy()
        cands['_gap'] = (cands['_date'] - irow['_date']).abs().dt.days
        best = cands.loc[cands['_gap'].idxmin()]
        if best['_gap'] > max_gap_days:
            continue
        img_rows.append(irow); rna_rows.append(best)
    img_out = pd.DataFrame(img_rows).drop(columns=['_date']).reset_index(drop=True)
    rna_out = pd.DataFrame(rna_rows).drop(columns=['_date', '_gap']).reset_index(drop=True)
    assert (img_out['patient_id'] == rna_out['patient_id']).all()
    n_prox = (img_out['visit_key'] != rna_out['visit_key']).sum()
    if n_prox:
        print(f'  proximity-matched {n_prox} visit(s) <= {max_gap_days}d')
    return img_out, rna_out


def load_img_base_for_matched(img_m):
    wsi = pd.read_csv(WSI_META)
    wsi['visit_date'] = pd.to_datetime(
        wsi['Date Sample Collected'], dayfirst=True, errors='coerce').dt.date.astype(str)
    wsi['slide_id']   = wsi['IMAGE_VSI'].str.replace('.vsi', '', regex=False)
    wsi['patient_id'] = wsi['deidentified_master_patient_id']
    visit_slides = {}
    for _, row in wsi.iterrows():
        key = (row['patient_id'], row['visit_date'])
        visit_slides.setdefault(key, []).append(row['slide_id'])
    rows = []
    for _, irow in img_m.iterrows():
        key  = (irow['patient_id'], irow['visit_date'])
        vecs = []
        for sid in visit_slides.get(key, []):
            h5p = os.path.join(EMB_BASE, f'{sid}.h5')
            if os.path.exists(h5p):
                with h5py.File(h5p, 'r') as h:
                    vecs.append(h['features'][:])
        if not vecs:
            continue
        vec = np.mean(vecs, axis=0)
        rows.append({'visit_key': irow['visit_key'],
                     **{f'f{i}': float(v) for i, v in enumerate(vec)}})
    df  = pd.DataFrame(rows)
    dim = len([c for c in df.columns if c.startswith('f')])
    print(f'  prism2_base: {len(df)} visits loaded  dim={dim}')
    return df


def load_bulkformer_emb():
    print('  Loading BulkFormer embeddings ...')
    emb = torch.load(BULKFORMER_PT, map_location='cpu').numpy().astype(np.float32)
    log_tpm = pd.read_csv(LOG_TPM_CSV, usecols=['SampleID'])
    assert len(emb) == len(log_tpm)
    df = pd.DataFrame(emb, columns=EMB_COLS)
    df['SampleID'] = log_tpm['SampleID'].values
    print(f'  BulkFormer: {emb.shape[0]} x {emb.shape[1]}')
    return df


def join_bulkformer(rna_m, emb_df):
    rna_indexed = rna_m.reset_index(drop=True)
    merged = rna_indexed[['SampleID']].merge(
        emb_df[['SampleID'] + EMB_COLS], on='SampleID', how='left')
    has_emb = merged[EMB_COLS[0]].notna()
    if (~has_emb).sum():
        print(f'  Dropped {(~has_emb).sum()} visit(s) with no BulkFormer embedding')
    bf_df = pd.concat([
        rna_indexed[['visit_key', 'patient_id', 'visit_date', 'label', 'fold']],
        merged[EMB_COLS],
    ], axis=1)[has_emb].reset_index(drop=True)
    return bf_df, has_emb


# ── Clinical feature loader ────────────────────────────────────────────────────

def load_clinical_for_visits(visit_df):
    """
    Load patient-level clinical features and join to visit_df by patient_id.
    Returns X_clin (n_visits × N_CLIN) as float32; NaN for missing values.
    Imputation must be done per fold in run_arm (no imputation here).
    """
    meta = pd.read_csv(METADATA_CSV)
    pid_set = set(visit_df['patient_id'])
    meta = meta[meta['deidentified_master_patient_id'].isin(pid_set)].copy()

    for col in CLINICAL_NUMERIC:
        meta[col] = pd.to_numeric(meta[col], errors='coerce')

    meta['stool_freq']    = meta['stool_freq'].map(STOOL_FREQ_MAP)
    meta['fecal_urgency'] = meta['fecal_urgency'].map(FECAL_URGENCY_MAP)
    meta['da6m']          = meta['da6m'].map(DA6M_MAP)
    meta['smk']           = meta['smk'].map({'Yes': 1, 'No': 0})

    for col in CLINICAL_BINARY:
        meta[col] = meta[col].map({'Y': 1, 'N': 0})

    clin = meta[['deidentified_master_patient_id'] + CLINICAL_COLS].copy()
    clin = clin.rename(columns={'deidentified_master_patient_id': 'patient_id'})

    merged = visit_df[['patient_id']].merge(clin, on='patient_id', how='left')
    X_clin = merged[CLINICAL_COLS].values.astype(np.float32)

    coverage = np.isfinite(X_clin).mean(axis=0)
    n_pat_with = (merged.groupby('patient_id')[CLINICAL_COLS[0]]
                  .first().notna().sum())
    print(f'  Clinical: {n_pat_with}/{visit_df["patient_id"].nunique()} patients '
          f'have metadata  mean coverage={coverage.mean():.1%}')
    return X_clin


# ── CV runner ─────────────────────────────────────────────────────────────────

def run_arm(name, df, X, n_clin=0):
    """
    Run RF and MLP for one arm.
    X layout: [clinical (n_clin cols) | other features]
    Clinical columns are imputed per fold (median, training fold only).
    MLP also applies StandardScaler per fold.
    """
    y     = df['label'].values
    folds = df['fold'].values
    vkeys = df['visit_key'].values
    pids  = df['patient_id'].values

    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold; va = folds == fold
        X_tr = X[tr].copy(); X_va = X[va].copy()

        # Impute clinical NaN using training-fold median
        if n_clin > 0:
            imp = SimpleImputer(strategy='median')
            X_tr[:, :n_clin] = imp.fit_transform(X_tr[:, :n_clin])
            X_va[:, :n_clin] = imp.transform(X_va[:, :n_clin])

        y_tr = y[tr]; y_va = y[va]

        # RF
        clf_rf = RandomForestClassifier(**RF_PARAMS)
        clf_rf.fit(X_tr, y_tr)
        uc_col = list(clf_rf.classes_).index(1)
        y_sc_rf = clf_rf.predict_proba(X_va)[:, uc_col]
        y_pr_rf = clf_rf.predict(X_va)

        auc_rf = roc_auc_score(y_va, y_sc_rf)
        ap_rf  = average_precision_score(y_va, y_sc_rf, pos_label=1)
        cr_rf  = classification_report(y_va, y_pr_rf,
                                        target_names=['CD', 'UC'], output_dict=True)
        cm_rf  = confusion_matrix(y_va, y_pr_rf)
        fold_results.append(dict(
            strategy=f'{name}_rf', fold=fold,
            n_train_visits=int(tr.sum()), n_val_visits=int(va.sum()),
            n_val_patients=int(pd.Series(pids[va]).nunique()),
            auc=round(auc_rf, 4), ap=round(ap_rf, 4),
            accuracy=round(cr_rf['accuracy'], 4),
            cd_f1=round(cr_rf['CD']['f1-score'], 4),
            uc_f1=round(cr_rf['UC']['f1-score'], 4),
            tn=int(cm_rf[0,0]), fp=int(cm_rf[0,1]),
            fn=int(cm_rf[1,0]), tp=int(cm_rf[1,1]),
        ))
        for vk, pid, yt, yp, ys in zip(vkeys[va], pids[va], y_va, y_pr_rf, y_sc_rf):
            all_preds.append(dict(visit_key=vk, patient_id=pid,
                                  strategy=f'{name}_rf', fold=fold,
                                  true_label=int(yt), pred_label=int(yp),
                                  prob_uc=round(float(ys), 5)))

        # MLP — scale after imputation
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_va_sc = sc.transform(X_va)
        sw = compute_sample_weight('balanced', y_tr)
        clf_mlp = MLPClassifier(**MLP_PARAMS)
        clf_mlp.fit(X_tr_sc, y_tr, sample_weight=sw)
        uc_col_m = list(clf_mlp.classes_).index(1)
        y_sc_mlp = clf_mlp.predict_proba(X_va_sc)[:, uc_col_m]
        y_pr_mlp = clf_mlp.predict(X_va_sc)

        auc_mlp = roc_auc_score(y_va, y_sc_mlp)
        ap_mlp  = average_precision_score(y_va, y_sc_mlp, pos_label=1)
        cr_mlp  = classification_report(y_va, y_pr_mlp,
                                         target_names=['CD', 'UC'], output_dict=True)
        cm_mlp  = confusion_matrix(y_va, y_pr_mlp)
        fold_results.append(dict(
            strategy=f'{name}_mlp', fold=fold,
            n_train_visits=int(tr.sum()), n_val_visits=int(va.sum()),
            n_val_patients=int(pd.Series(pids[va]).nunique()),
            auc=round(auc_mlp, 4), ap=round(ap_mlp, 4),
            accuracy=round(cr_mlp['accuracy'], 4),
            cd_f1=round(cr_mlp['CD']['f1-score'], 4),
            uc_f1=round(cr_mlp['UC']['f1-score'], 4),
            tn=int(cm_mlp[0,0]), fp=int(cm_mlp[0,1]),
            fn=int(cm_mlp[1,0]), tp=int(cm_mlp[1,1]),
        ))
        for vk, pid, yt, yp, ys in zip(vkeys[va], pids[va], y_va, y_pr_mlp, y_sc_mlp):
            all_preds.append(dict(visit_key=vk, patient_id=pid,
                                  strategy=f'{name}_mlp', fold=fold,
                                  true_label=int(yt), pred_label=int(yp),
                                  prob_uc=round(float(ys), 5)))

        print(f'  {name:<24}  fold {fold}:  '
              f'RF AUC={auc_rf:.4f}   MLP AUC={auc_mlp:.4f}  '
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
    global HISTOSCORE_CSV, EMB_BASE, CV_PATIENTS, OUT_DIR
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--histoscore_csv", type=str, default=HISTOSCORE_CSV)
    p.add_argument("--emb_base_dir",   type=str, default=EMB_BASE)
    p.add_argument("--cv_patients",    type=str, default=CV_PATIENTS)
    p.add_argument("--out_dir",        type=str, default=OUT_DIR)
    a = p.parse_args()
    HISTOSCORE_CSV = a.histoscore_csv
    EMB_BASE       = a.emb_base_dir
    CV_PATIENTS    = a.cv_patients
    OUT_DIR        = a.out_dir

    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)

    print('=== Loading matched cohort ===')
    img_df = load_histoscore_visits(cv_patients)
    rna_df = load_rna_visits(cv_patients)
    img_m, rna_m = common_visits(img_df, rna_df)

    print('\n=== Loading prism2_base ===')
    base_raw = load_img_base_for_matched(img_m)
    has_base = img_m['visit_key'].isin(set(base_raw['visit_key']))
    img_m = img_m[has_base.values].reset_index(drop=True)
    rna_m = rna_m[has_base.values].reset_index(drop=True)
    base_m = img_m[['visit_key']].merge(base_raw, on='visit_key', how='left')

    print('\n=== Loading BulkFormer embeddings ===')
    emb_df = load_bulkformer_emb()
    bf_m, has_emb = join_bulkformer(rna_m, emb_df)
    img_m  = img_m[has_emb.values].reset_index(drop=True)
    rna_m  = rna_m[has_emb.values].reset_index(drop=True)
    base_m = base_m[has_emb.values].reset_index(drop=True)

    n_vis = len(bf_m); n_pat = bf_m['patient_id'].nunique()
    print(f'\nCohort: {n_vis} visits / {n_pat} patients  '
          f"(CD {(bf_m['label']==0).sum()}, UC {(bf_m['label']==1).sum()})")

    print('\n=== Loading clinical features ===')
    X_clin = load_clinical_for_visits(img_m)

    rna_cols  = [c for c in rna_m.columns if c.startswith('r')]
    base_cols = [c for c in base_m.columns if c.startswith('f')]

    X_rna  = rna_m[rna_cols].values.astype(np.float32)
    X_base = base_m[base_cols].values.astype(np.float32)

    all_fold_results, all_preds = [], []

    # Clinical: imputation needed (n_clin = N_CLIN, all clinical columns)
    # Other arms: clinical is always in the first N_CLIN columns
    arms = [
        ('clinical_visit',      img_m, X_clin,                                  N_CLIN),
        ('img_clinical_visit',  img_m, np.hstack([X_clin, X_base]),              N_CLIN),
        ('rna_clinical_visit',  rna_m, np.hstack([X_clin, X_rna]),               N_CLIN),
        ('trimodal_visit',      rna_m, np.hstack([X_clin, X_base, X_rna]),       N_CLIN),
    ]

    for arm_name, arm_df, arm_X, arm_n_clin in arms:
        print(f'\n--- {arm_name}  ({arm_X.shape[1]}-d) ---')
        fr, preds = run_arm(arm_name, arm_df, arm_X, n_clin=arm_n_clin)
        all_fold_results += fr; all_preds += preds

    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'at20cm_multimodal_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'at20cm_multimodal_predictions.csv'), index=False)

    strategies = list(dict.fromkeys(r['strategy'] for r in all_fold_results))
    summary_list = []
    for strat in strategies:
        fr_s = [r for r in all_fold_results if r['strategy'] == strat]
        s = summarise(fr_s)
        s['strategy'] = strat
        s['n_visits'] = n_vis; s['n_patients'] = n_pat
        summary_list.append(s)

    with open(os.path.join(OUT_DIR, 'at20cm_multimodal_summary.json'), 'w') as f:
        json.dump(summary_list, f, indent=2)

    print('\n\n=== MULTIMODAL SUMMARY ===')
    print(f"{'Strategy':<34} {'AUC':<24} {'AP':<24} {'Accuracy'}")
    print('-' * 96)
    for s in summary_list:
        print(f"  {s['strategy']:<32} "
              f"AUC={s['mean_auc']:.4f}+/-{s['std_auc']:.4f}  "
              f"AP={s['mean_ap']:.4f}+/-{s['std_ap']:.4f}  "
              f"Acc={s['mean_acc']:.4f}")
    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
