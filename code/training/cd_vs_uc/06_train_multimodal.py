"""
Multimodal CD vs UC classifier: imaging embeddings + transcriptomics VST.

Unit of analysis
----------------
Patient-level mean pooling  : all colon slides / RNA samples for a patient are
  averaged into one vector each; fusion is at the patient level.

Site-matched (biopsy-paired): RNA and imaging are only paired when they come from
  the SAME anatomical location (e.g. both from "At 20 cm").  This gives 1,388
  patient-location instances from 993 patients and is the biologically faithful
  pairing.  For locations where a patient has multiple slides, those slides are
  mean-pooled.  The RF is then trained on these 1,388 instances (one per
  patient-location pair), still using patient-level fold assignments to avoid leakage.

Biopsy-site mapping validation (metadata check)
------------------------------------------------
    Patients with both RNA and imaging:  1,007
    Exact location match (same site set): 856 (85 %)
    Partial overlap:                       137
    No overlap:                             14  <- excluded from site-matched fusion
    Patient-location pairs with both:    1,388 from 993 patients

Fusion strategies evaluated
---------------------------
1. img_base_patmean      - imaging only, patient-mean
2. img_diag_patmean      - imaging only, patient-mean
3. rna_patmean           - RNA only, patient-mean
4. late_fusion_base_rna  - avg P(UC) from img_base + RNA (patient-mean)
5. concat_scaled         - StandardScaler per block, concat (patient-mean)
6. concat_pca128         - PCA-128 per block, concat (patient-mean)
7. site_matched_scaled   - same as concat_scaled but uses site-matched pairs
8. site_matched_pca128   - same as concat_pca128 but uses site-matched pairs

Inputs
------
- /home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv
- prism2_base / prism2_diagnostic embedding directories  (slide-level .h5)
- transcriptomics VST GCT  (sample-level)
- ibd_21183_omics_patient_mapping_genestack.csv
- GSF1478941_sample_combined_from1stRun.tsv__metadata.csv

Outputs  (under /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/)
--------
- multimodal_fold_metrics.csv         per-fold metrics for every fusion strategy
- multimodal_patient_predictions.csv  per-patient predicted label + prob_uc per strategy
- multimodal_summary.json             aggregated statistics for all strategies
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────────
TRANSCRIPTOMICS_DIR = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/'
                       'genestack/transcriptomics')
VST_GCT     = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF1491805_CombatSeq_vst_mtx_batch_corrected_'
              'alltissues_all3releases_header.gct')
MAPPING_CSV = os.path.join(TRANSCRIPTOMICS_DIR,
              'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
EMB_DIRS = {
    'prism2_base': ('/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
                    '20x_224px_0px_overlap/prism2_base'),
    'prism2_diagnostic': ('/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
                          '20x_224px_0px_overlap/prism2_diagnostic'),
}
OUT_DIR = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/06_07_multimodal_allsites/results'

COLON_LOCS = {
    'at 20 cm', 'At 20 cm', 'Cecum', 'Rectum',
    'Ascending Colon', 'Descending Colon', 'Sigmoid Colon', 'Transverse Colon',
}
RF_PARAMS = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)

# PCA dimensionality for equal-projection strategies
N_COMPONENTS = 128


# ── helper: normalise diagnosis string ────────────────────────────────────────
def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


# ── load slide-level embeddings and mean-pool to patient level ────────────────
def load_imaging_patient_mean(cv_patients, emb_dir):
    """
    Returns DataFrame with columns [patient_id, label, fold, f0, f1, ..., f(D-1)].
    Restricted to patients that have at least one slide embedding.
    """
    from pathlib import Path
    pat_df = cv_patients.copy()
    pat_df['label'] = (pat_df['diagnosis'] == 'Ulcerative colitis').astype(int)

    # scan slide IDs from cv_splits_slides
    slides_csv = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_slides.csv'
    slides = pd.read_csv(slides_csv)
    slides = slides[slides['patient_id'].isin(pat_df['patient_id'])]

    records = {}
    for pid, grp in slides.groupby('patient_id'):
        vecs = []
        for sid in grp['slide_id']:
            h5p = os.path.join(emb_dir, f'{sid}.h5')
            if os.path.exists(h5p):
                with h5py.File(h5p, 'r') as h:
                    vecs.append(h['features'][:])
        if vecs:
            records[pid] = np.mean(vecs, axis=0)

    dim = next(iter(records.values())).shape[0]
    feat_cols = [f'f{i}' for i in range(dim)]
    rows = []
    for _, row in pat_df.iterrows():
        pid = row['patient_id']
        if pid in records:
            rows.append({'patient_id': pid, 'label': row['label'], 'fold': row['fold'],
                         **dict(zip(feat_cols, records[pid]))} )
    df = pd.DataFrame(rows)
    print(f'  {os.path.basename(emb_dir)}: {len(df)} patients with embeddings  dim={dim}')
    return df


# ── load transcriptomics and mean-pool to patient level ───────────────────────
def load_rna_patient_mean(cv_patients):
    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(norm_dx)
    cv_pats = set(cv_patients['patient_id'])
    rna_colon = rna[
        rna['characteristics_bio_material'].isin(COLON_LOCS) &
        rna['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pats)
    ]
    sample_ids = set(rna_colon['SampleID'])

    print(f'  Loading RNA GCT for {len(sample_ids)} samples ...')
    with open(VST_GCT) as f:
        f.readline(); f.readline()
        header_cols = f.readline().strip().split('\t')
    keep = [0, 1] + [i for i, c in enumerate(header_cols) if c in sample_ids]
    keep_names = [header_cols[i] for i in keep]
    gct_df = pd.read_csv(VST_GCT, sep='\t', skiprows=2, header=0,
                         usecols=keep_names, index_col=0,
                         dtype={c: (str if c in ('Name','Description') else np.float32)
                                for c in keep_names})
    gct_df = gct_df.drop(columns=['Description'])
    X_df = gct_df.T.astype(np.float32)   # samples × genes

    pid_map = rna_colon.set_index('SampleID')['deidentified_master_patient_id']
    pat_df = cv_patients.set_index('patient_id')

    feat_cols = [f'r{i}' for i in range(X_df.shape[1])]
    rows = []
    for pid, grp in pid_map.groupby(pid_map):
        sids = [s for s in grp.index if s in X_df.index]
        if not sids or pid not in pat_df.index:
            continue
        vec = X_df.loc[sids].values.mean(axis=0)
        row = pat_df.loc[pid]
        rows.append({'patient_id': pid,
                     'label': int(row['diagnosis'] == 'Ulcerative colitis'),
                     'fold': int(row['fold']),
                     **dict(zip(feat_cols, vec))})
    df = pd.DataFrame(rows)
    print(f'  RNA: {len(df)} patients with RNA  dim={X_df.shape[1]}')
    return df


# ── single-modality RF (used inside late fusion and as reference) ─────────────
def run_single_modality_cv(name, pat_df):
    feat_cols = [c for c in pat_df.columns if c not in ('patient_id','label','fold')]
    X = pat_df[feat_cols].values.astype(np.float32)
    y = pat_df['label'].values
    folds = pat_df['fold'].values
    ids = pat_df['patient_id'].values

    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold; va = folds == fold
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X[tr], y[tr])
        proba   = clf.predict_proba(X[va])
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]
        y_pred  = clf.predict(X[va])
        y_val   = y[va]
        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred, target_names=['CD','UC'], output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)
        fold_results.append(dict(
            strategy=name, fold=fold,
            n_train=int(tr.sum()), n_val=int(va.sum()),
            auc=round(auc,4), ap=round(ap,4), accuracy=round(cr['accuracy'],4),
            cd_f1=round(cr['CD']['f1-score'],4), uc_f1=round(cr['UC']['f1-score'],4),
            tn=int(cm[0,0]), fp=int(cm[0,1]), fn=int(cm[1,0]), tp=int(cm[1,1]),
        ))
        for pid, yt, yp, ys in zip(ids[va], y_val, y_pred, y_score):
            all_preds.append(dict(patient_id=pid, strategy=name, fold=fold,
                                  true_label=int(yt), pred_label=int(yp),
                                  prob_uc=round(float(ys),5)))
        print(f'    {name} fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  Acc={cr["accuracy"]:.4f}')
    return fold_results, all_preds


# ── fusion helpers ────────────────────────────────────────────────────────────
def common_patients(img_df, rna_df):
    """Inner join on patient_id; return aligned DataFrames."""
    common = set(img_df['patient_id']) & set(rna_df['patient_id'])
    img = img_df[img_df['patient_id'].isin(common)].sort_values('patient_id').reset_index(drop=True)
    rna = rna_df[rna_df['patient_id'].isin(common)].sort_values('patient_id').reset_index(drop=True)
    assert (img['patient_id'] == rna['patient_id']).all()
    return img, rna


def feat(df):
    return df[[c for c in df.columns if c not in ('patient_id','label','fold')]].values.astype(np.float32)


def make_combined_df(img_df, rna_df, X_combined):
    base = img_df[['patient_id','label','fold']].copy()
    feat_cols = [f'mm{i}' for i in range(X_combined.shape[1])]
    return pd.concat([base, pd.DataFrame(X_combined, columns=feat_cols)], axis=1)


# ── fusion strategies ─────────────────────────────────────────────────────────
def strategy_concat_scaled(img_df, rna_df):
    """Scale each block to unit variance, then concatenate."""
    img_a, rna_a = common_patients(img_df, rna_df)
    Xi = feat(img_a); Xr = feat(rna_a)
    # fit scaler on training portion is handled fold-by-fold below
    return img_a[['patient_id','label','fold']], Xi, Xr


def run_concat_scaled_cv(name, img_df, rna_df):
    img_a, rna_a = common_patients(img_df, rna_df)
    Xi = feat(img_a); Xr = feat(rna_a)
    y = img_a['label'].values
    folds = img_a['fold'].values
    ids = img_a['patient_id'].values

    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold; va = folds == fold
        # scale each block using train-fold statistics only
        sc_i = StandardScaler().fit(Xi[tr]); sc_r = StandardScaler().fit(Xr[tr])
        Xtr = np.hstack([sc_i.transform(Xi[tr]), sc_r.transform(Xr[tr])])
        Xva = np.hstack([sc_i.transform(Xi[va]), sc_r.transform(Xr[va])])
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(Xtr, y[tr])
        proba   = clf.predict_proba(Xva)
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]
        y_pred  = clf.predict(Xva)
        y_val   = y[va]
        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred, target_names=['CD','UC'], output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)
        fold_results.append(dict(
            strategy=name, fold=fold,
            n_train=int(tr.sum()), n_val=int(va.sum()),
            auc=round(auc,4), ap=round(ap,4), accuracy=round(cr['accuracy'],4),
            cd_f1=round(cr['CD']['f1-score'],4), uc_f1=round(cr['UC']['f1-score'],4),
            tn=int(cm[0,0]), fp=int(cm[0,1]), fn=int(cm[1,0]), tp=int(cm[1,1]),
        ))
        for pid, yt, yp, ys in zip(ids[va], y_val, y_pred, y_score):
            all_preds.append(dict(patient_id=pid, strategy=name, fold=fold,
                                  true_label=int(yt), pred_label=int(yp),
                                  prob_uc=round(float(ys),5)))
        print(f'    {name} fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  Acc={cr["accuracy"]:.4f}')
    return fold_results, all_preds


def run_concat_pca_cv(name, img_df, rna_df, n_components):
    """Project each modality to n_components via PCA (train-only fit), then concat."""
    img_a, rna_a = common_patients(img_df, rna_df)
    Xi = feat(img_a); Xr = feat(rna_a)
    y = img_a['label'].values
    folds = img_a['fold'].values
    ids = img_a['patient_id'].values

    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold; va = folds == fold
        pca_i = PCA(n_components=n_components, random_state=42).fit(Xi[tr])
        pca_r = PCA(n_components=n_components, random_state=42).fit(Xr[tr])
        Xtr = np.hstack([pca_i.transform(Xi[tr]), pca_r.transform(Xr[tr])])
        Xva = np.hstack([pca_i.transform(Xi[va]), pca_r.transform(Xr[va])])
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(Xtr, y[tr])
        proba   = clf.predict_proba(Xva)
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]
        y_pred  = clf.predict(Xva)
        y_val   = y[va]
        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred, target_names=['CD','UC'], output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)
        fold_results.append(dict(
            strategy=name, fold=fold,
            n_train=int(tr.sum()), n_val=int(va.sum()),
            auc=round(auc,4), ap=round(ap,4), accuracy=round(cr['accuracy'],4),
            cd_f1=round(cr['CD']['f1-score'],4), uc_f1=round(cr['UC']['f1-score'],4),
            tn=int(cm[0,0]), fp=int(cm[0,1]), fn=int(cm[1,0]), tp=int(cm[1,1]),
        ))
        for pid, yt, yp, ys in zip(ids[va], y_val, y_pred, y_score):
            all_preds.append(dict(patient_id=pid, strategy=name, fold=fold,
                                  true_label=int(yt), pred_label=int(yp),
                                  prob_uc=round(float(ys),5)))
        print(f'    {name} fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  Acc={cr["accuracy"]:.4f}')
    return fold_results, all_preds


def run_late_fusion_cv(name, img_fold_preds, rna_fold_preds, img_df, rna_df):
    """Average P(UC) from two single-modality models (patient-level)."""
    img_a, rna_a = common_patients(img_df, rna_df)
    common_pids  = set(img_a['patient_id'])
    y_map  = img_a.set_index('patient_id')['label'].to_dict()
    f_map  = img_a.set_index('patient_id')['fold'].to_dict()

    img_scores = (pd.DataFrame(img_fold_preds)
                  .query('patient_id in @common_pids')[['patient_id','fold','prob_uc']]
                  .rename(columns={'prob_uc':'img_prob'}))
    rna_scores = (pd.DataFrame(rna_fold_preds)
                  .query('patient_id in @common_pids')[['patient_id','fold','prob_uc']]
                  .rename(columns={'prob_uc':'rna_prob'}))
    merged = img_scores.merge(rna_scores, on=['patient_id','fold'], how='inner')

    fold_results, all_preds = [], []
    for fold in range(5):
        fdf = merged[merged['fold'] == fold]
        y_val   = np.array([y_map[p] for p in fdf['patient_id']])
        y_score = ((fdf['img_prob'] + fdf['rna_prob']) / 2).values
        y_pred  = (y_score >= 0.5).astype(int)
        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred, target_names=['CD','UC'], output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)
        fold_results.append(dict(
            strategy=name, fold=fold,
            n_train=int((np.array([f_map[p] for p in merged['patient_id']]) != fold).sum()),
            n_val=int(len(fdf)),
            auc=round(auc,4), ap=round(ap,4), accuracy=round(cr['accuracy'],4),
            cd_f1=round(cr['CD']['f1-score'],4), uc_f1=round(cr['UC']['f1-score'],4),
            tn=int(cm[0,0]), fp=int(cm[0,1]), fn=int(cm[1,0]), tp=int(cm[1,1]),
        ))
        for pid, yt, yp, ys in zip(fdf['patient_id'], y_val, y_pred, y_score):
            all_preds.append(dict(patient_id=pid, strategy=name, fold=fold,
                                  true_label=int(yt), pred_label=int(yp),
                                  prob_uc=round(float(ys),5)))
        print(f'    {name} fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  Acc={cr["accuracy"]:.4f}')
    return fold_results, all_preds


# ── site-matched data loader ──────────────────────────────────────────────────
def load_site_matched_pairs(cv_patients, emb_dir):
    """
    Build (patient_id, location, label, fold, img_vec, rna_vec) for every
    (patient, location) pair that has BOTH an imaging slide and an RNA sample.

    Returns a DataFrame with columns:
      patient_id, location, label, fold, img_f0..img_f(Di-1), rna_r0..rna_r(Dr-1)
    """
    slides_csv = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_slides.csv'
    slides_all = pd.read_csv(slides_csv)

    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna_meta = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna_meta['diagnosis_norm'] = rna_meta['diagnosis'].map(norm_dx)

    cv_pats = set(cv_patients['patient_id'])
    rna_meta = rna_meta[
        rna_meta['characteristics_bio_material'].isin(COLON_LOCS) &
        rna_meta['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna_meta['Sample QC'] != 'fail') &
        rna_meta['deidentified_master_patient_id'].isin(cv_pats)
    ].copy()

    loc_norm = {
        'at 20 cm': 'At 20 cm', 'At 20 cm': 'At 20 cm',
        'Cecum': 'Cecum', 'Rectum': 'Rectum',
        'Ascending Colon': 'Ascending Colon', 'Descending Colon': 'Descending Colon',
        'Sigmoid Colon': 'Sigmoid Colon', 'Transverse Colon': 'Transverse Colon',
    }
    rna_meta['location_norm']  = rna_meta['characteristics_bio_material'].map(loc_norm)
    slides_all['location_norm'] = slides_all['BIOSAMPLE_LOCATION'].map(loc_norm)

    # Load VST GCT for all needed samples
    sample_ids_needed = set(rna_meta['SampleID'])
    print(f'  Site-matched: loading GCT for {len(sample_ids_needed)} RNA samples ...')
    with open(VST_GCT) as f:
        f.readline(); f.readline()
        header_cols = f.readline().strip().split('\t')
    keep = [0, 1] + [i for i, c in enumerate(header_cols) if c in sample_ids_needed]
    keep_names = [header_cols[i] for i in keep]
    gct_df = pd.read_csv(VST_GCT, sep='\t', skiprows=2, header=0,
                         usecols=keep_names, index_col=0,
                         dtype={c: (str if c in ('Name','Description') else np.float32)
                                for c in keep_names})
    gct_df = gct_df.drop(columns=['Description'])
    X_rna = gct_df.T.astype(np.float32)   # samples × genes

    pat_fold = cv_patients.set_index('patient_id')[['fold','diagnosis']]
    rna_pid  = rna_meta.set_index('SampleID')['deidentified_master_patient_id']
    rna_loc  = rna_meta.set_index('SampleID')['location_norm']

    # Index slides by (patient, location)
    slide_by_loc = {}
    for _, row in slides_all.iterrows():
        key = (row['patient_id'], row['location_norm'])
        slide_by_loc.setdefault(key, []).append(row['slide_id'])

    # Index RNA by (patient, location)
    rna_by_loc = {}
    for sid in X_rna.index:
        if sid not in rna_pid.index: continue
        pid = rna_pid[sid]; loc = rna_loc[sid]
        rna_by_loc.setdefault((pid, loc), []).append(sid)

    # Build matched pairs
    Di = None  # imaging dim (detected from first h5)
    rows = []
    for (pid, loc), sids in rna_by_loc.items():
        if (pid, loc) not in slide_by_loc: continue
        if pid not in pat_fold.index: continue
        slide_ids = slide_by_loc[(pid, loc)]
        img_vecs = []
        for sid in slide_ids:
            h5p = os.path.join(emb_dir, f'{sid}.h5')
            if os.path.exists(h5p):
                with h5py.File(h5p, 'r') as h:
                    img_vecs.append(h['features'][:])
        if not img_vecs: continue
        img_vec = np.mean(img_vecs, axis=0)
        if Di is None: Di = img_vec.shape[0]
        rna_vec = X_rna.loc[sids].values.mean(axis=0)
        prow = pat_fold.loc[pid]
        label = int(prow['diagnosis'] == 'Ulcerative colitis')
        rows.append({'patient_id': pid, 'location': loc, 'label': label,
                     'fold': int(prow['fold']),
                     **{f'img_f{i}': v for i,v in enumerate(img_vec)},
                     **{f'rna_r{i}': v for i,v in enumerate(rna_vec)}})

    df = pd.DataFrame(rows)
    Dr = X_rna.shape[1]
    print(f'  Site-matched pairs: {len(df)} from {df["patient_id"].nunique()} patients  '
          f'img_dim={Di}  rna_dim={Dr}')
    return df, Di, Dr


def run_site_matched_cv(name, site_df, Di, Dr, strategy='scaled', n_comp=N_COMPONENTS):
    """CV on site-matched (patient, location) pairs; fold by patient to avoid leakage."""
    img_cols = [f'img_f{i}' for i in range(Di)]
    rna_cols = [f'rna_r{i}' for i in range(Dr)]
    Xi = site_df[img_cols].values.astype(np.float32)
    Xr = site_df[rna_cols].values.astype(np.float32)
    y = site_df['label'].values
    folds = site_df['fold'].values
    ids = site_df['patient_id'].values

    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold; va = folds == fold
        if strategy == 'scaled':
            sc_i = StandardScaler().fit(Xi[tr]); sc_r = StandardScaler().fit(Xr[tr])
            Xtr = np.hstack([sc_i.transform(Xi[tr]), sc_r.transform(Xr[tr])])
            Xva = np.hstack([sc_i.transform(Xi[va]), sc_r.transform(Xr[va])])
        else:  # pca
            pca_i = PCA(n_components=n_comp, random_state=42).fit(Xi[tr])
            pca_r = PCA(n_components=n_comp, random_state=42).fit(Xr[tr])
            Xtr = np.hstack([pca_i.transform(Xi[tr]), pca_r.transform(Xr[tr])])
            Xva = np.hstack([pca_i.transform(Xi[va]), pca_r.transform(Xr[va])])
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(Xtr, y[tr])
        proba   = clf.predict_proba(Xva)
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]
        y_pred  = clf.predict(Xva)
        y_val   = y[va]
        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred, target_names=['CD','UC'], output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)
        fold_results.append(dict(
            strategy=name, fold=fold,
            n_train=int(tr.sum()), n_val=int(va.sum()),
            auc=round(auc,4), ap=round(ap,4), accuracy=round(cr['accuracy'],4),
            cd_f1=round(cr['CD']['f1-score'],4), uc_f1=round(cr['UC']['f1-score'],4),
            tn=int(cm[0,0]), fp=int(cm[0,1]), fn=int(cm[1,0]), tp=int(cm[1,1]),
        ))
        for pid, yt, yp, ys in zip(ids[va], y_val, y_pred, y_score):
            all_preds.append(dict(patient_id=pid, strategy=name, fold=fold,
                                  true_label=int(yt), pred_label=int(yp),
                                  prob_uc=round(float(ys),5)))
        print(f'    {name} fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  Acc={cr["accuracy"]:.4f}')
    return fold_results, all_preds


# ── summary helper ─────────────────────────────────────────────────────────────
def summarise(fold_results):
    aucs = [r['auc'] for r in fold_results]
    aps  = [r['ap']  for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    return dict(mean_auc=round(np.mean(aucs),4), std_auc=round(np.std(aucs),4),
                mean_ap=round(np.mean(aps),4),  std_ap=round(np.std(aps),4),
                mean_acc=round(np.mean(accs),4), std_acc=round(np.std(accs),4))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)

    print('\n=== Loading imaging embeddings (patient-level mean) ===')
    img_base = load_imaging_patient_mean(cv_patients, EMB_DIRS['prism2_base'])
    img_diag = load_imaging_patient_mean(cv_patients, EMB_DIRS['prism2_diagnostic'])

    print('\n=== Loading RNA embeddings (patient-level mean) ===')
    rna_df = load_rna_patient_mean(cv_patients)

    n_matched = len(set(img_base['patient_id']) & set(rna_df['patient_id']))
    print(f'\nPatients with both imaging (base) and RNA: {n_matched}')

    # ── Biopsy-site mapping validation ────────────────────────────────────────
    print('\n=== Biopsy-site mapping validation ===')
    slides_csv = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_slides.csv'
    slides_all = pd.read_csv(slides_csv)
    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna_meta = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna_meta = rna_meta[
        rna_meta['characteristics_bio_material'].isin(COLON_LOCS) &
        (rna_meta['Sample QC'] != 'fail') &
        rna_meta['deidentified_master_patient_id'].isin(set(cv_patients['patient_id']))
    ].copy()

    loc_norm = {
        'at 20 cm': 'At 20 cm', 'At 20 cm': 'At 20 cm',
        'Cecum': 'Cecum', 'Rectum': 'Rectum',
        'Ascending Colon': 'Ascending Colon', 'Descending Colon': 'Descending Colon',
        'Sigmoid Colon': 'Sigmoid Colon', 'Transverse Colon': 'Transverse Colon',
    }
    rna_meta['location_norm'] = rna_meta['characteristics_bio_material'].map(loc_norm)
    slides_all['location_norm'] = slides_all['BIOSAMPLE_LOCATION'].map(loc_norm)

    rna_pairs  = set(zip(rna_meta['deidentified_master_patient_id'], rna_meta['location_norm']))
    img_pairs  = set(zip(slides_all['patient_id'], slides_all['location_norm']))
    both_pairs = rna_pairs & img_pairs
    print(f'  Patient-location pairs with BOTH RNA and imaging: {len(both_pairs)} '
          f'({len(set(p for p,_ in both_pairs))} patients)')
    print(f'  RNA-only pairs:     {len(rna_pairs - img_pairs)}')
    print(f'  Imaging-only pairs: {len(img_pairs - rna_pairs)}')

    all_fold_results = []
    all_preds        = []

    # ── 1. Single-modality baselines (patient-level) ───────────────────────────
    print('\n--- prism2_base (patient-mean, single-modality) ---')
    img_a_base, rna_a_base = common_patients(img_base, rna_df)
    fr, preds = run_single_modality_cv('img_base_patmean', img_a_base)
    all_fold_results += fr; all_preds += preds
    img_base_preds = preds

    print('\n--- prism2_diagnostic (patient-mean, single-modality) ---')
    img_a_diag, rna_a_diag = common_patients(img_diag, rna_df)
    fr, preds = run_single_modality_cv('img_diag_patmean', img_a_diag)
    all_fold_results += fr; all_preds += preds

    print('\n--- RNA VST (patient-mean, single-modality) ---')
    _, rna_aligned = common_patients(img_base, rna_df)
    fr, preds = run_single_modality_cv('rna_patmean', rna_aligned)
    all_fold_results += fr; all_preds += preds
    rna_preds = preds

    # ── 2. Late fusion ─────────────────────────────────────────────────────────
    print('\n--- Late fusion: avg P(UC) from img_base + RNA ---')
    fr, preds = run_late_fusion_cv('late_fusion_base_rna',
                                   img_base_preds, rna_preds, img_base, rna_df)
    all_fold_results += fr; all_preds += preds

    # ── 3. Concat + scaled ─────────────────────────────────────────────────────
    print('\n--- Concat + StandardScaler (img_base + RNA) ---')
    fr, preds = run_concat_scaled_cv('concat_scaled_base_rna', img_base, rna_df)
    all_fold_results += fr; all_preds += preds

    # ── 4. Concat + PCA-128 each ───────────────────────────────────────────────
    print(f'\n--- Concat + PCA-{N_COMPONENTS} each (img_base + RNA) ---')
    fr, preds = run_concat_pca_cv(f'concat_pca{N_COMPONENTS}_base_rna',
                                  img_base, rna_df, N_COMPONENTS)
    all_fold_results += fr; all_preds += preds

    print(f'\n--- Concat + PCA-{N_COMPONENTS} each (img_diag + RNA) ---')
    fr, preds = run_concat_pca_cv(f'concat_pca{N_COMPONENTS}_diag_rna',
                                  img_diag, rna_df, N_COMPONENTS)
    all_fold_results += fr; all_preds += preds

    # ── 5. Site-matched fusion ─────────────────────────────────────────────────
    print('\n--- Loading site-matched (patient × location) pairs ---')
    site_df_base, Di_base, Dr = load_site_matched_pairs(cv_patients, EMB_DIRS['prism2_base'])

    print('\n--- Site-matched Concat + Scaled (img_base + RNA) ---')
    fr, preds = run_site_matched_cv('site_matched_scaled_base', site_df_base, Di_base, Dr,
                                    strategy='scaled')
    all_fold_results += fr; all_preds += preds

    print(f'\n--- Site-matched Concat + PCA-{N_COMPONENTS} (img_base + RNA) ---')
    fr, preds = run_site_matched_cv(f'site_matched_pca{N_COMPONENTS}_base', site_df_base,
                                    Di_base, Dr, strategy='pca', n_comp=N_COMPONENTS)
    all_fold_results += fr; all_preds += preds

    # ── Save ───────────────────────────────────────────────────────────────────
    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'multimodal_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'multimodal_patient_predictions.csv'), index=False)

    strategies = pd.DataFrame(all_fold_results)['strategy'].unique().tolist()
    summary_list = []
    for strat in strategies:
        fr_s = [r for r in all_fold_results if r['strategy'] == strat]
        s = summarise(fr_s)
        s['strategy'] = strat
        summary_list.append(s)
    with open(os.path.join(OUT_DIR, 'multimodal_summary.json'), 'w') as f:
        json.dump(summary_list, f, indent=2)

    # ── Print comparison table ─────────────────────────────────────────────────
    print('\n\n=== MULTIMODAL FUSION COMPARISON (matched cohort, patient-level) ===')
    print(f"{'Strategy':<38} {'AUC':<20} {'AP':<20} {'Accuracy'}")
    print('-' * 95)
    for s in summary_list:
        print(f"{s['strategy']:<38} "
              f"{s['mean_auc']:.4f} ± {s['std_auc']:.4f}   "
              f"{s['mean_ap']:.4f} ± {s['std_ap']:.4f}   "
              f"{s['mean_acc']:.4f} ± {s['std_acc']:.4f}")
    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
