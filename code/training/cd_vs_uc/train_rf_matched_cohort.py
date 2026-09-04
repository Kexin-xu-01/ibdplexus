"""
Single-modality RF — matched cohort: same 605 patients, same splits.

Runs all four single-modality arms on exactly the patients that have Olink
proteomics data, so every arm uses an identical patient set and fold assignment.
This ensures the single-modality comparison in single_modality_comparison.py
is cohort-fair.

Cohort
------
  605 patients (subset of cv_splits_patients_at20cm_matched.csv) that have
  Olink plasma proteomics samples.  All four modalities are available for all 605.

Arms
----
  clinical_matched     — 15 pre-diagnosis clinical features (patient-level)
  rna_matched          — VST RNA-seq at 20 cm (visit-level, 17,963 genes)
  imaging_matched      — prism2_base embeddings at 20 cm (visit-level, 2,560-d)
  proteomics_matched   — Olink NPX plasma (visit-level, 2,938 proteins)

Normalisation
-------------
  RNA / imaging / proteomics: per-fold StandardScaler (fit on train only).
  Clinical:                   per-fold median imputation (fit on train only).

Outputs
-------
  at20cm_proteomics/results/matched_cohort_fold_metrics.csv
  at20cm_proteomics/results/matched_cohort_summary.json
"""

import os, re, json, warnings
import numpy as np
import pandas as pd
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────────
TDIR         = '/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics'
VST_GCT      = f'{TDIR}/GSF1491805_CombatSeq_vst_mtx_batch_corrected_alltissues_all3releases_header.gct'
RNA_MAP      = f'{TDIR}/ibd_21183_omics_patient_mapping_genestack.csv'
SAMPLE_META  = f'{TDIR}/GSF1478941_sample_combined_from1stRun.tsv__metadata.csv'
WSI_META     = '/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/IBD_meta_data_latest/wsi_metadata_raw.csv'
EMB_BASE     = ('/home/jovyan/kgbk271-ibd-volume/data/processed/'
                'tissue_threshold_15_filtered_no_darkspot_manual_knn/'
                '20x_224px_0px_overlap/prism2_base')
PROT_GCT     = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/'
                'proteomics/GSF1618983_NPX_below_lod_included.gct')
PROT_MAP     = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/'
                'proteomics/ibd_21183_omics_patient_mapping_proteomics_genestack.csv')
METADATA_CSV = '/home/jovyan/kgbk271-ibd-volume/metadata/merged_patient_metadata.csv'
CV_PATIENTS  = ('/home/jovyan/kgbk271-ibd-volume/training/'
                'cv_splits_patients_at20cm_matched.csv')
OUT_DIR      = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                'at20cm_proteomics/results')

AT20      = {'at 20 cm', 'At 20 cm'}
CDUC_RAW  = ["Crohn's disease", 'Ulcerative colitis']
RF_PARAMS = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)

# clinical features (same as 01_train_clinical.py)
NUMERIC_COLS  = ['rectal_bleed', 'first_abdopain', 'first_daily_bm', 'first_well_being']
BINARY_COLS   = ['cardiovascular_disease', 'hypertension', 'diabetes_type_i',
                 'diabetes_type_ii', 'ckd', 'hiv', 'tuberculosis']
STOOL_MAP     = {'Normal': 0, '1-2 stools more than normal': 1,
                 '3-4 stools more than normal': 2, '>4 stools more than normal': 3}
URGENCY_MAP   = {'Mild': 1, 'Moderate': 2, 'Moderately severe': 3, 'Severe': 4}
DA6M_MAP      = {'1. Absence of symptoms': 0, '2. Rarely active': 1,
                 '3. Sometimes active': 2, '4. Occasionally active': 3,
                 '5. Often active': 4, '6. Constantly active': 5}
FEATURE_NAMES = NUMERIC_COLS + ['stool_freq', 'fecal_urgency', 'da6m', 'smk'] + BINARY_COLS


# ── cohort: 605 patients with proteomics ──────────────────────────────────────

def get_prot_pids(splits):
    pmap = pd.read_csv(PROT_MAP)
    return set(pmap['deidentified_master_patient_id']) & set(splits['patient_id'])


# ── loaders ───────────────────────────────────────────────────────────────────

def load_clinical(splits, pids):
    cv = splits[splits['patient_id'].isin(pids)].copy()
    meta = pd.read_csv(METADATA_CSV)
    df = cv.merge(meta, left_on='patient_id',
                  right_on='deidentified_master_patient_id',
                  how='left', suffixes=('', '_meta'))
    df['label'] = (df['diagnosis'] == 'Ulcerative colitis').astype(int)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['stool_freq']    = df['stool_freq'].map(STOOL_MAP)
    df['fecal_urgency'] = df['fecal_urgency'].map(URGENCY_MAP)
    df['da6m']          = df['da6m'].map(DA6M_MAP)
    df['smk']           = df['smk'].map({'Yes': 1, 'No': 0})
    for c in BINARY_COLS:
        df[c] = df[c].map({'Y': 1, 'N': 0})
    print(f'  clinical: {len(df)} patients  '
          f"(CD {(df['label']==0).sum()}, UC {(df['label']==1).sum()})")
    return df[['patient_id', 'label', 'fold'] + FEATURE_NAMES].copy()


def load_rna(splits, pids):
    rna_map     = pd.read_csv(RNA_MAP)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = rna_map.merge(sample_meta[['SampleID', 'Sample QC']], on='SampleID', how='left')
    rna = rna[rna['characteristics_bio_material'].isin(AT20)
              & (rna['Sample QC'] != 'fail')
              & rna['deidentified_master_patient_id'].isin(pids)
              ].drop_duplicates('visit_encounter_id', keep='first')
    sample_ids = set(rna['SampleID'])
    print(f'  RNA: loading {len(sample_ids)} samples from VST GCT ...')

    with open(VST_GCT) as f:
        f.readline(); f.readline()
        hdr = f.readline().strip().split('\t')
    keep = ['Name'] + [c for c in hdr if c in sample_ids]
    gct = pd.read_csv(VST_GCT, sep='\t', skiprows=2, header=0,
                      usecols=keep, index_col=0,
                      dtype={c: (str if c == 'Name' else np.float32) for c in keep})
    X_df = gct.T.astype(np.float32).reset_index().rename(columns={'index': 'SampleID'})
    gene_names = gct.index.tolist()

    rna['SampleID'] = rna['SampleID'].astype(str)
    merged = X_df.merge(
        rna[['SampleID', 'deidentified_master_patient_id']],
        on='SampleID', how='inner'
    ).merge(
        splits[['patient_id', 'label' if 'label' in splits.columns else 'diagnosis', 'fold']],
        left_on='deidentified_master_patient_id', right_on='patient_id', how='inner'
    )
    if 'label' not in merged.columns:
        merged['label'] = (merged['diagnosis'] == 'Ulcerative colitis').astype(int)
    print(f'  RNA: {len(merged)} visits / {merged["patient_id"].nunique()} patients  '
          f"dim={len(gene_names)}")
    return merged, gene_names


def load_imaging(splits, pids):
    wsi = pd.read_csv(WSI_META)
    wsi = wsi[wsi['BIOSAMPLE_LOCATION'].isin(AT20)
              & wsi['diagnosis'].isin(CDUC_RAW)
              & wsi['deidentified_master_patient_id'].isin(pids)].copy()
    wsi['date']     = pd.to_datetime(wsi['Date Sample Collected'], dayfirst=True, errors='coerce')
    wsi['slide_id'] = wsi['IMAGE_VSI'].str.replace('.vsi', '', regex=False)
    wsi['visit_key'] = list(zip(wsi['deidentified_master_patient_id'],
                                wsi['date'].dt.date.astype(str)))

    cv = splits[splits['patient_id'].isin(pids)].set_index('patient_id')
    rows = []
    for (pid, vdate), grp in wsi.groupby('visit_key'):
        if pid not in cv.index:
            continue
        vecs = []
        for sid in grp['slide_id']:
            h5p = os.path.join(EMB_BASE, f'{sid}.h5')
            if os.path.exists(h5p):
                with h5py.File(h5p, 'r') as h:
                    vecs.append(h['features'][:])
        if not vecs:
            continue
        vec = np.mean(vecs, axis=0)
        row_cv = cv.loc[pid]
        label  = int(row_cv['diagnosis'] == 'Ulcerative colitis')
        rows.append({'patient_id': pid, 'visit_key': str((pid, vdate)),
                     'label': label, 'fold': int(row_cv['fold']),
                     **{f'emb{i}': float(v) for i, v in enumerate(vec)}})
    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c.startswith('emb')]
    print(f'  imaging: {len(df)} visits / {df["patient_id"].nunique()} patients  '
          f'dim={len(feat_cols)}')
    return df, feat_cols


def load_proteomics(splits, pids):
    pmap = pd.read_csv(PROT_MAP)
    pmap = pmap[pmap['deidentified_master_patient_id'].isin(pids)].copy()
    sample_ids = set(pmap['SampleID'].astype(str))
    with open(PROT_GCT) as f:
        f.readline(); f.readline()
        hdr = f.readline().strip().split('\t')
    keep = ['NAME'] + [c for c in hdr if c in sample_ids]
    gct = pd.read_csv(PROT_GCT, sep='\t', skiprows=2, header=0, usecols=keep,
                      dtype={c: (str if c == 'NAME' else np.float32) for c in keep}
                      ).set_index('NAME')
    protein_names = gct.index.tolist()
    X_df = gct.T.astype(np.float32).reset_index()
    X_df.columns = ['SampleID'] + protein_names
    X_df['SampleID'] = X_df['SampleID'].astype(str)
    pmap['SampleID'] = pmap['SampleID'].astype(str)
    merged = X_df.merge(
        pmap[['SampleID', 'deidentified_master_patient_id']], on='SampleID', how='inner'
    ).merge(
        splits[['patient_id', 'diagnosis', 'fold']],
        left_on='deidentified_master_patient_id', right_on='patient_id', how='inner'
    )
    merged['label'] = (merged['diagnosis'] == 'Ulcerative colitis').astype(int)
    print(f'  proteomics: {len(merged)} visits / {merged["patient_id"].nunique()} patients  '
          f'dim={len(protein_names)}')
    return merged, protein_names


# ── CV runner ─────────────────────────────────────────────────────────────────

def run_rf(name, df, feat_cols, normalise='scale'):
    """normalise: 'scale' (StandardScaler) or 'impute' (median only, for clinical)."""
    X_raw = df[feat_cols].values.astype(np.float32 if normalise == 'scale' else float)
    y     = df['label'].values
    folds = df['fold'].values
    pids  = df['patient_id'].values

    fold_results = []
    for fold in range(5):
        tr = folds != fold;  va = folds == fold

        X_tr_raw = X_raw[tr].copy();  X_va_raw = X_raw[va].copy()

        if normalise == 'scale':
            # median impute then standardise — both fit on train only
            med = np.nanmedian(X_tr_raw, axis=0)
            for j in range(X_tr_raw.shape[1]):
                X_tr_raw[np.isnan(X_tr_raw[:, j]), j] = med[j]
                X_va_raw[np.isnan(X_va_raw[:, j]), j] = med[j]
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr_raw)
            X_va = sc.transform(X_va_raw)
        else:
            imp  = SimpleImputer(strategy='median')
            X_tr = imp.fit_transform(X_tr_raw)
            X_va = imp.transform(X_va_raw)

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y[tr])
        proba   = clf.predict_proba(X_va)
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]
        y_pred  = clf.predict(X_va)
        y_val   = y[va]

        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred,
                                    target_names=['CD', 'UC'], output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)
        fold_results.append(dict(
            strategy=name, fold=fold,
            n_train=int(tr.sum()), n_val=int(va.sum()),
            n_val_patients=int(pd.Series(pids[va]).nunique()),
            auc=round(auc, 4), ap=round(ap, 4),
            accuracy=round(cr['accuracy'], 4),
            cd_f1=round(cr['CD']['f1-score'], 4),
            uc_f1=round(cr['UC']['f1-score'], 4),
            tn=int(cm[0, 0]), fp=int(cm[0, 1]),
            fn=int(cm[1, 0]), tp=int(cm[1, 1]),
        ))
        print(f'    {name}  fold {fold}: AUC={auc:.4f}  '
              f"F1-CD={cr['CD']['f1-score']:.4f}  F1-UC={cr['UC']['f1-score']:.4f}  "
              f'n_val={va.sum()}', flush=True)
    return fold_results


def summarise(fold_results):
    aucs  = [r['auc']  for r in fold_results]
    f1cds = [r['cd_f1'] for r in fold_results]
    f1ucs = [r['uc_f1'] for r in fold_results]
    accs  = [r['accuracy'] for r in fold_results]
    return dict(
        mean_auc=round(np.mean(aucs), 4),   std_auc=round(np.std(aucs, ddof=1), 4),
        mean_f1cd=round(np.mean(f1cds), 4), std_f1cd=round(np.std(f1cds, ddof=1), 4),
        mean_f1uc=round(np.mean(f1ucs), 4), std_f1uc=round(np.std(f1ucs, ddof=1), 4),
        mean_acc=round(np.mean(accs), 4),   std_acc=round(np.std(accs, ddof=1), 4),
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    splits = pd.read_csv(CV_PATIENTS)
    pids   = get_prot_pids(splits)
    # add label column to splits for convenience
    splits['label'] = (splits['diagnosis'] == 'Ulcerative colitis').astype(int)
    print(f'Matched cohort: {len(pids)} patients\n')

    all_fold_results = []
    summary_list     = []

    # ── Clinical ───────────────────────────────────────────────────────────────
    print('=== Clinical ===')
    clin_df = load_clinical(splits, pids)
    fr = run_rf('clinical_matched', clin_df, FEATURE_NAMES, normalise='impute')
    all_fold_results += fr
    s = summarise(fr); s['strategy'] = 'clinical_matched'
    s['n_obs'] = len(clin_df); s['n_patients'] = clin_df['patient_id'].nunique()
    s['note']  = 'patient-level'
    summary_list.append(s)

    # ── RNA ────────────────────────────────────────────────────────────────────
    print('\n=== RNA-seq (VST) ===')
    rna_df, gene_names = load_rna(splits, pids)
    fr = run_rf('rna_matched', rna_df, gene_names, normalise='scale')
    all_fold_results += fr
    s = summarise(fr); s['strategy'] = 'rna_matched'
    s['n_obs'] = len(rna_df); s['n_patients'] = rna_df['patient_id'].nunique()
    s['note']  = 'visit-level'
    summary_list.append(s)

    # ── Imaging ────────────────────────────────────────────────────────────────
    print('\n=== Imaging (prism2_base) ===')
    img_df, feat_cols = load_imaging(splits, pids)
    fr = run_rf('imaging_matched', img_df, feat_cols, normalise='scale')
    all_fold_results += fr
    s = summarise(fr); s['strategy'] = 'imaging_matched'
    s['n_obs'] = len(img_df); s['n_patients'] = img_df['patient_id'].nunique()
    s['note']  = 'visit-level'
    summary_list.append(s)

    # ── Proteomics ─────────────────────────────────────────────────────────────
    print('\n=== Proteomics (Olink) ===')
    prot_df, prot_cols = load_proteomics(splits, pids)
    fr = run_rf('proteomics_matched', prot_df, prot_cols, normalise='scale')
    all_fold_results += fr
    s = summarise(fr); s['strategy'] = 'proteomics_matched'
    s['n_obs'] = len(prot_df); s['n_patients'] = prot_df['patient_id'].nunique()
    s['note']  = 'visit-level'
    summary_list.append(s)

    # ── Save ───────────────────────────────────────────────────────────────────
    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'matched_cohort_fold_metrics.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'matched_cohort_summary.json'), 'w') as f:
        json.dump(summary_list, f, indent=2)

    print('\n\n=== MATCHED-COHORT SUMMARY (605 patients) ===')
    print(f"{'Strategy':<24} {'AUC':>20} {'F1-CD':>20} {'F1-UC':>20}")
    print('-' * 86)
    for s in sorted(summary_list, key=lambda x: -x['mean_auc']):
        print(f"  {s['strategy']:<22}  "
              f"AUC={s['mean_auc']:.4f}±{s['std_auc']:.4f}  "
              f"F1-CD={s['mean_f1cd']:.4f}±{s['std_f1cd']:.4f}  "
              f"F1-UC={s['mean_f1uc']:.4f}±{s['std_f1uc']:.4f}")
    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
