"""
CD vs UC classifiers using BulkFormer transcriptomic embeddings — visit-level cohort.

BulkFormer is a transformer-based encoder trained on bulk RNA-seq data.
Pre-computed 640-d embeddings derived from the CombatSeq log-TPM matrix
(same underlying data as the VST-based RNA arm).

The cohort is identical to 08c: proximity-matched histoscore imaging visits + RNA
visits (≤7 days, at-20-cm, 945 visits / 817 patients). BulkFormer embeddings are
joined to the matched RNA visits by SampleID. Visits missing a BulkFormer embedding
are dropped (expected to be ≤1 sample).

Arms
----
  bulkformer_visit      — 640-d BulkFormer embeddings
  rna_visit             — RNA VST (17,963 genes), same matched cohort
  concat_bf_histo_visit — BulkFormer + histological scores (640 + 11)
  concat_bf_base_visit  — BulkFormer + prism2_base (640 + 2,560)

Outputs (under OUT_DIR)
-----------------------
  at20cm_bulkformer_fold_metrics.csv
  at20cm_bulkformer_predictions.csv
  at20cm_bulkformer_summary.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import h5py
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
TRANSCRIPTOMICS_DIR = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/'
                       'genestack/transcriptomics')
WSI_META       = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/'
                  'IBD_meta_data_latest/wsi_metadata_raw.csv')
HISTOSCORE_CSV = '/home/jovyan/kgbk271-ibd-volume/results/prism2/prism2_histological_score.csv'
BULKFORMER_PT  = ('/home/jovyan/kgbk271-ibd-volume/data/processed/'
                  'bulk_rna_seq/bulkformer/transcriptomics_embeddings.pt')
LOG_TPM_CSV    = os.path.join(TRANSCRIPTOMICS_DIR,
                 'GSF1491803_CombatSeq_count_mtx_batch_corrected_'
                 'alltissues_all3releases_header_log_tpm_with_sampleID.csv')
VST_GCT        = os.path.join(TRANSCRIPTOMICS_DIR,
                 'GSF1491805_CombatSeq_vst_mtx_batch_corrected_'
                 'alltissues_all3releases_header.gct')
MAPPING_CSV    = os.path.join(TRANSCRIPTOMICS_DIR,
                 'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META    = os.path.join(TRANSCRIPTOMICS_DIR,
                 'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS    = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
EMB_BASE       = ('/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
                  '20x_224px_0px_overlap/prism2_base')
OUT_DIR        = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                  'concept_learning/results')

AT20       = {'at 20 cm', 'At 20 cm'}
CDUC_RAW   = ["Crohn's disease", 'Ulcerative colitis']
RF_PARAMS  = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
                  class_weight='balanced', n_jobs=-1, random_state=42)
EMB_DIM    = 640
EMB_COLS   = [f'e{i}' for i in range(EMB_DIM)]

HISTO_COLS = [
    'inflammation_involvement', 'crypt_architectural_distortion',
    'neutrophil_granulocytic_infiltration', 'crypt_abscesses',
    'lymphoid_aggregates', 'histiocytic_granulomas', 'mucin_depletion',
    'pyloric_gland_metaplasia', 'paneth_cell_metaplasia',
    'neuronal_hyperplasia', 'muscular_hypertrophy',
]


def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


# ── loaders (identical to 08c except load_rna_visits stores SampleID) ─────────

def load_histoscore_visits(cv_patients):
    """One row per (patient, visit-date). Slides from the same date are mean-pooled."""
    pat_df  = cv_patients.set_index('patient_id')
    cv_pids = set(pat_df.index)

    wsi = pd.read_csv(WSI_META)
    wsi = wsi[
        wsi['BIOSAMPLE_LOCATION'].isin(AT20) &
        wsi['diagnosis'].isin(CDUC_RAW) &
        wsi['deidentified_master_patient_id'].isin(cv_pids)
    ].copy()
    wsi['date']     = pd.to_datetime(wsi['Date Sample Collected'],
                                     dayfirst=True, errors='coerce')
    wsi['slide_id'] = wsi['IMAGE_VSI'].str.replace('.vsi', '', regex=False)
    wsi['visit_key'] = list(zip(wsi['deidentified_master_patient_id'],
                                wsi['date'].dt.date))

    scores = pd.read_csv(HISTOSCORE_CSV).set_index('slide')

    rows = []
    for (pid, vdate), grp in wsi.groupby('visit_key'):
        if pid not in pat_df.index:
            continue
        vecs = []
        for sid in grp['slide_id']:
            if sid in scores.index:
                vecs.append(scores.loc[sid, HISTO_COLS].values.astype(np.float32))
        if not vecs:
            continue
        vec = np.mean(vecs, axis=0)
        row = pat_df.loc[pid]
        rows.append({
            'visit_key':  str((pid, vdate)),
            'patient_id': pid,
            'visit_date': str(vdate),
            'label': int(row['diagnosis'] == 'Ulcerative colitis'),
            'fold':  int(row['fold']),
            **{c: float(v) for c, v in zip(HISTO_COLS, vec)},
        })

    df = pd.DataFrame(rows)
    print(f'  histoscore visits: {len(df)} visits / {df["patient_id"].nunique()} patients'
          f"  (CD {(df['label']==0).sum()}, UC {(df['label']==1).sum()})")
    return df


def load_rna_visits(cv_patients):
    """One row per RNA encounter. Stores SampleID for later BulkFormer join."""
    pat_df  = cv_patients.set_index('patient_id')
    cv_pids = set(pat_df.index)

    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})

    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(norm_dx)
    rna['date'] = pd.to_datetime(rna['sample_collected_date'], dayfirst=True, errors='coerce')
    rna['visit_key'] = list(zip(rna['deidentified_master_patient_id'],
                                rna['date'].dt.date))

    rna = rna[
        rna['characteristics_bio_material'].isin(AT20) &
        rna['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pids)
    ].copy()
    rna = rna.drop_duplicates(subset='visit_encounter_id', keep='first')

    sample_ids = set(rna['SampleID'])
    print(f'  Loading GCT for {len(sample_ids)} At-20-cm RNA visits ...')
    with open(VST_GCT) as f:
        f.readline(); f.readline()
        header_cols = f.readline().strip().split('\t')
    keep       = [0, 1] + [i for i, c in enumerate(header_cols) if c in sample_ids]
    keep_names = [header_cols[i] for i in keep]
    gct_df = pd.read_csv(VST_GCT, sep='\t', skiprows=2, header=0,
                         usecols=keep_names, index_col=0,
                         dtype={c: (str if c in ('Name', 'Description') else np.float32)
                                for c in keep_names})
    gct_df = gct_df.drop(columns=['Description'])
    X_df = gct_df.T.astype(np.float32)

    rows = []
    for _, enc_row in rna.iterrows():
        sid = enc_row['SampleID']
        pid = enc_row['deidentified_master_patient_id']
        if sid not in X_df.index or pid not in pat_df.index:
            continue
        vec  = X_df.loc[sid].values
        row  = pat_df.loc[pid]
        vdate = enc_row['date'].date() if not pd.isna(enc_row['date']) else None
        rows.append({
            'visit_key':          str((pid, vdate)),
            'patient_id':         pid,
            'visit_date':         str(vdate),
            'visit_encounter_id': enc_row['visit_encounter_id'],
            'SampleID':           sid,
            'label': int(row['diagnosis'] == 'Ulcerative colitis'),
            'fold':  int(row['fold']),
            **{f'r{i}': v for i, v in enumerate(vec)},
        })

    df = pd.DataFrame(rows)
    dim = len([c for c in df.columns if c.startswith('r')])
    print(f'  RNA visits: {len(df)} visits / {df["patient_id"].nunique()} patients'
          f"  (CD {(df['label']==0).sum()}, UC {(df['label']==1).sum()})  dim={dim}")
    return df


def common_visits(img_df, rna_df, max_gap_days=7):
    """Proximity join on (patient_id, date) — identical to 08b/08c."""
    img_df = img_df.copy(); rna_df = rna_df.copy()
    img_df['_date'] = pd.to_datetime(img_df['visit_date'])
    rna_df['_date'] = pd.to_datetime(rna_df['visit_date'])

    img_rows, rna_rows = [], []
    rna_by_pid = {pid: grp for pid, grp in rna_df.groupby('patient_id')}

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
    assert (img_out['fold']       == rna_out['fold']).all()

    n_prox = (img_out['visit_key'] != rna_out['visit_key']).sum()
    if n_prox:
        print(f'  proximity-matched {n_prox} visit(s) with date offset ≤ {max_gap_days}d')
    return img_out, rna_out


def load_img_base_for_matched(img_m):
    """
    Load prism2_base (2,560-d) H5 embeddings for visits already in img_m.
    Slides from the same visit are mean-pooled. Returns DataFrame with
    visit_key + f0..f2559, aligned to img_m row order where H5 files exist.
    """
    wsi = pd.read_csv(WSI_META)
    wsi['visit_date'] = pd.to_datetime(
        wsi['Date Sample Collected'], dayfirst=True, errors='coerce').dt.date.astype(str)
    wsi['slide_id']   = wsi['IMAGE_VSI'].str.replace('.vsi', '', regex=False)
    wsi['patient_id'] = wsi['deidentified_master_patient_id']

    # build lookup: (patient_id, visit_date) → [slide_ids]
    visit_slides = {}
    for _, row in wsi.iterrows():
        key = (row['patient_id'], row['visit_date'])
        visit_slides.setdefault(key, []).append(row['slide_id'])

    rows = []
    for _, irow in img_m.iterrows():
        key   = (irow['patient_id'], irow['visit_date'])
        vecs  = []
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
    """Load BulkFormer .pt and assign SampleIDs from the aligned log-TPM CSV."""
    print('  Loading BulkFormer embeddings ...')
    emb = torch.load(BULKFORMER_PT, map_location='cpu').numpy().astype(np.float32)
    log_tpm = pd.read_csv(LOG_TPM_CSV, usecols=['SampleID'])
    assert len(emb) == len(log_tpm), (
        f'Embedding rows ({len(emb)}) != log_tpm rows ({len(log_tpm)})')
    df = pd.DataFrame(emb, columns=EMB_COLS)
    df['SampleID'] = log_tpm['SampleID'].values
    print(f'  Embedding matrix: {emb.shape[0]} samples × {emb.shape[1]} dims')
    return df


def join_bulkformer(rna_m, emb_df):
    """
    Inner-join BulkFormer embeddings to the matched RNA visits by SampleID.
    Returns (bf_df, mask) where mask is the boolean index into rna_m of kept rows.
    """
    rna_indexed = rna_m.reset_index(drop=True)
    merged = rna_indexed[['SampleID']].merge(
        emb_df[['SampleID'] + EMB_COLS], on='SampleID', how='left')
    has_emb = merged[EMB_COLS[0]].notna()
    n_dropped = (~has_emb).sum()
    if n_dropped:
        print(f'  Dropped {n_dropped} visit(s) with no BulkFormer embedding')
    bf_df = pd.concat([
        rna_indexed[['visit_key', 'patient_id', 'visit_date',
                     'visit_encounter_id', 'label', 'fold']],
        merged[EMB_COLS],
    ], axis=1)[has_emb].reset_index(drop=True)
    return bf_df, has_emb


# ── feature extractors ────────────────────────────────────────────────────────

def feat_histo(df):
    return df[HISTO_COLS].values.astype(np.float32)


def feat_rna(df):
    cols = [c for c in df.columns if c.startswith('r')]
    return df[cols].values.astype(np.float32)


def feat_bf(df):
    return df[EMB_COLS].values.astype(np.float32)


def feat_base(df):
    cols = [c for c in df.columns if c.startswith('f')]
    return df[cols].values.astype(np.float32)


# ── CV runner ─────────────────────────────────────────────────────────────────

def run_unimodal(name, df, X):
    y     = df['label'].values
    folds = df['fold'].values
    vkeys = df['visit_key'].values
    pids  = df['patient_id'].values

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
        cr  = classification_report(y_val, y_pred,
                                    target_names=['CD', 'UC'], output_dict=True)
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
              f'Acc={cr["accuracy"]:.4f}  '
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


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)

    print('=== Loading At-20-cm imaging+RNA matched cohort (08c cohort) ===')
    img_df = load_histoscore_visits(cv_patients)
    rna_df = load_rna_visits(cv_patients)
    img_m, rna_m = common_visits(img_df, rna_df)

    print(f'Matched cohort (pre-BulkFormer): {len(img_m)} visits / '
          f"{img_m['patient_id'].nunique()} patients  "
          f"(CD {(img_m['label']==0).sum()}, UC {(img_m['label']==1).sum()})")

    print('\n=== Joining BulkFormer embeddings ===')
    emb_df = load_bulkformer_emb()
    bf_m, has_emb = join_bulkformer(rna_m, emb_df)

    # Restrict all modalities to the same visits
    img_m = img_m[has_emb.values].reset_index(drop=True)
    rna_m = rna_m[has_emb.values].reset_index(drop=True)

    n_vis = len(bf_m); n_pat = bf_m['patient_id'].nunique()
    print(f'Final cohort: {n_vis} visits / {n_pat} patients  '
          f"(CD {(bf_m['label']==0).sum()}, UC {(bf_m['label']==1).sum()})")

    assert len(img_m) == len(rna_m) == len(bf_m), 'Row count mismatch'
    assert (bf_m['patient_id'].values == rna_m['patient_id'].values).all()
    assert (bf_m['fold'].values == rna_m['fold'].values).all()

    # Load prism2_base for the same imaging visits
    print('\n=== Loading prism2_base embeddings ===')
    base_raw = load_img_base_for_matched(img_m)
    has_base = img_m['visit_key'].isin(set(base_raw['visit_key']))
    n_dropped = (~has_base).sum()
    if n_dropped:
        print(f'  Dropped {n_dropped} visit(s) with no prism2_base H5')
    # restrict all modalities to visits with prism2_base
    img_m  = img_m[has_base.values].reset_index(drop=True)
    rna_m  = rna_m[has_base.values].reset_index(drop=True)
    bf_m   = bf_m[has_base.values].reset_index(drop=True)
    base_m = img_m[['visit_key']].merge(base_raw, on='visit_key', how='left')

    n_vis = len(bf_m); n_pat = bf_m['patient_id'].nunique()
    print(f'Final cohort: {n_vis} visits / {n_pat} patients  '
          f"(CD {(bf_m['label']==0).sum()}, UC {(bf_m['label']==1).sum()})")

    X_bf    = feat_bf(bf_m)
    X_rna   = feat_rna(rna_m)
    X_histo = feat_histo(img_m)
    X_base  = feat_base(base_m)

    all_fold_results, all_preds = [], []

    print('\n--- bulkformer_visit ---')
    fr, preds = run_unimodal('bulkformer_visit', bf_m, X_bf)
    all_fold_results += fr; all_preds += preds

    print('\n--- rna_visit (matched cohort) ---')
    fr, preds = run_unimodal('rna_visit', rna_m, X_rna)
    all_fold_results += fr; all_preds += preds

    print('\n--- concat_bf_histo_visit ---')
    fr, preds = run_unimodal('concat_bf_histo_visit', bf_m,
                              np.hstack([X_bf, X_histo]))
    all_fold_results += fr; all_preds += preds

    print('\n--- concat_bf_base_visit ---')
    fr, preds = run_unimodal('concat_bf_base_visit', bf_m,
                              np.hstack([X_bf, X_base]))
    all_fold_results += fr; all_preds += preds

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'at20cm_bulkformer_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'at20cm_bulkformer_predictions.csv'), index=False)

    strategies = list(dict.fromkeys(r['strategy'] for r in all_fold_results))
    summary_list = []
    for strat in strategies:
        fr_s = [r for r in all_fold_results if r['strategy'] == strat]
        s = summarise(fr_s)
        s['strategy'] = strat
        s['n_visits']   = n_vis
        s['n_patients'] = n_pat
        summary_list.append(s)

    with open(os.path.join(OUT_DIR, 'at20cm_bulkformer_summary.json'), 'w') as f:
        json.dump(summary_list, f, indent=2)

    print('\n\n=== BULKFORMER SUMMARY ===')
    print(f"{'Strategy':<28} {'Visits':<8} {'Patients':<10} "
          f"{'AUC':<22} {'AP':<22} {'Accuracy'}")
    print('-' * 100)
    for s in summary_list:
        print(f"  {s['strategy']:<26} {s['n_visits']:<8} {s['n_patients']:<10} "
              f"AUC={s['mean_auc']:.4f}±{s['std_auc']:.4f}  "
              f"AP={s['mean_ap']:.4f}±{s['std_ap']:.4f}  "
              f"Acc={s['mean_acc']:.4f}")
    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()