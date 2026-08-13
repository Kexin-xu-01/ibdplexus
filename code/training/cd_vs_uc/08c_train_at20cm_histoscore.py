"""
CD vs UC classifiers using prism2 histological scores — visit-level cohort.

Replaces the dense prism2_base (2,560-d) embedding with 11 interpretable
histological probability scores per slide (output of the PRISM2 diagnostic head):

  inflammation_involvement
  crypt_architectural_distortion
  neutrophil_granulocytic_infiltration
  crypt_abscesses
  lymphoid_aggregates
  histiocytic_granulomas
  mucin_depletion
  pyloric_gland_metaplasia
  paneth_cell_metaplasia
  neuronal_hyperplasia
  muscular_hypertrophy

Within-visit aggregation: mean across slides from the same (patient, date).
Patient-level CV split identical to 08b; all arms restricted to the same
matched visit cohort for fair modality comparison.

Arms
----
  img_histoscore_visit   — 11 histological scores, visit-level
  rna_visit              — RNA VST (17,963 genes), same matched cohort
  concat_histoscore_visit — 11 scores + 17,963 genes (concat)

Outputs (under OUT_DIR)
-----------------------
  at20cm_histoscore_fold_metrics.csv
  at20cm_histoscore_predictions.csv
  at20cm_histoscore_summary.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
TRANSCRIPTOMICS_DIR = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/'
                       'genestack/transcriptomics')
WSI_META      = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/'
                 'IBD_meta_data_latest/wsi_metadata_raw.csv')
VST_GCT       = os.path.join(TRANSCRIPTOMICS_DIR,
                'GSF1491805_CombatSeq_vst_mtx_batch_corrected_'
                'alltissues_all3releases_header.gct')
MAPPING_CSV   = os.path.join(TRANSCRIPTOMICS_DIR,
                'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META   = os.path.join(TRANSCRIPTOMICS_DIR,
                'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS   = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
HISTOSCORE_CSV = '/home/jovyan/kgbk271-ibd-volume/results/prism2/prism2_histological_score.csv'
OUT_DIR       = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                 'concept_learning/results')

AT20       = {'at 20 cm', 'At 20 cm'}
CDUC_RAW   = ["Crohn's disease", 'Ulcerative colitis']
CDUC_OMICS = ["Crohn's Disease", "Crohn's disease",
              'Ulcerative colitis', 'Ulcerative Colitis']
RF_PARAMS  = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
                  class_weight='balanced', n_jobs=-1, random_state=42)

HISTO_COLS = [
    'inflammation_involvement',
    'crypt_architectural_distortion',
    'neutrophil_granulocytic_infiltration',
    'crypt_abscesses',
    'lymphoid_aggregates',
    'histiocytic_granulomas',
    'mucin_depletion',
    'pyloric_gland_metaplasia',
    'paneth_cell_metaplasia',
    'neuronal_hyperplasia',
    'muscular_hypertrophy',
]


def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


# ── loaders ───────────────────────────────────────────────────────────────────

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
    print(f'  histoscore visits: {len(df)} visits from {df["patient_id"].nunique()} patients'
          f"  (CD {(df['label']==0).sum()}, UC {(df['label']==1).sum()})  dim={len(HISTO_COLS)}")
    return df


def load_rna_visits(cv_patients):
    """Identical to 08b — one row per RNA encounter."""
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
            'visit_key':        str((pid, vdate)),
            'patient_id':       pid,
            'visit_date':       str(vdate),
            'visit_encounter_id': enc_row['visit_encounter_id'],
            'label': int(row['diagnosis'] == 'Ulcerative colitis'),
            'fold':  int(row['fold']),
            **{f'r{i}': v for i, v in enumerate(vec)},
        })

    df = pd.DataFrame(rows)
    dim = len([c for c in df.columns if c.startswith('r')])
    print(f'  RNA visits: {len(df)} visits from {df["patient_id"].nunique()} patients'
          f"  (CD {(df['label']==0).sum()}, UC {(df['label']==1).sum()})  dim={dim}")
    return df


def common_visits(img_df, rna_df, max_gap_days=7):
    """Proximity join on (patient_id, date) — identical to 08b."""
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


def feat_img(df):
    return df[HISTO_COLS].values.astype(np.float32)


def feat_rna(df):
    cols = [c for c in df.columns if c.startswith('r')]
    return df[cols].values.astype(np.float32)


# ── CV runners ────────────────────────────────────────────────────────────────

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


def run_concat(name, img_m, rna_m):
    """img_m and rna_m must already be proximity-aligned."""
    Xi    = feat_img(img_m)
    Xr    = feat_rna(rna_m)
    X     = np.hstack([Xi, Xr])
    y     = img_m['label'].values
    folds = img_m['fold'].values
    vkeys = img_m['visit_key'].values
    pids  = img_m['patient_id'].values

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

    print('=== Loading At-20-cm visit-level data ===')
    img_df = load_histoscore_visits(cv_patients)
    rna_df = load_rna_visits(cv_patients)

    img_m, rna_m = common_visits(img_df, rna_df)
    n_vis = len(img_m); n_pat = img_m['patient_id'].nunique()
    print(f'Matched cohort: {n_vis} visits from {n_pat} patients  '
          f"(CD {(img_m['label']==0).sum()}, UC {(img_m['label']==1).sum()})")

    all_fold_results, all_preds = [], []

    print('\n--- img_histoscore_visit ---')
    X_img = feat_img(img_m)
    fr, preds = run_unimodal('img_histoscore_visit', img_m, X_img)
    all_fold_results += fr; all_preds += preds

    print('\n--- rna_visit (matched cohort) ---')
    X_rna = feat_rna(rna_m)
    fr, preds = run_unimodal('rna_visit', rna_m, X_rna)
    all_fold_results += fr; all_preds += preds

    print('\n--- concat_histoscore_visit ---')
    fr, preds = run_concat('concat_histoscore_visit', img_m, rna_m)
    all_fold_results += fr; all_preds += preds

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'at20cm_histoscore_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'at20cm_histoscore_predictions.csv'), index=False)

    strategies = list(dict.fromkeys(r['strategy'] for r in all_fold_results))
    summary_list = []
    for strat in strategies:
        fr_s = [r for r in all_fold_results if r['strategy'] == strat]
        s = summarise(fr_s)
        s['strategy'] = strat
        s['n_visits']   = n_vis
        s['n_patients'] = n_pat
        summary_list.append(s)

    with open(os.path.join(OUT_DIR, 'at20cm_histoscore_summary.json'), 'w') as f:
        json.dump(summary_list, f, indent=2)

    print('\n\n=== HISTOSCORE SUMMARY ===')
    print(f"{'Strategy':<30} {'Visits':<8} {'Patients':<10} "
          f"{'AUC':<22} {'AP':<22} {'Accuracy'}")
    print('-' * 100)
    for s in summary_list:
        print(f"  {s['strategy']:<28} {s['n_visits']:<8} {s['n_patients']:<10} "
              f"AUC={s['mean_auc']:.4f}±{s['std_auc']:.4f}  "
              f"AP={s['mean_ap']:.4f}±{s['std_ap']:.4f}  "
              f"Acc={s['mean_acc']:.4f}")
    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
