"""
PCA-compressed RNA arms — control for BulkFormer dimensionality.

Tests whether the BulkFormer + prism2_base gain over RNA + prism2_base
is due to representation quality or simply smaller feature dimensionality.

PCA(640) compresses RNA-seq VST to the same 640-d as BulkFormer, then fuses
with prism2_base (2,560-d) to give a 3,200-d concat — identical shape to
the concat_bf_base_visit arm in 08f.

PCA is fitted on the TRAINING fold only and applied to the validation fold,
preventing leakage.

Same 945-visit / 817-patient matched cohort as 08c/08f.

Arms
----
  pca640_rna_visit      — PCA(640) of RNA VST only
  concat_pca640_base    — PCA(640) RNA + prism2_base (same dim as BulkFormer arm)

Outputs (under OUT_DIR)
-----------------------
  at20cm_pca_rna_fold_metrics.csv
  at20cm_pca_rna_predictions.csv
  at20cm_pca_rna_summary.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import h5py
from sklearn.decomposition import PCA
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
N_PCA      = 640
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


# ── loaders (identical to 08f) ─────────────────────────────────────────────────

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
                     'fold': int(row['fold'])})
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
    print(f'  Loading GCT for {len(sample_ids)} RNA visits ...')
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
        print(f'  proximity-matched {n_prox} visit(s) ≤ {max_gap_days}d')
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


# ── CV runner with per-fold PCA ───────────────────────────────────────────────

def run_pca_arm(name, df, X_raw, X_base, n_pca=N_PCA):
    """
    Per-fold: fit PCA on training RNA, transform both splits, concat with X_base.
    X_base may be None (PCA-only arm) or a 2D array.
    """
    y     = df['label'].values
    folds = df['fold'].values
    vkeys = df['visit_key'].values
    pids  = df['patient_id'].values

    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold; va = folds == fold

        # PCA fitted on training fold only
        pca = PCA(n_components=n_pca, random_state=42)
        X_tr_pca = pca.fit_transform(X_raw[tr])
        X_va_pca = pca.transform(X_raw[va])
        var_explained = pca.explained_variance_ratio_.sum()

        if X_base is not None:
            X_tr = np.hstack([X_tr_pca, X_base[tr]])
            X_va = np.hstack([X_va_pca, X_base[va]])
        else:
            X_tr = X_tr_pca
            X_va = X_va_pca

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
            n_train_visits=int(tr.sum()), n_val_visits=int(va.sum()),
            n_val_patients=int(pd.Series(pids[va]).nunique()),
            pca_var_explained=round(float(var_explained), 4),
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
        print(f'    {name}  fold {fold}: AUC={auc:.4f}  var={var_explained:.3f}  '
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

    print('=== Loading matched cohort (same as 08f) ===')
    img_df = load_histoscore_visits(cv_patients)
    rna_df = load_rna_visits(cv_patients)
    img_m, rna_m = common_visits(img_df, rna_df)

    print('\n=== Loading prism2_base ===')
    base_raw = load_img_base_for_matched(img_m)
    has_base = img_m['visit_key'].isin(set(base_raw['visit_key']))
    img_m = img_m[has_base.values].reset_index(drop=True)
    rna_m = rna_m[has_base.values].reset_index(drop=True)
    base_m = img_m[['visit_key']].merge(base_raw, on='visit_key', how='left')

    n_vis = len(img_m); n_pat = img_m['patient_id'].nunique()
    print(f'Cohort: {n_vis} visits / {n_pat} patients  '
          f"(CD {(img_m['label']==0).sum()}, UC {(img_m['label']==1).sum()})")

    rna_cols = [c for c in rna_m.columns if c.startswith('r')]
    base_cols = [c for c in base_m.columns if c.startswith('f')]
    X_rna  = rna_m[rna_cols].values.astype(np.float32)
    X_base = base_m[base_cols].values.astype(np.float32)

    all_fold_results, all_preds = [], []

    print(f'\n--- pca{N_PCA}_rna_visit (PCA only, no imaging) ---')
    fr, preds = run_pca_arm(f'pca{N_PCA}_rna_visit', rna_m, X_rna,
                             X_base=None, n_pca=N_PCA)
    all_fold_results += fr; all_preds += preds

    print(f'\n--- concat_pca{N_PCA}_base (PCA RNA + prism2_base) ---')
    fr, preds = run_pca_arm(f'concat_pca{N_PCA}_base', img_m, X_rna,
                             X_base=X_base, n_pca=N_PCA)
    all_fold_results += fr; all_preds += preds

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'at20cm_pca_rna_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'at20cm_pca_rna_predictions.csv'), index=False)

    strategies = list(dict.fromkeys(r['strategy'] for r in all_fold_results))
    summary_list = []
    for strat in strategies:
        fr_s = [r for r in all_fold_results if r['strategy'] == strat]
        s = summarise(fr_s)
        s['strategy'] = strat
        s['n_visits']   = n_vis
        s['n_patients'] = n_pat
        var_mean = np.mean([r['pca_var_explained'] for r in fr_s])
        s['mean_pca_var'] = round(float(var_mean), 4)
        summary_list.append(s)

    with open(os.path.join(OUT_DIR, 'at20cm_pca_rna_summary.json'), 'w') as f:
        json.dump(summary_list, f, indent=2)

    print('\n\n=== PCA RNA SUMMARY ===')
    print(f"{'Strategy':<28} {'Visits':<8} {'Patients':<10} {'PCA var':<10}"
          f"{'AUC':<22} {'AP':<22} {'Accuracy'}")
    print('-' * 110)
    for s in summary_list:
        print(f"  {s['strategy']:<26} {s['n_visits']:<8} {s['n_patients']:<10} "
              f"{s['mean_pca_var']:<10.3f}"
              f"AUC={s['mean_auc']:.4f}±{s['std_auc']:.4f}  "
              f"AP={s['mean_ap']:.4f}±{s['std_ap']:.4f}  "
              f"Acc={s['mean_acc']:.4f}")
    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
