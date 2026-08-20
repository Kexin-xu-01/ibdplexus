"""
MLP classifier — all 10 arms on the at-20-cm matched cohort.

Drop-in replacement for the RF experiments in 08b/08c/08f/08g.
Key differences vs RF:
  - StandardScaler fitted on training fold only (required for MLP)
  - Class imbalance handled via sample_weight (balanced) in fit()
  - PCA(640) still fitted on training fold only (same as 08g)

MLP: 2 hidden layers (256, 128), ReLU, Adam, L2 α=1e-3, early stopping.

Arms (same 945-visit / 817-patient matched cohort as 08f/08g)
-----
  img_base_visit          — prism2 base (2,560-d)
  img_histoscore_visit    — histo scores (11-d)
  bulkformer_visit        — BulkFormer embeddings (640-d)
  rna_visit               — RNA-seq VST (17,963 genes)
  pca640_rna_visit        — PCA(640) of RNA VST
  concat_raw_visit        — RNA + prism2 base
  concat_histoscore_visit — RNA + histo scores
  concat_bf_base_visit    — BulkFormer + prism2 base
  concat_bf_histo_visit   — BulkFormer + histo scores
  concat_pca640_base      — PCA-640 RNA + prism2 base

Outputs
-------
  concept_learning/results/at20cm_mlp_fold_metrics.csv
  concept_learning/results/at20cm_mlp_predictions.csv
  concept_learning/results/at20cm_mlp_summary.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import h5py
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
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

AT20      = {'at 20 cm', 'At 20 cm'}
CDUC_RAW  = ["Crohn's disease", 'Ulcerative colitis']
EMB_DIM   = 640
EMB_COLS  = [f'e{i}' for i in range(EMB_DIM)]
N_PCA     = 640
HISTO_COLS = [
    'inflammation_involvement', 'crypt_architectural_distortion',
    'neutrophil_granulocytic_infiltration', 'crypt_abscesses',
    'lymphoid_aggregates', 'histiocytic_granulomas', 'mucin_depletion',
    'pyloric_gland_metaplasia', 'paneth_cell_metaplasia',
    'neuronal_hyperplasia', 'muscular_hypertrophy',
]

MLP_PARAMS = dict(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    alpha=1e-3,
    batch_size=64,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
)


def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


# ── loaders ───────────────────────────────────────────────────────────────────

def load_histoscore_visits(cv_patients):
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


def load_bulkformer_emb():
    print('  Loading BulkFormer embeddings ...')
    emb = torch.load(BULKFORMER_PT, map_location='cpu').numpy().astype(np.float32)
    log_tpm = pd.read_csv(LOG_TPM_CSV, usecols=['SampleID'])
    assert len(emb) == len(log_tpm)
    df = pd.DataFrame(emb, columns=EMB_COLS)
    df['SampleID'] = log_tpm['SampleID'].values
    print(f'  BulkFormer: {emb.shape[0]} × {emb.shape[1]}')
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


# ── CV runner ─────────────────────────────────────────────────────────────────

def run_arm(name, df, X, pca_dim=None):
    """
    MLP with per-fold StandardScaler.
    If pca_dim is set, PCA(pca_dim) is fitted on training RNA (first pca_dim columns
    of X) before scaling and concatenating with remaining columns.
    """
    y     = df['label'].values
    folds = df['fold'].values
    vkeys = df['visit_key'].values
    pids  = df['patient_id'].values

    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold; va = folds == fold

        if pca_dim is not None:
            # X = [RNA_cols | base_cols]; PCA only on the first pca_dim RNA cols
            X_rna_part  = X[:, :pca_dim]      # raw RNA VST portion
            X_rest       = X[:, pca_dim:]      # prism2_base portion (may be empty)
            pca = PCA(n_components=pca_dim, random_state=42)
            X_rna_tr = pca.fit_transform(X_rna_part[tr])
            X_rna_va = pca.transform(X_rna_part[va])
            X_tr_raw = np.hstack([X_rna_tr, X_rest[tr]]) if X_rest.shape[1] else X_rna_tr
            X_va_raw = np.hstack([X_rna_va, X_rest[va]]) if X_rest.shape[1] else X_rna_va
        else:
            X_tr_raw = X[tr]
            X_va_raw = X[va]

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr_raw)
        X_va = sc.transform(X_va_raw)

        sw = compute_sample_weight('balanced', y[tr])
        clf = MLPClassifier(**MLP_PARAMS)
        clf.fit(X_tr, y[tr], sample_weight=sw)

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

    rna_cols   = [c for c in rna_m.columns if c.startswith('r')]
    base_cols  = [c for c in base_m.columns if c.startswith('f')]
    histo_cols = HISTO_COLS

    X_rna   = rna_m[rna_cols].values.astype(np.float32)
    X_base  = base_m[base_cols].values.astype(np.float32)
    X_histo = img_m[histo_cols].values.astype(np.float32)
    X_bf    = bf_m[EMB_COLS].values.astype(np.float32)

    all_fold_results, all_preds = [], []

    print(f'\n--- img_base_visit ---')
    fr, preds = run_arm('img_base_visit', img_m, X_base)
    all_fold_results += fr; all_preds += preds

    print(f'\n--- img_histoscore_visit ---')
    fr, preds = run_arm('img_histoscore_visit', img_m, X_histo)
    all_fold_results += fr; all_preds += preds

    print(f'\n--- bulkformer_visit ---')
    fr, preds = run_arm('bulkformer_visit', bf_m, X_bf)
    all_fold_results += fr; all_preds += preds

    print(f'\n--- rna_visit ---')
    fr, preds = run_arm('rna_visit', rna_m, X_rna)
    all_fold_results += fr; all_preds += preds

    print(f'\n--- pca640_rna_visit ---')
    fr, preds = run_arm('pca640_rna_visit', rna_m, X_rna, pca_dim=N_PCA)
    all_fold_results += fr; all_preds += preds

    print(f'\n--- concat_raw_visit ---')
    fr, preds = run_arm('concat_raw_visit', rna_m,
                         np.hstack([X_rna, X_base]))
    all_fold_results += fr; all_preds += preds

    print(f'\n--- concat_histoscore_visit ---')
    fr, preds = run_arm('concat_histoscore_visit', rna_m,
                         np.hstack([X_rna, X_histo]))
    all_fold_results += fr; all_preds += preds

    print(f'\n--- concat_bf_base_visit ---')
    fr, preds = run_arm('concat_bf_base_visit', bf_m,
                         np.hstack([X_bf, X_base]))
    all_fold_results += fr; all_preds += preds

    print(f'\n--- concat_bf_histo_visit ---')
    fr, preds = run_arm('concat_bf_histo_visit', bf_m,
                         np.hstack([X_bf, X_histo]))
    all_fold_results += fr; all_preds += preds

    # PCA arm: pass RNA as first N_PCA cols, base as trailing cols
    # run_arm will apply PCA to X[:, :N_PCA] and concat with X[:, N_PCA:]
    print(f'\n--- concat_pca640_base ---')
    fr, preds = run_arm('concat_pca640_base', img_m,
                         np.hstack([X_rna, X_base]), pca_dim=N_PCA)
    all_fold_results += fr; all_preds += preds

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'at20cm_mlp_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'at20cm_mlp_predictions.csv'), index=False)

    strategies = list(dict.fromkeys(r['strategy'] for r in all_fold_results))
    summary_list = []
    for strat in strategies:
        fr_s = [r for r in all_fold_results if r['strategy'] == strat]
        s = summarise(fr_s)
        s['strategy'] = strat
        s['n_visits'] = n_vis; s['n_patients'] = n_pat
        summary_list.append(s)

    with open(os.path.join(OUT_DIR, 'at20cm_mlp_summary.json'), 'w') as f:
        json.dump(summary_list, f, indent=2)

    print('\n\n=== MLP SUMMARY ===')
    print(f"{'Strategy':<28} {'AUC':<24} {'AP':<24} {'Accuracy'}")
    print('-' * 90)
    for s in summary_list:
        print(f"  {s['strategy']:<26} "
              f"AUC={s['mean_auc']:.4f}±{s['std_auc']:.4f}  "
              f"AP={s['mean_ap']:.4f}±{s['std_ap']:.4f}  "
              f"Acc={s['mean_acc']:.4f}")
    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
