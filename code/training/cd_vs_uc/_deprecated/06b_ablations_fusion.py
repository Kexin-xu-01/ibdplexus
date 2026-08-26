"""
Ablation experiments for multimodal fusion.

Extends 06_train_multimodal.py with three targeted experiments:

  concat_raw         - raw concatenation (no scaling, no PCA)
                       tests whether RNA's 17,963 dims dominate imaging's 2,560

  img_base_pca128    - unimodal imaging only, PCA-128 projection
  rna_pca128         - unimodal RNA only, PCA-128 projection

  These two unimodal PCA runs isolate which modality loses more from PCA-128
  compression, explaining the drop in concat_pca128 vs concat_scaled.

All experiments use the same patient-level mean-pooling and identical 5-fold splits
as 06_train_multimodal.py.

Outputs (appended / written alongside multimodal results)
---------
  multimodal_ablation_fold_metrics.csv
  multimodal_ablation_patient_predictions.csv
  multimodal_ablation_summary.json
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
EMB_BASE    = ('/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
               '20x_224px_0px_overlap/prism2_base')
OUT_DIR     = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/06_07_multimodal_allsites/results'

COLON_LOCS = {
    'at 20 cm', 'At 20 cm', 'Cecum', 'Rectum',
    'Ascending Colon', 'Descending Colon', 'Sigmoid Colon', 'Transverse Colon',
}
RF_PARAMS = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)
N_COMPONENTS = 128


def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


# ── data loaders (identical to 06_train_multimodal.py) ───────────────────────

def load_imaging_patient_mean(cv_patients, emb_dir):
    slides = pd.read_csv('/home/jovyan/kgbk271-ibd-volume/training/cv_splits_slides.csv')
    slides = slides[slides['patient_id'].isin(set(cv_patients['patient_id']))]
    pat_df = cv_patients.copy()
    pat_df['label'] = (pat_df['diagnosis'] == 'Ulcerative colitis').astype(int)
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
                         **dict(zip(feat_cols, records[pid]))})
    df = pd.DataFrame(rows)
    print(f'  prism2_base: {len(df)} patients  dim={dim}')
    return df


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
    print(f'  Loading GCT for {len(sample_ids)} samples ...')
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
    X_df = gct_df.T.astype(np.float32)
    pid_map = rna_colon.set_index('SampleID')['deidentified_master_patient_id']
    pat_df  = cv_patients.set_index('patient_id')
    feat_cols = [f'r{i}' for i in range(X_df.shape[1])]
    rows = []
    for pid, grp in pid_map.groupby(pid_map):
        sids = [s for s in grp.index if s in X_df.index]
        if not sids or pid not in pat_df.index: continue
        vec = X_df.loc[sids].values.mean(axis=0)
        row = pat_df.loc[pid]
        rows.append({'patient_id': pid,
                     'label': int(row['diagnosis'] == 'Ulcerative colitis'),
                     'fold': int(row['fold']),
                     **dict(zip(feat_cols, vec))})
    df = pd.DataFrame(rows)
    print(f'  RNA: {len(df)} patients  dim={X_df.shape[1]}')
    return df


def common_patients(df_a, df_b):
    common = set(df_a['patient_id']) & set(df_b['patient_id'])
    a = df_a[df_a['patient_id'].isin(common)].sort_values('patient_id').reset_index(drop=True)
    b = df_b[df_b['patient_id'].isin(common)].sort_values('patient_id').reset_index(drop=True)
    assert (a['patient_id'] == b['patient_id']).all()
    return a, b


def feat(df):
    return df[[c for c in df.columns if c not in ('patient_id','label','fold')]].values.astype(np.float32)


# ── generic CV runner ─────────────────────────────────────────────────────────

def run_cv(name, pat_df, X_override=None):
    """
    Train RF on pat_df features (or X_override if provided).
    Returns (fold_results, all_preds).
    """
    X = X_override if X_override is not None else feat(pat_df)
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
        print(f'    {name}  fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  Acc={cr["accuracy"]:.4f}')
    return fold_results, all_preds


def run_pca_cv(name, base_df, n_comp):
    """Unimodal RF on PCA-n_comp projection (fit on train fold only)."""
    X_full = feat(base_df)
    y = base_df['label'].values
    folds = base_df['fold'].values
    ids = base_df['patient_id'].values
    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold; va = folds == fold
        pca = PCA(n_components=n_comp, random_state=42).fit(X_full[tr])
        Xtr = pca.transform(X_full[tr])
        Xva = pca.transform(X_full[va])
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
        ev   = pca.explained_variance_ratio_.sum()
        fold_results.append(dict(
            strategy=name, fold=fold,
            n_train=int(tr.sum()), n_val=int(va.sum()),
            auc=round(auc,4), ap=round(ap,4), accuracy=round(cr['accuracy'],4),
            cd_f1=round(cr['CD']['f1-score'],4), uc_f1=round(cr['UC']['f1-score'],4),
            tn=int(cm[0,0]), fp=int(cm[0,1]), fn=int(cm[1,0]), tp=int(cm[1,1]),
            pca_var_explained=round(float(ev),4),
        ))
        for pid, yt, yp, ys in zip(ids[va], y_val, y_pred, y_score):
            all_preds.append(dict(patient_id=pid, strategy=name, fold=fold,
                                  true_label=int(yt), pred_label=int(yp),
                                  prob_uc=round(float(ys),5)))
        print(f'    {name}  fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  '
              f'Acc={cr["accuracy"]:.4f}  PCA_var={ev:.3f}')
    return fold_results, all_preds


def run_concat_raw_cv(name, img_df, rna_df):
    """Raw concatenation — no scaling, no PCA."""
    img_a, rna_a = common_patients(img_df, rna_df)
    Xi = feat(img_a); Xr = feat(rna_a)
    X = np.hstack([Xi, Xr])
    return run_cv(name, img_a, X_override=X)


def run_concat_scaled_fold_cv(name, img_df, rna_df):
    """StandardScaler per block per fold (already done in 06, replicated for reference)."""
    img_a, rna_a = common_patients(img_df, rna_df)
    Xi = feat(img_a); Xr = feat(rna_a)
    y = img_a['label'].values
    folds = img_a['fold'].values
    ids = img_a['patient_id'].values
    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold; va = folds == fold
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
            strategy=name, fold=fold, n_train=int(tr.sum()), n_val=int(va.sum()),
            auc=round(auc,4), ap=round(ap,4), accuracy=round(cr['accuracy'],4),
            cd_f1=round(cr['CD']['f1-score'],4), uc_f1=round(cr['UC']['f1-score'],4),
            tn=int(cm[0,0]), fp=int(cm[0,1]), fn=int(cm[1,0]), tp=int(cm[1,1]),
        ))
        for pid, yt, yp, ys in zip(ids[va], y_val, y_pred, y_score):
            all_preds.append(dict(patient_id=pid, strategy=name, fold=fold,
                                  true_label=int(yt), pred_label=int(yp),
                                  prob_uc=round(float(ys),5)))
        print(f'    {name}  fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  Acc={cr["accuracy"]:.4f}')
    return fold_results, all_preds


def summarise(fold_results):
    aucs = [r['auc'] for r in fold_results]
    aps  = [r['ap']  for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    return dict(mean_auc=round(np.mean(aucs),4), std_auc=round(np.std(aucs),4),
                mean_ap=round(np.mean(aps),4),   std_ap=round(np.std(aps),4),
                mean_acc=round(np.mean(accs),4),  std_acc=round(np.std(accs),4))


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)

    print('=== Loading data ===')
    img_df = load_imaging_patient_mean(cv_patients, EMB_BASE)
    rna_df = load_rna_patient_mean(cv_patients)

    img_a, rna_a = common_patients(img_df, rna_df)
    print(f'Matched patients: {len(img_a)}')

    all_fold_results = []
    all_preds        = []

    # ── Experiment 1: raw concatenation ──────────────────────────────────────
    print('\n--- concat_raw (no scaling, no PCA) ---')
    fr, preds = run_concat_raw_cv('concat_raw', img_df, rna_df)
    all_fold_results += fr; all_preds += preds

    # ── Experiment 2: concat_scaled (reference from 06, for direct comparison) ─
    print('\n--- concat_scaled (reference) ---')
    fr, preds = run_concat_scaled_fold_cv('concat_scaled', img_df, rna_df)
    all_fold_results += fr; all_preds += preds

    # ── Experiment 3: unimodal imaging, PCA-128 ───────────────────────────────
    print(f'\n--- img_base_pca{N_COMPONENTS} (unimodal imaging, PCA projection) ---')
    fr, preds = run_pca_cv(f'img_base_pca{N_COMPONENTS}', img_a, N_COMPONENTS)
    all_fold_results += fr; all_preds += preds

    # ── Experiment 4: unimodal RNA, PCA-128 ──────────────────────────────────
    print(f'\n--- rna_pca{N_COMPONENTS} (unimodal RNA, PCA projection) ---')
    fr, preds = run_pca_cv(f'rna_pca{N_COMPONENTS}', rna_a, N_COMPONENTS)
    all_fold_results += fr; all_preds += preds

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'multimodal_ablation_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'multimodal_ablation_patient_predictions.csv'), index=False)

    strategies = [r['strategy'] for r in all_fold_results]
    strats_unique = list(dict.fromkeys(strategies))
    summary_list = []
    for strat in strats_unique:
        fr_s = [r for r in all_fold_results if r['strategy'] == strat]
        s = summarise(fr_s); s['strategy'] = strat
        summary_list.append(s)
    with open(os.path.join(OUT_DIR, 'multimodal_ablation_summary.json'), 'w') as f:
        json.dump(summary_list, f, indent=2)

    # ── Print comparison ──────────────────────────────────────────────────────
    # pull prior results for context
    prior_path = os.path.join(OUT_DIR, 'multimodal_summary.json')
    prior = {}
    if os.path.exists(prior_path):
        for s in json.load(open(prior_path)):
            prior[s['strategy']] = s

    context = {
        'img_base_patmean':    prior.get('img_base_patmean'),
        'rna_patmean':         prior.get('rna_patmean'),
        'concat_pca128_base_rna': prior.get('concat_pca128_base_rna'),
    }

    print('\n\n=== ABLATION RESULTS (patient-level, 5-fold CV) ===')
    print(f"{'Strategy':<32} {'Dim':<12} {'AUC':<20} {'AP':<20} {'Accuracy'}")
    print('-' * 100)

    dims = {
        'img_base_patmean':       '2,560',
        'rna_patmean':            '17,963',
        'concat_raw':             '20,523',
        'concat_scaled':          '20,523',
        f'img_base_pca{N_COMPONENTS}': f'{N_COMPONENTS} (PCA)',
        f'rna_pca{N_COMPONENTS}':      f'{N_COMPONENTS} (PCA)',
        'concat_pca128_base_rna': f'2×{N_COMPONENTS}',
    }

    rows_to_print = [
        ('img_base_patmean',    context.get('img_base_patmean')),
        ('rna_patmean',         context.get('rna_patmean')),
        ('concat_raw',          summarise([r for r in all_fold_results if r['strategy']=='concat_raw'])),
        ('concat_scaled',       summarise([r for r in all_fold_results if r['strategy']=='concat_scaled'])),
        ('concat_pca128_base_rna', context.get('concat_pca128_base_rna')),
        (f'img_base_pca{N_COMPONENTS}', summarise([r for r in all_fold_results if r['strategy']==f'img_base_pca{N_COMPONENTS}'])),
        (f'rna_pca{N_COMPONENTS}',      summarise([r for r in all_fold_results if r['strategy']==f'rna_pca{N_COMPONENTS}'])),
    ]
    for strat, s in rows_to_print:
        if s is None: continue
        print(f"{strat:<32} {dims.get(strat,'?'):<12} "
              f"{s['mean_auc']:.4f} ± {s['std_auc']:.4f}   "
              f"{s['mean_ap']:.4f} ± {s['std_ap']:.4f}   "
              f"{s['mean_acc']:.4f} ± {s['std_acc']:.4f}")

    # PCA variance explained (from fold_results that have it)
    print('\n--- PCA variance explained (mean across folds) ---')
    for strat in [f'img_base_pca{N_COMPONENTS}', f'rna_pca{N_COMPONENTS}']:
        evs = [r['pca_var_explained'] for r in all_fold_results
               if r['strategy'] == strat and 'pca_var_explained' in r]
        if evs:
            print(f'  {strat:<30}: {np.mean(evs)*100:.1f}% variance retained')

    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
