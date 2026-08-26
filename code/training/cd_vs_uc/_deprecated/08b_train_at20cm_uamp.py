"""
Re-run CD vs UC classifiers restricted to At-20-cm biopsies, using UAMP
slide-level features instead of raw prism2 patch embeddings.

Feature source
--------------
  /home/jovyan/kgbk271-ibd-volume/results/prism2/uamp_report.csv
  11 slide-level pathology scores (inflammation_involvement, crypt_architectural_distortion, …)
  averaged across all At-20-cm slides per patient.

Data split
----------
  Same cv_splits_patients.csv / cv_splits_slides.csv used by the rest of the pipeline.

Outputs  (under /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/)
--------
  at20cm_uamp_fold_metrics.csv
  at20cm_uamp_patient_predictions.csv
  at20cm_uamp_summary.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
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
SLIDES_CSV  = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_slides.csv'
UAMP_CSV    = '/home/jovyan/kgbk271-ibd-volume/results/prism2/prism2_histological_score.csv'
OUT_DIR     = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/08_09_at20cm_site_controlled/results'

AT20 = {'at 20 cm', 'At 20 cm'}
RF_PARAMS = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)
N_COMPONENTS = 128

UAMP_FEATURES = [
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


# ── loaders ───────────────────────────────────────────────────────────────────

def load_uamp_20cm(cv_patients):
    slides = pd.read_csv(SLIDES_CSV)
    slides = slides[slides['BIOSAMPLE_LOCATION'].isin(AT20) &
                    slides['patient_id'].isin(set(cv_patients['patient_id']))]

    uamp = pd.read_csv(UAMP_CSV)
    # join on slide id
    merged = slides.merge(uamp, left_on='slide_id', right_on='slide', how='inner')

    pat_df = cv_patients.set_index('patient_id')
    rows = []
    for pid, grp in merged.groupby('patient_id'):
        if pid not in pat_df.index:
            continue
        vec = grp[UAMP_FEATURES].values.mean(axis=0)
        row = pat_df.loc[pid]
        rows.append({'patient_id': pid,
                     'label': int(row['diagnosis'] == 'Ulcerative colitis'),
                     'fold': int(row['fold']),
                     **dict(zip(UAMP_FEATURES, vec))})

    df = pd.DataFrame(rows)
    print(f'  uamp At-20-cm: {len(df)} patients  '
          f"(CD {(df['label']==0).sum()}, UC {(df['label']==1).sum()})  "
          f'dim={len(UAMP_FEATURES)}')
    return df


def load_rna_20cm(cv_patients):
    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(norm_dx)
    cv_pats = set(cv_patients['patient_id'])
    rna_20 = rna[
        rna['characteristics_bio_material'].isin(AT20) &
        rna['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pats)
    ]
    sample_ids = set(rna_20['SampleID'])
    print(f'  Loading GCT for {len(sample_ids)} At-20-cm RNA samples ...')
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
    pid_map = rna_20.set_index('SampleID')['deidentified_master_patient_id']
    pat_df  = cv_patients.set_index('patient_id')
    rows = []
    for pid, grp in pid_map.groupby(pid_map):
        sids = [s for s in grp.index if s in X_df.index]
        if not sids or pid not in pat_df.index: continue
        vec = X_df.loc[sids].values.mean(axis=0)
        row = pat_df.loc[pid]
        rows.append({'patient_id': pid,
                     'label': int(row['diagnosis'] == 'Ulcerative colitis'),
                     'fold': int(row['fold']),
                     **{f'r{i}': v for i,v in enumerate(vec)}})
    df = pd.DataFrame(rows)
    print(f'  RNA At-20-cm: {len(df)} patients  '
          f"(CD {(df['label']==0).sum()}, UC {(df['label']==1).sum()})  dim={X_df.shape[1]}")
    return df


def rna_patient_ids_20cm(cv_patients):
    """Return set of patient IDs that have passing At-20-cm RNA-seq — no GCT load."""
    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(norm_dx)
    cv_pats = set(cv_patients['patient_id'])
    rna_20 = rna[
        rna['characteristics_bio_material'].isin(AT20) &
        rna['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pats)
    ]
    ids = set(rna_20['deidentified_master_patient_id'])
    print(f'  RNA At-20-cm patients (metadata only): {len(ids)}')
    return ids


def common_patients(a, b):
    common = set(a['patient_id']) & set(b['patient_id'])
    a = a[a['patient_id'].isin(common)].sort_values('patient_id').reset_index(drop=True)
    b = b[b['patient_id'].isin(common)].sort_values('patient_id').reset_index(drop=True)
    assert (a['patient_id'] == b['patient_id']).all()
    return a, b


def feat(df):
    return df[[c for c in df.columns if c not in ('patient_id','label','fold')]].values.astype(np.float32)


# ── CV runners ────────────────────────────────────────────────────────────────

def run_unimodal(name, df):
    X = feat(df); y = df['label'].values; folds = df['fold'].values; ids = df['patient_id'].values
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


def run_unimodal_paired_eval(name, df, paired_ids):
    """Train on all imaging patients; evaluate only on those in paired_ids."""
    paired_mask = df['patient_id'].isin(paired_ids).values
    X = feat(df); y = df['label'].values; folds = df['fold'].values; ids = df['patient_id'].values
    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold
        va = (folds == fold) & paired_mask
        if va.sum() == 0:
            print(f'    {name}  fold {fold}: no paired patients in validation — skipped')
            continue
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
        print(f'    {name}  fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  Acc={cr["accuracy"]:.4f}  '
              f'(n_val_paired={va.sum()})')
    return fold_results, all_preds


def run_concat(name, img_df, rna_df, pca=False):
    img_a, rna_a = common_patients(img_df, rna_df)
    Xi = feat(img_a); Xr = feat(rna_a)
    y = img_a['label'].values; folds = img_a['fold'].values; ids = img_a['patient_id'].values
    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold; va = folds == fold
        if pca:
            n_img = min(N_COMPONENTS, Xi.shape[1] - 1)
            pca_i = PCA(n_components=n_img, random_state=42).fit(Xi[tr])
            pca_r = PCA(n_components=N_COMPONENTS, random_state=42).fit(Xr[tr])
            Xtr = np.hstack([pca_i.transform(Xi[tr]), pca_r.transform(Xr[tr])])
            Xva = np.hstack([pca_i.transform(Xi[va]), pca_r.transform(Xr[va])])
        else:
            Xtr = np.hstack([Xi[tr], Xr[tr]])
            Xva = np.hstack([Xi[va], Xr[va]])
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
        print(f'    {name}  fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  Acc={cr["accuracy"]:.4f}')
    return fold_results, all_preds


def summarise(fold_results):
    aucs = [r['auc'] for r in fold_results]; aps = [r['ap'] for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    return dict(mean_auc=round(np.mean(aucs),4), std_auc=round(np.std(aucs),4),
                mean_ap=round(np.mean(aps),4),   std_ap=round(np.std(aps),4),
                mean_acc=round(np.mean(accs),4),  std_acc=round(np.std(accs),4))


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)

    print('=== Loading At-20-cm imaging data ===')
    img_df = load_uamp_20cm(cv_patients)
    rna_ids = rna_patient_ids_20cm(cv_patients)
    paired_n = int(img_df['patient_id'].isin(rna_ids).sum())
    print(f'  Paired (imaging + RNA): {paired_n} patients')

    all_fold_results, all_preds = [], []

    print('\n--- uamp_20cm (unimodal, histological scores At-20-cm only) ---')
    fr, preds = run_unimodal('uamp_20cm', img_df)
    all_fold_results += fr; all_preds += preds

    print('\n--- uamp_20cm_paired (train: all imaging; eval: paired with RNA only) ---')
    fr, preds = run_unimodal_paired_eval('uamp_20cm_paired', img_df, rna_ids)
    all_fold_results += fr; all_preds += preds

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'at20cm_uamp_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'at20cm_uamp_patient_predictions.csv'), index=False)

    strategies = list(dict.fromkeys(r['strategy'] for r in all_fold_results))
    summary_list = []
    for strat in strategies:
        fr_s = [r for r in all_fold_results if r['strategy'] == strat]
        s = summarise(fr_s); s['strategy'] = strat
        s['n_patients'] = paired_n if 'paired' in strat else len(img_df)
        summary_list.append(s)
    with open(os.path.join(OUT_DIR, 'at20cm_uamp_summary.json'), 'w') as f:
        json.dump(summary_list, f, indent=2)

    print('\n\n=== RESULTS SUMMARY ===')
    print(f"{'Strategy':<35} {'N pts':<8} {'AUC':<22} {'AP':<22} {'Accuracy'}")
    print('-'*105)
    for strat in strategies:
        fr_s = [r for r in all_fold_results if r['strategy'] == strat]
        s = summarise(fr_s)
        n = paired_n if 'paired' in strat else len(img_df)
        print(f"  {strat:<33} N={n:<5} "
              f"AUC={s['mean_auc']:.4f}±{s['std_auc']:.4f}  "
              f"AP={s['mean_ap']:.4f}±{s['std_ap']:.4f}  "
              f"Acc={s['mean_acc']:.4f}")

    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
