"""
RNA-only MLP ablation: input representation × architecture.

Isolates two hypotheses for why MLP underperforms RF on RNA-seq VST:
  (A) Input representation: VST vs log-TPM
  (B) Architecture: flat (256,128) vs gradual bottleneck (2048,512,128)

Configs
-------
  vst_flat       — VST + StandardScaler   → MLP(256, 128)          [current baseline]
  logtpm_flat    — log-TPM + StandardScaler → MLP(256, 128)
  vst_bottle     — VST + StandardScaler   → MLP(2048, 512, 128)
  logtpm_bottle  — log-TPM + StandardScaler → MLP(2048, 512, 128)

Same 945-visit / 817-patient cohort, same 5-fold patient-level CV.
"""

import os, json, warnings
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report)

warnings.filterwarnings('ignore')

TRANSCRIPTOMICS_DIR = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/'
                       'genestack/transcriptomics')
WSI_META       = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/'
                  'IBD_meta_data_latest/wsi_metadata_raw.csv')
HISTOSCORE_CSV = '/home/jovyan/kgbk271-ibd-volume/results/prism2/prism2_histological_score.csv'
VST_GCT        = os.path.join(TRANSCRIPTOMICS_DIR,
                 'GSF1491805_CombatSeq_vst_mtx_batch_corrected_'
                 'alltissues_all3releases_header.gct')
LOG_TPM_CSV    = os.path.join(TRANSCRIPTOMICS_DIR,
                 'GSF1491803_CombatSeq_count_mtx_batch_corrected_'
                 'alltissues_all3releases_header_log_tpm_with_sampleID.csv')
MAPPING_CSV    = os.path.join(TRANSCRIPTOMICS_DIR,
                 'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META    = os.path.join(TRANSCRIPTOMICS_DIR,
                 'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS    = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
OUT_DIR        = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
                  'concept_learning/results')

AT20     = {'at 20 cm', 'At 20 cm'}
CDUC_RAW = ["Crohn's disease", 'Ulcerative colitis']

CONFIGS = {
    'vst_flat':      dict(hidden_layer_sizes=(256, 128),     alpha=1e-3),
    'logtpm_flat':   dict(hidden_layer_sizes=(256, 128),     alpha=1e-3),
    'vst_bottle':    dict(hidden_layer_sizes=(2048, 512, 128), alpha=1e-2),
    'logtpm_bottle': dict(hidden_layer_sizes=(2048, 512, 128), alpha=1e-2),
}
MLP_BASE = dict(activation='relu', solver='adam', batch_size=64,
                max_iter=500, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20, random_state=42)

RF_AUC = 0.816  # RF baseline for rna_visit


def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


def load_cohort_meta(cv_patients):
    """Return list of (sid, pid, fold, label) for at-20-cm RNA visits."""
    pat_df  = cv_patients.set_index('patient_id')
    cv_pids = set(pat_df.index)
    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(norm_dx)
    rna = rna[
        rna['characteristics_bio_material'].isin(AT20) &
        rna['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pids)
    ].drop_duplicates(subset='visit_encounter_id', keep='first').copy()
    records = []
    for _, row in rna.iterrows():
        pid = row['deidentified_master_patient_id']
        if pid not in pat_df.index: continue
        pr = pat_df.loc[pid]
        records.append(dict(
            SampleID=row['SampleID'], patient_id=pid,
            fold=int(pr['fold']),
            label=int(pr['diagnosis'] == 'Ulcerative colitis'),
        ))
    return pd.DataFrame(records)


def load_vst(meta_df):
    sample_ids = set(meta_df['SampleID'])
    with open(VST_GCT) as f:
        f.readline(); f.readline()
        header_cols = f.readline().strip().split('\t')
    keep       = [0, 1] + [i for i, c in enumerate(header_cols) if c in sample_ids]
    keep_names = [header_cols[i] for i in keep]
    gct = pd.read_csv(VST_GCT, sep='\t', skiprows=2, header=0,
                      usecols=keep_names, index_col=0,
                      dtype={c: (str if c in ('Name', 'Description') else np.float32)
                             for c in keep_names})
    gct = gct.drop(columns=['Description'])
    X_df = gct.T.astype(np.float32)
    rows = [X_df.loc[sid].values for sid in meta_df['SampleID'] if sid in X_df.index]
    sids = [sid for sid in meta_df['SampleID'] if sid in X_df.index]
    idx = meta_df[meta_df['SampleID'].isin(set(sids))].copy()
    idx = idx.set_index('SampleID').loc[sids].reset_index()
    print(f'  VST: {len(idx)} visits, {len(rows[0])} genes')
    return idx, np.array(rows, dtype=np.float32)


def load_logtpm(meta_df):
    sample_ids = set(meta_df['SampleID'])
    df = pd.read_csv(LOG_TPM_CSV, index_col=0)
    df.index = df.index.astype(str)
    present = [sid for sid in meta_df['SampleID'] if sid in df.index]
    idx = meta_df[meta_df['SampleID'].isin(set(present))].copy()
    idx = idx.set_index('SampleID').loc[present].reset_index()
    X = df.loc[present].values.astype(np.float32)
    print(f'  log-TPM: {len(idx)} visits, {X.shape[1]} genes')
    return idx, X


def run_cv(name, idx_df, X, mlp_params):
    y     = idx_df['label'].values
    folds = idx_df['fold'].values
    pids  = idx_df['patient_id'].values
    fold_aucs = []
    for fold in range(5):
        tr = folds != fold; va = folds == fold
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_va = sc.transform(X[va])
        sw = compute_sample_weight('balanced', y[tr])
        clf = MLPClassifier(**{**MLP_BASE, **mlp_params})
        clf.fit(X_tr, y[tr], sample_weight=sw)
        y_score = clf.predict_proba(X_va)[:, list(clf.classes_).index(1)]
        auc = roc_auc_score(y[va], y_score)
        fold_aucs.append(auc)
        print(f'    {name}  fold {fold}: AUC={auc:.4f}  '
              f'({va.sum()} visits / {pd.Series(pids[va]).nunique()} patients)')
    m, s = np.mean(fold_aucs), np.std(fold_aucs, ddof=1)
    print(f'  → {name}: AUC={m:.4f}±{s:.4f}  (RF baseline: {RF_AUC:.3f})')
    return m, s


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)

    print('=== Loading cohort metadata ===')
    meta = load_cohort_meta(cv_patients)
    print(f'  {len(meta)} visits / {meta["patient_id"].nunique()} patients')

    print('\n=== Loading VST features ===')
    meta_vst, X_vst = load_vst(meta)

    print('\n=== Loading log-TPM features ===')
    meta_ltpm, X_ltpm = load_logtpm(meta)

    # Align to common samples
    common_sids = set(meta_vst['SampleID']) & set(meta_ltpm['SampleID'])
    meta_vst  = meta_vst[meta_vst['SampleID'].isin(common_sids)].reset_index(drop=True)
    meta_ltpm = meta_ltpm[meta_ltpm['SampleID'].isin(common_sids)].reset_index(drop=True)
    # reorder log-tpm to match vst order
    ltpm_idx = {s: i for i, s in enumerate(meta_ltpm['SampleID'])}
    order = [ltpm_idx[s] for s in meta_vst['SampleID']]
    X_ltpm = X_ltpm[order]
    meta_ltpm = meta_vst  # same aligned meta

    n = len(meta_vst)
    print(f'\nAligned cohort: {n} visits / {meta_vst["patient_id"].nunique()} patients')
    print(f'VST range:    [{X_vst.min():.2f}, {X_vst.max():.2f}]')
    print(f'log-TPM range:[{X_ltpm.min():.2f}, {X_ltpm.max():.2f}]')

    results = []
    for name, params in CONFIGS.items():
        print(f'\n--- {name} ---')
        X = X_vst if name.startswith('vst') else X_ltpm
        m, s = run_cv(name, meta_vst, X, params)
        results.append({'config': name, 'mean_auc': round(m, 4), 'std_auc': round(s, 4),
                        'architecture': str(params['hidden_layer_sizes']),
                        'alpha': params['alpha'],
                        'input': 'VST' if name.startswith('vst') else 'log-TPM'})

    print('\n\n=== RNA MLP ABLATION SUMMARY ===')
    print(f"{'Config':<20} {'Input':<10} {'Architecture':<22} {'α':<8} {'AUC'}")
    print('-'*75)
    for r in results:
        print(f"  {r['config']:<18} {r['input']:<10} {r['architecture']:<22} "
              f"{r['alpha']:<8} {r['mean_auc']:.4f}±{r['std_auc']:.4f}")
    print(f"\n  RF baseline: 0.8160±0.0334")

    with open(os.path.join(OUT_DIR, 'at20cm_rna_mlp_ablation.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {OUT_DIR}/at20cm_rna_mlp_ablation.json')


if __name__ == '__main__':
    main()
