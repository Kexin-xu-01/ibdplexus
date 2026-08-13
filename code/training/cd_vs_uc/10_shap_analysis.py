"""
SHAP feature importance for CD vs UC classifiers.

Arms analysed
-------------
  rna_20cm        — RNA VST, At-20-cm only          (genes named)
  rna_patmean     — RNA VST, all colon sites         (comparison)
  img_base_20cm   — prism2_base, At-20-cm only
  concat_raw_20cm — raw concat fusion, At-20-cm only

Method
------
For each arm we re-run the 5-fold CV.  After fitting each fold's RF we call
  TreeExplainer(clf).shap_values(X_val)[uc_class_index]
which gives one SHAP value per patient per feature.  We take mean(|SHAP|)
across all validation patients (aggregated over folds) to get a global
importance score per feature.

For RNA arms the feature index maps to a gene name read from the GCT header.
For imaging arms the feature index is an embedding dimension (dim_NNNN).
For the fusion arm the first 2,560 features are imaging dims, the rest genes.

Outputs  (under /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/)
--------
  shap_rna_20cm_top500.csv          — gene, mean_abs_shap, rank
  shap_rna_patmean_top500.csv
  shap_img_base_20cm_top500.csv
  shap_concat_raw_20cm_top500.csv
  shap_rna_20cm_vs_patmean.csv      — side-by-side comparison
  shap_summary.json                 — top-20 per arm
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import h5py
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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
SLIDES_CSV  = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_slides.csv'
EMB_BASE    = ('/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
               '20x_224px_0px_overlap/prism2_base')
OUT_DIR     = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/10_11_shap_analysis/data'

AT20 = {'at 20 cm', 'At 20 cm'}
COLON_LOCS = {
    'at 20 cm', 'At 20 cm', 'Cecum', 'Rectum',
    'Ascending Colon', 'Descending Colon', 'Sigmoid Colon', 'Transverse Colon',
}
RF_PARAMS = dict(n_estimators=300, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)


def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


# ── loaders ────────────────────────────────────────────────────────────────────

def load_img(cv_patients, locations=None):
    slides = pd.read_csv(SLIDES_CSV)
    if locations:
        slides = slides[slides['BIOSAMPLE_LOCATION'].isin(locations)]
    slides = slides[slides['patient_id'].isin(set(cv_patients['patient_id']))]
    pat_df = cv_patients.set_index('patient_id')
    records = {}
    for pid, grp in slides.groupby('patient_id'):
        vecs = []
        for sid in grp['slide_id']:
            h5p = os.path.join(EMB_BASE, f'{sid}.h5')
            if os.path.exists(h5p):
                with h5py.File(h5p, 'r') as h:
                    vecs.append(h['features'][:])
        if vecs:
            records[pid] = np.mean(vecs, axis=0)
    if not records:
        raise RuntimeError('No imaging embeddings found')
    dim = next(iter(records.values())).shape[0]
    rows = []
    for pid, vec in records.items():
        row = pat_df.loc[pid]
        rows.append({'patient_id': pid,
                     'label': int(row['diagnosis'] == 'Ulcerative colitis'),
                     'fold': int(row['fold']),
                     **{f'f{i}': v for i,v in enumerate(vec)}})
    df = pd.DataFrame(rows)
    tag = 'AT20' if locations == AT20 else 'all-colon'
    print(f'  img {tag}: {len(df)} patients  dim={dim}')
    return df, [f'f{i}' for i in range(dim)]


def load_rna(cv_patients, locations):
    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(norm_dx)
    cv_pats = set(cv_patients['patient_id'])
    rna_sel = rna[
        rna['characteristics_bio_material'].isin(locations) &
        rna['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pats)
    ]
    sample_ids = set(rna_sel['SampleID'])
    print(f'  Loading GCT for {len(sample_ids)} RNA samples ...')

    # read gene names from GCT header
    with open(VST_GCT) as f:
        f.readline(); f.readline()
        header_cols = f.readline().strip().split('\t')
    gene_ids_col = 'Name'   # row index = gene ID
    keep_names = [header_cols[0], header_cols[1]] + \
                 [c for c in header_cols[2:] if c in sample_ids]

    gct_df = pd.read_csv(VST_GCT, sep='\t', skiprows=2, header=0,
                         usecols=keep_names, index_col=0,
                         dtype={c: (str if c in (header_cols[0], header_cols[1])
                                    else np.float32)
                                for c in keep_names})
    gene_names = list(gct_df.index)          # gene ID = row Name
    gct_df = gct_df.drop(columns=['Description'])
    X_df = gct_df.T.astype(np.float32)      # samples × genes

    pid_map = rna_sel.set_index('SampleID')['deidentified_master_patient_id']
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
    tag = 'AT20' if locations == AT20 else 'all-colon'
    print(f'  RNA {tag}: {len(df)} patients  dim={len(gene_names)}')
    return df, feat_cols, gene_names


def common_patients(a, b):
    common = set(a['patient_id']) & set(b['patient_id'])
    a = a[a['patient_id'].isin(common)].sort_values('patient_id').reset_index(drop=True)
    b = b[b['patient_id'].isin(common)].sort_values('patient_id').reset_index(drop=True)
    assert (a['patient_id'] == b['patient_id']).all()
    return a, b


def feat(df, cols):
    return df[cols].values.astype(np.float32)


# ── SHAP runner ────────────────────────────────────────────────────────────────

def run_shap(name, X, y, folds, feat_names, n_background=100):
    """
    5-fold CV + SHAP.  Returns array shape (n_patients, n_features)
    where each row is the SHAP value for that patient's contribution
    to P(UC), taken from the fold where that patient was in validation.
    """
    print(f'\n  [{name}] Running 5-fold SHAP ... (n_features={X.shape[1]})')
    shap_vals = np.zeros_like(X, dtype=np.float32)   # one SHAP row per patient
    aucs = []

    for fold in range(5):
        tr = folds != fold; va = folds == fold
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_tr)
        uc_col = list(clf.classes_).index(1)

        auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, uc_col])
        aucs.append(auc)
        print(f'    fold {fold}: AUC={auc:.4f}  n_val={va.sum()}', flush=True)

        # SHAP: use a random background subsample for speed
        bg_idx = np.random.RandomState(42).choice(X_tr.shape[0],
                                                   min(n_background, X_tr.shape[0]),
                                                   replace=False)
        explainer = shap.TreeExplainer(
            clf,
            data=X_tr[bg_idx],
            feature_perturbation='interventional',
        )
        sv = explainer.shap_values(X_va)
        # sv is list[n_classes] each shape (n_val, n_feat)
        if isinstance(sv, list):
            sv_uc = sv[uc_col]
        else:
            # newer shap returns (n_val, n_feat, n_classes)
            sv_uc = sv[:, :, uc_col]
        shap_vals[va] = sv_uc.astype(np.float32)

    mean_auc = float(np.mean(aucs))
    print(f'  [{name}] mean AUC={mean_auc:.4f}')

    mean_abs = np.abs(shap_vals).mean(axis=0)   # shape (n_features,)
    order    = np.argsort(mean_abs)[::-1]
    result   = pd.DataFrame({
        'feature': [feat_names[i] for i in order],
        'feature_idx': order,
        'mean_abs_shap': mean_abs[order],
        'rank': np.arange(1, len(order)+1),
    })
    return result, shap_vals, mean_auc


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)
    np.random.seed(42)

    # ── Load data ──────────────────────────────────────────────────────────────
    print('=== Loading data ===')
    img_20_df, img_feat_names = load_img(cv_patients, locations=AT20)
    rna_20_df, rna_feat_cols, gene_names = load_rna(cv_patients, AT20)
    rna_all_df, _, _ = load_rna(cv_patients, COLON_LOCS)

    img_20_a, rna_20_a = common_patients(img_20_df, rna_20_df)

    summary = {}

    # ══ ARM 1: rna_20cm ═══════════════════════════════════════════════════════
    print('\n=== ARM: rna_20cm ===')
    X = feat(rna_20_df, rna_feat_cols)
    y = rna_20_df['label'].values; folds = rna_20_df['fold'].values
    df_shap, _, auc = run_shap('rna_20cm', X, y, folds, gene_names)
    df_shap.to_csv(os.path.join(OUT_DIR, 'shap_rna_20cm_top500.csv'),
                   index=False, float_format='%.6f')
    summary['rna_20cm'] = {
        'auc': round(auc, 4),
        'top20': df_shap.head(20)[['feature','mean_abs_shap']].to_dict('records'),
    }
    print(f'  Saved shap_rna_20cm_top500.csv')

    # ══ ARM 2: rna_patmean ════════════════════════════════════════════════════
    print('\n=== ARM: rna_patmean (all-sites) ===')
    X = feat(rna_all_df, rna_feat_cols)
    y = rna_all_df['label'].values; folds = rna_all_df['fold'].values
    df_shap_all, _, auc = run_shap('rna_patmean', X, y, folds, gene_names)
    df_shap_all.to_csv(os.path.join(OUT_DIR, 'shap_rna_patmean_top500.csv'),
                       index=False, float_format='%.6f')
    summary['rna_patmean'] = {
        'auc': round(auc, 4),
        'top20': df_shap_all.head(20)[['feature','mean_abs_shap']].to_dict('records'),
    }
    print(f'  Saved shap_rna_patmean_top500.csv')

    # ══ ARM 3: img_base_20cm ══════════════════════════════════════════════════
    print('\n=== ARM: img_base_20cm ===')
    X = feat(img_20_df, img_feat_names)
    y = img_20_df['label'].values; folds = img_20_df['fold'].values
    df_shap_img, _, auc = run_shap('img_base_20cm', X, y, folds, img_feat_names)
    df_shap_img.to_csv(os.path.join(OUT_DIR, 'shap_img_base_20cm_top500.csv'),
                       index=False, float_format='%.6f')
    summary['img_base_20cm'] = {
        'auc': round(auc, 4),
        'top20': df_shap_img.head(20)[['feature','mean_abs_shap']].to_dict('records'),
    }
    print(f'  Saved shap_img_base_20cm_top500.csv')

    # ══ ARM 4: concat_raw_20cm ════════════════════════════════════════════════
    print('\n=== ARM: concat_raw_20cm ===')
    n_img = len(img_feat_names)
    n_rna = len(gene_names)
    concat_feat_names = [f'img_{f}' for f in img_feat_names] + gene_names
    X_img = feat(img_20_a, img_feat_names)
    X_rna = feat(rna_20_a, rna_feat_cols)
    X_cat = np.hstack([X_img, X_rna])
    y = img_20_a['label'].values; folds = img_20_a['fold'].values
    df_shap_cat, shap_cat, auc = run_shap('concat_raw_20cm', X_cat, y, folds,
                                          concat_feat_names)
    df_shap_cat.to_csv(os.path.join(OUT_DIR, 'shap_concat_raw_20cm_top500.csv'),
                       index=False, float_format='%.6f')

    # modality-level summary for fusion arm
    mean_abs_full = np.abs(shap_cat).mean(axis=0)
    img_total = mean_abs_full[:n_img].sum()
    rna_total = mean_abs_full[n_img:].sum()
    img_frac  = img_total / (img_total + rna_total)
    rna_frac  = rna_total / (img_total + rna_total)
    print(f'  Fusion SHAP split: imaging={img_frac*100:.1f}%  RNA={rna_frac*100:.1f}%')

    # top RNA genes in fusion arm
    rna_importance = pd.DataFrame({
        'feature': gene_names,
        'mean_abs_shap_fusion': mean_abs_full[n_img:],
    }).sort_values('mean_abs_shap_fusion', ascending=False)

    summary['concat_raw_20cm'] = {
        'auc': round(auc, 4),
        'imaging_shap_fraction': round(img_frac, 4),
        'rna_shap_fraction': round(rna_frac, 4),
        'top20': df_shap_cat.head(20)[['feature','mean_abs_shap']].to_dict('records'),
    }
    print(f'  Saved shap_concat_raw_20cm_top500.csv')

    # ══ Comparison table: rna_20cm vs rna_patmean ════════════════════════════
    print('\n=== Building rna_20cm vs rna_patmean comparison ===')
    top500_20  = df_shap.set_index('feature')[['mean_abs_shap','rank']].rename(
        columns={'mean_abs_shap':'shap_20cm','rank':'rank_20cm'})
    top500_all = df_shap_all.set_index('feature')[['mean_abs_shap','rank']].rename(
        columns={'mean_abs_shap':'shap_allsites','rank':'rank_allsites'})
    cmp = top500_20.join(top500_all, how='outer').reset_index()
    cmp.rename(columns={'index': 'gene'}, inplace=True)
    cmp = cmp.sort_values('rank_20cm', na_position='last')
    cmp['rank_change'] = cmp['rank_allsites'] - cmp['rank_20cm']  # +ve = went down in all-sites
    cmp.to_csv(os.path.join(OUT_DIR, 'shap_rna_20cm_vs_patmean.csv'),
               index=False, float_format='%.6f')
    print(f'  Saved shap_rna_20cm_vs_patmean.csv')

    # print top-20 comparison to console
    print('\n  Top-20 genes: rna_20cm  vs  rna_patmean (all-sites)')
    print(f"  {'Rank':>4}  {'Gene':<20}  {'SHAP@20cm':>10}  {'rank@all':>9}  {'SHAP@all':>10}")
    print('  ' + '-'*62)
    for _, r in cmp.head(20).iterrows():
        print(f"  {int(r.rank_20cm):>4}  {r.gene:<20}  {r.shap_20cm:>10.5f}  "
              f"{'' if pd.isna(r.rank_allsites) else int(r.rank_allsites):>9}  "
              f"{'' if pd.isna(r.shap_allsites) else r.shap_allsites:>10.5f}")

    # ══ Save summary JSON ════════════════════════════════════════════════════
    with open(os.path.join(OUT_DIR, 'shap_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nAll outputs saved to {OUT_DIR}/')

    # ══ Print top genes per arm ══════════════════════════════════════════════
    for arm, data in summary.items():
        print(f'\n── Top 10 features: {arm} (AUC={data["auc"]}) ──')
        for i, r in enumerate(data['top20'][:10], 1):
            print(f'  {i:>3}. {r["feature"]:<25}  {r["mean_abs_shap"]:.5f}')


if __name__ == '__main__':
    main()
