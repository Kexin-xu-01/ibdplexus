"""
CD vs UC — At-20-cm, visit-level, ssGSEA pathway scores as RNA features.

Compresses 17,963-gene VST expression into pathway-level NES scores
(single-sample GSEA via gseapy), then trains the same visit-level RF
classifier used in 08b_train_at20cm_visit_level.py.

Three arms — all on the same matched cohort (hist + RNA, 945 visits / 817 pts):
  pathway_hallmark_visit   — 50 MSigDB Hallmark 2020 NES scores
  pathway_kegg_visit       — 320 KEGG 2021 Human NES scores
  pathway_combined_visit   — 50 + 320 concatenated (370 features)

Gene format
-----------
  VST GCT uses Ensembl IDs as row index; Description column has gene symbols.
  ssGSEA gene sets use symbols — genes are renamed before scoring.
  Duplicate symbols (rare) → keep the row with highest median expression.

Outputs (under OUT_DIR)
-----------------------
  at20cm_pathway_visit_fold_metrics.csv
  at20cm_pathway_visit_predictions.csv
  at20cm_pathway_visit_summary.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import gseapy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
TRANSCRIPTOMICS_DIR = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/'
                       'genestack/transcriptomics')
VST_GCT     = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF1491805_CombatSeq_vst_mtx_batch_corrected_'
              'alltissues_all3releases_header.gct')
MAPPING_CSV = os.path.join(TRANSCRIPTOMICS_DIR,
              'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
WSI_META    = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/'
               'IBD_meta_data_latest/wsi_metadata_raw.csv')
HIST_CSV    = '/home/jovyan/kgbk271-ibd-volume/results/prism2/prism2_histological_score.csv'
CV_PATIENTS = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
OUT_DIR     = ('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/'
               '08_09_at20cm_site_controlled/results')

AT20      = {'at 20 cm', 'At 20 cm'}
CDUC_RAW  = ["Crohn's disease", 'Ulcerative colitis']
RF_PARAMS = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)

LIBRARIES = {
    'hallmark': 'MSigDB_Hallmark_2020',
    'kegg':     'KEGG_2021_Human',
}


def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


# ── data loading ──────────────────────────────────────────────────────────────

def get_rna_visit_meta(cv_patients):
    """Return visit-level metadata for at-20-cm RNA samples (no expression loaded)."""
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

    rows = []
    for _, r in rna.iterrows():
        pid = r['deidentified_master_patient_id']
        if pid not in pat_df.index:
            continue
        pat   = pat_df.loc[pid]
        vdate = r['date'].date() if not pd.isna(r['date']) else None
        rows.append({
            'SampleID':   r['SampleID'],
            'patient_id': pid,
            'visit_date': str(vdate),
            'visit_key':  str((pid, vdate)),
            'visit_encounter_id': r['visit_encounter_id'],
            'label': int(pat['diagnosis'] == 'Ulcerative colitis'),
            'fold':  int(pat['fold']),
        })
    return pd.DataFrame(rows)


def load_vst_symbol(sample_ids):
    """Load VST GCT for given sample IDs; return genes-as-symbols × samples DataFrame."""
    with open(VST_GCT) as f:
        f.readline(); f.readline()
        header_cols = f.readline().strip().split('\t')

    keep_idx   = [0, 1] + [i for i, c in enumerate(header_cols) if c in set(sample_ids)]
    keep_names = [header_cols[i] for i in keep_idx]

    gct = pd.read_csv(VST_GCT, sep='\t', skiprows=2, header=0,
                      usecols=keep_names, index_col=0,
                      dtype={c: (str if c in ('Name', 'Description') else np.float32)
                             for c in keep_names})

    # rename index: Ensembl → gene symbol
    symbol_map = gct['Description'].to_dict()
    gct = gct.drop(columns=['Description'])

    # deduplicate symbols: keep row with highest median expression
    gct.index = gct.index.map(symbol_map)
    gct = gct[~gct.index.isna()]
    gct = gct[~gct.index.duplicated(keep=False) |
              gct.index.isin(
                  gct.groupby(gct.index).apply(lambda g: g.median(axis=1).idxmax())
              )]
    # simpler dedup: keep first occurrence (nearly identical for duplicates)
    gct = gct[~gct.index.duplicated(keep='first')]

    return gct.astype(np.float32)   # genes × samples


def run_ssgsea(expr_df, library_name, threads=4):
    """
    Run single-sample GSEA on expr_df (genes × samples).
    Returns samples × pathways NES DataFrame.
    """
    ss = gseapy.ssgsea(
        expr_df,
        gene_sets=library_name,
        outdir=None,
        min_size=5,
        max_size=500,
        scale=True,
        permutation_num=0,
        threads=threads,
        no_plot=True,
    )
    scores = ss.res2d.pivot(index='Name', columns='Term', values='NES')
    scores.index.name = 'SampleID'
    return scores   # samples × pathways


def build_visit_df(meta_df, scores_df, prefix):
    """Merge visit metadata with ssGSEA scores; return one row per visit."""
    merged = meta_df.merge(scores_df, left_on='SampleID', right_index=True, how='inner')
    score_cols = scores_df.columns.tolist()
    renamed    = {c: f'{prefix}_{c}' for c in score_cols}
    merged     = merged.rename(columns=renamed)
    print(f'  {prefix}: {len(merged)} visits from '
          f'{merged["patient_id"].nunique()} patients  '
          f"(CD {(merged['label']==0).sum()}, UC {(merged['label']==1).sum()})  "
          f'dim={len(score_cols)}')
    return merged


# ── visit matching (copied from 08b / 08c) ────────────────────────────────────

def load_hist_visits(cv_patients):
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
    hist  = pd.read_csv(HIST_CSV)
    HFEAT = [c for c in hist.columns if c != 'slide']
    wsi   = wsi.merge(hist, left_on='slide_id', right_on='slide', how='inner')
    rows  = []
    for (pid, vdate), grp in wsi.groupby('visit_key'):
        if pid not in pat_df.index:
            continue
        vec = grp[HFEAT].values.mean(axis=0)
        row = pat_df.loc[pid]
        rows.append({'visit_key': str((pid, vdate)), 'patient_id': pid,
                     'visit_date': str(vdate),
                     'label': int(row['diagnosis'] == 'Ulcerative colitis'),
                     'fold':  int(row['fold']),
                     **dict(zip(HFEAT, vec))})
    return pd.DataFrame(rows)


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
    assert (img_out['fold']       == rna_out['fold']).all()
    n_prox = (img_out['visit_key'] != rna_out['visit_key']).sum()
    if n_prox:
        print(f'  proximity-matched {n_prox} visit(s) ≤{max_gap_days}d')
    return img_out, rna_out


# ── CV runner ─────────────────────────────────────────────────────────────────

def feat_cols(df):
    skip = {'visit_key', 'patient_id', 'visit_date', 'visit_encounter_id',
            'SampleID', 'label', 'fold'}
    return [c for c in df.columns if c not in skip]


def run_unimodal(name, df):
    cols  = feat_cols(df)
    X     = df[cols].values.astype(np.float32)
    y     = df['label'].values
    folds = df['fold'].values
    vkeys = df['visit_key'].values
    pids  = df['patient_id'].values

    fold_results, all_preds = [], []
    for fold in range(5):
        tr = folds != fold
        va = folds == fold
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X[tr], y[tr])
        proba   = clf.predict_proba(X[va])
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]
        y_pred  = clf.predict(X[va])
        y_val   = y[va]
        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred, target_names=['CD', 'UC'],
                                    output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)
        fold_results.append(dict(
            strategy=name, fold=fold,
            n_train_visits=int(tr.sum()), n_val_visits=int(va.sum()),
            n_val_patients=int(pd.Series(pids[va]).nunique()),
            auc=round(auc,4), ap=round(ap,4),
            accuracy=round(cr['accuracy'],4),
            cd_f1=round(cr['CD']['f1-score'],4),
            uc_f1=round(cr['UC']['f1-score'],4),
            tn=int(cm[0,0]), fp=int(cm[0,1]),
            fn=int(cm[1,0]), tp=int(cm[1,1]),
        ))
        for vk, pid, yt, yp, ys in zip(vkeys[va], pids[va], y_val, y_pred, y_score):
            all_preds.append(dict(visit_key=vk, patient_id=pid, strategy=name,
                                  fold=fold, true_label=int(yt), pred_label=int(yp),
                                  prob_uc=round(float(ys), 5)))
        print(f'    {name}  fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  '
              f'Acc={cr["accuracy"]:.4f}  '
              f'({va.sum()} visits / {pd.Series(pids[va]).nunique()} patients)')
    return fold_results, all_preds


def summarise(fold_results):
    aucs = [r['auc'] for r in fold_results]
    aps  = [r['ap']  for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    return dict(
        mean_auc=round(np.mean(aucs),4), std_auc=round(np.std(aucs, ddof=1),4),
        mean_ap=round(np.mean(aps),  4), std_ap=round(np.std(aps,  ddof=1),4),
        mean_acc=round(np.mean(accs), 4), std_acc=round(np.std(accs, ddof=1),4),
    )


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)

    # ── 1. Visit metadata ─────────────────────────────────────────────────────
    print('=== Loading RNA visit metadata ===')
    rna_meta = get_rna_visit_meta(cv_patients)
    print(f'  {len(rna_meta)} at-20-cm RNA visits  '
          f'({rna_meta["patient_id"].nunique()} patients)')

    # ── 2. Load VST expression for those samples ───────────────────────────────
    print('\n=== Loading VST expression (at-20-cm samples) ===')
    sample_ids = rna_meta['SampleID'].tolist()
    expr = load_vst_symbol(sample_ids)
    print(f'  Expression matrix: {expr.shape[0]} genes × {expr.shape[1]} samples')

    # ── 3. ssGSEA per library ─────────────────────────────────────────────────
    scores = {}
    for key, lib_name in LIBRARIES.items():
        print(f'\n=== ssGSEA: {lib_name} ===')
        scores[key] = run_ssgsea(expr, lib_name)
        print(f'  Scores: {scores[key].shape[0]} samples × {scores[key].shape[1]} pathways')

    # ── 4. Build visit-level feature DataFrames ────────────────────────────────
    print('\n=== Building visit feature DataFrames ===')
    hall_visits = build_visit_df(rna_meta, scores['hallmark'], prefix='H')
    kegg_visits = build_visit_df(rna_meta, scores['kegg'],     prefix='K')

    # combined: join hallmark + kegg columns (same sample order)
    comb_visits = hall_visits.merge(
        kegg_visits[[c for c in kegg_visits.columns if c.startswith('K_')]
                    + ['SampleID']],
        on='SampleID', how='inner')
    print(f'  combined: {len(comb_visits)} visits  '
          f'dim={len([c for c in comb_visits.columns if c.startswith(("H_","K_"))])}')

    # ── 5. Restrict to matched cohort (hist + RNA proximity join) ─────────────
    print('\n=== Matching to hist cohort (same 945/817 as img_base_visit) ===')
    hist_df = load_hist_visits(cv_patients)

    _, hall_m = common_visits(hist_df, hall_visits)
    _, kegg_m = common_visits(hist_df, kegg_visits)
    _, comb_m = common_visits(hist_df, comb_visits)

    n_vis = len(hall_m)
    n_pat = hall_m['patient_id'].nunique()
    print(f'Matched cohort: {n_vis} visits from {n_pat} patients  '
          f"(CD {(hall_m['label']==0).sum()}, UC {(hall_m['label']==1).sum()})")

    # ── 6. Run classifiers ────────────────────────────────────────────────────
    all_fold_results, all_preds = [], []

    print('\n--- pathway_hallmark_visit (50 Hallmark NES scores) ---')
    fr, preds = run_unimodal('pathway_hallmark_visit', hall_m)
    all_fold_results += fr; all_preds += preds

    print('\n--- pathway_kegg_visit (320 KEGG NES scores) ---')
    fr, preds = run_unimodal('pathway_kegg_visit', kegg_m)
    all_fold_results += fr; all_preds += preds

    print('\n--- pathway_combined_visit (Hallmark + KEGG, 370 scores) ---')
    fr, preds = run_unimodal('pathway_combined_visit', comb_m)
    all_fold_results += fr; all_preds += preds

    # ── 7. Save ───────────────────────────────────────────────────────────────
    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(OUT_DIR, 'at20cm_pathway_visit_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'at20cm_pathway_visit_predictions.csv'), index=False)

    strategies = list(dict.fromkeys(r['strategy'] for r in all_fold_results))
    summary_list = []
    for strat in strategies:
        fr_s = [r for r in all_fold_results if r['strategy'] == strat]
        s = summarise(fr_s)
        s['strategy']   = strat
        s['n_visits']   = n_vis
        s['n_patients'] = n_pat
        summary_list.append(s)
    with open(os.path.join(OUT_DIR, 'at20cm_pathway_visit_summary.json'), 'w') as f:
        json.dump(summary_list, f, indent=2)

    # ── 8. Summary table ──────────────────────────────────────────────────────
    print('\n\n=== RESULTS SUMMARY ===')
    print(f"{'Strategy':<32} {'Visits':<8} {'Patients':<10} "
          f"{'AUC':<22} {'AP':<22} {'Accuracy'}")
    print('-' * 108)

    try:
        with open(os.path.join(OUT_DIR, 'at20cm_visit_summary.json')) as f:
            prev = {r['strategy']: r for r in json.load(f)}
        for strat in ['img_base_visit', 'rna_visit', 'concat_raw_visit']:
            r = prev.get(strat, {})
            if r:
                print(f"  {strat:<30} {r['n_visits']:<8} {r['n_patients']:<10} "
                      f"AUC={r['mean_auc']:.4f}±{r['std_auc']:.4f}  "
                      f"AP={r['mean_ap']:.4f}±{r['std_ap']:.4f}  "
                      f"Acc={r['mean_acc']:.4f}")
        with open(os.path.join(OUT_DIR, 'at20cm_hist_visit_summary.json')) as f:
            hv = json.load(f)[0]
        print(f"  {'hist_visit':<30} {hv['n_visits']:<8} {hv['n_patients']:<10} "
              f"AUC={hv['mean_auc']:.4f}±{hv['std_auc']:.4f}  "
              f"AP={hv['mean_ap']:.4f}±{hv['std_ap']:.4f}  "
              f"Acc={hv['mean_acc']:.4f}")
        print()
    except FileNotFoundError:
        pass

    for s in summary_list:
        print(f"  {s['strategy']:<30} {s['n_visits']:<8} {s['n_patients']:<10} "
              f"AUC={s['mean_auc']:.4f}±{s['std_auc']:.4f}  "
              f"AP={s['mean_ap']:.4f}±{s['std_ap']:.4f}  "
              f"Acc={s['mean_acc']:.4f}")

    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
