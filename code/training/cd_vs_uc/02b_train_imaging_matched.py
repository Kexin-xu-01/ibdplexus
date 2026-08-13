"""
Re-run imaging Random Forest (CD vs UC) restricted to the 997 patients that also
have colon RNAseq, enabling a head-to-head comparison with transcriptomics_vst.

The original 02_train_random_forest.py uses all 1,250 patients in the imaging CV
split (2,121 slides). This script filters to the 997-patient matched subset
(1,758 slides) so that both imaging and transcriptomics models are evaluated on
an identical patient cohort with identical fold assignments.

Inputs
------
- /home/jovyan/kgbk271-ibd-volume/training/cv_splits_slides.csv
- /home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv
- /home/jovyan/shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/
      ibd_21183_omics_patient_mapping_genestack.csv
      GSF1478941_sample_combined_from1stRun.tsv__metadata.csv
- prism2_base/ and prism2_diagnostic/ embedding directories

Outputs  (under /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/)
--------
- prism2_base_matched_fold_metrics.csv
- prism2_base_matched_slide_predictions.csv
- prism2_base_matched_summary.json
- prism2_diagnostic_matched_fold_metrics.csv
- prism2_diagnostic_matched_slide_predictions.csv
- prism2_diagnostic_matched_summary.json
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
MAPPING_CSV  = os.path.join(TRANSCRIPTOMICS_DIR,
               'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META  = os.path.join(TRANSCRIPTOMICS_DIR,
               'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS  = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
CV_SLIDES    = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_slides.csv'
EMB_DIRS = {
    'prism2_base': ('/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
                    '20x_224px_0px_overlap/prism2_base'),
    'prism2_diagnostic': ('/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
                          '20x_224px_0px_overlap/prism2_diagnostic'),
}
OUT_DIR = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/02_04_imaging_allsites/results'

COLON_LOCS = {
    'at 20 cm', 'At 20 cm', 'Cecum', 'Rectum',
    'Ascending Colon', 'Descending Colon', 'Sigmoid Colon', 'Transverse Colon',
}
RF_PARAMS = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)


def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


def get_rna_patients(cv_patients):
    """Return set of patient IDs that have colon RNAseq (= transcriptomics cohort)."""
    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(norm_dx)
    cv_pats = set(cv_patients['patient_id'])
    matched = rna[
        rna['characteristics_bio_material'].isin(COLON_LOCS) &
        rna['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pats)
    ]
    return set(matched['deidentified_master_patient_id'])


def load_embeddings(slide_ids, emb_dir):
    X, valid_ids = [], []
    for sid in slide_ids:
        path = os.path.join(emb_dir, f'{sid}.h5')
        if os.path.exists(path):
            with h5py.File(path, 'r') as h:
                X.append(h['features'][:])
            valid_ids.append(sid)
    return np.array(X, dtype=np.float32), valid_ids


def run_cv(model_name, emb_dir, slides_df):
    tag = f'{model_name}_matched'
    print(f"\n{'='*60}")
    print(f"  {tag}  (CD=0, UC=1, positive=UC)")
    print(f"{'='*60}")

    id2label = dict(zip(slides_df['slide_id'], slides_df['label']))
    id2diag  = dict(zip(slides_df['slide_id'], slides_df['diagnosis']))
    fold_results, all_preds = [], []

    for fold in range(5):
        train_df = slides_df[slides_df['fold'] != fold]
        val_df   = slides_df[slides_df['fold'] == fold]

        X_train, train_ids = load_embeddings(train_df['slide_id'].tolist(), emb_dir)
        X_val,   val_ids   = load_embeddings(val_df['slide_id'].tolist(),   emb_dir)
        y_train = np.array([id2label[s] for s in train_ids])
        y_val   = np.array([id2label[s] for s in val_ids])

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train, y_train)

        proba   = clf.predict_proba(X_val)
        uc_col  = list(clf.classes_).index(1)
        y_score = proba[:, uc_col]
        y_pred  = clf.predict(X_val)

        auc = roc_auc_score(y_val, y_score)
        ap  = average_precision_score(y_val, y_score, pos_label=1)
        cr  = classification_report(y_val, y_pred,
                                    target_names=['CD', 'UC'], output_dict=True)
        cm  = confusion_matrix(y_val, y_pred)

        fold_res = dict(
            fold=fold, n_train=int(len(y_train)), n_val=int(len(y_val)),
            auc=round(auc, 4), ap=round(ap, 4), accuracy=round(cr['accuracy'], 4),
            cd_precision=round(cr['CD']['precision'], 4),
            cd_recall=round(cr['CD']['recall'], 4),
            cd_f1=round(cr['CD']['f1-score'], 4),
            uc_precision=round(cr['UC']['precision'], 4),
            uc_recall=round(cr['UC']['recall'], 4),
            uc_f1=round(cr['UC']['f1-score'], 4),
            tn=int(cm[0, 0]), fp=int(cm[0, 1]),
            fn=int(cm[1, 0]), tp=int(cm[1, 1]),
        )
        fold_results.append(fold_res)
        print(f"  Fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  Acc={cr['accuracy']:.4f}  "
              f"CD-F1={cr['CD']['f1-score']:.4f}  UC-F1={cr['UC']['f1-score']:.4f}")

        for sid, yt, yp, ys in zip(val_ids, y_val, y_pred, y_score):
            all_preds.append(dict(slide_id=sid, fold=fold, diagnosis=id2diag[sid],
                                  true_label=int(yt), pred_label=int(yp),
                                  prob_uc=round(float(ys), 5)))

    aucs = [r['auc'] for r in fold_results]
    aps  = [r['ap']  for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    print(f"\n  AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  AP:       {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    pd.DataFrame(fold_results).to_csv(
        os.path.join(OUT_DIR, f'{tag}_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, f'{tag}_slide_predictions.csv'), index=False)

    summary = dict(
        model=tag, cohort='matched_rna_patients',
        n_patients=int(slides_df['patient_id'].nunique()),
        n_slides=int(len(slides_df)),
        mean_auc=round(float(np.mean(aucs)), 4),
        std_auc=round(float(np.std(aucs)), 4),
        mean_ap=round(float(np.mean(aps)), 4),
        std_ap=round(float(np.std(aps)), 4),
        mean_acc=round(float(np.mean(accs)), 4),
        std_acc=round(float(np.std(accs)), 4),
        folds=fold_results,
        label_encoding={'CD': 0, 'UC': 1},
        positive_class='UC', score_column='prob_uc',
        rf_params=RF_PARAMS,
    )
    with open(os.path.join(OUT_DIR, f'{tag}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cv_patients = pd.read_csv(CV_PATIENTS)
    rna_pats    = get_rna_patients(cv_patients)
    print(f"Patients with colon RNAseq (transcriptomics cohort): {len(rna_pats)}")

    slides_all     = pd.read_csv(CV_SLIDES)
    slides_all['label'] = (slides_all['diagnosis'] == 'Ulcerative colitis').astype(int)
    slides_matched = slides_all[slides_all['patient_id'].isin(rna_pats)].copy()
    print(f"Imaging slides (all):     {len(slides_all)} from {slides_all['patient_id'].nunique()} patients")
    print(f"Imaging slides (matched): {len(slides_matched)} from {slides_matched['patient_id'].nunique()} patients")

    results = {}
    for model_name, emb_dir in EMB_DIRS.items():
        results[model_name] = run_cv(model_name, emb_dir, slides_matched)

    rna_s = json.load(open(os.path.join('/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/03_05_transcriptomics_allsites', 'results', 'transcriptomics_vst_summary.json')))

    print('\n\n=== HEAD-TO-HEAD (matched 997-patient cohort) ===')
    print(f"{'Model':<30} {'Modality':<14} {'N':<8} {'AUC':<20} {'AP':<20} {'Accuracy'}")
    for name, s, mod, n in [
        ('transcriptomics_vst',          rna_s,          'RNAseq VST', '1,728 samp'),
        ('prism2_base_matched',          results['prism2_base'],       'WSI embed',  f"{len(slides_matched)} slides"),
        ('prism2_diagnostic_matched',    results['prism2_diagnostic'], 'WSI embed',  f"{len(slides_matched)} slides"),
    ]:
        print(f"{name:<30} {mod:<14} {n:<8} "
              f"{s['mean_auc']:.4f}±{s['std_auc']:.4f}   "
              f"{s['mean_ap']:.4f}±{s['std_ap']:.4f}   "
              f"{s['mean_acc']:.4f}±{s['std_acc']:.4f}")

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
