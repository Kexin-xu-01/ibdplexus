"""
Clinical variables CD vs UC classifier — at-20-cm proximity-matched cohort.

Cohort
------
Identical to 08b_train_at20cm_visit_level.py: patients whose at-20-cm H&E slide
and at-20-cm RNA-seq sample were collected within 7 days of each other. This is
the same ~817-patient / ~945-visit matched cohort used in the base visit series.

Clinical features operate at patient level (one row per patient, using patient-level
metadata). Where a patient has multiple matched visits the fold assignment and label
come from cv_splits_patients.csv — the clinical metadata does not change per visit.

Features (pre-diagnosis only — no treatment, surgery, or disease-specific scores)
--------
  Symptoms:      rectal_bleed, first_abdopain, first_daily_bm, stool_freq,
                 first_well_being, fecal_urgency, da6m
  Lifestyle:     smk (smoking)
  Labs:          hemoglobin_text, erythrocyte_sedimentation_text, neutrophil_text,
                 white_blood_text, platelets_text  (numeric extracted by regex)
  Comorbidities: cardiovascular_disease, hypertension, diabetes_type_i,
                 diabetes_type_ii, ckd, hiv, tuberculosis

Excluded (post-diagnosis / label-leaking)
-----------------------------------------
  Disease-specific scores: SCDAI, SES-CD, Mayo, PGA
  Montreal classification: crohn_s_disease_phenotype, cd_first_cd_location,
    uc_first_uc_phenotype
  Treatment: aminosalicylates, immunomodulators, biologics, corticosteroids, etc.
  Surgery:   number_of_ibd_surgeries, colectomy, resect_sb, ostomy_presence
  Demographics: age_at_diagnosis, gender, race_and_ethnicity
    (epidemiological associations, not diagnostic criteria — NICE NG130 / ECCO 2019)

Missing values
--------------
  Median imputation fitted on training fold only (no leakage into validation).

Outputs  /home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/results/
--------
  clinical_fold_metrics.csv
  clinical_patient_predictions.csv
  clinical_summary.json
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
TRANSCRIPTOMICS_DIR = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/'
                       'genestack/transcriptomics')
WSI_META    = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/'
               'IBD_meta_data_latest/wsi_metadata_raw.csv')
MAPPING_CSV = os.path.join(TRANSCRIPTOMICS_DIR,
              'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS  = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients_at20cm_matched.csv'
METADATA_CSV = '/home/jovyan/kgbk271-ibd-volume/metadata/merged_patient_metadata.csv'
OUT_DIR      = '/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/results'

AT20     = {'at 20 cm', 'At 20 cm'}
CDUC_RAW = ["Crohn's disease", 'Ulcerative colitis']
RF_PARAMS = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=2,
                 class_weight='balanced', n_jobs=-1, random_state=42)

# ── Feature definitions ────────────────────────────────────────────────────────
NUMERIC_COLS = [
    'rectal_bleed',     # 0–3 (bleeding severity)
    'first_abdopain',   # 0–3 (abdominal pain)
    'first_daily_bm',   # continuous (daily bowel movements delta)
    'first_well_being', # 0–4 (general wellbeing)
]

STOOL_FREQ_MAP = {
    'Normal': 0,
    '1-2 stools more than normal': 1,
    '3-4 stools more than normal': 2,
    '>4 stools more than normal': 3,
    'Not applicable, I have an ostomy': np.nan,
}

FECAL_URGENCY_MAP = {
    'Mild': 1,
    'Moderate': 2,
    'Moderately severe': 3,
    'Severe': 4,
    'Not applicable, I have an ostomy': np.nan,
}

DA6M_MAP = {
    '1. Absence of symptoms': 0,
    '2. Rarely active': 1,
    '3. Sometimes active': 2,
    '4. Occasionally active': 3,
    '5. Often active': 4,
    '6. Constantly active': 5,
}

BINARY_COLS = [
    'cardiovascular_disease', 'hypertension',
    'diabetes_type_i', 'diabetes_type_ii',
    'ckd', 'hiv', 'tuberculosis',
]

FEATURE_NAMES = (
    NUMERIC_COLS
    + ['stool_freq', 'fecal_urgency', 'da6m', 'smk']
    + BINARY_COLS
)


# ── Cohort: replicate 08b proximity-matched patient set ───────────────────────

def _norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


def _img_visit_dates(cv_pids):
    """Return DataFrame of (patient_id, visit_date) for at-20-cm H&E visits."""
    wsi = pd.read_csv(WSI_META)
    wsi = wsi[
        wsi['BIOSAMPLE_LOCATION'].isin(AT20) &
        wsi['diagnosis'].isin(CDUC_RAW) &
        wsi['deidentified_master_patient_id'].isin(cv_pids)
    ].copy()
    wsi['date'] = pd.to_datetime(wsi['Date Sample Collected'], dayfirst=True, errors='coerce')
    return (wsi[['deidentified_master_patient_id', 'date']]
            .rename(columns={'deidentified_master_patient_id': 'patient_id'})
            .dropna(subset=['date'])
            .drop_duplicates())


def _rna_visit_dates(cv_pids):
    """Return DataFrame of (patient_id, visit_date) for at-20-cm RNA visits (QC pass)."""
    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(_norm_dx)
    rna['date'] = pd.to_datetime(rna['sample_collected_date'], dayfirst=True, errors='coerce')
    rna = rna[
        rna['characteristics_bio_material'].isin(AT20) &
        rna['diagnosis_norm'].isin(CDUC_RAW) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pids)
    ].drop_duplicates(subset='visit_encounter_id', keep='first')
    return (rna[['deidentified_master_patient_id', 'date']]
            .rename(columns={'deidentified_master_patient_id': 'patient_id'})
            .dropna(subset=['date'])
            .drop_duplicates())


def get_matched_patients(cv_patients, max_gap_days=7):
    """
    Replicate the proximity-match from 08b_train_at20cm_visit_level.py.
    Returns the set of patient_ids that have at least one (H&E, RNA) visit
    pair from at-20-cm collected within max_gap_days of each other.
    """
    cv_pids = set(cv_patients['patient_id'])
    img_dates = _img_visit_dates(cv_pids)
    rna_dates = _rna_visit_dates(cv_pids)

    rna_by_pid = {pid: grp for pid, grp in rna_dates.groupby('patient_id')}

    matched_pids = set()
    for _, irow in img_dates.iterrows():
        pid = irow['patient_id']
        if pid not in rna_by_pid:
            continue
        gaps = (rna_by_pid[pid]['date'] - irow['date']).abs().dt.days
        if gaps.min() <= max_gap_days:
            matched_pids.add(pid)

    print(f'  Proximity-matched patients (≤{max_gap_days}d): {len(matched_pids)}')
    return matched_pids


# ── Feature matrix ─────────────────────────────────────────────────────────────

def extract_numeric(series):
    """Extract first numeric token from text field; NaN for non-numeric entries."""
    def _parse(v):
        if pd.isna(v):
            return np.nan
        m = re.search(r'[-+]?\d*\.?\d+', str(v))
        return float(m.group()) if m else np.nan
    return series.map(_parse)


def build_feature_matrix(cv_patients, matched_pids):
    cv_match = cv_patients[cv_patients['patient_id'].isin(matched_pids)].copy()
    meta     = pd.read_csv(METADATA_CSV)

    df = cv_match.merge(
        meta, left_on='patient_id', right_on='deidentified_master_patient_id',
        how='left', suffixes=('', '_meta'),
    )

    # Label: CD=0, UC=1
    df['label'] = (df['diagnosis'] == 'Ulcerative colitis').astype(int)

    # Encode features
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['stool_freq']    = df['stool_freq'].map(STOOL_FREQ_MAP)
    df['fecal_urgency'] = df['fecal_urgency'].map(FECAL_URGENCY_MAP)
    df['da6m']          = df['da6m'].map(DA6M_MAP)
    df['smk']           = df['smk'].map({'Yes': 1, 'No': 0})

    for col in BINARY_COLS:
        df[col] = df[col].map({'Y': 1, 'N': 0})

    out = df[['patient_id', 'label', 'fold'] + FEATURE_NAMES].copy()

    cd_n = (out['label'] == 0).sum()
    uc_n = (out['label'] == 1).sum()
    print(f'  Cohort: {len(out)} patients  (CD {cd_n}, UC {uc_n})')
    print(f'  Feature coverage:')
    for col in FEATURE_NAMES:
        n   = out[col].notna().sum()
        pct = n / len(out)
        print(f'    {col:<38} {n:>4}/{len(out)}  ({pct:.1%})')

    return out


# ── CV ─────────────────────────────────────────────────────────────────────────

def run_cv(df):
    X_raw = df[FEATURE_NAMES].values.astype(float)
    y     = df['label'].values
    folds = df['fold'].values
    ids   = df['patient_id'].values

    fold_results, all_preds = [], []

    for fold in range(5):
        tr = folds != fold
        va = folds == fold

        # Fit imputer on training fold only — no leakage
        imp  = SimpleImputer(strategy='median')
        X_tr = imp.fit_transform(X_raw[tr])
        X_va = imp.transform(X_raw[va])

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
            strategy='clinical', fold=fold,
            n_train=int(tr.sum()), n_val=int(va.sum()),
            auc=round(auc, 4), ap=round(ap, 4),
            accuracy=round(cr['accuracy'], 4),
            cd_f1=round(cr['CD']['f1-score'], 4),
            uc_f1=round(cr['UC']['f1-score'], 4),
            tn=int(cm[0, 0]), fp=int(cm[0, 1]),
            fn=int(cm[1, 0]), tp=int(cm[1, 1]),
        ))
        for pid, yt, yp, ys in zip(ids[va], y_val, y_pred, y_score):
            all_preds.append(dict(
                patient_id=pid, strategy='clinical', fold=fold,
                true_label=int(yt), pred_label=int(yp),
                prob_uc=round(float(ys), 5),
            ))

        print(f'    clinical  fold {fold}: AUC={auc:.4f}  AP={ap:.4f}  '
              f'Acc={cr["accuracy"]:.4f}  '
              f'(n_train={tr.sum()}, n_val={va.sum()})')

    return fold_results, all_preds


def summarise(fold_results):
    aucs = [r['auc'] for r in fold_results]
    aps  = [r['ap']  for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    return dict(
        mean_auc=round(np.mean(aucs), 4), std_auc=round(np.std(aucs, ddof=1), 4),
        mean_ap=round(np.mean(aps),   4), std_ap=round(np.std(aps,  ddof=1), 4),
        mean_acc=round(np.mean(accs), 4), std_acc=round(np.std(accs, ddof=1), 4),
    )


# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv_patients = pd.read_csv(CV_PATIENTS)
    matched_pids = set(cv_patients['patient_id'])
    cd_n = (cv_patients['diagnosis'] == "Crohn's disease").sum()
    uc_n = (cv_patients['diagnosis'] == "Ulcerative colitis").sum()
    print(f'=== CV split: {len(cv_patients)} patients  (CD {cd_n}, UC {uc_n}) ===')

    print('\n=== Building clinical feature matrix ===')
    df = build_feature_matrix(cv_patients, matched_pids)

    print(f'\n--- clinical ({len(FEATURE_NAMES)} features, 5-fold patient-level CV) ---')
    fold_results, all_preds = run_cv(df)

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(fold_results).to_csv(
        os.path.join(OUT_DIR, 'clinical_fold_metrics.csv'), index=False)
    pd.DataFrame(all_preds).to_csv(
        os.path.join(OUT_DIR, 'clinical_patient_predictions.csv'), index=False)

    s = summarise(fold_results)
    s['strategy']   = 'clinical'
    s['n_patients'] = len(df)
    s['n_features'] = len(FEATURE_NAMES)
    s['features']   = FEATURE_NAMES

    with open(os.path.join(OUT_DIR, 'clinical_summary.json'), 'w') as f:
        json.dump([s], f, indent=2)

    print(f'\n=== CLINICAL SUMMARY ===')
    print(f"  N={s['n_patients']}  features={s['n_features']}")
    print(f"  AUC  {s['mean_auc']:.4f} ± {s['std_auc']:.4f}")
    print(f"  AP   {s['mean_ap']:.4f} ± {s['std_ap']:.4f}")
    print(f"  Acc  {s['mean_acc']:.4f} ± {s['std_acc']:.4f}")
    print(f'\nResults saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
