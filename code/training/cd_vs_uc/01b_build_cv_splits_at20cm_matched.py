"""
Build a patient-level CV split file restricted to the at-20-cm matched cohort.

Filters cv_splits_patients.csv (1,250 patients, 5-fold) down to the 817 patients
that have at least one proximity-matched imaging + RNA visit (≤7 days apart) at
the at-20-cm biopsy site.  Fold assignments are inherited unchanged.

Mirrors the cohort assembly logic in 08b_train_at20cm_visit_level.py exactly:
  - Imaging: wsi_metadata_raw.csv filtered to AT20 / CD+UC / CV patients,
    slides that have a prism2_base H5 embedding, grouped by (patient, date)
  - RNA: omics mapping + sample QC filtered to AT20 / CD+UC / CV patients,
    deduplicated to one row per encounter
  - Match: for each imaging visit find the closest RNA visit from the same patient
    within 7 days; patients with ≥1 such match are included

Output
------
  /home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients_at20cm_matched.csv
  Columns: patient_id, diagnosis, gender, age_at_diagnosis, fold
"""

import os
import pandas as pd

TRANSCRIPTOMICS_DIR = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/'
                       'genestack/transcriptomics')
WSI_META    = ('/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/'
               'IBD_meta_data_latest/wsi_metadata_raw.csv')
MAPPING_CSV = os.path.join(TRANSCRIPTOMICS_DIR,
              'ibd_21183_omics_patient_mapping_genestack.csv')
SAMPLE_META = os.path.join(TRANSCRIPTOMICS_DIR,
              'GSF1478941_sample_combined_from1stRun.tsv__metadata.csv')
CV_PATIENTS = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv'
EMB_BASE    = ('/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
               '20x_224px_0px_overlap/prism2_base')
OUT_CSV     = '/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients_at20cm_matched.csv'

AT20        = {'at 20 cm', 'At 20 cm'}
CDUC_RAW    = ["Crohn's disease", 'Ulcerative colitis']
MAX_GAP     = 7   # days


def norm_dx(d):
    if isinstance(d, str):
        if 'Crohn' in d:      return "Crohn's disease"
        if 'Ulcerative' in d: return 'Ulcerative colitis'
    return None


def main():
    cv_patients = pd.read_csv(CV_PATIENTS)
    pat_df  = cv_patients.set_index('patient_id')
    cv_pids = set(pat_df.index)

    # ── Imaging visits ────────────────────────────────────────────────────────
    wsi = pd.read_csv(WSI_META)
    wsi = wsi[
        wsi['BIOSAMPLE_LOCATION'].isin(AT20) &
        wsi['diagnosis'].isin(CDUC_RAW) &
        wsi['deidentified_master_patient_id'].isin(cv_pids)
    ].copy()
    wsi['date']     = pd.to_datetime(wsi['Date Sample Collected'], dayfirst=True, errors='coerce')
    wsi['slide_id'] = wsi['IMAGE_VSI'].str.replace('.vsi', '', regex=False)
    wsi = wsi[wsi['slide_id'].apply(
        lambda s: os.path.exists(os.path.join(EMB_BASE, f'{s}.h5')))]
    print(f'Slides with H5 embedding: {len(wsi)}')

    img_visits = []
    for (pid, vdate), _ in wsi.groupby(
            [wsi['deidentified_master_patient_id'], wsi['date'].dt.date]):
        if pid not in pat_df.index or pd.isna(vdate):
            continue
        img_visits.append({'patient_id': pid, 'visit_date': str(vdate)})
    img_df = pd.DataFrame(img_visits)
    img_df['_date'] = pd.to_datetime(img_df['visit_date'])
    print(f'Imaging visits: {len(img_df)}, patients: {img_df["patient_id"].nunique()}')

    # ── RNA visits ────────────────────────────────────────────────────────────
    mapping     = pd.read_csv(MAPPING_CSV)
    sample_meta = pd.read_csv(SAMPLE_META).rename(columns={'Name': 'SampleID'})
    rna = mapping.merge(
        sample_meta[['SampleID', 'diagnosis', 'Sample QC']], on='SampleID', how='left')
    rna['diagnosis_norm'] = rna['diagnosis'].map(norm_dx)
    rna['date'] = pd.to_datetime(rna['sample_collected_date'], dayfirst=True, errors='coerce')
    rna = rna[
        rna['characteristics_bio_material'].isin(AT20) &
        rna['diagnosis_norm'].isin(["Crohn's disease", 'Ulcerative colitis']) &
        (rna['Sample QC'] != 'fail') &
        rna['deidentified_master_patient_id'].isin(cv_pids)
    ].drop_duplicates(subset='visit_encounter_id', keep='first').copy()
    rna = rna.dropna(subset=['date'])
    rna_df = rna.rename(columns={'deidentified_master_patient_id': 'patient_id',
                                  'date': '_date'})[['patient_id', '_date']]
    print(f'RNA visits: {len(rna_df)}, patients: {rna_df["patient_id"].nunique()}')

    # ── Proximity match ───────────────────────────────────────────────────────
    rna_by_pid = {pid: grp for pid, grp in rna_df.groupby('patient_id')}
    matched_pids, matched_visits = set(), 0
    for _, irow in img_df.iterrows():
        pid = irow['patient_id']
        if pid not in rna_by_pid:
            continue
        cands = rna_by_pid[pid].copy()
        cands['_gap'] = (cands['_date'] - irow['_date']).abs().dt.days
        if cands['_gap'].min() <= MAX_GAP:
            matched_pids.add(pid)
            matched_visits += 1

    print(f'Matched visits: {matched_visits}, patients: {len(matched_pids)}')

    # ── Write output ──────────────────────────────────────────────────────────
    sub = (cv_patients[cv_patients['patient_id'].isin(matched_pids)]
           .copy().reset_index(drop=True))
    print(f'\nOutput: {len(sub)} patients')
    print(sub['diagnosis'].value_counts().to_string())
    print(sub['fold'].value_counts().sort_index().to_string())
    sub.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')


if __name__ == '__main__':
    main()
