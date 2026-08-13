"""
Build patient-level 5-fold stratified cross-validation splits for colon CD vs UC.

Inputs
------
- /home/jovyan/shared-data/ibd_plexus_sparc_raw/image/IBD_meta_data_latest/wsi_metadata_raw.csv
- /home/jovyan/shared-data/ibd_plexus_sparc_processed/omics_samples.csv
- /home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/
      20x_224px_0px_overlap/prism2_base/   (used to find available slide IDs)

Outputs
-------
- /home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv
- /home/jovyan/kgbk271-ibd-volume/training/cv_splits_slides.csv

Stratification
--------------
MultilabelStratifiedKFold on: diagnosis × sex × age-at-diagnosis bin.
All slides for a patient are assigned to the same fold (no leakage).
Patients with conflicting CD/UC labels are excluded.
"""

import os
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# ── paths ─────────────────────────────────────────────────────────────────────
WSI_META   = '/home/jovyan/shared-data/ibd_plexus_sparc_raw/image/IBD_meta_data_latest/wsi_metadata_raw.csv'
OMICS_CSV  = '/home/jovyan/shared-data/ibd_plexus_sparc_processed/omics_samples.csv'
H5_DIR     = ('/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed/'
              '20x_224px_0px_overlap/prism2_base')
OUT_DIR    = '/home/jovyan/kgbk271-ibd-volume/training'
N_FOLDS    = 5
RANDOM_SEED = 42

COLON_LOCS = {
    'at 20 cm', 'At 20 cm', 'Cecum', 'Rectum',
    'Ascending Colon', 'Descending Colon', 'Sigmoid Colon', 'Transverse Colon',
}


def age_bin(a):
    if pd.isna(a):  return (0, 0, 0)   # Missing
    if a < 20:      return (1, 0, 0)
    if a <= 35:     return (0, 1, 0)
    return (0, 0, 1)


def sex_bins(g):
    if g == 'Female': return (1, 0)
    if g == 'Male':   return (0, 1)
    return (0, 0)                       # Missing


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load WSI metadata and filter to colon slides
    wsi = pd.read_csv(WSI_META)
    wsi = wsi[wsi['BIOSAMPLE_LOCATION'].isin(COLON_LOCS)].copy()
    wsi['slide_id'] = wsi['IMAGE_VSI'].str.replace('.vsi', '', regex=False)

    # 2. Keep only slides with h5 embeddings
    h5_ids = {f[:-3] for f in os.listdir(H5_DIR) if f.endswith('.h5')}
    slides = wsi[wsi['slide_id'].isin(h5_ids)].copy()
    print(f"Colon slides with embeddings: {len(slides)}")

    # 3. Exclude patients with conflicting CD/UC labels
    dx_per_pat = slides.groupby('deidentified_master_patient_id')['diagnosis'].nunique()
    ambiguous  = set(dx_per_pat[dx_per_pat > 1].index)
    print(f"Excluding {len(ambiguous)} patients with conflicting labels")
    slides = slides[~slides['deidentified_master_patient_id'].isin(ambiguous)]

    # 4. Deduplicate slide IDs (keep first)
    slides = slides.drop_duplicates(subset='slide_id', keep='first').copy()
    print(f"Slides after dedup: {len(slides)}")

    # 5. Patient-level diagnosis table
    pat_dx = (slides.groupby('deidentified_master_patient_id')['diagnosis']
              .first().reset_index()
              .rename(columns={'deidentified_master_patient_id': 'patient_id'}))

    # 6. Merge patient-level metadata from omics
    omics = pd.read_csv(OMICS_CSV,
                        usecols=['deidentified_master_patient_id',
                                 'birth_year', 'age_at_diagnosis', 'gender'])
    omics_pat = (omics.groupby('deidentified_master_patient_id').first().reset_index()
                 .rename(columns={'deidentified_master_patient_id': 'patient_id'}))
    patients = pat_dx.merge(omics_pat, on='patient_id', how='left')

    # 7. Build stratification label matrix
    patients['dx_cd'] = (patients['diagnosis'] == "Crohn's disease").astype(int)
    patients[['sex_female', 'sex_male']] = pd.DataFrame(
        patients['gender'].map(sex_bins).tolist(), index=patients.index)
    patients[['age_lt20', 'age_20_35', 'age_gt35']] = pd.DataFrame(
        patients['age_at_diagnosis'].map(age_bin).tolist(), index=patients.index)

    strat_cols = ['dx_cd', 'sex_female', 'sex_male', 'age_lt20', 'age_20_35', 'age_gt35']
    Y = patients[strat_cols].values

    # 8. Run MultilabelStratifiedKFold
    mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_assignments = np.zeros(len(patients), dtype=int)
    for fold, (_, val_idx) in enumerate(mskf.split(np.zeros(len(patients)), Y)):
        fold_assignments[val_idx] = fold
    patients['fold'] = fold_assignments

    # 9. Propagate fold to slide level
    slides_fold = slides[['deidentified_master_patient_id', 'slide_id', 'diagnosis',
                           'BIOSAMPLE_LOCATION', 'MACROSCOPIC_APPEARANCE']].copy()
    slides_fold = slides_fold.merge(
        patients[['patient_id', 'fold', 'gender', 'age_at_diagnosis']],
        left_on='deidentified_master_patient_id', right_on='patient_id', how='left')

    # 10. Sanity checks
    assert (patients.groupby('patient_id')['fold'].nunique() > 1).sum() == 0, \
        "Patient leakage detected!"
    assert slides_fold['slide_id'].nunique() == len(slides_fold), \
        "Duplicate slide IDs in output!"
    print("✓ No patient leakage | ✓ No duplicate slides")

    # 11. Save
    patients[['patient_id', 'diagnosis', 'gender', 'age_at_diagnosis', 'fold']].to_csv(
        os.path.join(OUT_DIR, 'cv_splits_patients.csv'), index=False)
    slides_fold[['patient_id', 'slide_id', 'diagnosis', 'BIOSAMPLE_LOCATION',
                 'MACROSCOPIC_APPEARANCE', 'gender', 'age_at_diagnosis', 'fold']].to_csv(
        os.path.join(OUT_DIR, 'cv_splits_slides.csv'), index=False)

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  cv_splits_patients.csv  — {len(patients)} patients")
    print(f"  cv_splits_slides.csv    — {len(slides_fold)} slides")
    print(f"\nDiagnosis distribution:")
    print(patients['diagnosis'].value_counts().to_string())
    print(f"\nFold balance (patients):")
    print(pd.crosstab(patients['fold'], patients['diagnosis']).to_string())


if __name__ == '__main__':
    main()
