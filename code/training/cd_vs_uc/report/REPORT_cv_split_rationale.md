# Cross-Validation Split Design: Rationale

**Files:**
- `cv_splits_patients.csv` — 1,250 patients, full cohort
- `cv_splits_patients_at20cm_matched.csv` — 817 patients, at-20-cm matched cohort
- `01_build_cv_splits.py` — builds the full split
- `01b_build_cv_splits_at20cm_matched.py` — builds the matched subset

---

## 1. Patient-Level Splitting

The unit of splitting is the **patient**, not the visit or slide.

A single patient can contribute multiple visits over time. If visit-level splitting were used, different visits from the same patient could appear in both the training set and the validation set. Because the same patient's visits share:

- identical germline genetics
- highly correlated disease trajectory
- potentially shared imaging characteristics (same staining protocol, same endoscopist)

any model trained on one visit from a patient would have an unfair advantage predicting another visit from the same patient. This would inflate AUC and make the result non-generalisable to new patients.

Patient-level splitting ensures that **all visits from a patient land in exactly one fold**. The classifier is therefore evaluated on patients it has never seen during training — the correct test of generalisation.

---

## 2. Five-Fold Cross-Validation

Five folds were chosen as a balance between:

- **Variance**: more folds → smaller validation sets → noisier per-fold AUC estimates. Five folds gives ~160–170 patients per validation fold, which is sufficient to estimate AUC with reasonable precision.
- **Bias**: fewer folds → models trained on less data → pessimistic AUC estimates. With 817 matched patients, 5-fold leaves ~654 patients for training per fold (~80%).
- **Compute**: 5 folds × 10 arms × 2 classifiers = 100 training runs. Ten folds would double this.

Fold assignments are fixed in `cv_splits_patients.csv` and never recomputed. All experiments use the same assignments, making all arm comparisons directly comparable.

---

## 3. Why Stratify on Diagnosis

CD:UC prevalence in this cohort is approximately 2:1 (CD 67%, UC 33%). Without stratification, random chance could place more UC patients in some folds than others, introducing fold-to-fold variation that has nothing to do with model performance.

Stratification by diagnosis ensures that every fold has the same ~2:1 CD:UC ratio, so AUC variation across folds reflects genuine uncertainty in the model rather than label imbalance artefacts.

In the full 1,250-patient split this gives exactly 168 CD and 82 UC per fold. In the 817-patient matched subset, folds are slightly uneven (159–170 total) because not all 250 patients per fold had a matched visit — but the CD:UC ratio is preserved within each fold.

---

## 4. Why Build the Split Once on the Full Cohort

The split was built on all 1,250 patients with CV-eligible diagnoses, not just the 817 with matched imaging+RNA visits. This design has two advantages:

**Consistency across modalities.** Some arms use imaging only, some RNA only, some both. If each arm used its own split, fold 0 in the imaging-only arm would contain different patients than fold 0 in the RNA-only arm, making cross-arm comparisons invalid. A single shared split ensures every arm is evaluated on identical held-out patient sets.

**Extensibility.** Any future arm — a new modality, a new biopsy site, a new feature set — can reuse the same fold assignments as long as patients overlap with the 1,250-patient base. No re-splitting required.

---

## 5. The Matched Cohort Subset (cv_splits_patients_at20cm_matched.csv)

The 817-patient file is a filtered view of the full split, not an independently constructed split. It exists for convenience — scripts can load it and skip the cohort-assembly step — but the fold numbers are identical to those in the full file.

**Matching criteria** (replicated from `08b_train_at20cm_visit_level.py`):
- Biopsy site: at 20 cm only (`{'at 20 cm', 'At 20 cm'}`)
- Diagnosis: CD or UC
- QC: RNA sample QC ≠ 'fail'
- Imaging: at least one slide with a prism2_base H5 embedding
- Proximity: closest imaging and RNA visit from the same patient ≤ 7 days apart

The 7-day window was chosen because the median gap between genuinely unpaired visits from the same patient is ~500 days, so a 7-day threshold recovers date-entry discrepancies (e.g. off-by-one) without pairing visits from different clinic encounters.

---

## 6. Fold Size Imbalance in the Matched Subset

| Fold | Patients |
|------|---------|
| 0 | 160 |
| 1 | 159 |
| 2 | 163 |
| 3 | 170 |
| 4 | 165 |

The imbalance (±11 patients across folds) is a consequence of the matching filter being applied after the split was fixed. It cannot be corrected without re-splitting, which would break comparability with the full 1,250-patient cohort. The imbalance is small enough (~7%) that it has negligible effect on AUC estimation.

---

## 7. What the Split Does Not Protect Against

**Global preprocessing leakage.** CombatSeq batch correction and VST normalisation were applied to all 3,289 RNA samples before any CV split was defined. This means the normalisation parameters (dispersion estimates, batch offsets) were estimated on data that includes held-out samples. All arms are affected equally, so relative comparisons remain valid, but absolute AUC values may be marginally optimistic relative to a fully prospective deployment.

**BulkFormer pre-training.** BulkFormer was pre-trained on TCGA (not IBD), so it carries no IBD label information. However, it was applied to all 3,288 IBD samples globally before CV splits. This is analogous to using a pre-trained ImageNet model for inference — the encoder parameters do not depend on IBD labels, so this is not a leakage concern in the usual sense.
