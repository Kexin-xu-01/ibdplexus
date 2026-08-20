# Clinical Variable Feature Selection Plan — CD vs UC

## Objective

Train a clinical-variable-only classifier on the same 823-patient CV split (5-fold,
patient-level), using only variables that a gastroenterologist would have access to
**at the time of diagnosis** — before any disease-specific classification, treatment,
or surgery is assigned.

Metadata source: `/home/jovyan/kgbk271-ibd-volume/metadata/merged_patient_metadata.csv`
CV splits: `/home/jovyan/kgbk271-ibd-volume/training/cv_splits_patients.csv`

---

## Variables Considered and Decisions

### EXCLUDED — Post-Diagnosis or Circular (label leakage)

These variables are assigned **after** a CD or UC label is given, or directly encode
the diagnosis:

| Variable | Reason |
|---|---|
| `crohn_s_disease_phenotype` (B1/B2/B3) | Montreal CD classification — requires CD diagnosis |
| `cd_first_cd_phenotype`, `cd_first_cd_location` | CD-specific Montreal classification |
| `uc_first_uc_phenotype` | UC-specific extent classification (E1/E2/E3) |
| `cd_pga`, `uc_pga` (and `_cat`) | Physician global assessments labeled by disease type |
| `cd_first_scdai_score/category`, `cd_first_ses_score` | SCDAI/SES-CD only computed after CD diagnosis |
| `uc_first_mayo_*` scores | Mayo scores only computed after UC diagnosis |
| `cd_extraintestinal_manifestation`, `uc_extraintestinal_manifestation` | Disease-labeled EIM columns |
| `aminosalicylates`, `immunomodulators`, `biologics`, `corticosteroids`, `jak_inhibitor`, `antibiotics`, `combination_treatment_category` | Treatment started after diagnosis; 5-ASA near-exclusively UC — direct label leakage |
| `number_of_ibd_surgeries`, `resect_sb`, `colectomy`, `ostomy_presence`, `location_resection` | Post-diagnosis surgeries; small-bowel resection implies CD |
| `time_to_next_surgery_from_diagnosis_years` | Explicitly post-diagnosis |
| `disease_dur` | Duration since diagnosis |

### EXCLUDED — Demographics (not diagnostic criteria per guidelines)

Age, gender, and race are **epidemiological associations**, not clinical diagnostic
criteria. No major IBD guideline (NICE NG130, ECCO 2019, ACG, BSG) uses these to
distinguish CD from UC.

**Data evidence of sampling imbalance:**

| Variable | Observation | Verdict |
|---|---|---|
| `age_at_diagnosis` | CD mean 25.8y vs UC 29.9y; CD:UC ratio in <18 group is 3.5:1 vs 2:1 overall — inflated by SPARC recruitment of pediatric-onset CD | Sampling artifact + epidemiology; **exclude** |
| `gender` | CD 56.7% F vs UC 50.7% F; Chi2 p=0.076 — **not statistically significant** | Not a diagnostic criterion; **exclude** |
| `race_and_ethnicity` | White vs non-White: p=1.0 (no difference). Within small non-White groups (n≈27–52) ratios vary wildly — unreliable at this sample size | Not a diagnostic criterion; potential bias; **exclude** |
| `race_emr`, `ethnicity_emr`, `birth_year` | Same reasoning | **exclude** |

**Clinical guidelines justification:**
NICE NG130 (Crohn's disease), NICE CG166 (UC), ECCO 2019, and ACG guidelines base
CD vs UC differentiation on endoscopy, histology, imaging, and presenting symptoms —
not patient demographics. Clinicians do not conclude "probably CD because patient is
young and female."

### EXCLUDED — Too Sparse (< 25% coverage)

| Variable | Coverage | Decision |
|---|---|---|
| `creactive_text` | 0.6% | Exclude |
| `crp` | 19.2% | Exclude |
| `wbc` | 8.8% | Exclude |
| `fcp` | 25.5% | Borderline — exclude for now |
| `fecal_calprotectin_text` | 16.1% | Exclude |

---

## INCLUDED Feature Set (17 features, pre-diagnosis)

### Presenting Symptoms (85–90% coverage)

| Feature | Type | Coverage | Clinical rationale |
|---|---|---|---|
| `rectal_bleed` | Ordinal 0–3 | 85% | **Strongest discriminator**: UC is defined by continuous mucosal inflammation from rectum; CD may have little/no rectal bleeding |
| `first_abdopain` | Ordinal 0–3 | 84% | Abdominal pain more common/severe in CD (transmural, RLQ, small bowel involvement) |
| `first_daily_bm` | Continuous | 85% | Large separation: UC mean 1.74 vs CD mean −0.49 (bloody diarrhea pattern vs CD constipation/obstruction mix) |
| `stool_freq` | Ordinal categorical | 85% | Stool frequency category (Normal / 1-2 more / 3-4 more / >4 more / ostomy) |
| `first_well_being` | Ordinal 0–4 | 84% | Overall wellbeing — disease burden proxy |

### Symptoms with Moderate Coverage (50–55%)

| Feature | Type | Coverage | Clinical rationale |
|---|---|---|---|
| `fecal_urgency` | Ordinal categorical | 55% | Rectal urgency is a hallmark of UC proctitis; less characteristic of CD |
| `da6m` | Ordinal categorical | 51% | Disease activity pattern over 6 months (relapsing vs constant) |

### Labs (~40% coverage — include with missingness handling)

| Feature | Type | Coverage | Clinical rationale |
|---|---|---|---|
| `hemoglobin_text` | Numeric string | 40% | UC → blood-loss iron-deficiency anemia; CD → malabsorption (B12, folate); both cause anemia but different profiles |
| `erythrocyte_sedimentation_text` | Numeric string | 40% | ESR elevated in active IBD; CD tends higher than UC for same clinical severity |
| `neutrophil_text` | Numeric string | 40% | Neutrophilia as acute inflammation marker |
| `white_blood_text` | Numeric string | 39% | WBC elevation in active disease |
| `platelets_text` | Numeric string | 40% | Reactive thrombocytosis more common in UC |

**Note:** These columns contain mixed formats (numeric strings + text like "Negative",
"Adequate"). Require regex-based numeric extraction with NaN for non-numeric values.

### Comorbidities (~86% coverage)

| Feature | Type | Coverage | Rationale |
|---|---|---|---|
| `cardiovascular_disease` | Binary Y/N | 86% | Baseline health status |
| `hypertension` | Binary Y/N | 86% | Baseline health status |
| `diabetes_type_i` | Binary Y/N | 86% | T1D is an autoimmune disease — immune dysregulation overlap |
| `diabetes_type_ii` | Binary Y/N | 86% | Metabolic comorbidity |
| `ckd` | Binary Y/N | 86% | Baseline renal function |
| `hiv` | Binary Y/N | 86% | Immune status baseline |
| `tuberculosis` | Binary Y/N | 86% | Relevant for anti-TNF safety but also baseline immune history |

### Smoking (Borderline — Include with caveat)

| Feature | Type | Coverage | Notes |
|---|---|---|---|
| `smk` | Binary Yes/No | 86% | Current smoking: protective in UC, risk factor in CD. Epidemiological, but ECCO guidelines mention it as a modifiable factor. CD 10% smokers vs UC 4%. Weak signal. |

**Decision: Include smoking** as a weak clinical variable. It is the one demographic/
lifestyle variable that has a biological mechanism (nicotine modulates mucosal immunity)
and is sometimes factored into clinical thinking, even if not a formal diagnostic criterion.

---

## Encoding Plan

| Type | Variables | Encoding |
|---|---|---|
| Binary comorbidities | `cardiovascular_disease`, `hypertension`, `diabetes_type_i`, `diabetes_type_ii`, `ckd`, `hiv`, `tuberculosis` | Y=1, N=0, NaN=NaN |
| Binary | `smk` | Yes=1, No=0, NaN=NaN |
| Ordinal numeric | `rectal_bleed`, `first_abdopain`, `first_daily_bm`, `first_well_being` | Already numeric, keep as-is |
| Ordinal categorical | `stool_freq` | Normal=0, 1-2 more=1, 3-4 more=2, >4 more=3, ostomy=NaN |
| Ordinal categorical | `fecal_urgency` | None=0, Mild=1, Moderate=2, Moderately severe=3, Severe=4, ostomy=NaN |
| Ordinal categorical | `da6m` | 1=0, 2=1, 3=2, 4=3, 5=4, 6=5 |
| Numeric strings (labs) | `hemoglobin_text`, `erythrocyte_sedimentation_text`, `neutrophil_text`, `white_blood_text`, `platelets_text` | Regex extract float; non-numeric → NaN |

Missing values: RandomForest handles NaN natively via sklearn's `missing_values=np.nan`
with `max_features='sqrt'`. For features with >50% missingness (labs), consider
running sensitivity analysis with and without them.

---

## Model Setup (matching existing pipeline)

```
RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
)
```

- Label: CD=0, UC=1
- Positive class: UC
- CV: 5-fold patient-level (same `cv_splits_patients.csv`)
- Metrics: AUC-ROC, AUC-PR, balanced accuracy (matching existing arms)
- SHAP: TreeExplainer (interventional, 100 background samples) — same as other arms

---

## Next Step

Write training script: `clinical/train_clinical.py`
- Follow structure of `02_train_random_forest.py`
- Output results to `/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/clinical/`
