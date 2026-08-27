# CD vs UC — Normalisation Audit and Leakage-Free Re-run
## At-20-cm Biopsies, Sample Level, RNA-seq

---

## 1. Background

All prior CD vs UC transcriptomics classifiers in this project loaded expression
values from pre-computed matrices that apply normalisation and batch correction
across the entire dataset before any train/test split. This document audits
every available normalisation for cross-sample data leakage, identifies the
clean alternative, and reports the re-run results.

---

## 2. Leakage Audit of Available Normalisations

### What counts as leakage?

Any normalisation step that computes a statistic jointly across training and
held-out (test) samples leaks test-set information into the training
representation. Even if no label information is transferred, the test samples'
expression distributions influence how training samples are scaled — inflating
apparent generalisation performance.

### Matrix inventory

| GCT file | Method | Cross-sample? | Leakage |
|---|---|---|---|
| `GSF1491805_CombatSeq_vst_mtx_batch_corrected_*` | DESeq2 VST + CombatSeq | Yes (both steps) | **YES** |
| `GSF1491803_CombatSeq_count_mtx_batch_corrected_*` | CombatSeq corrected counts | Yes | **YES** |
| `GSF1485554_vst_count_mtx_df.gct` | DESeq2 VST only | Yes | **YES** |
| `GSF1491807_combined_raw_count_mtx_header.gct` | Raw counts | No | Clean |
| `GSF2048892_combined_TPM_matrix_with_header.gct` | TPM | No | **Clean** |

### Why VST leaks

DESeq2 Variance Stabilizing Transformation fits a mean–dispersion trend across
the full sample set. The transformation applied to each sample depends on this
shared trend, meaning the dispersion parameters estimated from held-out test
samples influence the VST values of training samples.

### Why CombatSeq leaks

CombatSeq (and the original ComBat) models per-batch means and variances from
the pooled dataset, then subtracts the estimated batch offsets. Both the
location and scale corrections are estimated from a combined pool that includes
test samples. Any change in the test-set batch composition alters the correction
applied to training samples.

### Why TPM is clean

TPM (Transcripts Per Million) is defined entirely within a single sample:

```
TPM_i = (count_i / gene_length_kb_i) / Σ_j(count_j / gene_length_kb_j) × 10⁶
```

No statistics from other samples enter the calculation. Each sample's TPM
vector is invariant to what other samples are present in the dataset. No batch
correction is applied.

---

## 3. Additional Safeguards in the Re-run

Beyond switching to TPM, the re-run script (`08_train_at20cm_tpm_sample_level.py`)
applies two further measures to ensure complete separation of train and test:

### Gene filtering — per fold, training samples only

Minimum expression filtering (mean log₂(TPM+1) > 0.5) is computed from
training-fold samples inside each CV loop. Test-fold expression values are
never used to decide which genes appear as features. This means the gene set
is determined solely by what is expressed in the training data.

### Sample-level analysis

Each RNA sample is one row in the feature matrix. Patients with multiple
at-20-cm samples contribute proportionally to training and evaluation (within
their assigned fold). No patient-level averaging is applied before training.

### Patient-level CV splits

All samples from the same patient land in the same fold, using the fixed
`cv_splits_patients.csv` assignment. No patient appears in both training
and validation within a fold.

---

## 4. Cohort

| Property | Value |
|---|---|
| Biopsy location | At 20 cm (rectosigmoid) |
| Diagnoses | Crohn's disease (CD), Ulcerative colitis (UC) |
| QC filter | `Sample QC != fail` |
| RNA samples | 1,038 (817-patient cohort) |
| Patients | 817 (CD 548, UC 269) |
| CV | 5-fold stratified, patient-level |
| Normalisation | log₂(TPM+1), per sample |
| Gene filter | mean log₂(TPM+1) > 0.5 on training fold, ~17,480 genes/fold |

---

## 5. Results

### Per-fold metrics

| Fold | Train samples | Val samples | Val patients | Genes used | AUC | AP | Accuracy |
|---|---|---|---|---|---|---|---|
| 0 | 832 | 206 | 160 | 17,476 | 0.8448 | 0.7676 | 0.7573 |
| 1 | 833 | 205 | 159 | 17,495 | 0.8132 | 0.6491 | 0.7902 |
| 2 | 838 | 200 | 163 | 17,442 | 0.8330 | 0.6938 | 0.7400 |
| 3 | 817 | 221 | 170 | 17,411 | 0.7844 | 0.5506 | 0.7421 |
| 4 | 832 | 206 | 165 | 17,485 | 0.8038 | 0.6730 | 0.7427 |

### Summary

| Metric | Mean ± SD |
|---|---|
| AUC | **0.8158 ± 0.0238** |
| AP | 0.6668 ± 0.0787 |
| Accuracy | 0.7545 ± 0.0209 |
| Mean genes/fold | 17,462 |

---

## 6. Comparison to Previous Batch-Corrected Run

### 6a. Matched comparison (same 817 patients, same 1,038 samples, sample level)

To isolate the effect of normalisation from all other variables, both VST and TPM
analyses are restricted to the same 817-patient cohort (`cohorts/at20cm_rna_visit_817_patients.csv`)
— the patients from the original `rna_visit` analysis (RNA+imaging matched within 7 days).
Both run at sample level with identical RF parameters and CV splits.

| | `rna_vst_sample` (VST + CombatSeq, leaky) | `rna_tpm_sample` (TPM, clean) |
|---|---|---|
| Normalisation | CombatSeq + DESeq2 VST | log₂(TPM+1), per sample |
| Batch correction | Yes — fitted on all samples | None |
| Level | Sample | Sample |
| Samples | 1,038 | 1,038 |
| Patients | 817 | 817 |
| Genes/fold | 17,963 (all retained) | ~17,462 (per-fold filter) |
| **AUC** | 0.8222 ± 0.0300 | **0.8158 ± 0.0238** |
| **AP** | 0.6708 ± 0.0750 | **0.6668 ± 0.0787** |
| **Accuracy** | 0.7580 | **0.7545** |
| **ΔAUC (VST − TPM)** | **+0.006** | — |

### 6b. Historical context: original patient-level visit analysis

The original `rna_visit` result (`08b_train_at20cm_visit_level.py`) averaged RNA
samples per patient and used a different cohort definition (proximity-matched to
imaging visits). It is not directly comparable but is included for context.

| | `rna_visit` (CombatSeq + VST, patient mean) | `rna_tpm_sample` (TPM, sample level) |
|---|---|---|
| Level | Visit (patient mean) | Sample |
| Samples / visits | 945 | 1,038 |
| Patients | 817 | 817 |
| **AUC** | 0.8160 ± 0.0334 | **0.8158 ± 0.0238** |

### Interpretation

1. **The leakage inflates VST performance by ~0.006 AUC** on this cohort.
   This is the pure normalisation effect, isolated by holding patients, samples,
   granularity, and CV protocol identical. The gap is modest, indicating the
   discriminative signal is driven by genuine biology, not normalisation artefacts.

2. **TPM matches the original VST visit-level result.** At AUC 0.8158 vs the
   original 0.8160, the leakage-free approach recovers essentially identical
   performance even without batch correction or VST.

3. **TPM is the defensible result to report.** It is methodologically clean,
   reproducible without access to the full dataset at inference time, and
   sacrifices nothing in predictive performance.

---

## 7. Files

| File | Description |
|---|---|
| `08_train_at20cm_tpm_sample_level.py` | TPM training script (leakage-free) |
| `08b_train_at20cm_vst_sample_level.py` | Matched VST training script (leaky reference) |
| `08_at20cm_tpm_sample_level/results/at20cm_tpm_sample_fold_metrics.csv` | TPM per-fold metrics |
| `08_at20cm_tpm_sample_level/results/at20cm_tpm_sample_predictions.csv` | TPM per-sample predictions |
| `08_at20cm_tpm_sample_level/results/at20cm_tpm_sample_summary.json` | TPM summary |
| `08_at20cm_vst_sample_level/results/at20cm_vst_sample_fold_metrics.csv` | VST per-fold metrics |
| `08_at20cm_vst_sample_level/results/at20cm_vst_sample_predictions.csv` | VST per-sample predictions |
| `08_at20cm_vst_sample_level/results/at20cm_vst_sample_summary.json` | VST summary |

Script location: `/home/jovyan/ibdplexus/code/training/cd_vs_uc/`

Results location: `/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/`
