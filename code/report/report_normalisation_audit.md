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
| RNA samples | 1,068 |
| Patients | 847 (CD 565, UC 282) |
| CV | 5-fold stratified, patient-level |
| Normalisation | log₂(TPM+1), per sample |
| Gene filter | mean log₂(TPM+1) > 0.5 on training fold, ~17,480 genes/fold |

---

## 5. Results

### Per-fold metrics

| Fold | Train samples | Val samples | Val patients | Genes used | AUC | AP | Accuracy | CD F1 | UC F1 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 850 | 218 | 172 | 17,488 | 0.8445 | 0.7592 | 0.7615 | 0.8365 | 0.5593 |
| 1 | 858 | 210 | 164 | 17,508 | 0.8257 | 0.6816 | 0.8095 | 0.8765 | 0.5833 |
| 2 | 862 | 206 | 169 | 17,459 | 0.8328 | 0.7073 | 0.7233 | 0.8106 | 0.4865 |
| 3 | 844 | 224 | 173 | 17,435 | 0.7878 | 0.5707 | 0.7321 | 0.8137 | 0.5238 |
| 4 | 858 | 210 | 169 | 17,508 | 0.7838 | 0.6492 | 0.7238 | 0.8117 | 0.4821 |

### Summary

| Metric | Mean ± SD |
|---|---|
| AUC | **0.8149 ± 0.0275** |
| AP | 0.6736 ± 0.0702 |
| Accuracy | 0.7500 ± 0.0367 |
| Mean genes/fold | 17,480 |

---

## 6. Comparison to Previous Batch-Corrected Run

The prior visit-level result (`rna_visit`, `08b_train_at20cm_visit_level.py`) used
CombatSeq + VST on a cohort matched to imaging visits (RNA + H&E slide within 7 days).

| | Previous: `rna_visit` (CombatSeq + VST) | New: `rna_tpm_sample` (TPM only) |
|---|---|---|
| Normalisation | CombatSeq + DESeq2 VST | log₂(TPM+1), per sample |
| Batch correction | Yes (all samples) | None |
| Level | Visit (RNA+imaging matched) | Sample (RNA only) |
| Samples / visits | 945 | 1,068 |
| Patients | 817 | 847 |
| **AUC** | 0.8160 ± 0.0334 | **0.8149 ± 0.0275** |
| **AP** | 0.6687 ± 0.0685 | **0.6736 ± 0.0702** |
| **Accuracy** | 0.7493 | **0.7500** |

### Interpretation

The two runs produce statistically indistinguishable performance (ΔAUC = −0.001,
ΔAP = +0.005). This has two implications:

1. **The batch correction and VST were not artificially inflating results.**
   The leakage was present but its practical effect on this cohort was
   negligible — the discriminative signal is driven by genuine biology, not
   normalisation artefacts.

2. **The TPM result is the defensible one to report.** It is methodologically
   clean: no cross-sample information, no batch correction, per-fold gene
   filtering. The near-identical AUC means nothing is sacrificed by using the
   correct approach.

---

## 7. Files

| File | Description |
|---|---|
| `08_train_at20cm_tpm_sample_level.py` | Training script |
| `results/at20cm_tpm_sample_fold_metrics.csv` | Per-fold AUC / AP / confusion matrix |
| `results/at20cm_tpm_sample_predictions.csv` | Per-sample predictions with `prob_uc` |
| `results/at20cm_tpm_sample_summary.json` | Aggregated summary with metadata |

Script location: `/home/jovyan/ibdplexus/code/training/cd_vs_uc/08_train_at20cm_tpm_sample_level.py`

Results location: `/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/08_at20cm_tpm_sample_level/results/`
