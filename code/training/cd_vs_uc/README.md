# CD vs UC Classification

Random Forest and MLP classifiers distinguishing Crohn's disease (CD) from ulcerative
colitis (UC) using PRISM2 imaging embeddings, VST/TPM transcriptomics, clinical features,
Olink proteomics, and multimodal fusion. Trained on the at-20-cm proximity-matched cohort
(945 visits / 817 patients) with 5-fold patient-level cross-validation.

For deeper project context (data sources, cohort details, RF params, SHAP schema,
plot conventions), see [`CLAUDE.md`](CLAUDE.md).

---

## Canonical pipeline

**Visit-level · at-20-cm biopsies · manual-KNN-filtered patches**

| Property | Value |
|---|---|
| Cohort | 945 visits / 817 patients (CD 632, UC 313) |
| Site | at-20-cm biopsies only |
| Imaging embeddings | `prism2_base` from `tissue_threshold_15_filtered_no_darkspot_manual_knn` |
| Histological scores | `results/prism2_manual_knn/prism2_histological_score.csv` |
| RNA | VST batch-corrected, 17,963 genes |
| CV | 5-fold patient-level (`cv_splits_patients.csv`) |
| Classifier | RF, 500 trees, `class_weight='balanced'` |

---

## Execution order

Run scripts in this order. Every `train_*.py` reads `cv_splits_patients.csv` — build that first.

```
build_cv_splits/01_build_cv_splits.py                   → cv_splits_patients.csv, cv_splits_slides.csv
build_cv_splits/01b_build_cv_splits_at20cm_matched.py   → cv_splits_at20cm_matched.csv

──────────────────────────────────────────────────────────────────────────
Single-modality classifiers (independent; run in any order)

train_rf.py                     ← PRIMARY: 10-arm RF, manual-KNN patches
train_mlp.py                    MLP counterpart to train_rf.py
train_rf_proteomics.py          Olink plasma NPX (1,031 samples / 605 patients)
clinical/01_train_clinical.py   Clinical features only
rna/03_train_transcriptomics_tpm.py          Full-cohort RNA classifier
rna/08_train_at20cm_tpm_sample_level.py      At-20-cm RNA (sample-level)
rna/08b_train_at20cm_vst_sample_level.py     At-20-cm RNA VST variant

──────────────────────────────────────────────────────────────────────────
Multimodal fusion (require single-modality results as arms)

train_multimodal.py              image + RNA + clinical fusion
train_multimodal_histoscore.py   histoscore + RNA + clinical fusion
train_rf_matched_cohort.py       All 4 modalities on the 605-patient proteomics cohort

──────────────────────────────────────────────────────────────────────────
SHAP + interpretability

clinical/02_shap_clinical.py    → clinical/03_plot_clinical.py
shap_rf_proteomics.py            SHAP for proteomics RF

──────────────────────────────────────────────────────────────────────────
Diagnostic UMAPs

rna/03b_umap_rna_tpm.py          RNA-seq UMAP with metadata colouring
rna/umap_bulkformer.py           BulkFormer embedding UMAP

──────────────────────────────────────────────────────────────────────────
Comparison plots (after training)

plot/single_modality_comparison.py     Clinical · Proteomics · RNA · Prism2
plot/multimodal_comparison.py          Fusion arms, RF vs MLP
plot/compare_proteomics_arms.py        Proteomics vs other modalities
plot/manual_knn_arm_comparison.py      manual_knn vs no_darkspot
plot/at20cm_venn_availability.py       Data availability venn
plot/cv_stratification_check.py        CV fold balance check
plot/visit_gaps.py, visit_structure.py Visit timing diagnostics
```

---

## Directory layout

```
training/cd_vs_uc/
├── train_rf.py                    PRIMARY: 10-arm RF, manual-KNN, visit-level
├── train_mlp.py                   MLP version of train_rf.py
├── train_multimodal.py            clinical + image + RNA fusion
├── train_multimodal_histoscore.py histoscore + RNA + clinical fusion
├── train_rf_proteomics.py         Olink proteomics RF
├── train_rf_matched_cohort.py     4-modality RF on 605-patient cohort
├── shap_rf_proteomics.py          SHAP for the proteomics RF
├── version_utils.py               shared helpers (next_versioned_path, log_version)
│
├── build_cv_splits/               CV split construction (run once)
│   ├── 01_build_cv_splits.py
│   └── 01b_build_cv_splits_at20cm_matched.py
│
├── clinical/                      Clinical-feature RF + SHAP
│   ├── 01_train_clinical.py
│   ├── 02_shap_clinical.py
│   └── 03_plot_clinical.py
│
├── rna/                           RNA-only classifiers + UMAPs
│   ├── 03_train_transcriptomics_tpm.py
│   ├── 03b_umap_rna_tpm.py
│   ├── 08_train_at20cm_tpm_sample_level.py
│   ├── 08b_train_at20cm_vst_sample_level.py
│   └── umap_bulkformer.py
│
├── plot/                          Comparison / diagnostic figures
├── report/                        Written analyses (Markdown)
├── cohorts/                       Reference cohort CSVs
├── jobs/                          Kubernetes job specs
│   ├── job_rf_at20cm_visit_manual_knn.yaml    ← canonical
│   ├── job_rf_at20cm_visit_no_darkspot.yaml
│   ├── job_mlp_at20cm_visit_manual_knn.yaml
│   ├── job_multimodal_at20cm_visit_manual_knn.yaml
│   └── _deprecated/
│
└── _deprecated/                   Historical pipelines (02–12 series, image/, multimodal/)
```

---

## Kubernetes jobs (active)

| Job file | What it runs |
|----------|-------------|
| `jobs/job_rf_at20cm_visit_manual_knn.yaml` | `train_rf.py` — canonical 10-arm RF |
| `jobs/job_rf_at20cm_visit_no_darkspot.yaml` | `train_rf.py` on the earlier `no_darkspot` dataset |
| `jobs/job_mlp_at20cm_visit_manual_knn.yaml` | `train_mlp.py` |
| `jobs/job_multimodal_at20cm_visit_manual_knn.yaml` | `train_multimodal.py` |

Deprecated jobs (histoscore, MLP-no_darkspot, old at-20-cm visit-level, RNA UMAP) are in `jobs/_deprecated/`.

---

## Outputs

Each `train_*.py` writes fold metrics + predictions + summary JSON under
`/home/jovyan/kgbk271-ibd-volume/training/cd_vs_uc/<arm>/results/`.
See `CLAUDE.md` for the full output directory schema.

---

## Deprecated pipeline

The original all-sites / patient-level / VST arms (scripts `02–12_*`) and their variants
live in `_deprecated/`. They are still runnable and documented in `report/` — see
`_deprecated/README.md` if regenerating historical results.
