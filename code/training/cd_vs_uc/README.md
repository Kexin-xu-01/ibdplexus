# CD vs UC Classification Pipeline

Random Forest classifiers distinguishing Crohn's disease (CD) from ulcerative colitis (UC)
using Virchow2 slide embeddings, CombatSeq/VST transcriptomics, and multimodal fusion.
The analysis progresses through three experimental arms, culminating in a site-controlled
cohort that removes a biopsy-protocol confound, plus SHAP feature importance.

---

## Directories

| Path | Role |
|------|------|
| `ibdplexus/code/training/cd_vs_uc/` | **This directory** — all scripts |
| `kgbk271-ibd-volume/training/cd_vs_uc/` | **Output root** — all results, reports, plots |
| `kgbk271-ibd-volume/training/` | Shared CV split files (`cv_splits_patients.csv`, `cv_splits_slides.csv`) |
| `shared-data/ibd_plexus_sparc_raw/genestack/transcriptomics/` | Raw GCT + metadata (read-only) |
| `kgbk271-ibd-volume/data/processed/trident_processed/…/prism2_base/` | Virchow2 H5 embeddings (read-only) |
| `jobs/` | Kubernetes job specs (see `jobs/` section below) |

---

## Scripts

Scripts are numbered to reflect execution order. The suffix letter (`b`) marks a variant
of the same experiment arm run in parallel with its sibling, not a prerequisite.
Superseded and experimental scripts are in `_deprecated/`.

### Active scripts

| Script | What it does | Outputs |
|--------|-------------|---------|
| `01_build_cv_splits.py` | Builds patient-level 5-fold stratified CV splits; filters to colon sites, deduplicates multi-site patients | `../cv_splits_patients.csv`, `../cv_splits_slides.csv` |
| `01b_build_cv_splits_at20cm_matched.py` | Matched variant: at-20-cm cohort with proximity-paired imaging + RNA (≤7 days) | `../cv_splits_at20cm_matched.csv` |
| `02_train_random_forest.py` | RF on prism2_base and prism2_diagnostic embeddings; 1,250-patient full cohort | `02_04_imaging_allsites/results/` |
| `02b_train_imaging_matched.py` | Imaging RF restricted to matched cohort (997 patients with RNA) | `02_04_imaging_allsites/results/` |
| `03_train_transcriptomics_tpm.py` | RF on log-TPM gene expression; per-sample (not patient-mean) | `03_tpm/results/` |
| `03b_umap_rna_tpm.py` | UMAP of log-TPM features (PCA-50 → UMAP-2) with clinical metadata colouring | `results/rna/umap_tpm/` |
| `08_train_at20cm_tpm_sample_level.py` | At-20-cm RF using log-TPM features at sample level | `08_09_at20cm_site_controlled/results/` |
| `12b_compare_filter_conditions.py` | Compares model performance across patch filter conditions (unfiltered, laplacian, no_darkspot) | console + CSVs |
| `version_utils.py` | Helper module (`next_versioned_path`, `log_version`) used by all report scripts | — |

### `clinical/` — Clinical feature modelling

| Script | What it does |
|--------|-------------|
| `01_train_clinical.py` | RF on clinical features (demographics, disease scores, medications) |
| `02_shap_clinical.py` | SHAP importance for clinical feature model |
| `03_plot_clinical.py` | Bar plots + beeswarm from clinical SHAP output |

### `plot/` — Standalone comparison figures

All scripts in `plot/` are run on demand after the relevant training arms complete.
No strict ordering dependency between them.

| Script | What it produces |
|--------|-----------------|
| `at20cm_venn_availability.py` | Venn diagram of imaging / RNA / clinical data availability at 20 cm |
| `cv_stratification_check.py` | Verify CV fold stratification balance |
| `multi_slide_thumbnails.py` | Grid of slide thumbnails for a patient |
| `pca640_confusion_matrix.py` | Confusion matrix for the PCA-640 RNA arm |
| `venn_all_vs_at20cm.py` | Venn: all-sites vs at-20-cm cohort overlap |
| `visit_gaps.py` | Distribution of imaging–RNA time gaps |
| `visit_structure.py` | Visit timeline visualisation per patient |

### `jobs/` — Kubernetes job specs

| Job file | What it runs |
|----------|-------------|
| `job_rf_no_darkspot.yaml` | RF training on `no_darkspot` filtered dataset |
| `job_mlp_no_darkspot.yaml` | MLP training on `no_darkspot` filtered dataset |
| `job_umap_rna.yaml` | RNA UMAP (03b_umap_rna_tpm.py or umap_rna.py) |
| `_deprecated/job_rf_tissue_threshold_15.yaml` | Superseded (tissue threshold only, no artefact filter) |
| `_deprecated/job_rf_tissue_threshold_15_filtered.yaml` | Superseded (intensity filter only) |

### `_deprecated/` — Historical pipeline

The original all-sites arms (04–07), site-controlled arms (08–09 variants), SHAP analysis
(10–11), and pathway enrichment (12) scripts are in `_deprecated/`. They document the full
experimental history and can be re-run, but the active pipeline has moved to the TPM-based
and matched-cohort variants above. See `report/` for written summaries of findings.

---

## Original pipeline arms (scripts now in `_deprecated/`)

These ran as the primary analysis; results are stable and documented in `report/`.

### Arm 1 — Imaging, all colon sites  →  `02_04_imaging_allsites/`

| Script | What it does |
|--------|-------------|
| `_deprecated/04_generate_reports.py` | Generates PDF report + PPTX for the imaging arm |

### Arm 2 — Transcriptomics, all colon sites  →  `03_05_transcriptomics_allsites/`

| Script | What it does |
|--------|-------------|
| `_deprecated/03_train_transcriptomics.py` | RF on VST gene expression (patient-mean across biopsy sites) |
| `_deprecated/05_generate_transcriptomics_reports.py` | PDF report + PPTX for transcriptomics vs imaging head-to-head |

### Arm 3 — Multimodal fusion  →  `06_07_multimodal_allsites/`

| Script | What it does |
|--------|-------------|
| `_deprecated/06_train_multimodal.py` | RF on raw concatenation of imaging + RNA |
| `_deprecated/06b_ablations_fusion.py` | Ablation: raw concat vs PCA-compressed fusion |
| `_deprecated/07_generate_multimodal_reports.py` | PDF report + PPTX for multimodal study |

### Arm 4 — At-20-cm site-controlled  →  `08_09_at20cm_site_controlled/`

| Script | What it does |
|--------|-------------|
| `_deprecated/08_train_at20cm_only.py` | All modalities restricted to at-20-cm biopsies |
| `_deprecated/08b_train_at20cm_uamp.py` | UAMP dimensionality reduction variant |
| `_deprecated/08c–08h_*` | Additional modality variants (histoscore, pathway, BulkFormer, MLP, PCA-RNA) |
| `_deprecated/09_generate_at20cm_reports.py` | PDF report + PPTX for site-controlled analysis |

### SHAP feature importance  →  `10_11_shap_analysis/`

| Script | What it does |
|--------|-------------|
| `_deprecated/10_shap_analysis.py` | TreeExplainer (interventional) for four arms; ranked feature tables |
| `_deprecated/10b–10e_*` | SHAP for visit-level, histoscore, pathway, and hallmark variants |
| `_deprecated/11_shap_plots.py` | 5 publication-quality SHAP figures |
| `_deprecated/11b–11f_*` | Bar plots and beeswarm for each series |
| `_deprecated/12_pathway_enrichment_visit.py` | ORA via g:Profiler + Enrichr for top SHAP genes |

---

## Output folder structure

Each output folder is named `{script numbers}_{experiment_arm}` so the folder name
directly identifies which scripts produced and reported it.

```
kgbk271-ibd-volume/training/cd_vs_uc/
│
├── 02_04_imaging_allsites/
│   ├── results/                          ← written by 02, 02b
│   │   ├── prism2_base_fold_metrics.csv
│   │   ├── prism2_base_slide_predictions.csv
│   │   ├── prism2_base_summary.json
│   │   ├── prism2_base_matched_fold_metrics.csv
│   │   ├── prism2_base_matched_slide_predictions.csv
│   │   ├── prism2_base_matched_summary.json
│   │   ├── prism2_diagnostic_fold_metrics.csv
│   │   ├── prism2_diagnostic_slide_predictions.csv
│   │   ├── prism2_diagnostic_summary.json
│   │   ├── prism2_diagnostic_matched_fold_metrics.csv
│   │   ├── prism2_diagnostic_matched_slide_predictions.csv
│   │   └── prism2_diagnostic_matched_summary.json
│   └── reports/                          ← written by 04
│       ├── pipeline_results_report_v1.pdf
│       └── pipeline_results_slides_v1.pptx
│
├── 03_05_transcriptomics_allsites/
│   ├── results/                          ← written by 03
│   │   ├── transcriptomics_vst_fold_metrics.csv
│   │   ├── transcriptomics_vst_sample_predictions.csv
│   │   └── transcriptomics_vst_summary.json
│   └── reports/                          ← written by 05
│       ├── transcriptomics_report_v1.pdf
│       └── transcriptomics_slides_v1.pptx
│
├── 06_07_multimodal_allsites/
│   ├── results/                          ← written by 06, 06b
│   │   ├── multimodal_fold_metrics.csv
│   │   ├── multimodal_patient_predictions.csv
│   │   ├── multimodal_summary.json
│   │   ├── multimodal_ablation_fold_metrics.csv
│   │   ├── multimodal_ablation_patient_predictions.csv
│   │   └── multimodal_ablation_summary.json
│   └── reports/                          ← written by 07
│       ├── multimodal_report_v1.pdf
│       └── multimodal_slides_v1.pptx
│
├── 08_09_at20cm_site_controlled/
│   ├── results/                          ← written by 08, 08b
│   │   ├── at20cm_fold_metrics.csv
│   │   ├── at20cm_patient_predictions.csv
│   │   ├── at20cm_summary.json
│   │   ├── at20cm_uamp_fold_metrics.csv
│   │   ├── at20cm_uamp_patient_predictions.csv
│   │   └── at20cm_uamp_summary.json
│   └── reports/                          ← written by 09
│       ├── at20cm_report_v1.pdf
│       └── at20cm_slides_v1.pptx
│
├── 10_11_shap_analysis/
│   ├── data/                             ← written by 10
│   │   ├── shap_rna_20cm_top500.csv
│   │   ├── shap_rna_patmean_top500.csv
│   │   ├── shap_img_base_20cm_top500.csv
│   │   ├── shap_concat_raw_20cm_top500.csv
│   │   ├── shap_rna_20cm_vs_patmean.csv
│   │   └── shap_summary.json
│   └── plots/                            ← written by 11
│       ├── shap_rna_20cm_bar.pdf         — top-20 genes, At-20-cm RNA arm
│       ├── shap_rna_allsites_bar.pdf     — top-20 genes, all-sites RNA arm (HOX confound visible)
│       ├── shap_rank_comparison.pdf      — dumbbell: rank shift between the two RNA arms
│       ├── shap_fusion_split.pdf         — imaging vs RNA SHAP fraction in fusion model
│       └── shap_panel.pdf               — combined 4-panel figure
│
├── VERSIONS.md                           ← append-only changelog (managed by version_utils.py)
└── patient_metadata_subset.csv           ← analysis cohort snapshot
```

---

## Code → output mapping

| Script(s) | Reads from | Writes to |
|-----------|-----------|-----------|
| `01` | shared-data, trident H5s | `../cv_splits_patients.csv`, `../cv_splits_slides.csv` |
| `02` | cv_splits | `02_04.../results/` (prism2_base + diagnostic, full cohort) |
| `02b` | cv_splits, `03_05.../results/transcriptomics_vst_summary.json` | `02_04.../results/` (matched cohort) |
| `03` | cv_splits, GCT | `03_05.../results/` |
| `04` | `02_04.../results/` | `02_04.../reports/` |
| `05` | `03_05.../results/`, `02_04.../results/` | `03_05.../reports/` |
| `06` | cv_splits, trident H5s, GCT | `06_07.../results/` (main fusion) |
| `06b` | cv_splits, trident H5s, GCT, `06_07.../results/multimodal_summary.json` | `06_07.../results/` (ablations) |
| `07` | `06_07.../results/`, `03_05.../results/`, `02_04.../results/` | `06_07.../reports/` |
| `08` | cv_splits, trident H5s, GCT, `06_07.../results/` | `08_09.../results/` |
| `08b` | cv_splits, trident H5s, GCT | `08_09.../results/` (UAMP variant) |
| `09` | `08_09.../results/`, `06_07.../results/` | `08_09.../reports/` |
| `10` | cv_splits, trident H5s, GCT | `10_11.../data/` |
| `11` | `10_11.../data/` | `10_11.../plots/` |

---

## Execution order

Run in numerical order. `b`-suffix scripts can run in parallel with their sibling
once the arm's prerequisite training is complete.

### Active pipeline

```
01 / 01b                # build CV splits (once — never re-run without good reason)
02  ─┐
02b  ├─ imaging arms (parallel)
03   ├─ transcriptomics TPM
03b  ┘  RNA UMAP (after 03)
08                      # at-20-cm TPM sample-level
12b                     # filter condition comparison (after 02 + image_qc pipeline)
```

For the `clinical/` arm:
```
clinical/01_train_clinical.py  →  02_shap_clinical.py  →  03_plot_clinical.py
```

### Historical pipeline (scripts in `_deprecated/`)

```
01                      # build CV splits (once)
02  →  04               # imaging arm
02b                     # imaging matched (needs 03 results)
03  →  05               # transcriptomics (VST) arm
06  →  06b  →  07       # multimodal arm  (06b needs 06 summary)
08  →  09               # site-controlled arm  (needs 06 results)
08b–08h                 # modality variants (independent of each other)
10  →  11               # SHAP analysis
12                      # pathway enrichment
```

Report versions are managed automatically by `version_utils.py` — re-running a report
script increments to `_v2`, `_v3`, etc. and appends a line to `VERSIONS.md`.
