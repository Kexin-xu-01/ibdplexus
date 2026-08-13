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

---

## Scripts

Scripts are numbered to reflect execution order. The suffix letter (`b`) marks a variant
of the same experiment arm run in parallel with its sibling, not a prerequisite.

### Setup

| Script | What it does | Outputs |
|--------|-------------|---------|
| `01_build_cv_splits.py` | Builds patient-level 5-fold stratified CV splits; filters to colon sites, deduplicates patients with multi-site biopsies | `../cv_splits_patients.csv`, `../cv_splits_slides.csv` |
| `version_utils.py` | Helper module (`next_versioned_path`, `log_version`) used by all report scripts | — |

### Arm 1 — Imaging, all colon sites  →  `02_04_imaging_allsites/`

| Script | What it does |
|--------|-------------|
| `02_train_random_forest.py` | Trains RF on prism2_base and prism2_diagnostic embeddings; 1,250-patient full cohort |
| `02b_train_imaging_matched.py` | Re-runs imaging RF restricted to the 997 patients who also have RNA data (matched cohort for fair comparison) |
| `04_generate_reports.py` | Generates PDF report + PPTX for the imaging arm |

### Arm 2 — Transcriptomics, all colon sites  →  `03_05_transcriptomics_allsites/`

| Script | What it does |
|--------|-------------|
| `03_train_transcriptomics.py` | Trains RF on VST gene expression (patient-mean across biopsy sites); 997 patients |
| `05_generate_transcriptomics_reports.py` | Generates PDF report + PPTX for transcriptomics vs imaging head-to-head |

### Arm 3 — Multimodal fusion, all colon sites  →  `06_07_multimodal_allsites/`

| Script | What it does |
|--------|-------------|
| `06_train_multimodal.py` | Trains RF on raw concatenation of imaging + RNA features; 828 matched patients |
| `06b_ablations_fusion.py` | Ablation study: raw concat vs PCA-compressed fusion; reads `multimodal_summary.json` from same results dir |
| `07_generate_multimodal_reports.py` | Generates PDF report + PPTX for the multimodal study |

### Arm 4 — Site-controlled (At-20-cm only)  →  `08_09_at20cm_site_controlled/`

| Script | What it does |
|--------|-------------|
| `08_train_at20cm_only.py` | Re-runs all modalities restricted to At-20-cm (rectosigmoid junction) biopsies, removing the cecum site confound; reads all-sites multimodal results for comparison |
| `08b_train_at20cm_uamp.py` | Variant: applies UAMP dimensionality reduction before fusion |
| `09_generate_at20cm_reports.py` | Generates PDF report + PPTX for the site-controlled analysis |

### SHAP feature importance  →  `10_11_shap_analysis/`

| Script | What it does |
|--------|-------------|
| `10_shap_analysis.py` | Runs 5-fold CV + `TreeExplainer` (interventional, 100 background samples) for arms `rna_20cm`, `rna_patmean`, `img_base_20cm`, `concat_raw_20cm`; outputs ranked feature tables + cross-arm comparison |
| `11_shap_plots.py` | Produces 5 publication-quality figures from the SHAP tables |

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

```
01                      # build CV splits (once)
02  →  04               # imaging arm
02b                     # imaging matched (needs 03 results for comparison table in 02b)
03  →  05               # transcriptomics arm
06  →  06b  →  07       # multimodal arm  (06b needs 06 summary)
08  →  09               # site-controlled arm  (needs 06 results)
08b                     # UAMP variant (independent of 08)
10  →  11               # SHAP analysis
```

Report versions are managed automatically by `version_utils.py` — re-running a report
script increments to `_v2`, `_v3`, etc. and appends a line to `VERSIONS.md`.
