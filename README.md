# IBD Plexus — CD vs UC Multimodal Classification

Multimodal machine-learning pipeline for distinguishing **Crohn's Disease (CD)** from
**Ulcerative Colitis (UC)** using the SPARC IBD Plexus cohort. Three modalities are
evaluated individually and in combination:

| Modality | Feature set |
|---|---|
| **Imaging** | PRISM2 whole-slide image embeddings (prism2_base 2,560-d; prism2_diagnostic 3,072-d; histological probability scores 11-d) |
| **Transcriptomics** | RNA-seq VST batch-corrected gene expression (17,963 genes) |
| **Multimodal** | Early-fusion concatenation of imaging + RNA features |

All classifiers are Random Forests with 5-fold **patient-level** stratified CV.
The primary metric is AUROC (mean ± SD across folds).

---

## Repository Structure

```
code/
  encode_image/          # WSI preprocessing and embedding
  prism2/                # PRISM2 report generation, UAMP scoring, UMAP visualisation
  training/cd_vs_uc/     # Classification pipeline (see below)
```

---

## Pipeline: `code/training/cd_vs_uc/`

Scripts are numbered by stage; suffix letters (`b`, `c`, …) indicate parallel
analysis series on the same cohort.

### Stage 01 — CV splits
| Script | Description |
|---|---|
| `01_build_cv_splits.py` | Build 5-fold patient-level CV splits; save `cv_splits_patients.csv` |

### Stages 02–07 — All-sites cohorts
| Script | Description |
|---|---|
| `02_train_random_forest.py` | Imaging-only RF (all biopsy sites) |
| `02b_train_imaging_matched.py` | Imaging RF on the RNA-matched cohort |
| `03_train_transcriptomics.py` | RNA-only RF |
| `06_train_multimodal.py` | Multimodal fusion (6 strategies) |
| `06b_ablations_fusion.py` | PCA / scaling ablations |
| `04`, `05`, `07` | Report generation for each cohort |

### Stage 08 — At-20-cm site-restricted analysis
Restricts all arms to the **at-20-cm** biopsy site to remove the biopsy-protocol
confound introduced by differential ileal sampling in CD.

| Script | Description |
|---|---|
| `08_train_at20cm_only.py` | Imaging-only, at-20-cm |
| `08b_train_at20cm_visit_level.py` | Visit-level: img_base + RNA + concat — **945 visits / 817 patients** |
| `08c_train_at20cm_histoscore.py` | Replaces prism2_base with 11-d histological scores (*concept learning* series) |
| `08d`, `08e` | Pathway-score and hallmark+histoscore variants |

### Stages 10–11 — SHAP feature importance
| Script | Description |
|---|---|
| `10b_shap_analysis_visit.py` | TreeExplainer SHAP for base visit-level arms |
| `10c_shap_analysis_histoscore.py` | SHAP for histoscore arms; saves per-sample arrays for beeswarm |
| `11b_shap_plots_visit.py` | Bar plots + direction labels (↑UC / ↑CD) for base series |
| `11c_shap_plots_histoscore.py` | Bar plots, score distributions, fusion panel for histoscore series |
| `11d_beeswarm_histoscore.py` | SHAP beeswarm plots (per-sample, feature-value coloured) |

### Stage 12 — Pathway enrichment
| Script | Description |
|---|---|
| `12_pathway_enrichment_visit.py` | ORA for CD-up / UC-up SHAP genes via g:Profiler (GO:BP, KEGG, Reactome, WikiPathways) + Enrichr cross-check |

### RNA UMAP
| Script | Description |
|---|---|
| `umap_rna.py` | PCA(50) → UMAP on all 3,289 RNA samples; coloured by diagnosis, biopsy location, macroscopic appearance, disease activity |
| `job_umap_rna.yaml` | Kubernetes job spec to run `umap_rna.py` |

### Comparison figures (`plot/`)
| Script | Description |
|---|---|
| `concept_learning_5arm_comparison.py` | 5-arm AUC bar chart: prism2_base vs histoscores vs RNA vs two multimodal arms |
| `at20cm_visit_3arm_comparison.py` | 3-arm visit-level AUC chart |
| `at20cm_full_comparison.py` | Full cohort comparison across all sites |

---

## WSI Preprocessing: `code/encode_image/`

| Script | Description |
|---|---|
| `convert_vsi_to_tiff.py` | Convert VSI → TIFF using BioFormats |
| `correct_mpp.py` | Fix microns-per-pixel metadata |
| `run_prism2.py` | Extract PRISM2 embeddings (prism2_base, prism2_diagnostic) |
| `qc_patch_counts.py` | QC: verify patch counts per slide |
| `jobs/` | Kubernetes job YAMLs for each encoder (Virchow2, GigaPath, TITAN, …) |

---

## PRISM2 Analysis: `code/prism2/`

| Script | Description |
|---|---|
| `run_prism2_reports.py` | Generate free-text histology reports from Virchow2 features |
| `run_prism2_uamp.py` | Score 11 histological features per slide (UAMP yes/no probabilities) |
| `umap_embeddings.py` | UMAP of slide-level embeddings |
| `umap_reports.py` | UMAP coloured by report-extracted + ground-truth metadata |
| `umap_patch_viewer.py` | Interactive HTML patch-level UMAP viewer |

---

## Key Results (At-20-cm visit-level, 945 visits / 817 patients)

| Arm | AUC (mean ± SD) |
|---|---|
| Imaging — prism2_base (2,560-d) | 0.793 ± 0.022 |
| Imaging — histological scores (11-d) | 0.733 ± 0.027 |
| RNA-seq only (17,963 genes) | 0.816 ± 0.033 |
| RNA + prism2_base (multimodal) | 0.842 ± 0.019 |
| RNA + histological scores (multimodal) | 0.821 ± 0.030 |

---

## Dependencies

```
python >= 3.10
scikit-learn
numpy  pandas  matplotlib
shap
umap-learn
gprofiler-official
gseapy
plotly
```

---

## Data Availability

This project uses de-identified data from the
[SPARC IBD Plexus](https://www.ibdplexus.org/) consortium. Raw data access
requires registration and data use agreement through the IBD Plexus portal.
Derived outputs (embeddings, fold metrics, SHAP summaries) are stored on the
project's institutional compute cluster and are available to consortium members.
