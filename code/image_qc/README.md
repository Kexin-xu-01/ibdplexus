# Patch Image QC Pipeline

Quality control pipeline for whole-slide image (WSI) patch extraction, applied to the KGBK271-IBD cohort. Starting from **865,070 patches across 9,954 slides** extracted at 20× magnification (224 px tiles, Virchow2 embeddings), the pipeline removes blurry, artifact-laden, and visually anomalous patches.

**Final result: 838,562 patches retained (96.94%)**

---

## Pipeline Overview

```
Input
  WSI TIFFs:    tiff_mpp_corrected/
  Patch coords: tissue_threshold_15/20x_224px_0px_overlap/patches/*_patches.h5
  Features:     tissue_threshold_15/20x_224px_0px_overlap/features_virchow2/*.h5
  GeoJSON:      tissue_threshold_15/contours_geojson/*.geojson

  [Step 0]  Threshold exploration  (01_blur_analysis.py, correlations.py)
      │
  [Step 1]  Laplacian blur filter  → laplacien_t100/*_patches.h5
      │
  [Step 2]  GrandQC artifact masks → grandqc/mpp1/grandqc_masks/
      │
  [Step 2a] Dark-spot patch filter → tissue_threshold_15_filtered_no_darkspot/
      │
  [Step 3]  UMAP annotation + KNN → tissue_threshold_15_filtered_no_darkspot_manual_knn/
```

---

## Directory Structure

```
image_qc/
├── pefim_utils.py              # Utility library: clustering, grid visualisation helpers
├── patch_diversity_select.py   # Diverse patch selection; bad-patch characterisation
├── correlations.py             # Step 0: QC metric correlation scatter matrix
├── report_patch_filtering.md   # Full pipeline report with per-stage counts
│
├── 01_laplacien/               # Step 1: Laplacian blur filter
│   ├── 01_blur_analysis.py     # Step 0: threshold calibration strip plot
│   ├── 02_filter_laplacien.py  # Main filter (parallel, multiprocessing)
│   ├── 03_inspect_rejected.py  # Post-filter visualisation of rejected patches
│   ├── job_01_filter.yaml      # K8s job: run 02_filter_laplacien.py
│   └── job_02_inspect_rejected.yaml
│
├── 02_grandqc/                 # Step 2: GrandQC artifact segmentation
│   ├── 01_run_grandqc.py       # GPU inference — produces per-slide artifact masks
│   ├── 02_filter_grandqc_darkspot.py  # Step 2a: removes dark-spot patches from H5s
│   ├── 03_show_artifact_examples.py   # Visualisation: artifact class example grid
│   ├── job_01_grandqc.yaml     # K8s job: run 01_run_grandqc.py on GPU
│   └── _deprecated/
│       └── filter_grandqc_oof.py   # OOF filter (not applied in production)
│
├── 03_prism2_knn/              # Step 3: Interactive UMAP + KNN manual exclusion
│   ├── 01_multi_slide_umap.py  # Build interactive UMAP HTML viewer
│   ├── 02_qc_find_nn.py        # KNN expansion of manually annotated bad patches
│   ├── 03_apply_knn_exclusion.py  # Apply merged exclusion list to feature H5s
│   └── report_patch_qc.md      # Detailed KNN methodology and iteration log
│
└── _deprecated/                # Superseded or unused top-level scripts/folders
    ├── blur_analysis.py        # Replaced by 01_laplacien/01_blur_analysis.py
    ├── artifact_analysis.py    # Early prototype, replaced by 02_grandqc/01_run_grandqc.py
    ├── entropy/                # Entropy threshold exploration (filter not applied)
    └── otsu/                   # Otsu tissue-fraction exploration (filter not applied)
```

---

## Step 0 — Threshold Exploration

Before applying any filter, candidate thresholds are calibrated visually by inspecting example patches at several cut-points.

**Scripts**
- `01_laplacien/01_blur_analysis.py` — samples 60 slides × 40 patches, sorts by Laplacian variance, generates a strip plot with a score histogram and per-threshold rows of example patches
- `correlations.py` — samples 60 slides × 40 patches, computes Pearson and Spearman correlations between Laplacian variance, Shannon entropy, and Otsu tissue fraction

**Outputs**
| File | Location |
|------|----------|
| `blur_threshold_analysis.png` | `tissue_threshold_15_remove_artifact/laplacien/` |
| `qc_metric_correlations.png` | `tissue_threshold_15_remove_artifact/` |

**Rationale:** Thresholds are chosen from data, not fixed heuristics. The strip plots let you confirm visually that the chosen cut-point cleanly separates problematic patches from good ones without over-removing tissue. The correlation matrix informs whether multiple filters are complementary or redundant — entropy and Otsu were found to add little over Laplacian alone (see `_deprecated/`).

---

## Step 1 — Laplacian Blur Filter (`01_laplacien/`)

**Script:** `02_filter_laplacien.py`  
**K8s job:** `job_01_filter.yaml` (8 CPU / 32 GiB)

Each patch is read from its WSI via OpenSlide, converted to grayscale, and the variance of the Laplacian is computed. Patches with Laplacian variance < 100 are discarded as blurry. Processing runs in parallel across workers; `--start` / `--end` allow range-based chunking for large cohorts.

**Inputs**
- `tissue_threshold_15/20x_224px_0px_overlap/patches/*_patches.h5` (patch coordinates)
- WSI TIFFs from `tiff_mpp_corrected/`

**Outputs**
| File | Location |
|------|----------|
| `*_patches.h5` (filtered coords) | `tissue_threshold_15_remove_artifact/laplacien_t100/` |

**Removed:** 4,977 patches (0.57%)

**Rationale:** Blurry patches add noise to embedding models. Laplacian variance is a fast, parameter-light focus measure — images with low variance are predominantly uniform (out-of-focus or blank). t=100 was selected from the Step 0 strip plot.

### Optional — Inspect Rejected Patches

**Script:** `03_inspect_rejected.py`  
**K8s job:** `job_02_inspect_rejected.yaml` (8 CPU / 32 GiB)

Collects all Laplacian-rejected patches, organises them into five LV-range buckets (0–10, 10–25, 25–50, 50–75, 75–100), and generates visualisation grids to verify the filter is not discarding genuine tissue.

**Outputs**
| File | Location |
|------|----------|
| Ranked grid + per-bucket grids | `tissue_threshold_15_remove_artifact/laplacien_t100_rejected/plots/` |
| Individual rejected patch PNGs | `tissue_threshold_15_remove_artifact/laplacien_t100_rejected/patches/lv_XXX-XXX/` |

---

## Step 2 — GrandQC Artifact Segmentation (`02_grandqc/`)

**Script:** `01_run_grandqc.py`  
**K8s job:** `job_01_grandqc.yaml` (1 GPU, 8 CPU / 32 GiB)

Runs a pretrained EfficientNet-B0 U-Net model (GrandQC) at MPP=1.0 to produce per-slide pixel-level artifact masks. For each slide:
1. The tissue GeoJSON contour is rasterised into a binary mask at MPP=10
2. The GrandQC model performs batched GPU inference (batch=32) with 8 parallel OpenSlide reader threads
3. Each pixel is assigned one of 8 artifact classes:

| Class | Label |
|-------|-------|
| 0 | Clean tissue |
| 1 | Tissue fold |
| 2 | Dark spots |
| 3 | Pen marks |
| 4 | Air bubble / edge |
| 5 | Out-of-focus |
| 6 | Other |
| 7 | Background |

The script is restart-safe: slides whose mask already exists are skipped.

**Inputs**
- WSI TIFFs from `tiff_mpp_corrected/`
- GeoJSON contours from `tissue_threshold_15/contours_geojson/`
- GrandQC model state dict

**Outputs**
| File | Location |
|------|----------|
| `{stem}_artifact_mask.png` (uint8, values 0–7) | `tissue_threshold_15_remove_artifact/grandqc/mpp1/grandqc_masks/` |
| `{stem}_p_s.npy` (level-0 pixels per model cell) | `tissue_threshold_15_remove_artifact/grandqc/mpp1/grandqc_masks/` |
| `{stem}_overlay.jpg` (colorised overlay) | `tissue_threshold_15_remove_artifact/grandqc/mpp1/grandqc_overlays/` |

**Rationale:** Pixel-level artifact detection is more precise than patch-level heuristics. GrandQC is a pathology-specific model trained on diverse WSI artifact types, making it usable without fine-tuning. The `_p_s.npy` file stores the pixel-to-mask-cell scaling factor needed to map mask coordinates back onto patch coordinates.

### Step 2a — Dark-Spot Patch Filter

**Script:** `02_filter_grandqc_darkspot.py`

For each patch in the feature H5 files, computes the fraction of overlapping mask pixels classified as dark spots (class 2). Patches where > 10% of pixels are dark spots are removed. Slides with no mask pass through unchanged.

**Inputs:** `tissue_threshold_15_filtered/20x_224px_0px_overlap/features_virchow2/*.h5`  
**Outputs:** `tissue_threshold_15_filtered_no_darkspot/20x_224px_0px_overlap/features_virchow2/*.h5`  
**Removed:** 1,727 patches (0.20%)

### Optional — Artifact Examples Visualisation

**Script:** `03_show_artifact_examples.py`

Samples up to 30 slides and produces a grid of example patches for each artifact class — useful for sanity-checking the GrandQC model on this cohort.

**Output:** `tissue_threshold_15_remove_artifact/grandqc_artifact_examples.png`

---

## Step 3 — Manual KNN Exclusion (`03_prism2_knn/`)

A semi-manual, iterative approach to catch artifact patches that pass automated filters (unusual staining, ink marks, torn tissue, etc.).

### Step 3a — Build Interactive UMAP Viewer

**Script:** `01_multi_slide_umap.py`

Loads Virchow2 features from filtered H5 files, runs PCA (2560 → 50d) then UMAP (50 → 2d), and generates a self-contained interactive HTML viewer with patch thumbnails embedded as base64 JPEGs. The viewer supports:

- Colouring by metadata (diagnosis, disease activity, macroscopic appearance, tissue site, gender)
- Lasso / box selection of patches
- **Mark Bad** — flags selected patches red
- **Find NN** — highlights K-nearest neighbours in UMAP 2-D space
- **Export CSV** — downloads `bad_patches.csv` (slide\_id, x, y) for the marked patches
- **Reset All** — clears all marks

**Outputs**
| File | Location |
|------|----------|
| `{out_name}.html` + `.html.gz` | `results/umap_patch_viewer/tissue_threshold_15/` |

**Rationale:** Virchow2 embeddings cluster patches by visual similarity. Artifact patches cluster together and are identifiable by hovering over outlier groups, so a small number of manually selected seeds represents a much larger set of similar patches.

### Step 3b — Find Nearest Neighbours

**Script:** `02_qc_find_nn.py`

Given the exported `bad_patches.csv` (seed patches annotated in the UMAP viewer), searches the full Virchow2 feature space for nearest neighbours using scikit-learn `NearestNeighbors`. Three search modes:

| Mode | Flag | Description |
|------|------|-------------|
| Fixed-K | default, `--k 10` | Return K nearest per seed |
| Radius | `--dist_threshold` | Return all within L2 distance |
| Per-patch | `--thresholds_csv` | Individual cutoff per seed |

Generates an optional `nn_gallery.html` showing neighbours in distance-binned rows with per-patch threshold sliders and a lightbox image viewer — useful for deciding how far to expand each seed.

**Outputs**
| File | Description |
|------|-------------|
| `exclusion_list.csv` | slide\_id, x, y, reason |
| `nn_gallery.html` (optional) | Interactive gallery with threshold sliders |

**Results from two annotation iterations:**
- Iter 1: 59 seed patches, d ≤ 40 (uniform) → 13,524 excluded
- Iter 2: 13 seed patches, per-patch d = 4–40 → 2,685 excluded (6,541 net-new after overlap)

### Step 3c — Apply Exclusion List

**Script:** `03_apply_knn_exclusion.py`

Applies the merged `exclusion_list_merged.csv` to all filtered feature H5 files, removing every listed (slide\_id, x, y) entry. Output H5 files use the same gzip-4 compression and key layout as the inputs.

**Inputs:** `tissue_threshold_15_filtered_no_darkspot/20x_224px_0px_overlap/features_virchow2/*.h5`  
**Exclusion list:** `image_preprocessing/patch_qc/prism2_knn/exclusion_list_merged.csv`  
**Outputs:** `tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/features_virchow2/*.h5`  
**Removed (net new):** 6,541 patches (0.76%)

---

## Filtering Summary

| Step | Filter | Removed | % of total |
|------|--------|--------:|----------:|
| 1 | Laplacian blur (LV < 100) | 4,977 | 0.57% |
| — | Faint/white intensity (mean ≥ 211.8) — applied in `prism2/make_filtered_features.py` | 13,263 | 1.53% |
| 2a | GrandQC dark spots (> 10% pixels) | 1,727 | 0.20% |
| 3c | Manual KNN exclusion (net new) | 6,541 | 0.76% |
| **—** | **Final retained** | **838,562** | **96.94%** |

---

## Utility Scripts

### `pefim_utils.py`

Shared utility library used by `patch_diversity_select.py`. Provides:
- `get_topk_frames` — selects upper-quartile highest-confidence frames from a list
- `create_image_grid` — assembles a PIL image grid from a list of paths
- `find_unique_features` / `computer_clusters_from_unique_set` — greedy clustering with distance threshold, saving per-cluster grids and a pickle of cluster metadata

### `patch_diversity_select.py`

Dual-mode patch selection using Virchow2 features:

**Default mode (good-patch selection):** Per-slide — loads features from H5, optionally excludes bad patches, applies top-quality quantile filter, runs DBSCAN clustering, writes one representative per cluster to `--out_csv`. Produces a diverse, high-quality patch subset.

**`--invert` mode (bad-patch characterisation):** Cross-slide — collects all excluded patches, runs PCA (→50d) then DBSCAN globally, writes cluster representatives to `--out_csv`. Used for artefact archetype discovery: at d=20, 5 broad artefact archetypes were identified. Representatives can also be fed directly into `qc_find_nn.py` as `--bad_csv` to seed Step 3 without manual UMAP annotation.

---

## Infrastructure

All compute jobs run as Kubernetes Batch Jobs in the `ibd-plexus-research` namespace using `harbor.csis.astrazeneca.net/azimuth-demo/cpu-codeserver:latest`. GPU jobs (GrandQC) request one NVIDIA GPU. CPU jobs use 4–8 cores and 16–32 GiB RAM. Three persistent volumes are mounted into every job: `kgbk271-ibd-volume`, `shared-data`, and `kgbk271-ibd-workspace`.

---

## Deprecated Scripts (`_deprecated/`)

**`_deprecated/`** — top-level prototypes and entire filter branches not used in production:

| Script / Folder | Reason deprecated |
|-----------------|-------------------|
| `blur_analysis.py` | Early standalone version; superseded by `01_laplacien/01_blur_analysis.py` |
| `artifact_analysis.py` | Early GrandQC prototype combining tissue masking and artifact detection in one script; superseded by `02_grandqc/01_run_grandqc.py` |
| `entropy/` | Shannon entropy threshold exploration and filter; correlation analysis showed entropy added little over Laplacian alone |
| `otsu/` | Otsu tissue-fraction threshold exploration and filter; similarly found redundant with Laplacian |

**`02_grandqc/_deprecated/`** — unused filter within the GrandQC step:

| Script | Reason deprecated |
|--------|-------------------|
| `filter_grandqc_oof.py` | Out-of-focus filter using GrandQC class 5; not applied in the final pipeline (Laplacian already removes most OOF patches) |
