# Patch Filtering Pipeline Report — PRISM2 Virchow2 Features

**Date:** 2026-08-26  
**Cohort:** KGBK271-IBD  
**Original patch pool:** 865,070 patches across 9,954 slides  
**Feature set:** Virchow2 2560-d, tissue threshold 15, 20× 224 px 0 px overlap  

---

## Overview

Patches were filtered through four sequential quality control stages to remove artefacts, blurry regions, dark-spot contamination, and manually identified poor-quality patches. Each stage is fully reproducible from the scripts listed below.

```
Original pool (865,070)
    │
    ├─ Laplacian blur filter (t100)
    ├─ Faint/white intensity filter (≥211.8)
    │       → tissue_threshold_15_filtered/          846,830 patches  (−2.11%)
    │
    ├─ GrandQC dark spots filter (class 3, >10%)
    │       → tissue_threshold_15_filtered_no_darkspot/   845,103 patches  (−0.20%)
    │
    └─ Manual KNN exclusion (iter1 + iter2)
            → tissue_threshold_15_filtered_no_darkspot_manual_knn/   838,562 patches  (−0.76%)
```

---

## Filter stages

### Stage 1 — Laplacian blur filter (t100)

Patches with Laplacian variance below 100 are removed as too blurry to carry reliable morphological signal.  Laplacian variance measures the sharpness of a greyscale image; low values indicate out-of-focus or motion-blurred patches.

**Script:** `code/image_qc/laplacien/filter_laplacien.py`  
**Threshold:** Laplacian variance < 100

### Stage 2 — Faint/white intensity filter

Patches with mean RGB intensity ≥ 211.8 (the 98th percentile across the cohort) are removed as near-white or background patches that slipped through the tissue mask.

**Script:** `code/prism2/make_filtered_features.py` (combined with Stage 1)  
**Threshold:** mean RGB ≥ 211.8

### Stage 3 — GrandQC dark spots filter

GrandQC (MPP 1.0 model) generates per-slide artefact masks with seven classes. Patches where more than 10% of the overlapping mask pixels are classified as dark spots (class 3) are removed.  Additionally, 8 slides with extreme dark-spot contamination (visually confirmed) are excluded entirely.

**Script:** `code/image_qc/grandqc/filter_grandqc_darkspot.py`  
**Threshold:** dark-spot fraction > 10%

GrandQC class mapping:

| Class | Label |
|-------|-------|
| 1 | Clean tissue |
| 2 | Tissue fold |
| **3** | **Dark spots ← removed** |
| 4 | Pen marks |
| 5 | Air bubble / slide edge |
| 6 | Out-of-focus |
| 7 | Background |

Fully excluded slides (all patches removed):

```
10502321HE101  11055940HE1  11007548HE1  10924448HE1  10537759HE101
10965446HE1    11005300HE1  11025121HE1
```

### Stage 4 — Manual KNN exclusion (iter1 + iter2)

Semi-automatic removal of artefact patches not captured by pixel-level filters. A human identified exemplar bad patches (pen marks, tissue folds, staining failures, edge artefacts) via interactive UMAP lasso selection, then nearest-neighbour search in Virchow2 feature space propagated each exclusion to similar patches across the full cohort.

Two iterations were performed:

| Iteration | Bad patches annotated | Threshold | Strategy | Patches excluded |
|---|---|---|---|---|
| Iter 1 | 59 (manual, UMAP) | d ≤ 40 (uniform) | Fixed radius NN | 13,524 |
| Iter 2 | 13 (manual, cleaned UMAP) | d = 4–40 (per-patch) | Per-patch threshold NN | 2,685 |
| **Merged** | **72** | — | Dedup union | **15,875** |

Of the 15,875 patches in the merged exclusion list, 9,334 were already removed by prior filters.  The net additional patches removed at this stage is **6,541**.

**Scripts:**
- `code/image_qc/prism2_knn/multi_slide_umap.py` — interactive UMAP viewer with lasso selection
- `code/image_qc/prism2_knn/qc_find_nn.py` — full-pool NN search + interval gallery with per-patch threshold UI
- `code/image_qc/prism2_knn/apply_knn_exclusion.py` — applies merged exclusion list to filtered H5 files

See `code/image_qc/prism2_knn/report_patch_qc.md` for full KNN pipeline detail.

---

## Summary statistics

| Filter | Patches removed | % of total (865,070) |
|---|---:|---:|
| Laplacian blur (t100) | 4,977 | 0.57% |
| Faint/white intensity (≥211.8) | 13,263 | 1.53% |
| **Subtotal — Laplacian + intensity** | **18,240** | **2.11%** |
| *Kept after Laplacian + intensity* | *846,830* | *97.89%* |
| | | |
| GrandQC dark spots (class 3, >10%) | 1,727 | 0.20% |
| *Kept after dark spots* | *845,103* | *97.69%* |
| | | |
| Manual KNN exclusion (iter1 + iter2) | 6,541 ¹ | 0.76% |
| ***Kept after KNN exclusion*** | ***838,562*** | ***96.94%*** |
| | | |
| **Total removed (all filters)** | **26,508** | **3.06%** |

¹ Exclusion list contains 15,875 patches; 9,334 were already removed by prior filters and are not double-counted.

---

## Output directories

All outputs under `/home/jovyan/kgbk271-ibd-volume/data/processed/`.  Each directory contains `20x_224px_0px_overlap/features_virchow2/*.h5` files with `coords` (M, 2) int64 and `features` (M, 2560) float32, gzip-4 compressed.

| Directory | Patches | Filters applied |
|---|---:|---|
| `tissue_threshold_15/` | 865,070 | Tissue threshold only |
| `tissue_threshold_15_filtered/` | 846,830 | + Laplacian + intensity |
| `tissue_threshold_15_filtered_no_darkspot/` | 845,103 | + GrandQC dark spots |
| **`tissue_threshold_15_filtered_no_darkspot_manual_knn/`** | **838,562** | **+ Manual KNN exclusion** |

---

## Artefact type discovery

Separately from the sequential filtering, artefact patches were clustered in Virchow2 feature space (PCA 50d + DBSCAN) to characterise the types of artefacts present in the cohort.  The full bad-patch pool (Laplacian-rejected + manual annotations + NN-expanded exclusion lists = 16,992 patches) was clustered at two granularities:

| Distance threshold | Clusters | Coverage | Use |
|---|---:|---:|---|
| d = 10 | 86 | 3,890 / 16,992 | Fine-grained artefact subtypes |
| d = 20 | 5 | 16,920 / 16,992 | 5 broad artefact archetypes |

The d=20 result shows that **nearly all artefacts in the cohort fall into 5 broad morphological groups**.  Cluster representative patches are available in `image_preprocessing/patch_qc/artefact_clusters/bad_cluster_reps_d20.csv` and can seed future automated NN exclusion rounds without manual UMAP annotation.

**Scripts:**
- `code/image_qc/pefim_utils.py` — original clustering utilities (endoscopy)
- `code/image_qc/patch_diversity_select.py` — adapted for Virchow2 patches; supports both good-patch diversity selection (default) and bad-patch artefact clustering (`--invert`)

Gallery HTMLs (thumbnails for each cluster):
- `image_preprocessing/patch_qc/artefact_clusters/artefact_gallery_d10.html`
- `image_preprocessing/patch_qc/artefact_clusters/artefact_gallery_d20.html`

---

## Usage in downstream pipelines

Load the final filtered features directly from `tissue_threshold_15_filtered_no_darkspot_manual_knn/`:

```python
import h5py
import numpy as np
from pathlib import Path

FEAT_DIR = Path(
    "/home/jovyan/kgbk271-ibd-volume/data/processed/"
    "tissue_threshold_15_filtered_no_darkspot_manual_knn/"
    "20x_224px_0px_overlap/features_virchow2"
)

with h5py.File(FEAT_DIR / f"{slide_id}.h5") as f:
    feats  = f["features"][:]   # (M, 2560) float32
    coords = f["coords"][:]     # (M, 2)    int64
```

To add a future exclusion round without rerunning all filters:

```bash
# 1. Annotate new bad patches in the cleaned UMAP
python code/image_qc/prism2_knn/multi_slide_umap.py \
    --exclusion_csv image_preprocessing/patch_qc/prism2_knn/exclusion_list_merged.csv \
    --out_name all_slides_umap_iter3.html

# 2. Export bad patches, run NN search with per-patch thresholds
python code/image_qc/prism2_knn/qc_find_nn.py \
    --bad_csv bad_patches_iter3.csv \
    --dist_threshold 40 \
    --html_out nn_gallery_iter3.html \
    --out_csv exclusion_list_iter3.csv

# 3. Merge with existing exclusion list
python - << 'EOF'
import pandas as pd
e_old = pd.read_csv("exclusion_list_merged.csv")
e_new = pd.read_csv("exclusion_list_iter3.csv")
merged = pd.concat([e_old, e_new]).drop_duplicates(subset=["slide_id","x","y"])
merged.to_csv("exclusion_list_merged_iter3.csv", index=False)
print(f"Combined: {len(merged):,} patches excluded")
EOF

# 4. Apply to filtered features
python code/image_qc/prism2_knn/apply_knn_exclusion.py \
    --excl_csv exclusion_list_merged_iter3.csv \
    --out_dir .../tissue_threshold_15_filtered_no_darkspot_manual_knn_iter3/...
```
