# Patch Quality Control Report — PRISM2 Virchow2 Features

**Date:** 2026-08-25  
**Cohort:** KGBK271-IBD  
**Feature set:** Virchow2 2560-d, tissue threshold 15, 20× 224 px 0 px overlap  
**Total patch pool:** 865,070 patches across 3,318 slides

---

## Objective

Remove poor-quality patches (artefacts, pen marks, edge effects, staining failures) from the Virchow2 feature pool before downstream model training.  The approach is semi-automatic: a human identifies exemplar bad patches visually in a UMAP, then nearest-neighbour search in feature space propagates the exclusion to similar patches across the full cohort.

---

## Tools developed

All scripts live in `code/image_qc/prism2_knn/`.

### `multi_slide_umap.py`
Builds a joint interactive HTML UMAP viewer across all slides.

- Loads Virchow2 features from per-slide H5 files; proportionally subsamples up to `--max_patches` (default 10,000) patches across all slides.  
- Runs PCA (2560 → 50 d) then UMAP (50 → 2 d).  
- Embeds patch thumbnails (extracted from pyramidal TIFFs via OpenSlide) as base64 JPEG.  
- Supports `--exclusion_csv` to filter bad patches before sampling, so replotted UMAPs reflect the cleaned pool.
- Supports `--out_name` to write the output HTML under a custom filename (e.g. `all_slides_umap_iter2.html`).

**QC toolbar** added to the viewer HTML:

| Control | Function |
|---|---|
| Lasso / Box select | Plotly native selection tools |
| Mark Bad | Adds lasso-selected patches to the bad set (shown in red) |
| Find NN (K=) | Expands the bad set by K nearest neighbours in UMAP 2-D space |
| Export CSV | Downloads `bad_patches.csv` with `global_index, slide_id, x, y, umap_x, umap_y` |
| Reset All | Clears all markings |

### `qc_find_nn.py`
Takes the exported bad-patch CSV and searches for nearest neighbours in the **full** Virchow2 feature space (all 865k patches), outputting an exclusion list and an optional gallery HTML.

Key arguments:

| Argument | Description |
|---|---|
| `--bad_csv` | CSV exported from the UMAP viewer |
| `--dist_threshold` | L2 distance radius for neighbour search (recommended; replaces fixed-K) |
| `--k` | Fixed-K fallback when `--dist_threshold` is not set |
| `--thresholds_csv` | Per-patch threshold CSV exported from the gallery UI |
| `--n_intervals` | Number of distance bins in gallery (default 5) |
| `--samples_per_interval` | Thumbnails shown per bin (default 5) |
| `--html_out` | Path for gallery HTML output |
| `--out_csv` | Path for exclusion list CSV |

#### Gallery HTML — per-patch threshold UI

The gallery HTML embeds a `QUERIES` JSON array containing, for each bad patch, the distance bin edges and per-bin patch counts.  The interface provides:

- **Sticky global controls bar** — a single threshold value can be applied to all queries at once.  
- **Per-query threshold slider + numeric input** — each bad patch can be assigned an independent distance cutoff; bins beyond the threshold dim out in real time.  
- **Live exclusion count** — each row shows the estimated patches excluded at the current threshold; the global bar shows the running pre-deduplication total.  
- **Export thresholds CSV** — downloads `per_patch_thresholds.csv` with columns `slide_id, x, y, threshold`.

The exported CSV is then fed back to `qc_find_nn.py` via `--thresholds_csv` to generate a per-patch-calibrated exclusion list.

---

## Workflow

```
── Iter 1 ──────────────────────────────────────────────────────────────────────

1. python multi_slide_umap.py
        → tissue_threshold_15/all_slides_umap.html

2. Open HTML, lasso-select poor-quality patches → Mark Bad → Export CSV
        → bad_patches_manual_all_slides_umap.csv  (59 patches)

3. python qc_find_nn.py \
       --bad_csv bad_patches_manual_all_slides_umap.csv \
       --dist_threshold 40 \
       --html_out nn_gallery_dist40_intervals.html \
       --out_csv exclusion_list_dist40.csv
        → 13,524 patches excluded

── Iter 2 ──────────────────────────────────────────────────────────────────────

4. python multi_slide_umap.py \
       --exclusion_csv exclusion_list_dist40.csv \
       --out_name all_slides_umap_iter2.html
        → cleaned UMAP (iter1 exclusions applied)

5. Open all_slides_umap_iter2.html, select remaining artefacts → Export CSV
        → bad_patches_iter2.csv  (13 patches)

6. python qc_find_nn.py \
       --bad_csv bad_patches_iter2.csv \
       --dist_threshold 40 \
       --html_out nn_gallery_dist40_intervals.html \
       --out_csv exclusion_list_dist40.csv
        → gallery with per-patch threshold sliders

7. Adjust per-query thresholds in gallery, Export thresholds CSV
        → per_patch_thresholds.csv

8. python qc_find_nn.py \
       --bad_csv bad_patches_iter2.csv \
       --thresholds_csv per_patch_thresholds.csv \
       --out_csv exclusion_list_per_patch_thresh.csv
        → 2,685 patches excluded (per-patch calibrated)
```

---

## Iter 1 — Manual annotation and threshold exploration

| Item | Value |
|---|---|
| Bad patches manually marked | 59 |
| Slides represented | 55 |
| Source | Interactive lasso selection in `all_slides_umap.html` |

Patches selected were predominantly artefacts: pen marks, tissue folds, staining failures, and edge-of-tissue regions containing minimal tissue.

### Threshold exploration

| Threshold | Patches excluded | Slides affected | % of pool |
|---|---|---|---|
| K = 10 (fixed) | 587 | — | 0.07% |
| K = 50 (fixed) | 2,116 | — | 0.24% |
| d ≤ 40 | **13,524** | 2,282 | **1.56%** |
| d ≤ 50 | 201,791 | 3,314 | 23.3% |

d = 50 was judged too aggressive (removes 23% of the pool, including patches visible in the gallery as normal tissue at d > 20).  The interval gallery (bins 0–8–16–24–32–40) confirmed that patches at d > 32 are largely normal-appearing tissue that differs from the query only in staining intensity, not quality.

### Iter 1 selected threshold: **d = 40**

| Metric | Value |
|---|---|
| Median neighbours per query | 1,742 |
| Maximum neighbours per query | 6,478 |
| **Total excluded** | **13,524 (1.56% of pool)** |
| Slides with ≥1 patch removed | 2,282 / 3,318 (68.8%) |

---

## Iter 2 — Second-pass annotation with per-patch thresholds

After plotting the cleaned UMAP (`all_slides_umap_iter2.html`, iter1 exclusions applied), 13 additional bad patches were identified across 10 slides.

### Per-patch threshold calibration

Rather than applying a single global distance cutoff, each query patch was assigned an individual threshold via the gallery UI sliders.  The rationale: artefact types differ in how tightly they cluster in feature space — pen marks cluster very tightly (low threshold appropriate) while staining failures have broader neighbourhoods.

| slide_id | x | y | Threshold | Notes |
|---|---|---|---|---|
| 10601053HE1 | 15456 | 18144 | 32.0 | |
| 10713332HE1 | 18816 | 26880 | 24.0 | |
| 10731074HE1 | 8064 | 14112 | 15.6 | tight cluster |
| 10731074HE1 | 6720 | 17472 | 16.0 | tight cluster |
| 10799426HE1 | 6720 | 16800 | 40.0 | broad neighbourhood |
| 10816881HE1 | 24192 | 6048 | 25.2 | |
| 10822592HE1 | 23520 | 8736 | 29.6 | |
| 10835454HE1 | 10752 | 13440 | 0.0 | excluded only the single patch (isolated artefact) |
| 10988610HE1 | 33600 | 27552 | 38.0 | |
| 10991797HE1 | 34944 | 6720 | 30.4 | |
| 11007122HE1 | 16128 | 13440 | 30.8 | |
| 11007122HE1 | 6720 | 16128 | 40.0 | broad neighbourhood |
| 11007122HE1 | 8736 | 17472 | 25.2 | |
| 11021944HE1 | 20832 | 8736 | 4.0 | very tight cluster |

### Iter 2 results

| Metric | Value |
|---|---|
| Bad patches annotated (iter 2) | 13 |
| Global radius searched (max threshold) | 40.0 |
| Median neighbours per query (after per-patch filter) | 114 |
| Maximum neighbours per query | 1,415 |
| **Total excluded (iter 2)** | **2,685 (0.31% of pool)** |

Per-patch calibration reduced the exclusion count from 28,563 (uniform d=40) to 2,685, eliminating over-exclusion for queries with tight artefact clusters.

---

## Cumulative exclusion

| Iteration | Exclusion list | Patches excluded |
|---|---|---|
| Iter 1 | `iter1/filtering/exclusion_list_dist40.csv` | 13,524 |
| Iter 2 | `iter2/filtering/exclusion_list_per_patch_thresh.csv` | 2,685 |
| **Combined (union)** | merge both CSVs and dedup on `(slide_id, x, y)` | **~16,200 est.** |

To merge:

```python
import pandas as pd

e1 = pd.read_csv("iter1/filtering/exclusion_list_dist40.csv")
e2 = pd.read_csv("iter2/filtering/exclusion_list_per_patch_thresh.csv")
combined = pd.concat([e1, e2]).drop_duplicates(subset=["slide_id", "x", "y"])
combined.to_csv("exclusion_list_combined.csv", index=False)
print(f"Combined: {len(combined):,} patches excluded")
```

---

## Output files

All files under `image_preprocessing/patch_qc/prism2_knn/`.

```
iter1/
  bad_patches/bad_patches_manual_all_slides_umap.csv   # 59 manually marked patches
  filtering/exclusion_list_dist40.csv                  # 13,524 patches, d ≤ 40
  filtering/nn_gallery_dist40_intervals.html           # interval gallery (d ≤ 40)
  filtering/nn_gallery_dist50_intervals.html           # comparison gallery (d ≤ 50)
  umap/all_slides_umap.html                            # original UMAP (unfiltered)

iter2/
  bad_patches/bad_patches_iter2.csv                    # 13 patches from cleaned UMAP
  filtering/per_patch_thresholds.csv                   # per-query thresholds from UI
  filtering/exclusion_list_dist40.csv                  # 28,563 patches, uniform d ≤ 40
  filtering/exclusion_list_per_patch_thresh.csv        # 2,685 patches, per-patch calibrated
  filtering/nn_gallery_dist40_intervals.html           # interval gallery with threshold sliders
  umap/all_slides_umap_iter2.html                      # cleaned UMAP (iter1 exclusions applied)
```

---

## Usage for downstream pipelines

```python
import pandas as pd, h5py, numpy as np

excl = pd.read_csv("exclusion_list_combined.csv")
excl_set = set(zip(excl["slide_id"], excl["x"].astype(int), excl["y"].astype(int)))

with h5py.File(f"{slide_id}.h5") as f:
    feats  = f["features"][:]
    coords = f["coords"][:]

keep = np.array([(slide_id, int(cx), int(cy)) not in excl_set for cx, cy in coords])
feats  = feats[keep]
coords = coords[keep]
```

To run another iteration:

```bash
# Replot UMAP with combined exclusions
python multi_slide_umap.py \
  --exclusion_csv exclusion_list_combined.csv \
  --out_name all_slides_umap_iter3.html

# Run NN search for new bad patches
python qc_find_nn.py \
  --bad_csv bad_patches_iter3.csv \
  --dist_threshold 40 \
  --html_out nn_gallery_iter3.html \
  --out_csv exclusion_list_iter3_dist40.csv

# Adjust per-patch thresholds in gallery, then apply:
python qc_find_nn.py \
  --bad_csv bad_patches_iter3.csv \
  --thresholds_csv per_patch_thresholds_iter3.csv \
  --out_csv exclusion_list_iter3.csv
```
