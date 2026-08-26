# PRISM2 Pipeline

End-to-end scripts for running [PRISM2](https://huggingface.co/paige-ai/prism2) on IBD-PLEXUS whole-slide images. PRISM2 is a vision-language model that takes pre-computed Virchow2 patch embeddings as input and produces slide-level embeddings, free-text pathology reports, and P(Yes) scores for 11 structured histological concepts (UAMP terms).

**Model weights:** `/home/jovyan/shared-data/users/kexin/models/VLM/prism2` (offline, no download needed)  
**Python environment:** `prism2` conda env for all GPU steps; `trident` env for CPU visualisation steps

---

## Datasets

All datasets live under `/home/jovyan/kgbk271-ibd-volume/data/processed/` and contain Virchow2 patch features at `20x_224px_0px_overlap/features_virchow2/`. Results land under `results/prism2_<dataset>/`.

| Short name | Directory suffix | Description |
|---|---|---|
| `trident_processed` | `trident_processed` | Original TRIDENT output, no additional QC |
| `tissue_threshold_15` | `tissue_threshold_15` | ≥15% tissue coverage filter only |
| `filtered` | `tissue_threshold_15_filtered` | + Laplacian blur (t=100) + faint-patch intensity (p98) |
| `no_darkspot` | `tissue_threshold_15_filtered_no_darkspot` | + GrandQC dark-spot removal |
| **`manual_knn`** | `tissue_threshold_15_filtered_no_darkspot_manual_knn` | **+ manual KNN artefact curation — current best** |

---

## Pipeline Overview

```
[01] intensity_filter.py          patch_intensity.parquet
[02] cluster_qc.py                cluster_qc.html  ← manual curation here
[03] make_filtered_features.py    tissue_threshold_15_filtered/features_virchow2/
         │
         ▼  (image_qc pipeline adds dark-spot and KNN filters → manual_knn dataset)
         │
[04] run_prism2_uamp.py           prism2_histological_score.csv
[05] run_prism2_uamp_repeat5.py   reproducibility_stats.csv        ← optional
[06] run_prism2_confidence_rating.py  confidence_rating.csv         ← optional
[07] run_prism2_reports.py        prism2_reports.jsonl / .csv
[08] collect_prism2_results.py    (backfill only — run if JSONL is missing)
         │
         ▼
[09] umap_embeddings.py           umap_prism2_{base,diagnostic}.html + _coords.npz
[10] umap_uamp_scores.py          umap_scores_prism2_{base,diagnostic}.html
[11] umap_reports.py              umap_prism2_diagnostic_reports.html + PNGs/PDFs
[12] umap_patch_viewer.py         <slide>_umap.html  (per-slide drill-down)
[13] multi_slide_umap.py          all_slides_umap.html  (cross-cohort QC)
         │
         ▼
[14] prism2_saliency.py           <slide>.h5 + <slide>.png  (base or yes/no target)
[15] prism2_saliency_uamp.py      <slide>/<uamp_term>.h5 + .png  (all 11 terms)
         │
[16] compare_tissue_filter.py     comparison_report.html  (post-hoc, no_filter vs filtered)
```

Steps 04 and 07 are independent and can run in parallel. Steps 05 and 06 are optional reproducibility studies.

---

## Step-by-Step Commands

### Step 01 — Patch intensity filter

No arguments. Edit the hardcoded paths at the top of the script if switching datasets.

```bash
conda activate trident
python 01_intensity_filter.py
```

**Output:** `results/cluster_qc/patch_intensity.parquet`  
**Rationale:** Computes mean RGB intensity per patch (via a tiny pyramid-level crop) for use as a faint-tissue filter in step 03. Threshold p98 = 211.8 was chosen from the distribution.

---

### Step 02 — Cluster QC

```bash
conda activate trident
python 02_cluster_qc.py \
    [--feat_dir PATH]       # default: tissue_threshold_15/features_virchow2
    [--tiff_dir PATH]       # default: tiff_mpp_corrected/
    [--out_dir PATH]        # default: results/cluster_qc/
    [--n_clusters 20]       # K-means clusters
    [--pca_components 50]   # PCA dims before clustering
    [--pca_sample 50000]    # patches used to fit PCA
    [--seed 42]
```

**Output:** `results/cluster_qc/patch_clusters.parquet` + `cluster_qc.html`  
**Rationale:** PCA → MiniBatch K-means groups patches by visual similarity. The HTML lets you inspect 9 representatives per cluster and copy artifact cluster IDs to feed downstream exclusion scripts.

---

### Step 03 — Build filtered feature set

No arguments. Edit `INTENSITY_THRESHOLD` (default 211.8) and paths at the top if needed.

```bash
conda activate trident
python 03_make_filtered_features.py
```

**Inputs:**
- `tissue_threshold_15/features_virchow2/` — original features
- `tissue_threshold_15_remove_artifact/laplacien_t100/` — Laplacian-filtered coords (from `image_qc/`)
- `results/cluster_qc/patch_intensity.parquet` — from step 01

**Output:** `tissue_threshold_15_filtered/20x_224px_0px_overlap/features_virchow2/`

---

### Step 04 — UAMP histological scoring *(GPU required)*

Scores 11 UAMP histological terms per slide using PRISM2 `yes_no_score()`. Resumes from an existing CSV — safe to re-run after interruption.

```bash
conda activate prism2
python 04_run_prism2_uamp.py \
    --feat_dir PATH         # default: trident_processed/features_virchow2
    --results_root PATH     # default: results/prism2/
    [--batch_size 4]        # slides per forward pass
    [--gpu 0]
```

**Example for manual_knn dataset:**
```bash
python 04_run_prism2_uamp.py \
    --feat_dir /home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/features_virchow2 \
    --results_root /home/jovyan/kgbk271-ibd-volume/results/prism2_manual_knn
```

**Output:** `<results_root>/prism2_histological_score.csv` — one row per slide, one column per UAMP term (P(Yes) ∈ [0,1])

**K8s job:** `jobs/02_uamp/job_prism2_umap_manual_knn.yaml`

---

### Step 05 — Reproducibility study *(optional, GPU)*

Runs UAMP scoring N times on the same dataset to estimate run-to-run variance.

```bash
conda activate prism2
python 05_run_prism2_uamp_repeat5.py \
    --feat_dir PATH         # default: no_darkspot/features_virchow2
    --out_dir PATH          # default: results/prism2_no_darkspot/repeat_five_times/
    [--n_runs 5]
    [--batch_size 4]
    [--gpu 0]
```

**Output:** `run_N/prism2_histological_score.csv` per run + `reproducibility_stats.csv` (per-slide per-term SD) + `reproducibility_summary.csv`

**K8s job:** `jobs/02_uamp/job_prism2_umap_repeat5_tissue_threshold_15_filtered_no_darkspot.yaml`

---

### Step 06 — Confidence rating *(optional, GPU)*

Uses stochastic free-text generation (`do_sample=True`) to ask PRISM2 to output a self-reported probability for each UAMP concept. More expensive than step 05 but gives a richer confidence signal.

```bash
conda activate prism2
python 06_run_prism2_confidence_rating.py \
    --feat_dir PATH
    --out_dir PATH          # default: results/.../confidence_rating/
    [--n_runs 5]
    [--batch_size 4]
    [--temperature 0.7]
    [--max_new_tokens 8]
    [--gpu 0]
```

**Output:** `run_N/confidence_rating.csv` per run + `reproducibility_stats.csv` + `reproducibility_summary.csv`

**K8s job:** `jobs/03_confidence_rating/job_prism2_confidence_rating_tissue_threshold_15_filtered_no_darkspot.yaml`

---

### Step 07 — Free-text report generation *(GPU required)*

Generates a natural-language pathology report for every slide. Safe to interrupt and resume — already-generated slides are skipped.

```bash
conda activate prism2
python 07_run_prism2_reports.py \
    --feat_dir PATH         # default: trident_processed/features_virchow2
    --results_root PATH     # default: results/prism2/
    [--prompt "Write a report"]
    [--out_dir PATH]        # auto: <results_root>/reports/<prompt_slug>/
    [--max_new_tokens 200]
    [--batch_size 4]
    [--gpu 0]
```

To run yes/no scoring (cheaper than text generation):
```bash
python 07_run_prism2_reports.py --yes_no --prompt "Is there active inflammation?"
```

**Output:**
- `<out_dir>/<slide>.txt` — per-slide report text
- `<results_root>/prism2_reports.jsonl` — appended incrementally
- `<results_root>/prism2_reports.csv` — rebuilt on each run

**K8s job:** `jobs/04_reports/job_prism2_ibd_report.yaml`

---

### Step 08 — Backfill reports JSONL *(repair utility)*

Only needed if `prism2_reports.jsonl` was lost or was not generated during step 07 (e.g. a run predating the JSONL logic). Safe to re-run; skips already-indexed entries.

```bash
conda activate trident
python 08_collect_prism2_results.py \
    [--reports_dir PATH]    # default: results/prism2/reports/
    [--out_dir PATH]        # default: results/prism2/
```

**Output:** Appends to `prism2_reports.jsonl`, rebuilds `prism2_reports.csv`

---

### Step 09 — UMAP of slide-level embeddings *(CPU)*

Runs PCA → UMAP on PRISM2 base and/or diagnostic embeddings and saves interactive Plotly HTML with a clinical metadata colour dropdown.

```bash
conda activate trident
python 09_umap_embeddings.py \
    [--job_dir PATH]        # TRIDENT job dir; default: trident_processed
    [--out_dir PATH]        # default: <job_dir>/../../results/prism2/umap/
    [--embeddings prism2_base prism2_diagnostic]
    [--n_neighbors 30]
    [--min_dist 0.25]
```

**Example for manual_knn:**
```bash
python 09_umap_embeddings.py \
    --job_dir /home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn \
    --out_dir /home/jovyan/kgbk271-ibd-volume/results/prism2/umap/manual_knn
```

**Output per embedding type:**
- `umap_<embed>.html` — interactive Plotly scatter with colour dropdown
- `umap_<embed>.pdf` — multi-page static PDF
- `umap_<embed>_coords.npz` — coordinate cache for steps 10–11

**K8s job:** `jobs/06_umap/job_umap_manual_knn.yaml`

---

### Step 10 — Overlay UAMP scores on UMAP *(CPU)*

Requires `prism2_histological_score.csv` from step 04 and `*_coords.npz` from step 09.

```bash
conda activate trident
python 10_umap_uamp_scores.py \
    --uamp_csv PATH         # prism2_histological_score.csv
    --coords_npz PATH [PATH ...]  # one or more *_coords.npz files
    --out_dir PATH
    [--meta_csv PATH]       # optional slide_metadata.csv
```

**Example:**
```bash
python 10_umap_uamp_scores.py \
    --uamp_csv results/prism2_manual_knn/prism2_histological_score.csv \
    --coords_npz results/prism2/umap/manual_knn/umap_prism2_base_coords.npz \
                 results/prism2/umap/manual_knn/umap_prism2_diagnostic_coords.npz \
    --out_dir results/prism2/umap/manual_knn
```

**Output:** `umap_scores_<embed_name>.html` per coords file — one tab per UAMP term (Viridis) plus clinical GT variables

**K8s job:** `jobs/06_umap/job_umap_scores_manual_knn.yaml`

---

### Step 11 — Overlay report features on UMAP *(CPU)*

No CLI args. Reads hardcoded paths — edit the `DATA_ROOT` constants at the top of the script to switch datasets.

```bash
conda activate trident
python 11_umap_reports.py
```

**Requires:** Steps 07, 09, and 10 to have completed. Reads `prism2_reports.jsonl`, `umap_prism2_diagnostic_coords.npz`, `slide_metadata.csv`, and `prism2_histological_score.csv`.

**Output** (all under `results/prism2/umap/`):
- `umap_prism2_diagnostic_reports.html` — interactive HTML with report-extracted categorical variables and UAMP scores
- Per-variable PNG + PDF
- Side-by-side report-vs-ground-truth comparison plots with agreement percentages
- `results/metadata/slide_report_features.csv`

**K8s job:** `jobs/06_umap/job_umap_reports.yaml`

---

### Step 12 — Per-slide patch UMAP *(CPU)*

Interactive UMAP of a single slide's patch embeddings with hover-to-show-thumbnail. Useful for per-slide drill-down after identifying interesting slides from step 09.

```bash
conda activate trident
python 12_umap_patch_viewer.py \
    [--feat_dir PATH]
    [--slides SLIDE1 SLIDE2 ...]  # explicit slide stems
    [--n_slides 5]                # or first N slides
    [--pca_components 50]
    [--umap_neighbors 15]
    [--umap_min_dist 0.1]
    [--thumb_px 224]
    [--out_dir PATH]
```

**Output:** `<out_dir>/<slide>_umap.html` per slide

---

### Step 13 — Cross-cohort patch UMAP *(CPU)*

Combined patch-level UMAP across all slides with metadata colouring and QC annotation tools. This is the same viewer used for manual KNN curation in `image_qc/03_prism2_knn/`.

```bash
conda activate trident
python 13_multi_slide_umap.py \
    [--feat_dir PATH]           # default: tissue_threshold_15/features_virchow2
    [--tiff_dir PATH]
    [--meta_csv PATH]
    [--out_dir PATH]
    [--max_patches 10000]       # total patches sampled across all slides
    [--n_slides N]              # limit to N slides
    [--pca_components 50]
    [--umap_neighbors 15]
    [--umap_min_dist 0.1]
    [--thumb_px 160]
    [--jpeg_quality 72]
    [--seed 42]
```

**Output:** `<out_dir>/all_slides_umap.html` + `.html.gz`

---

### Step 14 — Per-slide saliency *(GPU required)*

Computes gradient × input saliency for a single slide or all slides. Two targets:
- `base` — L2 norm of the PRISM2 base embedding (fast, Perceiver only)
- `yesno` — log-odds P(Yes)/P(No) for a specific question (full model, slow)

```bash
conda activate prism2
# Single slide, base target:
python 14_prism2_saliency.py \
    --slide SLIDE_STEM \
    --target base \
    [--job_dir PATH] \
    [--gpu 0]

# Single slide, yes/no target:
python 14_prism2_saliency.py \
    --slide SLIDE_STEM \
    --target yesno \
    --question "Is there active inflammation?" \
    [--gpu 0]

# All slides:
python 14_prism2_saliency.py --all --target base [--gpu 0]
```

**Output:** `<job_dir>/20x_224px_0px_overlap/prism2_saliency_<target>/<slide>.h5` + `<slide>.png`

---

### Step 15 — UAMP saliency for one slide *(GPU required)*

Computes saliency for all 11 UAMP terms on a single named slide. First scores P(Yes) for each term, then backpropagates the yes/no log-odds to get per-tile importance. Use `--compute-only` first (saves H5 files), then `--viz-only` to regenerate figures without re-running inference.

```bash
conda activate prism2
# Full run (inference + figures):
python 15_prism2_saliency_uamp.py \
    --slide SLIDE_STEM \
    [--gpu 0]

# Split into two passes (useful on shared GPU):
python 15_prism2_saliency_uamp.py --slide SLIDE_STEM --compute-only [--gpu 0]
python 15_prism2_saliency_uamp.py --slide SLIDE_STEM --viz-only
```

Default slide: `10407210HE1`. Feat dir and TIFF dir are hardcoded to `trident_processed` — edit the constants at the top to change dataset.

**Output:** `prism2_saliency_uamp/<slide>/<term>.h5` + `<term>.png` per UAMP term (11 files each)

**K8s job:** `jobs/07_saliency/job_prism2_saliency_umap.yaml`

---

### Step 16 — Filter condition comparison *(CPU)*

Post-hoc comparison of unfiltered vs `tissue_threshold_15_filtered` datasets. Requires RF training results and UMAP coordinates from both conditions to already exist. No CLI args — edit the `DATA_ROOT` constants at the top of the script.

```bash
conda activate trident
python 16_compare_tissue_filter.py
```

**Output** (all under `results/comparison_no_filter_vs_filtered/`):
- `comparison_rf.csv` — AUC delta table
- `comparison_histoscore.csv` — UAMP score distribution shift
- `comparison_umap.csv` — UMAP structure comparison
- `comparison_report.html` — Plotly bar charts + scatter UMAP

---

## Running on Kubernetes

All GPU jobs use the `prism2` environment. All CPU jobs use the `trident` environment. Jobs run in the `ibd-plexus-research` namespace.

```
jobs/
├── 01_embeddings/     # PRISM2 base + diagnostic embeddings (GPU)
│   ├── job_prism2_embeddings_manual_knn.yaml    ← active
│   └── _deprecated/   # earlier dataset iterations
├── 02_uamp/           # UAMP histological scoring (GPU)
│   ├── job_prism2_umap_manual_knn.yaml          ← active
│   ├── job_prism2_umap_repeat5_*.yaml           ← reproducibility study
│   └── _deprecated/
├── 03_confidence_rating/  # Diagnostic confidence scoring (GPU)
│   └── job_prism2_confidence_rating_*.yaml
├── 04_reports/        # Free-text report generation (GPU)
│   └── job_prism2_ibd_report.yaml
├── 05_patch_viewer/   # Patch-level UMAP with thumbnails (CPU)
├── 06_umap/           # Slide-level UMAP + score overlays (CPU)
│   ├── job_umap_manual_knn.yaml                 ← umap_embeddings.py
│   ├── job_umap_scores_manual_knn.yaml          ← umap_uamp_scores.py
│   ├── job_umap_reports.yaml                    ← umap_reports.py
│   └── _deprecated/
└── 07_saliency/       # Saliency maps (GPU)
    └── job_prism2_saliency_umap.yaml
```

### Standard run order for manual_knn

Steps 1 and 2 can run in parallel.

```bash
# Step 1 — embeddings (GPU, ~2 h)
kubectl apply -f jobs/01_embeddings/job_prism2_embeddings_manual_knn.yaml

# Step 2 — UAMP scoring (GPU, ~4 h) — run in parallel with step 1
kubectl apply -f jobs/02_uamp/job_prism2_umap_manual_knn.yaml

# Step 3 — UMAP (CPU) — wait for step 1 to finish
kubectl apply -f jobs/06_umap/job_umap_manual_knn.yaml

# Step 4 — UAMP score overlays (CPU) — wait for steps 2 and 3
kubectl apply -f jobs/06_umap/job_umap_scores_manual_knn.yaml
```

| Step | Output |
|------|--------|
| 1 | `manual_knn/.../prism2_base/*.h5` and `prism2_diagnostic/*.h5` |
| 2 | `results/prism2_manual_knn/prism2_histological_score.csv` |
| 3 | `results/prism2/umap/manual_knn/umap_prism2_{base,diagnostic}.html` + `*_coords.npz` |
| 4 | `results/prism2/umap/manual_knn/umap_scores_prism2_{base,diagnostic}.html` |

---

## QC Utilities

### `01_intensity_filter.py` + `02_cluster_qc.py` + `03_make_filtered_features.py`

These three scripts build the `tissue_threshold_15_filtered` dataset. They need to be re-run only if you are creating a new filtered dataset from scratch. The `manual_knn` dataset (current best) was built on top of `tissue_threshold_15_filtered` using the additional image QC steps in `image_qc/`.

### `08_collect_prism2_results.py`

Only needed if `prism2_reports.jsonl` is missing. Scans the `reports/` directory for existing `.txt` files and reconstructs the JSONL index.

### `16_compare_tissue_filter.py`

Post-hoc analysis. Only meaningful after running the full pipeline on both `trident_processed` (unfiltered) and `tissue_threshold_15_filtered`, and after training RF classifiers on both. Paths are hardcoded — edit before running.
