# PRISM2 Pipeline

End-to-end scripts for running [PRISM2](https://huggingface.co/paige-ai/prism2) on IBD-PLEXUS whole-slide images. PRISM2 is a vision-language model that takes pre-computed Virchow2 patch embeddings as input and produces slide-level embeddings, free-text pathology reports, and P(Yes) scores for 11 structured histological concepts (UAMP terms).

**Model weights:** `/home/jovyan/shared-data/users/kexin/models/VLM/prism2` (offline, no download needed)  
**Python environment:** `prism2` conda env for all GPU steps; `trident` env for CPU visualisation steps

> **Note — preprocessing:** The intensity filter, cluster QC, and filtered-feature build that produce the `tissue_threshold_15_filtered` dataset live in `code/image_qc/04_intensity_cluster/`. Run those first (see `image_qc/README.md`) before running this pipeline.

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
(Preprocessing — see image_qc/04_intensity_cluster/)
  01_intensity_filter.py  →  patch_intensity.parquet
  02_cluster_qc.py        →  cluster_qc.html  ← manual curation step
  03_make_filtered_features.py  →  tissue_threshold_15_filtered/features_virchow2/
         │
         ▼  (image_qc adds dark-spot and KNN filters → manual_knn dataset)
         │
[01] run_prism2_uamp.py           prism2_histological_score.csv
[02] run_prism2_uamp_repeat5.py   reproducibility_stats.csv        ← optional
[03] run_prism2_confidence_rating.py  confidence_rating.csv         ← optional
[04] run_prism2_reports.py        prism2_reports.jsonl / .csv
         │
         ▼
[05] umap_embeddings.py           umap_prism2_{base,diagnostic}.html + _coords.npz
[06] umap_uamp_scores.py          umap_scores_prism2_{base,diagnostic}.html
[07] umap_reports.py              umap_prism2_diagnostic_reports.html + PNGs
[08] umap_patch_viewer.py         <slide>_umap.html  (per-slide drill-down)
[09] multi_slide_umap.py          all_slides_umap.html  (cross-cohort QC)
         │
         ▼
[10] prism2_saliency.py           <slide>.h5 + <slide>.png  (base or yes/no target)
[11] prism2_saliency_uamp.py      <slide>/<uamp_term>.h5 + .png  (all 11 terms)
         │
[12] compare_tissue_filter.py     comparison_report.html  (post-hoc, no_filter vs filtered)
[13] run_prism2_temperature_sampling.py  temperature sampling study
[14] compare_histological_scores.py  cross-dataset score comparison
[15] plot_temperature_scores.py   plots from step 13
[16] attention_heatmap.py         patch attention visualisation
```

Steps 01 and 04 are independent and can run in parallel. Steps 02 and 03 are optional reproducibility/confidence studies.

---

## Step-by-Step Commands

### Step 01 — UAMP histological scoring *(GPU required)*

Scores 11 UAMP histological terms per slide using PRISM2 `yes_no_score()`. Resumes from an existing CSV — safe to re-run after interruption.

```bash
conda activate prism2
python 01_run_prism2_uamp.py \
    --feat_dir PATH         # default: trident_processed/features_virchow2
    --results_root PATH     # default: results/prism2/
    [--batch_size 4]        # slides per forward pass
    [--gpu 0]
```

**Example for manual_knn dataset:**
```bash
python 01_run_prism2_uamp.py \
    --feat_dir /home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn/20x_224px_0px_overlap/features_virchow2 \
    --results_root /home/jovyan/kgbk271-ibd-volume/results/prism2_manual_knn
```

**Output:** `<results_root>/prism2_histological_score.csv` — one row per slide, one column per UAMP term (P(Yes) ∈ [0,1])

**K8s job:** `jobs/02_uamp/job_prism2_umap_manual_knn.yaml`

---

### Step 02 — Reproducibility study *(optional, GPU)*

Runs UAMP scoring N times on the same dataset to estimate run-to-run variance.

```bash
conda activate prism2
python 02_run_prism2_uamp_repeat5.py \
    --feat_dir PATH         # default: no_darkspot/features_virchow2
    --out_dir PATH          # default: results/prism2_no_darkspot/repeat_five_times/
    [--n_runs 5]
    [--batch_size 4]
    [--gpu 0]
```

**Output:** `run_N/prism2_histological_score.csv` per run + `reproducibility_stats.csv` + `reproducibility_summary.csv`

---

### Step 03 — Confidence rating *(optional, GPU)*

Uses stochastic free-text generation (`do_sample=True`) to ask PRISM2 to output a self-reported probability for each UAMP concept.

```bash
conda activate prism2
python 03_run_prism2_confidence_rating.py \
    --feat_dir PATH
    --out_dir PATH          # default: results/.../confidence_rating/
    [--n_runs 5]
    [--batch_size 4]
    [--temperature 0.7]
    [--max_new_tokens 8]
    [--gpu 0]
```

**Output:** `run_N/confidence_rating.csv` per run + `reproducibility_stats.csv`

---

### Step 04 — Free-text report generation *(GPU required)*

Generates a natural-language pathology report for every slide. Safe to interrupt and resume — already-generated slides are skipped.

```bash
conda activate prism2
python 04_run_prism2_reports.py \
    --feat_dir PATH         # default: trident_processed/features_virchow2
    --results_root PATH     # default: results/prism2/
    [--prompt "Write a report"]
    [--out_dir PATH]        # auto: <results_root>/reports/<prompt_slug>/
    [--max_new_tokens 200]
    [--batch_size 4]
    [--gpu 0]
```

**Output:**
- `<out_dir>/<slide>.txt` — per-slide report text
- `<results_root>/prism2_reports.jsonl` — appended incrementally
- `<results_root>/prism2_reports.csv` — rebuilt on each run

**K8s job:** `jobs/04_reports/job_prism2_ibd_report.yaml`

---

### Step 05 — UMAP of slide-level embeddings *(CPU)*

Runs PCA → UMAP on PRISM2 base and/or diagnostic embeddings and saves interactive Plotly HTML with a clinical metadata colour dropdown.

```bash
conda activate trident
python 05_umap_embeddings.py \
    [--job_dir PATH]        # TRIDENT job dir; default: trident_processed
    [--out_dir PATH]        # default: <job_dir>/../../results/prism2/umap/
    [--embeddings prism2_base prism2_diagnostic]
    [--n_neighbors 30]
    [--min_dist 0.25]
```

**Example for manual_knn:**
```bash
python 05_umap_embeddings.py \
    --job_dir /home/jovyan/kgbk271-ibd-volume/data/processed/tissue_threshold_15_filtered_no_darkspot_manual_knn \
    --out_dir /home/jovyan/kgbk271-ibd-volume/results/prism2/umap/manual_knn
```

**Output per embedding type:**
- `umap_<embed>.html` — interactive Plotly scatter with colour dropdown
- `umap_<embed>.pdf` — multi-page static PDF
- `umap_<embed>_coords.npz` — coordinate cache for steps 06–07

**K8s job:** `jobs/06_umap/job_umap_manual_knn.yaml`

---

### Step 06 — Overlay UAMP scores on UMAP *(CPU)*

```bash
conda activate trident
python 06_umap_uamp_scores.py \
    --uamp_csv PATH         # prism2_histological_score.csv
    --coords_npz PATH [PATH ...]  # one or more *_coords.npz files
    --out_dir PATH
    [--meta_csv PATH]       # optional slide_metadata.csv
```

**Example:**
```bash
python 06_umap_uamp_scores.py \
    --uamp_csv results/prism2_manual_knn/prism2_histological_score.csv \
    --coords_npz results/prism2/umap/manual_knn/umap_prism2_base_coords.npz \
                 results/prism2/umap/manual_knn/umap_prism2_diagnostic_coords.npz \
    --out_dir results/prism2/umap/manual_knn
```

**Output:** `umap_scores_<embed_name>.html` per coords file

**K8s job:** `jobs/06_umap/job_umap_scores_manual_knn.yaml`

---

### Step 07 — Overlay report features on UMAP *(CPU)*

No CLI args. Reads hardcoded paths — edit the `DATA_ROOT` constants at the top of the script to switch datasets.

```bash
conda activate trident
python 07_umap_reports.py
```

**Requires:** Steps 04, 05, and 06 to have completed.

**Output** (all under `results/prism2/umap/`):
- `umap_prism2_diagnostic_reports.html`
- Per-variable PNG + PDF
- `results/metadata/slide_report_features.csv`

**K8s job:** `jobs/06_umap/job_umap_reports.yaml`

---

### Step 08 — Per-slide patch UMAP *(CPU)*

```bash
conda activate trident
python 08_umap_patch_viewer.py \
    [--feat_dir PATH]
    [--slides SLIDE1 SLIDE2 ...]
    [--n_slides 5]
    [--out_dir PATH]
```

**Output:** `<out_dir>/<slide>_umap.html` per slide

---

### Step 09 — Cross-cohort patch UMAP *(CPU)*

Combined patch-level UMAP across all slides. Also used for manual KNN curation in `image_qc/03_prism2_knn/`.

```bash
conda activate trident
python 09_multi_slide_umap.py \
    [--feat_dir PATH]
    [--max_patches 10000]
    [--n_slides N]
    [--out_dir PATH]
```

**Output:** `<out_dir>/all_slides_umap.html` + `.html.gz`

---

### Step 10 — Per-slide saliency *(GPU required)*

```bash
conda activate prism2
# Base target:
python 10_prism2_saliency.py --slide SLIDE_STEM --target base [--gpu 0]
# Yes/no target:
python 10_prism2_saliency.py --slide SLIDE_STEM --target yesno \
    --question "Is there active inflammation?" [--gpu 0]
# All slides:
python 10_prism2_saliency.py --all --target base [--gpu 0]
```

**Output:** `prism2_saliency_<target>/<slide>.h5` + `<slide>.png`

---

### Step 11 — UAMP saliency for one slide *(GPU required)*

```bash
conda activate prism2
python 11_prism2_saliency_uamp.py --slide SLIDE_STEM [--gpu 0]
# Or split into two passes:
python 11_prism2_saliency_uamp.py --slide SLIDE_STEM --compute-only [--gpu 0]
python 11_prism2_saliency_uamp.py --slide SLIDE_STEM --viz-only
```

**Output:** `prism2_saliency_uamp/<slide>/<term>.h5` + `<term>.png` per UAMP term

---

### Step 12 — Filter condition comparison *(CPU)*

Post-hoc: compares unfiltered vs `tissue_threshold_15_filtered`. Edit `DATA_ROOT` constants at the top of the script.

```bash
conda activate trident
python 12_compare_tissue_filter.py
```

**Output** (under `results/comparison_no_filter_vs_filtered/`): comparison CSVs + `comparison_report.html`

---

### Steps 13–16 — Exploratory analyses

| Script | What it does |
|--------|-------------|
| `13_run_prism2_temperature_sampling.py` | UAMP scoring with non-default sampling temperature; launcher: `13_run_prism2_temperature_sampling.sh` |
| `14_compare_histological_scores.py` | Cross-dataset histological score comparison |
| `15_plot_temperature_scores.py` | Visualises output of step 13 |
| `16_attention_heatmap.py` | Patch attention heatmap visualisation |

---

## Running on Kubernetes

All GPU jobs use the `prism2` environment. All CPU jobs use the `trident` environment.

```
jobs/
├── 01_embeddings/     # PRISM2 base + diagnostic embeddings (GPU) — from encode_image/09_run_prism2.py
│   ├── job_prism2_embeddings_manual_knn.yaml    ← active
│   └── _deprecated/
├── 02_uamp/           # UAMP histological scoring (GPU) — script 01
│   ├── job_prism2_umap_manual_knn.yaml          ← active
│   ├── job_prism2_umap_repeat5_*.yaml
│   └── _deprecated/
├── 03_confidence_rating/  # Confidence scoring (GPU) — script 03
├── 04_reports/        # Free-text report generation (GPU) — script 04
├── 05_patch_viewer/   # Patch-level UMAP (CPU) — script 08
├── 06_umap/           # Slide-level UMAP + overlays (CPU) — scripts 05, 06, 07
│   ├── job_umap_manual_knn.yaml
│   ├── job_umap_scores_manual_knn.yaml
│   ├── job_umap_reports.yaml
│   └── _deprecated/
├── 07_saliency/       # Saliency maps (GPU) — scripts 10, 11
└── 08_temperature_sampling/  # Temperature sampling (GPU) — script 13
```

### Standard run order for manual_knn

```bash
# Step 1 — embeddings (GPU, ~2 h)
kubectl apply -f jobs/01_embeddings/job_prism2_embeddings_manual_knn.yaml

# Step 2 — UAMP scoring (GPU, ~4 h) — run in parallel with step 1
kubectl apply -f jobs/02_uamp/job_prism2_umap_manual_knn.yaml

# Step 3 — UMAP (CPU) — wait for step 1
kubectl apply -f jobs/06_umap/job_umap_manual_knn.yaml

# Step 4 — UAMP score overlays (CPU) — wait for steps 2 and 3
kubectl apply -f jobs/06_umap/job_umap_scores_manual_knn.yaml
```

---

## Deprecated scripts (`_deprecated/`)

| Script | Reason |
|--------|--------|
| `_deprecated/05_collect_prism2_results.py` | Repair utility — reconstructs `prism2_reports.jsonl` from existing `.txt` files if the JSONL was lost; not a normal pipeline step |
