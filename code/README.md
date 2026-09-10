# IBD-PLEXUS Codebase

End-to-end analysis pipeline for the SPARC IBD cohort — from raw whole-slide images
and RNA-seq counts to a CD-vs-UC classifier with SHAP interpretability.

**Data volume:** `/home/jovyan/kgbk271-ibd-volume/` (all inputs, intermediates, and outputs live here — see the volume's `README.md`).
**Conda environments:** `trident` (CPU) and `prism2` (GPU) — installed under `envs/` on the data volume.
**Cluster:** Kubernetes namespace `ibd-plexus-research`; GPU jobs use the `prism2` env, CPU jobs use `trident`.

---

## Pipeline Overview

```
┌───────────────────────────────────────────────────────────────────────────────┐
│  1. encode_image/      Raw VSI → LZW pyramidal TIFF → TRIDENT patches +       │
│                        Virchow2 features + PRISM2 slide embeddings            │
│                                                                               │
│                                     ▼                                         │
│  2. image_qc/          Filter patches: Laplacian blur, GrandQC dark spots,    │
│                        intensity threshold, manual KNN artefact curation      │
│                                                                               │
│                                     ▼                                         │
│  3. prism2/            PRISM2 slide-level analysis: UAMP scoring, free-text   │
│                        reports, UMAP visualisations, saliency, uncertainty    │
│                                                                               │
│                                     ▼                                         │
│  4. training/cd_vs_uc/ CD vs UC classification (imaging, RNA, clinical,       │
│                        proteomics, multimodal) + SHAP + plots                 │
└───────────────────────────────────────────────────────────────────────────────┘

  conch/                 Pilot experiments comparing CONCH v1 / TITAN against
                         PRISM2. Independent of the main pipeline.
```

---

## Execution Order

### First-time run from scratch

1. **`encode_image/`** (steps 01 → 09) — convert VSI slides, extract patches with TRIDENT, encode with Virchow2, produce PRISM2 slide embeddings. See `encode_image/README.md`.
2. **`image_qc/`** (steps 01 → 04) — successively filter patches. Each step produces a new dataset folder in `data/processed/`. Current best dataset: `tissue_threshold_15_filtered_no_darkspot_manual_knn`. See `image_qc/README.md`.
3. **`prism2/scoring/`** (01, 04) — run UAMP scoring and (optionally) free-text reports on the current dataset.
4. **`prism2/umap/`** (05 → 09) — build UMAP visualisations.
5. **`training/cd_vs_uc/build_cv_splits/`** (01, 01b) — build patient-level CV splits.
6. **`training/cd_vs_uc/`** root scripts — `train_rf.py` (imaging), `train_multimodal.py` (fusion), `train_mlp.py` (all arms), `train_rf_proteomics.py`.
7. **`training/cd_vs_uc/plot/`** — comparison figures.

### Everyday: analysing the current best dataset (`manual_knn`)

- Score: `prism2/scoring/01_run_prism2_uamp.py` → `results/prism2_manual_knn/prism2_histological_score.csv`
- Train: `training/cd_vs_uc/train_rf.py` or `train_multimodal.py` → `training/cd_vs_uc/<arm>_results/`
- Plot: scripts in `training/cd_vs_uc/plot/`

All K8s jobs for these live in `<subsystem>/jobs/`. See each subsystem's README for the exact `kubectl apply -f …` command.

---

## Directory Map

| Directory | Purpose | Entry point |
|-----------|---------|-------------|
| `encode_image/` | VSI → TIFF conversion + TRIDENT patch extraction + PRISM2 slide encoding | `encode_image/README.md` |
| `image_qc/` | Patch quality filtering — Laplacian, GrandQC, intensity, manual KNN | `image_qc/README.md` |
| `prism2/` | PRISM2 VLM inference and downstream analysis (UAMP, UMAP, saliency, uncertainty) | `prism2/README.md` |
| `training/cd_vs_uc/` | CD vs UC classifiers on imaging, RNA, clinical, proteomics, multimodal | `training/cd_vs_uc/README.md` |
| `conch/` | CONCH / TITAN pilot experiments | `conch/README.md` |

Every subsystem uses the same conventions:

- **Numbered scripts** — `NN_verb_noun.py` gives the intended pipeline order.
- **`_deprecated/`** subfolders hold superseded scripts (kept for reference, not maintained).
- **`jobs/`** subfolders hold Kubernetes YAML specs.

---

## Datasets

The QC pipeline produces successive dataset versions in `data/processed/`. The dataset name is the suffix appended to `results/prism2_`.

| Stage | Dataset | Filter added |
|-------|---------|-------------|
| 0 | `trident_processed` | (baseline TRIDENT extraction) |
| 1 | `tissue_threshold_15` | ≥15% tissue coverage |
| 2 | `tissue_threshold_15_filtered` | + Laplacian blur (LV<100) + faint intensity (p98) |
| 3 | `tissue_threshold_15_filtered_no_darkspot` | + GrandQC dark-spot removal |
| **4** | **`tissue_threshold_15_filtered_no_darkspot_manual_knn`** | **+ manual KNN artefact curation — current best** |

`838,562 / 865,070 (96.94%)` patches retained.

Results for stages 0–3 live in `results/_deprecated/prism2_*/`; stage 4 (`manual_knn`) is the current active dataset under `results/prism2_manual_knn/`.

---

## Conventions

- **Path handling** — scripts prefer `argparse` defaults that point at the current best dataset (`manual_knn`). Never hardcode a dataset path when a CLI arg exists.
- **Idempotence** — long-running scripts (TRIDENT, PRISM2 inference) skip slides that already have output. Safe to restart.
- **K8s jobs** — active jobs sit at the top of each `jobs/<subfolder>/` directory; retired ones move to `jobs/<subfolder>/_deprecated/`. Every job's `command:` block references a real script.
- **Results** — root-owned; write access is only available from inside a K8s job (which runs as root). Interactive sessions can write to `training/`, `metadata/`, `logs/`.

---

## Handover notes

For anyone taking over:

1. Read each subsystem's `README.md` before running anything — the READMEs are the source of truth for CLI args and expected outputs.
2. `training/cd_vs_uc/CLAUDE.md` has the deepest project context (data sources, cohorts, model configs, SHAP schema).
3. `results/patch_filtering_report.md` explains the impact of each QC stage on classifier performance.
4. Old experimental variants are in `_deprecated/`. The active pipeline is intentionally small.
