# CONCH / TITAN pilots

Exploratory comparisons between the primary PRISM2 pipeline and two alternative
pathology foundation models — **CONCH v1** (Mahmoodlab) and **TITAN** (Paige).
These scripts are pilots and are **not part of the production pipeline**.

**Dependencies:**
- CONCH: `pip install git+https://github.com/Mahmoodlab/CONCH.git`
- Model weights (offline): `/home/jovyan/shared-data/users/kexin/models/VLM/`
- Feature caches: precomputed embeddings live under `data/processed/*/features_conch_v15/` and TITAN features under `data/processed/trident_processed/20x_512px_0px_overlap/slide_features_titan/`
- Python env: `prism2` (GPU inference) or `trident` (CPU-only analysis / plotting)

**Depends on the main pipeline for:** PRISM2 attention heatmaps (from `code/prism2/saliency/16_attention_heatmap.py`) and PRISM2 UAMP scores (from `code/prism2/scoring/01_run_prism2_uamp.py`).

---

## Pipeline

Run scripts in numerical order. Each depends on the outputs of the previous.

```
[01] CONCH: aggregate PRISM2 attention onto CONCH tile grid, save top-K tile embeddings
       │
       ▼
[02] CONCH v1: zero-shot patch-level label scores (CLIP-style, 12 IBD UAMP labels)
[03] PRISM2:  free-form Phi-3 description of top-K high-attention patches
[04] TITAN:   slide-level zero-shot classification (11 UAMP terms)
       │
       ▼
[05] Visualise per-slide CONCH scores + PRISM2 attention (2-panel figure)
[06] TITAN cohort-level overview (violin, correlations, top slides)
[07] TITAN vs PRISM2 across the cohort (scatter, Pearson bar)
[08] CONCH attention-weighted slide scores vs PRISM2 (and side-by-side vs TITAN)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `01_conch_patch_embeddings.py` | Spatial join: map PRISM2 tile attention (672 px) onto CONCH tile grid (1536 px), select top-K CONCH tiles, save embeddings. |
| `02_conch_zero_shot_labels.py` | CONCH v1 CLIP-style zero-shot: rank each patch against 12 UAMP text prompts. |
| `03_prism2_patch_describe.py` | PRISM2 Phi-3 free-form description of the top-K high-attention tile regions. |
| `04_titan_zero_shot.py` | TITAN zero-shot classification, 11 UAMP terms, raw cosine similarity. |
| `05_visualize_patches.py` | Per-slide 2-panel figure: H&E patches + CONCH score heatmap. |
| `06_titan_overview.py` | TITAN cohort overview: violin, pairwise correlation, top-scoring slides. |
| `07_titan_vs_prism2.py` | TITAN vs PRISM2 scatter + Pearson bar across UAMP terms. |
| `08_conch_weighted_vs_prism2.py` | CONCH slide-level score (attention-weighted top-K) vs PRISM2, and side-by-side vs TITAN. |

## Kubernetes jobs

| Job | Runs |
|-----|------|
| `jobs/job_conch_zero_shot.yaml` | `02_conch_zero_shot_labels.py` on the current dataset |
| `jobs/job_titan_zero_shot.yaml` | `04_titan_zero_shot.py` on the current dataset |

## Outputs

All under `/home/jovyan/kgbk271-ibd-volume/results/`:
- `conch_zero_shot/` — per-patch scores + slide-level comparison figures
- `titan_zero_shot/` — slide-level TITAN scores + cohort overviews
