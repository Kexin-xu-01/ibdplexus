# encode_image

Converts raw VSI slides to LZW pyramidal TIFFs, extracts patch-level and slide-level
embeddings via TRIDENT + PRISM2, and runs post-extraction QC.

**Output root:** `/home/jovyan/kgbk271-ibd-volume/data/`  
**Python environments:** `trident` (extraction), `prism2` (PRISM2 encoding)

---

## Pipeline Overview

```
[01] correct_mpp.py             ┐
[02] convert_missing_to_tiff.py ├─ all write to tiff_mpp_corrected/ (run in parallel)
[03] convert_large_slides.py    │
[04] reconvert_corrupt_slides.py┘
         │
[05] run_trident_virchow2.sh   ─── standard: wait for all TIFFs to be ready
[06] run_trident_loop.sh        ── alternative: poll for new TIFFs concurrently with conversion
[07] run_trident_titan.sh       ── alternative encoder (TITAN, 512 px patches)
         │
[08] qc_patch_counts.py         ── patch count histogram + low-patch-count slide thumbnails
         │
[09] run_prism2.py / [10] run_prism2.sh
[11] run_prism2_tissue_threshold_15_filtered.sh
```

Steps 01–04 cover different slide subsets and can run simultaneously.  
Steps 05/06 are mutually exclusive — use 06 only when conversion is still ongoing.

---

## Step-by-Step

### Steps 01–04 — Slide conversion to `tiff_mpp_corrected/`

All four scripts write LZW-compressed, 256×256-tiled, variable-depth pyramidal TIFFs to:
```
/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected/
```
Each skips slides already present. Run whichever subset applies.

---

#### Step 01 — Correct MPP tags (`01_correct_mpp.py`)

The main conversion path. Copies TIFFs from `all_wsi_tiff/` byte-for-byte and patches only
the three TIFF resolution tags (XResolution, YResolution, ResolutionUnit) using the
correct values from `vsi_metadata.tsv`. No pixel decompression — JPEG quality is preserved.

No CLI args. Edit the constants at the top of the script to change paths.

```bash
conda activate trident
python 01_correct_mpp.py
```

**Inputs:**
- `/home/jovyan/shared-data/users/kexin/vsi_metadata.tsv` — correct MPP values per slide
- `…/ibd_plexus_sparc_raw/image/all_wsi_tiff/` — source TIFFs with wrong tags

---

#### Step 02 — Convert missing slides (`02_convert_missing_to_tiff.py`)

Converts the 431 slides present in `all_wsi_tiff/` but absent from `tiff_mpp_corrected/`.
Locates source VSIs under `S3-raw-data-Jul-2026/sparc-image-ffpe/`, converts to LZW
pyramidal TIFF via atomic temp-file writes.

No CLI args.

```bash
conda activate trident
python 02_convert_missing_to_tiff.py
```

**Input:** `.../S3-raw-data-Jul-2026/sparc-image-ffpe/**/*.vsi`

---

#### Step 03 — Convert large slides (`03_convert_large_slides.py`)

Handles 8 specific VSIs that are 64–113 GB uncompressed — too large to load into RAM.
Uses a disk-backed numpy memmap to read in strips, then builds pyramid levels in memory.

No CLI args. Slide list and paths are hardcoded.

```bash
conda activate trident
python 03_convert_large_slides.py
```

---

#### Step 04 — Reconvert corrupt slides (`04_reconvert_corrupt_slides.py`)

Hotfix for 4 slides whose `tiff_mpp_corrected/` entries are corrupt.
Overwrites via atomic temp-file rename, then verifies the output with OpenSlide.

No CLI args. Slide list and paths are hardcoded.

```bash
conda activate trident
python 04_reconvert_corrupt_slides.py
```

---

### Steps 05–07 — TRIDENT patch extraction

TRIDENT runs tissue segmentation, patch coordinate extraction, Virchow2 patch feature
encoding, and (optionally) slide-level feature aggregation in a single `--task all` call.
It skips slides whose output already exists — safe to re-run after adding new TIFFs.

**Output tree:**
```
data/processed/trident_processed/20x_224px_0px_overlap/
  patches/           *.h5  (patch coords, shape N×2)
  features_virchow2/ *.h5  (Virchow2 2560-d patch embeddings, shape N×2560)
  visualization/     *.jpg (slide thumbnails)
```

---

#### Step 05 — Standard TRIDENT run (`05_run_trident_virchow2.sh`)

Use when all TIFFs are ready before starting extraction.

```bash
bash 05_run_trident_virchow2.sh
```

Calls TRIDENT with `--task all --patch_encoder virchow2 --mag 20 --patch_size 224 --gpus 0 --skip_errors`.

---

#### Step 06 — Polling loop (`06_run_trident_loop.sh`)

Use when VSI conversion (steps 01–04) and TRIDENT extraction should run concurrently.
Polls `tiff_mpp_corrected/` every 60 seconds for new TIFFs, re-runs TRIDENT (skipping
already-processed slides), and exits after 10 consecutive rounds with nothing new.

```bash
bash 06_run_trident_loop.sh
```

---

#### Step 07 — TITAN encoder (`07_run_trident_titan.sh`)

Runs TRIDENT with the TITAN slide encoder instead of Virchow2.
Uses 512 px patches at 20×. Independent of steps 05/06 — can run in parallel.

```bash
bash 07_run_trident_titan.sh
```

**Output:** `data/processed/trident_processed/20x_512px_0px_overlap/slide_features_titan/`

---

### Step 08 — Patch count QC (`08_qc_patch_counts.py`)

Reads all `*_patches.h5` files, counts patches per slide, and generates:
1. A histogram of the patch count distribution (mean + median lines)
2. A thumbnail grid of slides below the 50-patch threshold (likely failed segmentation)

```bash
conda activate trident
python 08_qc_patch_counts.py [patch_dir] [viz_dir] [qc_out_dir]
```

All three positional args are optional. Defaults:

| Argument | Default |
|----------|---------|
| `patch_dir` | `trident_processed/20x_224px_0px_overlap/patches/` |
| `viz_dir` | `trident_processed/20x_224px_0px_overlap/visualization/` |
| `qc_out_dir` | `kgbk271-ibd-volume/training/qc/` |

**Output:** `patch_counts_<config>.png` and `low_patch_count_slides_<config>.png`

---

### Steps 09–11 — PRISM2 slide-level embeddings *(GPU required)*

Reads per-slide Virchow2 patch features, slices the 1280-d class token from the
2560-d concatenated output, and produces PRISM2 base (2560-d) and diagnostic
(3072-d) slide embeddings. Skips slides already encoded.

```bash
conda activate prism2
python 09_run_prism2.py \
    [--job_dir PATH]    # TRIDENT job dir; default: trident_processed
    [--batch_size 8]    # slides per forward pass; reduce if OOM
    [--gpu 0]
```

**Output:**
```
<job_dir>/20x_224px_0px_overlap/
  prism2_base/<slide>.h5        (2560-d, float32)
  prism2_diagnostic/<slide>.h5  (3072-d, float32)
```

**Shell launchers** for older datasets (moved to `_deprecated/` — use K8s jobs in `../prism2/jobs/01_embeddings/` for the current `manual_knn` dataset):

**Model path:** `/home/jovyan/shared-data/users/kexin/models/VLM/prism2`

---

## Utilities

### `check_mpp.py` — inspect TIFF resolution tags

Reads one TIFF and prints its XResolution, YResolution, ResolutionUnit, and the
derived MPP (µm/px). Useful for verifying a slide after conversion.

```bash
python check_mpp.py /path/to/slide.tiff
```

---

## Running on Kubernetes

Encoder jobs are in `jobs/`. All run in the `ibd-plexus-research` namespace and mount
the `kgbk271-ibd-volume`, `shared-data`, and `kgbk271-ibd-workspace` PVCs.

```
jobs/
├── job_trident_virchow2_patch_qc.yaml   # Virchow2 patch extraction + QC
├── job_trident_titan.yaml               # TITAN slide encoder
├── job_trident_chief.yaml               # CHIEF encoder
├── job_trident_gigapath.yaml            # GigaPath encoder
├── job_trident_madeleine.yaml           # Madeleine encoder
├── job_trident_feather.yaml             # Feather encoder
├── job_trident_feather_uni_v2.yaml      # Feather UNI-v2 encoder
├── job_trident_care.yaml                # CARE encoder
├── job_tissue_threshold_15.yaml         # Tissue threshold 15 extraction
└── submit_all_encoders.sh               # Submit one or all encoder jobs
```

To submit a single encoder:
```bash
bash jobs/submit_all_encoders.sh titan
```

To submit all encoders at once:
```bash
bash jobs/submit_all_encoders.sh
```

PRISM2 embedding extraction is **not** included in `submit_all_encoders.sh` — run it
separately via the K8s jobs in `../prism2/jobs/01_embeddings/`.

---

## Deprecated scripts (`_deprecated/`)

| Script | Reason |
|--------|--------|
| `convert_vsi_to_tiff.py` | Uses JPEG compression and a fixed 4-level pyramid; superseded by `02_convert_missing_to_tiff.py` which produces LZW pyramids matching the `tiff_mpp_corrected/` format |
| `check_tiff_permission.py` | Used `os.access()` which gives wrong results on NFS+ACL mounts; `01_correct_mpp.py` tries to actually open each file instead |
| `run_trident_virchow2_vsi.sh` | Ran TRIDENT directly on raw VSI files; superseded by the convert-then-extract approach |
| `10_run_prism2.sh` | Launcher for `trident_processed` (original unfiltered dataset); use K8s job in `../prism2/jobs/01_embeddings/` for current dataset |
| `11_run_prism2_tissue_threshold_15_filtered.sh` | Launcher for intermediate `tissue_threshold_15_filtered` dataset; superseded |
