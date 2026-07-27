#!/bin/bash
export TORCH_HOME=/alan-data/jinc/share/repo/kexin/.cache/torch

nohup python /alan-data/jinc/share/repo/kexin/models/trident/run_batch_of_slides.py \
    --task all \
    --wsi_dir /alan-data/jinc/share/repo/kexin/ibdplexus/data/raw/tiff_mpp_corrected \
    --job_dir /alan-data/jinc/share/repo/kexin/ibdplexus/data/processed/trident_processed \
    --patch_encoder virchow2 \
    --mag 20 \
    --patch_size 224 \
    --skip_errors

    #--wsi_dir /alan-data/jinc/share/data/ibd_clean_xavier/sparc-image-ffp/all_wsi_tiff \