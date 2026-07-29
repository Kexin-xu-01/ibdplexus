#!/bin/bash
# export TORCH_HOME=/alan-data/jinc/share/repo/kexin/.cache/torch

python3 /home/jovyan/models/trident/run_batch_of_slides.py \
    --task all \
    --wsi_dir /home/jovyan/kgbk271-ibd-datavol-1/data/raw/tiff_mpp_corrected \
    --job_dir /home/jovyan/kgbk271-ibd-datavol-1/data/processed/trident_processed \
    --patch_encoder virchow2 \
    --mag 20 \
    --patch_size 224 \
    --skip_errors

    #--wsi_dir /alan-data/jinc/share/data/ibd_clean_xavier/sparc-image-ffp/all_wsi_tiff \