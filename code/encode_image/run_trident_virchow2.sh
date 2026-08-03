#!/bin/bash

/home/jovyan/kgbk271-ibd-volume/envs/trident/bin/python /home/jovyan/shared-data/users/kexin/models/trident/run_batch_of_slides.py \
    --task all \
    --wsi_dir /home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected \
    --job_dir /home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed \
    --patch_encoder virchow2 \
    --mag 20 \
    --patch_size 224 \
    --gpus 0 \
    --skip_errors