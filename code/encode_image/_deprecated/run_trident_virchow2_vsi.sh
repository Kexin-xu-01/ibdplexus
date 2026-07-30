#!/bin/bash
#export TORCH_HOME=/home/jovyan/.cache/torch
#export PYTHONPATH=/home/jovyan/.local/lib/python3.10/site-packages:$PYTHONPATH

/home/jovyan/kgbk271-ibd-volume/envs/trident/bin/python /home/jovyan/models/trident/run_batch_of_slides.py \
    --task all \
    --wsi_dir /home/jovyan/shared-data/S3-raw-data-Jul-2026/sparc-image-ffpe/vsi-2021 \
    --job_dir /home/jovyan/kgbk271-ibd-volume/data/processed/vsi/trident_processed \
    --patch_encoder virchow2 \
    --mag 20 \
    --patch_size 224 \
    --gpus 0 \
    --skip_errors \
    --custom_list_of_wsis /home/jovyan/shared-data/vsi_2021_mpp.csv

    #--wsi_dir /alan-data/jinc/share/data/ibd_clean_xavier/sparc-image-ffp/all_wsi_tiff \