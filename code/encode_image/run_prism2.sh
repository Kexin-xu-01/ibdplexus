#!/bin/bash

TRANSFORMERS_CACHE=/home/jovyan/kgbk271-ibd-volume/huggingface_cache \
HF_HOME=/home/jovyan/kgbk271-ibd-volume/huggingface_cache \
/home/jovyan/kgbk271-ibd-volume/envs/prism2/bin/python \
    /home/jovyan/ibdplexus/code/encode_image/run_prism2.py \
    --gpu 0 \
    --batch_size 8 \
    --skip_errors
