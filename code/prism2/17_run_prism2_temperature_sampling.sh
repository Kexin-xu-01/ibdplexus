#!/bin/bash

TRANSFORMERS_CACHE=/home/jovyan/kgbk271-ibd-volume/huggingface_cache \
HF_HOME=/home/jovyan/kgbk271-ibd-volume/huggingface_cache \
/home/jovyan/kgbk271-ibd-volume/envs/prism2/bin/python \
    /home/jovyan/ibdplexus/code/prism2/17_run_prism2_temperature_sampling.py \
    --temperature 2.0 \
    --n_samples 50 \
    --batch_size 4 \
    --gpu 0 \
    --skip_errors
