#!/bin/bash
# Repeatedly runs trident as new mpp-corrected TIFFs appear.
# Trident skips already-processed slides automatically.
# Stops when no new slides appear for IDLE_ROUNDS consecutive rounds.

TRIDENT=/home/jovyan/kgbk271-ibd-volume/envs/trident/bin/python
SCRIPT=/home/jovyan/shared-data/users/kexin/models/trident/run_batch_of_slides.py
WSI_DIR=/home/jovyan/kgbk271-ibd-volume/data/raw/tiff_mpp_corrected
JOB_DIR=/home/jovyan/kgbk271-ibd-volume/data/processed/trident_processed
POLL_SECS=60   # wait between rounds if no new slides
IDLE_ROUNDS=10 # stop after this many consecutive rounds with no new slides

idle=0
last_count=0

echo "[LOOP] Starting trident loop. Polling every ${POLL_SECS}s, stopping after ${IDLE_ROUNDS} idle rounds."

while true; do
    current_count=$(ls "$WSI_DIR"/*.tiff 2>/dev/null | wc -l)
    echo "[LOOP] $(date '+%H:%M:%S') | TIFFs in WSI_DIR: $current_count"

    if [ "$current_count" -gt 0 ]; then
        $TRIDENT $SCRIPT \
            --task all \
            --wsi_dir "$WSI_DIR" \
            --job_dir "$JOB_DIR" \
            --patch_encoder virchow2 \
            --mag 20 \
            --patch_size 224 \
            --gpus 0 \
            --skip_errors \
            --clear_dead_locks

        if [ "$current_count" -eq "$last_count" ]; then
            idle=$((idle + 1))
            echo "[LOOP] No new slides this round ($idle/$IDLE_ROUNDS idle rounds)."
        else
            idle=0
            echo "[LOOP] Processed new slides (was $last_count, now $current_count)."
        fi
        last_count=$current_count
    else
        echo "[LOOP] No TIFFs yet, waiting..."
        idle=$((idle + 1))
    fi

    if [ "$idle" -ge "$IDLE_ROUNDS" ]; then
        echo "[LOOP] $IDLE_ROUNDS consecutive idle rounds. Done."
        break
    fi

    echo "[LOOP] Sleeping ${POLL_SECS}s..."
    sleep $POLL_SECS
done

echo "[LOOP] Finished at $(date)"
