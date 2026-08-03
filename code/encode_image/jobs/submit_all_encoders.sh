#!/bin/bash
# Submit all trident slide encoder jobs.
# Each job runs --task all: seg + coords + feat (patch) + feat (slide).
# Jobs skip already-processed slides automatically.
#
# REQUIREMENTS before submitting each encoder:
#   titan          : conch_v15 and TITAN models auto-downloaded on first run (needs internet)
#   chief          : ctranspath model auto-downloaded on first run (needs internet)
#   gigapath       : gigapath patch + slide models auto-downloaded on first run (needs internet)
#   madeleine      : conch_v1 model auto-downloaded on first run (needs internet)
#   feather        : conch_v15 auto-downloaded on first run (needs internet)
#   feather_uni_v2 : uni_v2 auto-downloaded on first run (needs internet)
#   care           : conch_v15 auto-downloaded on first run (needs internet)
#   prism2         : run separately with code/prism2/job_prism2_embeddings.yaml
#                    (requires shards 1-3 to be re-uploaded first)
#
# Models are cached to /home/jovyan/kgbk271-ibd-volume/hf_cache on first run.
# Subsequent runs use the cache offline.
#
# Usage:
#   bash submit_all_encoders.sh            # submit all
#   bash submit_all_encoders.sh titan      # submit one encoder by name

DIR="$(cd "$(dirname "$0")" && pwd)"

submit() {
    local name=$1
    local yaml=$2
    echo "Submitting $name ..."
    kubectl delete job "kgbk271-trident-${name}" -n ibd-plexus-research 2>/dev/null || true
    kubectl apply -f "$yaml"
}

TARGET="${1:-all}"

if [[ "$TARGET" == "all" || "$TARGET" == "titan" ]];          then submit titan          "$DIR/job_trident_titan.yaml"; fi
if [[ "$TARGET" == "all" || "$TARGET" == "chief" ]];          then submit chief          "$DIR/job_trident_chief.yaml"; fi
if [[ "$TARGET" == "all" || "$TARGET" == "gigapath" ]];       then submit gigapath       "$DIR/job_trident_gigapath.yaml"; fi
if [[ "$TARGET" == "all" || "$TARGET" == "madeleine" ]];      then submit madeleine      "$DIR/job_trident_madeleine.yaml"; fi
if [[ "$TARGET" == "all" || "$TARGET" == "feather" ]];        then submit feather        "$DIR/job_trident_feather.yaml"; fi
if [[ "$TARGET" == "all" || "$TARGET" == "feather_uni_v2" ]]; then submit feather-uni-v2 "$DIR/job_trident_feather_uni_v2.yaml"; fi
if [[ "$TARGET" == "all" || "$TARGET" == "care" ]];           then submit care           "$DIR/job_trident_care.yaml"; fi

echo ""
echo "Status:"
kubectl get jobs -n ibd-plexus-research | grep kgbk271-trident
