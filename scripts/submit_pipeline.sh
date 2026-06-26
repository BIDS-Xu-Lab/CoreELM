#!/bin/bash
# Usage:
#   bash scripts/submit_pipeline.sh configs/experiments/<name>.yaml
#   bash scripts/submit_pipeline.sh configs/experiments/<name>.yaml --with-embed
#   bash scripts/submit_pipeline.sh configs/experiments/<name>.yaml --after-embed <JID>
#
# --with-embed:       submit the one-time embed job and chain this experiment to it.
# --after-embed <JID> chain this experiment's prepare step to an already-submitted embed job.
# Omit both if embeddings/embeddings.npy already exists on the cluster.

set -euo pipefail

EXPERIMENT="${1:-}"
WITH_EMBED=0
EMBED_JID=""

i=1
while [ $i -le $# ]; do
    arg="${!i}"
    if [ "$arg" = "--with-embed" ]; then
        WITH_EMBED=1
    elif [ "$arg" = "--after-embed" ]; then
        i=$((i + 1))
        EMBED_JID="${!i}"
    fi
    i=$((i + 1))
done

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: bash scripts/submit_pipeline.sh <experiment.yaml> [--with-embed | --after-embed <JID>]"
    exit 1
fi

EXP_NAME=$(basename "$EXPERIMENT" .yaml)
echo "Submitting pipeline for experiment: $EXP_NAME"

if [ "$WITH_EMBED" -eq 1 ]; then
    EMBED_JID=$(sbatch --parsable scripts/embed.sh)
    echo "  embed           job $EMBED_JID"
fi

DEP_PREP=""
[ -n "$EMBED_JID" ] && DEP_PREP="--dependency=afterok:$EMBED_JID"

JID_PREP=$(sbatch --parsable $DEP_PREP scripts/prepare_dataset.sh "$EXPERIMENT")
echo "  prepare_dataset job $JID_PREP"

JID_TRAIN=$(sbatch --parsable --dependency=afterok:$JID_PREP scripts/train.sh "$EXPERIMENT")
echo "  train           job $JID_TRAIN"

JID_EVAL=$(sbatch --parsable --dependency=afterok:$JID_TRAIN scripts/evaluate.sh "$EXPERIMENT")
echo "  evaluate        job $JID_EVAL"

echo ""
echo "Chain: ${JID_PREP} → ${JID_TRAIN} → ${JID_EVAL}"
echo "Each job will email david.kaauwai@yale.edu on END or FAIL."
