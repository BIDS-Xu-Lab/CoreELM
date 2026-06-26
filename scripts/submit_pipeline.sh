#!/bin/bash
# Usage:
#   bash scripts/submit_pipeline.sh configs/experiments/<name>.yaml
#   bash scripts/submit_pipeline.sh configs/experiments/<name>.yaml --with-embed
#
# --with-embed: also submits the one-time embed job at the front of the chain.
# Omit it if embeddings/embeddings.npy already exists on the cluster.

set -euo pipefail

EXPERIMENT="${1:-}"
WITH_EMBED=0
for arg in "$@"; do
    [ "$arg" = "--with-embed" ] && WITH_EMBED=1
done

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: bash scripts/submit_pipeline.sh <experiment.yaml> [--with-embed]"
    exit 1
fi

EXP_NAME=$(basename "$EXPERIMENT" .yaml)
echo "Submitting pipeline for experiment: $EXP_NAME"

PREV_JID=""

if [ "$WITH_EMBED" -eq 1 ]; then
    JID_EMBED=$(sbatch --parsable scripts/embed.sh)
    echo "  embed           job $JID_EMBED"
    PREV_JID="$JID_EMBED"
fi

DEP_PREP=""
[ -n "$PREV_JID" ] && DEP_PREP="--dependency=afterok:$PREV_JID"

JID_PREP=$(sbatch --parsable $DEP_PREP scripts/prepare_dataset.sh "$EXPERIMENT")
echo "  prepare_dataset job $JID_PREP"

JID_TRAIN=$(sbatch --parsable --dependency=afterok:$JID_PREP scripts/train.sh "$EXPERIMENT")
echo "  train           job $JID_TRAIN"

JID_EVAL=$(sbatch --parsable --dependency=afterok:$JID_TRAIN scripts/evaluate.sh "$EXPERIMENT")
echo "  evaluate        job $JID_EVAL"

echo ""
echo "Chain: ${JID_PREP} → ${JID_TRAIN} → ${JID_EVAL}"
echo "Each job will email david.kaauwai@yale.edu on END or FAIL."
