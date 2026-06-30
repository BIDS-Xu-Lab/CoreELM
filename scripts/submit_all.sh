#!/bin/bash
# Submit all experiments across all variants using job arrays (3 sbatch calls total).
# Assumes graph data and embeddings are already built on the cluster.
# Usage: bash scripts/submit_all.sh

set -euo pipefail

echo "Submitting prepare array (32 experiments)..."
PREP_JID=$(sbatch --parsable scripts/prepare_array.sh)
echo "  prepare array job $PREP_JID"

echo "Submitting train array (32 experiments, aftercorr:prepare)..."
TRAIN_JID=$(sbatch --parsable --dependency=aftercorr:$PREP_JID scripts/train_array.sh)
echo "  train array job $TRAIN_JID"

echo "Submitting eval array (32 experiments, aftercorr:train)..."
EVAL_JID=$(sbatch --parsable --dependency=aftercorr:$TRAIN_JID scripts/eval_array.sh)
echo "  eval array job $EVAL_JID"

echo ""
echo "Pipeline: prepare[$PREP_JID] → train[$TRAIN_JID] → eval[$EVAL_JID]"
echo "Each job will email david.kaauwai@yale.edu on END or FAIL."
