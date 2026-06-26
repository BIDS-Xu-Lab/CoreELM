#!/bin/bash
# Submit all experiments, all gated on a single embed job.
# Usage: bash scripts/submit_all.sh

set -euo pipefail

echo "Submitting embed job..."
EMBED_JID=$(sbatch --parsable scripts/embed.sh)
echo "  embed job $EMBED_JID"
echo ""

for exp in chain1 chain2 chain3 chain4; do
    for mode in reconstruct generate; do
        bash scripts/submit_pipeline.sh \
            configs/experiments/${exp}_${mode}.yaml \
            --after-embed "$EMBED_JID"
        echo ""
    done
done

echo "All experiments queued. Embed job $EMBED_JID must complete before any prepare step starts."
