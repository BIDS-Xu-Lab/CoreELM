#!/bin/bash
# Submit all experiments across all variants, each variant gated on its own embed job.
# Usage: bash scripts/submit_all.sh

set -euo pipefail

VARIANTS=(
    configs/pipeline_no_labels_numbers.yaml
    configs/pipeline_no_labels_no_numbers.yaml
    configs/pipeline_labels_numbers.yaml
    configs/pipeline_labels_no_numbers.yaml
)

for variant in "${VARIANTS[@]}"; do
    VARIANT_NAME=$(basename "$variant" .yaml)
    echo "=== Variant: $VARIANT_NAME ==="

    echo "Submitting embed job..."
    EMBED_JID=$(sbatch --parsable scripts/embed.sh "$variant")
    echo "  embed job $EMBED_JID"
    echo ""

    for exp in chain1 chain2 chain3 chain4; do
        for mode in reconstruct generate; do
            bash scripts/submit_pipeline.sh \
                configs/experiments/${exp}_${mode}.yaml \
                --variant "$variant" \
                --after-embed "$EMBED_JID"
            echo ""
        done
    done

    echo ""
done

echo "All experiments queued across ${#VARIANTS[@]} variants."
