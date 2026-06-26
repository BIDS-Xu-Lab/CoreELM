#!/bin/bash
#SBATCH --job-name=ctELM_prepare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err

module load miniconda
conda activate ctELM_proj

EXPERIMENT=${1:-}

if [ -n "$EXPERIMENT" ]; then
    python prepare_graph_dataset.py --config configs/pipeline.yaml --experiment "$EXPERIMENT"
else
    python prepare_graph_dataset.py --config configs/pipeline.yaml
fi
