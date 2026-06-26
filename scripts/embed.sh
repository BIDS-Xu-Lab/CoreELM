#!/bin/bash
#SBATCH --job-name=ctELM_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err

module load miniconda
conda activate ctELM_proj

EXPERIMENT=${1:-}

if [ -n "$EXPERIMENT" ]; then
    python embed_abstracts.py --config configs/pipeline.yaml --experiment "$EXPERIMENT"
else
    python embed_abstracts.py --config configs/pipeline.yaml
fi
