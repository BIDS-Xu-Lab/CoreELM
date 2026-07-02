#!/bin/bash
#SBATCH --job-name=ctELM_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=david.kaauwai@yale.edu

module load miniconda
conda activate ctELM_proj
export HF_HOME="$(cd "$(dirname "$0")/.." && pwd)/.hf_cache"

VARIANT=${1:-}

if [ -n "$VARIANT" ]; then
    python embed_abstracts.py --config configs/pipeline.yaml --variant "$VARIANT"
else
    python embed_abstracts.py --config configs/pipeline.yaml
fi
