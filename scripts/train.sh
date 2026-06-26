#!/bin/bash
#SBATCH --job-name=ctELM_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module load miniconda
conda activate ctELM_proj

EXPERIMENT=${1:-}

if [ -n "$EXPERIMENT" ]; then
    torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE train.py --config configs/pipeline.yaml --experiment "$EXPERIMENT"
else
    torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE train.py --config configs/pipeline.yaml
fi
