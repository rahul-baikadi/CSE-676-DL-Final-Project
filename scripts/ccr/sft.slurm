#!/bin/bash
#SBATCH --job-name=dl676-sft
#SBATCH --clusters=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
source scripts/ccr/config.sh
eval "$ENV_ACTIVATE"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$PROJ_DIR"
python -c "import torch; assert torch.cuda.is_available(), 'GPU not visible to PyTorch'" || exit 1
nvidia-smi
python -m src.train --train_samples 100000 --epochs 3 --batch_size 64 --out checkpoints/pretrain
