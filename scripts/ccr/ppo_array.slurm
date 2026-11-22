#!/bin/bash
#SBATCH --job-name=dl676-ppo
#SBATCH --clusters=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm_logs/%x-%A_%a.out
source scripts/ccr/config.sh
eval "$ENV_ACTIVATE"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$PROJ_DIR"
python -c "import torch; assert torch.cuda.is_available(), 'GPU not visible to PyTorch'" || exit 1

# index: reward  beta   output dir
CONFIGS=(
  "rule 0.0  checkpoints/ppo_rule_b0"
  "rule 0.05 checkpoints/ppo_rule_b005"
  "rule 0.2  checkpoints/ppo_rule_b02"
  "llm  0.05 checkpoints/ppo_llm_b005"
)
SEEDS=(1 2 3)
CONFIG_ID=$((SLURM_ARRAY_TASK_ID % ${#CONFIGS[@]}))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / ${#CONFIGS[@]}))]}
read REWARD BETA OUT <<< "${CONFIGS[$CONFIG_ID]}"
OUT="${OUT}_s${SEED}"
echo "reward=$REWARD beta=$BETA seed=$SEED out=$OUT"
python -m src.ppo --init checkpoints/pretrain/best.pt --reward "$REWARD" --beta "$BETA" \
       --seed "$SEED" --steps 300 --out "$OUT"
