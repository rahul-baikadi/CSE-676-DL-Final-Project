#!/bin/bash
#SBATCH --job-name=dl676-eval
#SBATCH --clusters=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
set -e
source scripts/ccr/config.sh
eval "$ENV_ACTIVATE"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$PROJ_DIR"
python -c "import torch; assert torch.cuda.is_available(), 'GPU not visible to PyTorch'" || exit 1

# First checkpoint = baseline (SFT); best-of-N is computed from it.
CKPTS=(sft=checkpoints/pretrain/best.pt)
for run in ppo_rule_b0 ppo_rule_b005 ppo_rule_b02 ppo_llm_b005; do
  for seed in 1 2 3; do
    CKPTS+=("${run}_s${seed}=checkpoints/${run}_s${seed}/policy.pt")
  done
done

python -m src.evaluate --checkpoints "${CKPTS[@]}" \
  --best_of 4 16 --best_of_rewards rule llm --test_samples 500 --mauve --judge --out results
python scripts/plot_results.py results/metrics.json results/
