#!/bin/bash
#SBATCH --job-name=dl676-eval
#SBATCH --clusters=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out
source scripts/ccr/config.sh
eval "$ENV_ACTIVATE"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$PROJ_DIR"

JUDGE=""
[ -d "$HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct" ] && JUDGE="--judge"

# First checkpoint = baseline (SFT); best-of-N is computed from it.
python -m src.evaluate \
  --checkpoints sft=checkpoints/pretrain/best.pt \
                ppo_rule_b0=checkpoints/ppo_rule_b0/policy.pt \
                ppo_rule_b005=checkpoints/ppo_rule_b005/policy.pt \
                ppo_rule_b02=checkpoints/ppo_rule_b02/policy.pt \
                ppo_llm_b005=checkpoints/ppo_llm_b005/policy.pt \
  --best_of 4 16 --test_samples 500 --mauve $JUDGE --out results
python scripts/plot_results.py results/metrics.json results/
