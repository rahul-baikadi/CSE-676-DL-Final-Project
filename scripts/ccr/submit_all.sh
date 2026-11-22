#!/bin/bash
# Usage (from repo root on the login node):  bash scripts/ccr/submit_all.sh
# Chains SFT -> 12 PPO runs (4 configs x 3 seeds) -> evaluation. Each starts only if the previous succeeded.
set -e
source scripts/ccr/config.sh
cd "$PROJ_DIR"; mkdir -p slurm_logs

# Fail now rather than after hours of training.
[ -f "$ENV_DIR/bin/activate" ] || { echo "Missing Python env. Run scripts/ccr/setup_login_node.sh"; exit 1; }
for m in gpt2 gpt2-large microsoft--Phi-3.5-mini-instruct Qwen--Qwen2.5-7B-Instruct; do
  [ -d "$HF_HOME/hub/models--$m" ] || { echo "Missing model $m. Run scripts/ccr/setup_login_node.sh"; exit 1; }
done

ACCT=${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT}
SFT=$(sbatch --parsable $ACCT scripts/ccr/sft.slurm | cut -d';' -f1)
PPO=$(sbatch --parsable $ACCT --dependency=afterok:$SFT scripts/ccr/ppo_array.slurm | cut -d';' -f1)
EVAL=$(sbatch --parsable $ACCT --dependency=afterok:$PPO scripts/ccr/eval.slurm | cut -d';' -f1)
echo "SFT=$SFT  PPO=$PPO  EVAL=$EVAL"
echo "Watch: squeue -u \$USER --clusters=ub-hpc ;  tail -f slurm_logs/*.out"
