#!/bin/bash
# Usage (from repo root on the login node):  bash scripts/ccr/submit_all.sh
# Chains SFT -> 4 PPO runs (parallel) -> evaluation. Each starts only if the previous succeeded.
set -e
source scripts/ccr/config.sh
cd "$PROJ_DIR"; mkdir -p slurm_logs
ACCT=${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT}
SFT=$(sbatch --parsable $ACCT scripts/ccr/sft.slurm | cut -d';' -f1)
PPO=$(sbatch --parsable $ACCT --dependency=afterok:$SFT scripts/ccr/ppo_array.slurm | cut -d';' -f1)
EVAL=$(sbatch --parsable $ACCT --dependency=afterok:$PPO scripts/ccr/eval.slurm | cut -d';' -f1)
echo "SFT=$SFT  PPO=$PPO  EVAL=$EVAL"
echo "Watch: squeue -u \$USER --clusters=ub-hpc ;  tail -f slurm_logs/*.out"
