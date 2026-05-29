#!/bin/bash
#SBATCH --account=ACD115052
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=b0-baseline

# If uv was installed locally (e.g. due to home quota), add it to PATH.
[ -f "$(pwd)/.bin/uv" ] && export PATH="$(pwd)/.bin:$PATH"
[ -d "$(pwd)/.uv-cache" ]  && export UV_CACHE_DIR=$(pwd)/.uv-cache
[ -d "$(pwd)/.uv-python" ] && export UV_PYTHON_INSTALL_DIR=$(pwd)/.uv-python

source .venv/bin/activate

# B0 — naïve baseline (plan §4.0):
#   BASE 124M, 1 GPU, FP32, JIT, no BEAM
#   batch=4, ctx=512 → 2048 tokens/step
#   3 warmup + 30 measured steps  (--steps 33)
#
# Run ≥5 times (plan §6 rule 2); pass rep index as first argument.
#   for i in 0 1 2 3 4; do sbatch experiments/b0_baseline/run.slurm $i; done

REP=${1:-0}

# Load WANDB_API_KEY from .env if not already set
if [ -z "${WANDB_API_KEY}" ] && [ -f ".env" ]; then
  export WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' .env | cut -d'=' -f2-)
fi

WANDB_MODE=online \
WANDB_API_KEY=${WANDB_API_KEY} \
JIT=1 \
BEAM=0 \
python experiments/bench.py \
  --config  BASE \
  --batch   4 \
  --steps   33 \
  --warmup  3 \
  --lr      3e-4 \
  --exp     B0 \
  --rep     ${REP} \
  --data_dir data

# Sync wandb offline runs from a login node:
#   wandb sync wandb/offline-run-*
