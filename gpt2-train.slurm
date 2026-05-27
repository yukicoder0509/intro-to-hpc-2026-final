#!/bin/bash
#SBATCH --account=ACD115052
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:60:00

source .venv/bin/activate

python train_gpt2_wikipedia.py \
  --streaming \
  --max_train_samples 50000 \
  --num_train_epochs 1 \
  --block_size 256