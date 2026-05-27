#!/bin/bash
#SBATCH --account=ACD115052
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00

source .venv/bin/activate

python -c '
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en")
'

echo "task complete"

