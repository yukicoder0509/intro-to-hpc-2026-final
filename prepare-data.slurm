#!/bin/bash
#SBATCH --account=ACD115052
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --job-name=prepare-data

# If uv was installed locally (e.g. due to home quota), add it to PATH.
[ -f "$(pwd)/.bin/uv" ] && export PATH="$(pwd)/.bin:$PATH"
[ -d "$(pwd)/.uv-cache" ]  && export UV_CACHE_DIR=$(pwd)/.uv-cache
[ -d "$(pwd)/.uv-python" ] && export UV_PYTHON_INSTALL_DIR=$(pwd)/.uv-python

source .venv/bin/activate

python prepare_data.py \
  --dataset roneneldan/TinyStories \
  --output_dir data \
  --val_fraction 0.05

echo "Data preparation complete."
