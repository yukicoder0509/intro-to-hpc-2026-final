# Scaling LLM Pre-training on V100s

HPC final project — measuring and optimising GPT-2 pre-training throughput on
NCHC Taiwania-2 (V100 × 16, 2 nodes, InfiniBand).

See [plan.md](plan.md) for the full experiment design.

---

## Important: never run heavy work on the login node

The login node is shared. Running training or data-processing jobs directly
will consume its resources and affect all other users.

> **Always submit heavy work via `sbatch`.** This includes any benchmark run,
> data tokenisation, or multi-GPU job. The only things safe to run directly on
> the login node are: `git` commands, `sbatch`/`squeue`, `wandb sync`, and
> lightweight import checks (`python model.py`).

## Quick start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> **Cluster / quota note:** if `~/.local/bin` is unavailable, install into the
> project directory and set the cache redirects before any other step:
>
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=$(pwd)/.bin sh
> export PATH="$(pwd)/.bin:$PATH"
> export UV_CACHE_DIR=$(pwd)/.uv-cache
> export UV_PYTHON_INSTALL_DIR=$(pwd)/.uv-python
> ```

### 2. Create the virtual environment (Python 3.11)

```bash
uv venv .venv --python 3.11
```

uv downloads CPython 3.11 automatically if it is not already present.

### 3. Install dependencies

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install git+https://github.com/tinygrad/tinygrad.git
uv pip install tiktoken "wandb>=0.17"
```

### 4. Configure credentials

Create a `.env` file in the project root with your W&B API key:

```bash
echo "WANDB_API_KEY=your_key_here" > .env
```

The SLURM job scripts load this automatically. You can find your key at <https://wandb.ai/authorize>.

### 5. Prepare training data

```bash
sbatch prepare-data.slurm
```

### 6. Run the B0 baseline

Submit 5 repetitions (plan §6 requires ≥5 runs per config):

```bash
for i in 0 1 2 3 4; do sbatch experiments/b0_baseline/run.slurm $i; done
```

Results are written to `wandb/offline-run-*/`.
Sync to wandb after the runs finish:

```bash
source .venv/bin/activate
wandb sync wandb/offline-run-*
```

---

## Project layout

```
train-gpt2/
├── model.py                      shared: GPT model (TINY / BASE / WIDE)
├── data.py                       shared: nanoGPT-style memmap data loader
├── prepare_data.py               shared: tokenise corpus → data/*.bin
├── prepare-data.slurm            SLURM: one-time data tokenisation
├── plan.md                       full experiment design (B0 → E11)
├── requirements.txt
│
├── experiments/
│   ├── bench.py                  shared harness for single-GPU experiments
│   └── b0_baseline/
│       └── run.slurm             SLURM: B0 naïve baseline run
│
└── .venv/                        Python 3.11 venv (created by uv)
```

---

## Key benchmark commands

```bash
source .venv/bin/activate

# B0 — naïve baseline (FP32, no JIT)
WANDB_MODE=offline python experiments/bench.py --config BASE --batch 4 --exp B0 --rep 0

# E1 — JIT on
WANDB_MODE=offline JIT=1 python experiments/bench.py --config BASE --batch 4 --exp E1 --rep 0

# E3 — FP16
WANDB_MODE=offline HALF=1 python experiments/bench.py --config BASE --batch 4 --exp E3 --rep 0
```

Environment variables recognised by `bench.py`:

| Variable | Values | Effect |
|----------|--------|--------|
| `JIT` | `0` / `1` | Enable `torch.jit.script` |
| `HALF` | `0` / `1` | Enable FP16 via `torch.cuda.amp` |
| `BEAM` | `0` / `2` / `4` | tinygrad BEAM autotuning level (future) |
| `WANDB_MODE` | `online` / `offline` / `disabled` | wandb logging mode |
| `DEV` | `CUDA` | tinygrad device selection |

---

## Hardware reference (V100-SXM2)

| Metric | Value |
|--------|-------|
| FP32 peak | 15.7 TFLOP/s |
| FP16 tensor-core peak | 125 TFLOP/s |
| HBM2 bandwidth | ~900 GB/s |
| NVLink (intra-node) | ~300 GB/s |
| InfiniBand (inter-node) | ~12.5 GB/s |
| Memory / GPU | 32 GB |

MFU formula used in `bench.py`:

```
flops_per_step  = 6 × N_params × (batch × ctx)
MFU_fp32        = flops_per_step / step_time / 15.7e12
MFU_fp16        = flops_per_step / step_time / 125e12
```
