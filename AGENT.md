## Environment setup

This project requires **Python 3.11+** (tinygrad 0.13 needs ≥3.11).

> **Cluster note:** if your home directory is over quota, redirect uv's cache
> and Python installs into the project directory by setting the variables in
> Step 2 before running anything else.

### Step 1 — install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If the default install location (`~/.local/bin`) is unavailable (e.g. home
quota exceeded), install into the project directory instead:

```bash
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=$(pwd)/.bin sh
export PATH="$(pwd)/.bin:$PATH"
```

### Step 2 — redirect caches (only if home is over quota)

```bash
export UV_CACHE_DIR=$(pwd)/.uv-cache
export UV_PYTHON_INSTALL_DIR=$(pwd)/.uv-python
```

In a normal environment these variables are not needed; uv uses `~/.cache/uv`
and `~/.local/share/uv/python` by default.

### Step 3 — create the virtual environment

```bash
uv venv .venv --python 3.11        # downloads CPython 3.11 automatically
```

### Step 4 — install dependencies

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install git+https://github.com/tinygrad/tinygrad.git
uv pip install tiktoken "wandb>=0.17"
```

### Step 5 — smoke test

```bash
source .venv/bin/activate
python model.py
```

Expected output:
```
TINY: 29.9M params  (d=384, L=6, H=6, ctx=256)
BASE: 123.6M params  (d=768, L=12, H=12, ctx=512)
WIDE: 2646.0M params  (d=2560, L=32, H=20, ctx=512)
```

### Add or update packages

```bash
uv pip install <package>
uv pip freeze > requirements.txt   # update lockfile
```

---

## Job submission

All SLURM scripts activate `.venv` automatically — no manual activation needed
when submitting jobs.

> **Cluster constraint:** every 4 CPUs (`--cpus-per-task`) must be paired with
> 1 GPU (`--gpus-per-node`). Always keep the ratio `cpus : gpus = 4 : 1`.
> Examples: 4 CPU → 1 GPU, 8 CPU → 2 GPU, 16 CPU → 4 GPU.

| Script | Purpose |
|--------|---------|
| `prepare-data.slurm` | Tokenise TinyStories → `data/train.bin` + `data/val.bin` |
| `experiments/b0_baseline/run.slurm` | B0 naïve baseline benchmark |
| `gpt2-train.slurm` | Legacy HF fine-tune on Wikipedia |
| `download-wikipedia-data.slurm` | Download Wikipedia dataset |

### Submit a job

```bash
sbatch prepare-data.slurm
sbatch experiments/b0_baseline/run.slurm 0   # pass rep index 0–4 for ≥5 repeats
```

### Check status / view logs

```bash
squeue -a -u $USER
cat slurm-<jobid>.out
```
