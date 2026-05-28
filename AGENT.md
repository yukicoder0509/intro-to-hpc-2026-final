## Project design — read plan.md first

**Before writing or modifying any code, read `plan.md` in full.**

`plan.md` is the authoritative design document for this project. It specifies:

- The exact model architecture (§3): decoder-only GPT, pre-LayerNorm, fused `scaled_dot_product_attention(is_causal=True)`, weight-tied head, three fixed configs TINY/BASE/WIDE.
- The canonical step function (§3 step.py): `@TinyJit`-decorated, `sparse_categorical_crossentropy` without reshape, `opt.zero_grad()` → `loss.backward()` → `opt.step()` → `return loss.realize()`.
- The benchmark harness contract (§4.1): 3 warmup steps, ≥30 measured steps, GPU sync via `loss.numpy()` before reading the clock, results logged to wandb with the schema in §6.
- The experiment ladder (§4, E1–E11): each experiment changes exactly one variable. Never "improve" an experiment by changing more than its stated knob.
- The measurement protocol (§6): median + IQR, ≥5 fresh-process repeats per config, everything in wandb with the specified `config` and `summary` schema.
- Multi-GPU APIs to use (§2, §4 E6–E9): `Tensor.shard`, `TinyJit`, `BEAM`, `DEBUG` env vars — use tinygrad's native APIs, not PyTorch/Accelerate.

**Key constraints for all code changes:**

1. `model.py`, `data.py`, `train.py`, `experiments/bench.py` must stay tinygrad-only — no PyTorch imports.
2. The three model configs (TINY/BASE/WIDE) and their exact parameters are fixed — do not change them.
3. `bench.py` is the shared harness all experiments call. Keep it general; add experiment-specific flags via env vars (`JIT`, `BEAM`, `HALF`, `ZERO`, `TP`, `PP`, `GPUS`) not new scripts.
4. Every run must log to wandb using the schema in plan.md §6. Offline mode (`WANDB_MODE=offline`) is normal on compute nodes; sync afterward with `wandb sync`.
5. When in doubt about an implementation choice, the plan's pseudocode takes precedence over intuition.

---

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

## Important: never run heavy work on the login node

The login node is shared infrastructure. Running training, benchmarks, or
large data-processing jobs directly will consume its CPU/memory and affect
all other users. **Always submit heavy work via `sbatch`.**

What counts as heavy work:
- Any `python bench.py` / `python experiments/bench.py` run
- Any `python prepare_data.py` run on a full dataset
- Any multi-GPU or distributed job

What is safe on the login node:
- Editing files, `git` commands, `sbatch`, `squeue`, `wandb sync`
- `python model.py` (prints param counts, no compute)
- Quick syntax / import checks (`python -c "import model"`)

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
