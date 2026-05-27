# Scaling LLM Pre-training on V100s with tinygrad

### An HPC-Optimization Final Project — Experiment Design & Implementation Plan

**Course:** Introduction to HPC · **Team size:** 5 · **Platform:** NCHC Taiwania-class cluster (2× V100 nodes, Slurm)
**Framing:** We treat LLM pre-training as a *systems / throughput* problem, not a *model-quality* problem. The model never has to converge. Our deliverable is a measured, statistically-defensible optimization-and-scaling story: baseline → single-GPU optimization → and then a head-to-head comparison of **parallelization strategies** (data parallelism, ZeRO-sharded data parallelism, and tensor/model parallelism) scaled from 1 to 16 GPUs across 2 nodes.

---

## 0. Why this topic fits the rubric

The course rubric asks for (a) a baseline + cross-tests, (b) repeated runs with statistics and benchmarks, and (c) a talk centered on *improvement strategy and experimental method* while still showing results. LLM pre-training is an almost ideal vehicle:

- The core loop is a small number of large matmuls repeated thousands of times — a clean, reproducible kernel to profile and optimize.
- Every classic HPC lever applies directly: kernel autotuning, mixed precision / tensor cores, memory-bandwidth vs compute roofline, batch sizing for arithmetic intensity, data-parallel scaling, and intra- vs inter-node communication.
- Throughput (tokens/sec) and parallel efficiency are continuous, low-variance metrics — perfect for statistics and error bars.

Crucially, **we optimize the machine, not the loss.** We fix the model and the number of steps, then make each step faster and add more GPUs. Final validation loss is reported only to prove correctness (the loss must go down), never as a success metric.

### Hard constraints we design around

| Constraint | Value | Design consequence |
| --- | --- | --- |
| Running jobs / person | 1 | Develop on small single-node jobs; coordinate one shared 2-node job for headline runs |
| Resource ceiling | 2 nodes / 64 cores / **16× V100** | 8 GPUs/node; inter-node = InfiniBand, intra-node = NVLink |
| Wall-time / job | **1 hour** | No run converges; every benchmark is a fixed-step or fixed-time micro-run (seconds–minutes) |
| Talk | 15 min (10 report + 5 QA), warning at 8 min, hard stop | Rehearse to ~8.5 min; one plot per optimization |

> **Resource etiquette:** the QoS was loosened for everyone. Never leave idle multi-node allocations sitting, prefer the Judge/dev queue for smoke tests, and announce big 2-node runs in the group chat so only one runs at a time.
> 

---

## 1. The platform, quantitatively (this is our roofline)

A Taiwania-2-class node carries 8× V100-SXM2 with NVLink; nodes are joined by InfiniBand. The single-GPU peak numbers we optimize against (V100-SXM2):

| Quantity | Value | Used for |
| --- | --- | --- |
| FP32 peak | 15.7 TFLOP/s | FP32 roofline ceiling |
| FP16 (tensor core) peak | 125 TFLOP/s | mixed-precision ceiling |
| FP16 (vector, no TC) | ~31 TFLOP/s | fallback if TC codegen unavailable on sm_70 |
| HBM2 bandwidth | ~900 GB/s | roofline slope |
| NVLink (intra-node, per GPU) | ~300 GB/s aggregate | intra-node allreduce cost |
| InfiniBand (inter-node) | ~100 Gb/s ≈ 12.5 GB/s | inter-node allreduce cost — **~24× slower than NVLink** |
| Memory / GPU | **32 GB** (all NCHC V100s) | batch & checkpointing limits — generous headroom, so memory experiments must push batch/ctx or use WIDE |

**Roofline ridge points** (arithmetic intensity where compute-bound begins):

- FP32: 15.7e12 / 900e9 ≈ **17.4 FLOP/byte**
- FP16-TC: 125e12 / 900e9 ≈ **139 FLOP/byte**

Interpretation that drives the whole project:

- **Large GEMMs** (QKV projection, attention output, the 4× FFN) have high arithmetic intensity → compute-bound → they win from tensor cores and from bigger batches.
- **Elementwise/normalization ops** (LayerNorm, GELU, residual adds, softmax) are memory-bound → they win from *kernel fusion*, which tinygrad's scheduler does automatically.
- The **24× NVLink-vs-IB gap** is the entire reason inter-node scaling is hard and worth studying.

---

## 2. Why tinygrad

tinygrad is small enough to read end-to-end, has a PyTorch-like `Tensor` API, lazy evaluation, automatic kernel fusion, a JIT (`TinyJit`) that replays captured kernels, a built-in kernel autotuner (`BEAM`), first-class multi-GPU sharding (`Tensor.shard`), and rich built-in profiling (`DEBUG=2` prints per-kernel GFLOP/s and GB/s; `GlobalCounters` totals FLOPs). That profiling output is exactly the raw data an HPC project needs, with no extra tooling.

What tinygrad does **not** give us for free: cross-node training. `Tensor.shard` distributes across devices *visible to one process* (i.e. the 8 GPUs of one node). Spanning 2 nodes therefore requires us to add gradient synchronization ourselves (via MPI). **That gap is our headline HPC contribution**, not a limitation to hide.

Key APIs we rely on:

- `Tensor.shard(devices, axis=0)` / `.shard_()` — split a tensor across GPUs along an axis (data **or** weight axis) or replicate (`axis=None`). Note: sharding a weight along its *contraction* axis makes tinygrad insert a reduction collective — this is what lets us build **tensor (model) parallelism** without writing the collectives by hand.
- `TinyJit` — wrap the pure train step; first 2 calls warm up/capture, subsequent calls replay kernels (huge Python-overhead savings).
- `DEBUG=2/4`, `BEAM=2`, `PROFILE=1`, `GlobalCounters.reset()` / `.global_ops` — measurement and autotuning.

**Experiment tracking — Weights & Biases (wandb).** Every run logs to wandb instead of to ad-hoc text files. This is not optional polish; it is how we earn the rubric's "statistics + benchmark presentation" points cheaply: wandb auto-captures **system metrics** (per-GPU utilization, memory, power, temperature) with zero extra code — exactly the evidence an HPC project wants — and its **run groups**, **sweeps**, and **parallel-coordinates / scatter** views turn our factorial cross-tests into publication-ready figures we can screenshot straight into the deck. We log hyperparameters via `wandb.init(config=...)`, per-step metrics via `wandb.log(...)`, and final medians into `run.summary`. The wandb run table *is* our results table (§6) — exportable to CSV for the appendix. (Cluster caveat: compute nodes may lack internet, so we run `WANDB_MODE=offline` on the nodes and `wandb sync` from a login node — see §10.)

---

## 3. Model architecture (implement from scratch)

A standard decoder-only GPT (pre-LayerNorm, causal self-attention, 4× GELU MLP, weight-tied head). It is ~150 lines in tinygrad and is the right shape for a roofline story. We define **two fixed configs** so every experiment is comparable:

| Config | d_model | layers | heads | ctx | vocab | ~params | Role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **TINY** (dev) | 384 | 6 | 6 | 256 | 50257 (or char) | ~11 M | Fast iteration, correctness, JIT/BEAM tuning |
| **BASE** (scaling target) | 768 | 12 | 12 | 512 | 50257 | ~124 M | Headline throughput/MFU + 1→16 GPU scaling |
| **WIDE** (TP stress) | 2560 | 32 | 20 | 512 | 50257 | ~2.5 B | Sized so Adam state alone (~40 GB) **exceeds one 32 GB V100** — cannot run data-parallel on a single GPU. Used only in E9 to show when tensor parallelism is *necessary* |

Rationale: TINY fits trivially on one GPU and gives a sub-second step for rapid debugging. BASE (GPT-2-small shape) makes the GEMMs big enough to be compute-bound and is the right size for the throughput/MFU and scaling story, while still running many steps inside the 1-hour limit. **Note on 32 GB:** a 124 M model has comfortable memory headroom on a 32 GB V100, so the capacity ceiling is not the binding constraint for BASE. The memory-focused experiments (E5 checkpointing, E8 ZeRO) therefore deliberately push batch size and sequence length — or switch to WIDE — to make the 32 GB ceiling actually bite; ZeRO's per-GPU memory saving is still real and measurable on BASE even when nothing is near OOM. Don't enlarge BASE for the throughput/scaling work; the goal is measurement quality, not scale.

```python
# model.py  — decoder-only GPT in tinygrad (skeleton)
from tinygrad import Tensor, nn

class Attention:
  def __init__(self, dim, n_heads):
    self.n_heads, self.hd = n_heads, dim // n_heads
    self.qkv  = nn.Linear(dim, 3*dim, bias=False)
    self.proj = nn.Linear(dim, dim,  bias=False)
  def __call__(self, x):
    B, T, C = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q, k, v = [t.reshape(B, T, self.n_heads, self.hd).transpose(1, 2) for t in (q, k, v)]
    o = q.scaled_dot_product_attention(k, v, is_causal=True)   # fused, causal
    return self.proj(o.transpose(1, 2).reshape(B, T, C))

class Block:
  def __init__(self, dim, n_heads):
    self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
    self.attn = Attention(dim, n_heads)
    self.fc1, self.fc2 = nn.Linear(dim, 4*dim), nn.Linear(4*dim, dim)
  def __call__(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.fc2(self.fc1(self.ln2(x)).gelu())
    return x

class GPT:
  def __init__(self, vocab, ctx, dim, n_layers, n_heads):
    self.tok = nn.Embedding(vocab, dim)
    self.pos = nn.Embedding(ctx, dim)
    self.blocks = [Block(dim, n_heads) for _ in range(n_layers)]
    self.lnf  = nn.LayerNorm(dim)
    self.head = nn.Linear(dim, vocab, bias=False)
  def __call__(self, idx):
    B, T = idx.shape
    x = self.tok(idx) + self.pos(Tensor.arange(T))
    for b in self.blocks: x = b(x)
    return self.head(self.lnf(x))
```

```python
# step.py — the pure, JIT-able training step (the thing we benchmark)
from tinygrad import TinyJit, nn
opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=3e-4)

@TinyJit
def train_step(x, y):
  Tensor.training = True
  loss = model(x).sparse_categorical_crossentropy(y)
  opt.zero_grad(); loss.backward(); opt.step()
  return loss.realize()
```

### Data pipeline (must not be the bottleneck)

nanoGPT-style: pre-tokenize a fixed corpus **once** to a `uint16` `.bin`, `np.memmap` it, and sample random `(B, ctx+1)` windows. This makes data loading effectively free so that what we measure is *compute + communication*, not I/O.

- **Recommended corpus:** TinyStories or a FineWeb-Edu sample (~100 MB tokenized) with the GPT-2 BPE tokenizer (`tiktoken`). Good loss-decrease signal, trivial to ship.
- **Fallback:** character-level Shakespeare (vocab ≈ 65, no tokenizer dependency) for the TINY dev loop.
- Use the 32 CPU cores/node for the one-time tokenization and for a background prefetch thread that stages the next batch to a pinned host buffer.

---

## 4. The experiments

This section is written so a teammate can sit down and run each experiment without guessing. We first nail down **two reference baselines** that every other number is compared against, give the **fixed experiment recipe** every run follows, then specify each experiment with the same four parts: **Concept** (what idea we're testing, in two sentences), **What to alter** (the exact knob, with code/flags), **Run it** (the command), and **What the results should look like** (the shape of the data + how to read it).

> All illustrative numbers below are **order-of-magnitude expectations to validate against, not promises.** Your hardware/commit will differ. What must hold is the *shape* (direction and rough ratio); if you see the opposite shape, that's a finding to explain, not a number to fudge.
> 

### 4.0 Baseline design (do this first — everything else is relative to it)

We define **two** reference points, because "speedup" means different things for single-GPU work versus scaling.

**B0 — the naïve baseline (reference for the single-GPU ladder, rungs 1–5).**
The most unoptimized configuration a beginner would write. Everything in rungs 1–5 is reported as a speedup over B0.

| Setting | B0 value |
| --- | --- |
| Model | BASE (124 M) |
| Devices | 1× V100 |
| Precision | FP32 |
| JIT | **off** |
| BEAM | 0 (off) |
| Batch × ctx | 4 × 512 → 2048 tokens/step |
| Optimizer | AdamW, lr 3e-4 |
| Steps measured | 30 steady-state (after 3 warmup) |

**B* — the optimized single-GPU reference (the per-GPU unit for scaling, rungs 6–11).**
This is B0 after rungs 1–4 are applied (JIT on, BEAM cached, FP16, best single-GPU batch). Scaling efficiency in the parallelism experiments (rungs 6–11) is measured against **B* on 1 GPU**, because you scale the configuration you would actually run, not the naïve one.

**How to record a baseline (and any config):** run the harness, discard 3 warmup steps, time ≥30 steps with a GPU sync each step. The harness calls `wandb.init(config=...)` (capturing every knob) and writes **median ms/step**, **tokens/s**, and **MFU** to `run.summary` (schema in §6). Compute MFU once here and reuse the formula everywhere:

```
tokens_per_step = batch * ctx
flops_per_step  = 6 * N_params * tokens_per_step          # 6ND; N≈124e6 for BASE
achieved_flops  = flops_per_step / (median_ms_per_step/1000)
MFU_fp16        = achieved_flops / 125e12                  # vs FP16 tensor-core peak
MFU_fp32        = achieved_flops / 15.7e12                 # vs FP32 peak
```

*Illustrative B0:* with no JIT, hundreds of tiny kernels are dispatched from Python each step, so expect a **compute-starved** step — roughly **300–800 ms/step**, **~3–7k tokens/s**, **MFU ≈ 5–15%** (vs FP32 peak). The exact number doesn't matter; its *lowness* is the point the ladder will fix.

### 4.1 The fixed experiment recipe (every run obeys this)

1. Change **exactly one** variable from the stated reference (B0 or B*).
2. Keep model, dataset, step count, and seed identical across the comparison.
3. Discard 3 warmup steps; measure ≥30 steady-state steps; force a GPU sync (`loss.numpy()`) before reading the clock.
4. Repeat the whole run ≥5 times (fresh process); group them in wandb with the same `group=` so the dashboard shows mean±std bands automatically; report **median + IQR**.
5. Log everything to wandb — the config captures the full environment (commit, env vars, GPU/node IDs, job ID); `wandb.log` records per-step metrics; `run.summary` records the medians.
6. A plot or a paired before/after number is the deliverable — never a single unreplicated timing.

The harness wraps this so each experiment is one call:

```python
# bench.py (core loop, abbreviated)
import os, time, statistics, wandb
run = wandb.init(project="hpc-gpt-tinygrad", group=cfg.exp, name=f"{cfg.exp}-r{cfg.rep}",
                 mode=os.getenv("WANDB_MODE", "online"), config=vars(cfg))
wandb.config.update({"commit": git_sha(), "jit": os.getenv("JIT"), "beam": os.getenv("BEAM")})
dts = []
for i in range(cfg.warmup + cfg.steps):
  t0 = time.perf_counter()
  loss = train_step(x, y); loss.numpy()                 # sync before clock
  dt = (time.perf_counter() - t0) * 1e3
  if i >= cfg.warmup:                                    # discard warmup
    dts.append(dt)
    wandb.log({"ms_per_step": dt, "tokens_per_s": cfg.tokens/(dt/1e3), "loss": float(loss.numpy())})
run.summary.update({"ms_median": statistics.median(dts),
                    "ms_iqr": iqr(dts), "tokens_per_s": cfg.tokens/(statistics.median(dts)/1e3),
                    "mfu_fp16": mfu(statistics.median(dts))})
wandb.finish()
```

The ladder at a glance (fill the right column from your own runs). Experiments 1–5 are single-GPU; 6–10 are *parallelism strategies* — different ways to spread one model over many GPUs, each trading compute, memory, and communication differently.

| # | Experiment | One thing you alter | Illustrative result vs reference |
| --- | --- | --- | --- |
| 1 | JIT | `TinyJit` on | 2–10× faster step (bigger win on TINY) |
| 2 | BEAM autotune | `BEAM=2`, `BEAM=4` | 1.2–2× faster GEMMs; +seconds–minutes compile |
| 3 | Mixed precision | FP32 → FP16 | 2–4× faster; ~½ memory |
| 4 | Batch sweep | batch 2→OOM | MFU rises then plateaus |
| 5 | Grad checkpointing | recompute on | enables ~1.5–2× larger batch; ±20% net tok/s |
| 6 | **Data parallelism** (intra-node) | 1→2→4→8 GPUs, NVLink | ~85–95% efficiency; model replicated |
| 7 | **Data parallelism** (inter-node) | 8→16 GPUs, MPI/IB | naïve ~60–75%; tuned ~85% |
| 8 | **ZeRO** (sharded DP) | shard optimizer states/grads/params | ~same speed as DP, **much less memory/GPU** |
| 9 | **Tensor parallelism** | shard weight matrices across GPUs | enables huge layers; comms *inside* every block |
| 10 | **Pipeline parallelism** *(stretch)* | put layer-groups on different GPUs | scales by depth; "bubble" overhead to measure |

The three parallelism families answer different questions and are the analytical heart of the project:

- **Data parallelism (6–7):** every GPU holds the *whole model* and a *slice of the batch*. Cheap communication (one gradient allreduce/step), but the model and optimizer state are **replicated N times** → memory-wasteful.
- **ZeRO (8):** data parallelism that **stops replicating** — it shards the optimizer state (and optionally gradients and parameters) across the DP GPUs, recovering the wasted memory at the cost of extra communication. The throughput should match DP while peak memory/GPU drops sharply.
- **Tensor / pipeline parallelism (9–10):** *model parallelism* — a single model instance is **split across GPUs** (by weight matrix, or by layer). Needed when one layer (TP) or the whole model (PP) won't fit on a GPU; communication moves *inside* the forward/backward pass.

The capstone is **E11: the strategy bake-off** — run DP, ZeRO, and TP on the *same* BASE config and the *same* 8 GPUs, and plot tokens/s vs peak-memory-per-GPU. That single chart is the talk's punchline.

---

### Experiment 1 — JIT (kill Python dispatch overhead)

**Concept.** In eager mode tinygrad launches every kernel from Python, and that per-launch overhead dominates when each kernel is small. `TinyJit` records the kernel sequence on the first couple of calls and then *replays* it, removing Python from the hot loop. This isolates "framework overhead" from "real GPU compute."

**What to alter.** Only the JIT decorator on the train step.

```python
def train_step(x, y): ...        # B0: plain function (no decorator)
@TinyJit
def train_step(x, y): ...        # E1: identical body, decorated
```

**Run it.**

```bash
JIT=0 python bench.py --config BASE --steps 33     # B0 reference
JIT=1 python bench.py --config BASE --steps 33     # E1
```

**What the results should look like.** A step-time vs step-index plot: the first 2–3 JIT steps are *slower* than B0 (capture cost), then steady-state drops sharply and flattens. Steady-state speedup is **large for TINY (overhead-dominated, can be 10×+)** and **more modest for BASE (compute-dominated, ~1.5–3×)** — that contrast is itself a result worth a sentence. GPU utilization (from `nvidia-smi`/`PROFILE=1`) should rise noticeably. If BASE barely speeds up, that's expected and *good* — it means BASE is already compute-bound, which motivates rungs 2–3.

---

### Experiment 2 — BEAM kernel autotuning

**Concept.** For each kernel, many equivalent GPU implementations exist (tiling, thread/loop layout). `BEAM=k` searches `k` candidates per kernel and keeps the fastest *measured* one for this exact hardware. The win is real but comes with a one-time search cost — a classic HPC amortization tradeoff.

**What to alter.** The `BEAM` env var only (keep JIT on).

```bash
BEAM=0 JIT=1 python bench.py --config BASE --steps 33   # from E1
BEAM=2 JIT=1 python bench.py --config BASE --steps 33
BEAM=4 JIT=1 python bench.py --config BASE --steps 33
```

Cache results so repeats reuse the search: `export BEAM_CACHE=1` (and reuse tinygrad's on-disk cache between runs).

**What the results should look like.** Two numbers per BEAM level: **steady-state GFLOP/s** (read from `DEBUG=2` per-kernel output, or derived from ms/step) and **one-time compile/search time**. Expect steady-state **1.2–2×** over BEAM=0, with **BEAM=4 ≥ BEAM=2** in speed but a longer search (tens of seconds to a few minutes). Plot it as a small bar chart of steady-state speedup with the search time annotated on each bar. The teaching point: BEAM only pays off when the run is long enough to amortize the search — quantify the break-even step count.

---

### Experiment 3 — Mixed precision (FP16 / tensor cores)

**Concept.** V100 does FP32 at 15.7 TFLOP/s but FP16 with tensor cores at up to 125 TFLOP/s, and FP16 halves memory traffic. We compute the big matmuls in FP16 while keeping an FP32 master copy of the weights and a scaled loss for numerical stability.

**What to alter.** Cast model+activations to half; keep FP32 master weights and apply static loss scaling.

```python
HALF = getenv("HALF", 0)
if HALF:
  for p in nn.state.get_parameters(model): p.assign(p.half())
def train_step(x, y):
  Tensor.training = True
  loss = model(x.half() if HALF else x).sparse_categorical_crossentropy(y)
  (loss * LOSS_SCALE).backward()                    # scale up before backward
  # ...unscale grads by LOSS_SCALE inside the optimizer, step on FP32 master...
  return loss.realize()
```

**Run it.**

```bash
HALF=0 BEAM=2 JIT=1 python bench.py --config BASE --steps 200   # FP32 ref + loss curve
HALF=1 BEAM=2 JIT=1 python bench.py --config BASE --steps 200   # FP16
```

**What the results should look like.** Two deliverables: (1) a **throughput** number — expect **~2–4×** tokens/s and roughly **halved peak memory**; (2) a **correctness overlay** — plot the FP16 and FP32 loss curves over the first ~200 steps; they must track closely (small gap OK, divergence = a loss-scaling bug). **Mandatory check:** run `DEBUG=4` once and grep the generated code for tensor-core (HMMA/`wmma`) instructions to confirm whether Volta tensor cores are actually used. If they are, you're near the 125 TFLOP/s regime; if not, you're getting the gain purely from halved memory traffic + ~2× FP16 vector throughput — **state which regime your number reflects.**

---

### Experiment 4 — Batch / sequence sweep (arithmetic intensity)

**Concept.** Bigger batches make the matmuls bigger, raising arithmetic intensity (FLOP per byte moved) and pushing the workload from memory-bound toward the compute-bound side of the roofline. MFU should climb with batch until you saturate compute or hit OOM.

**What to alter.** Only the batch size (then, separately, ctx), at the best precision from E3.

```bash
for B in 2 4 8 16 32 64; do
  HALF=1 BEAM=2 JIT=1 python bench.py --config BASE --batch $B --steps 33 || break  # break on OOM
done
```

**What the results should look like.** An **MFU-vs-batch curve** (and tokens/s-vs-batch): rising, then a **knee** where it plateaus, then OOM. The plateau MFU is your single-GPU compute ceiling for this model (commonly landing somewhere around **25–40%** on a V100 for a 124 M model — report your actual). Mark the largest batch that fits in 32 GB. This curve directly visualizes the roofline ridge from §1 and tells rung 6 what per-GPU batch to scale with.

---

### Experiment 5 — Gradient checkpointing (memory ↔ compute trade)

**Concept.** Activations stored for the backward pass dominate memory. Checkpointing throws them away and *recomputes* them during backward — paying ~25–35% extra compute to free a lot of memory, which you then spend on a larger batch.

**What to alter.** Wrap each transformer block so its activations are recomputed in backward (toggle via a flag), then push batch past the E4 OOM point. With 32 GB, BASE at ctx 512 has lots of headroom, so make activation memory the binding constraint first: run E5 at **long context (e.g. ctx 1024–2048)** and/or large batch — activation memory grows with batch × ctx, so that's where checkpointing produces a visible saving. (Reuse the same long-ctx setting for the no-ckpt and ckpt runs so the comparison is fair.)

```bash
CKPT=0 HALF=1 python bench.py --config BASE --ctx 1024 --batch <E4_max>   # no-ckpt max batch
CKPT=1 HALF=1 python bench.py --config BASE --ctx 1024 --batch <larger>   # ckpt enables bigger batch
```

**What the results should look like.** A small **Pareto plot: peak memory (x) vs tokens/s (y)**, with and without checkpointing. Checkpointing should let you fit roughly **1.5–2× the batch**; net throughput may go up (bigger batch amortizes overhead) or down (recompute tax wins) by perhaps ±20% — **either direction is a valid result**, and the point is to show the tradeoff explicitly rather than to "win." State the recompute overhead you measured (extra ms/step at equal batch).

---

### Experiment 6 — Intra-node data parallelism (NVLink, 1→8 GPUs)

**Concept.** Replicate the model on each GPU, give each a different slice of the batch, and average gradients across GPUs every step (data parallelism). Inside one node these averages travel over NVLink (~300 GB/s), so communication is cheap relative to compute and scaling should be near-linear. tinygrad does the gradient allreduce implicitly once parameters are sharded.

**What to alter.** The number of GPUs and how the batch/params are sharded — model code is otherwise unchanged.

```python
from tinygrad import Device
GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(NGPU))   # NGPU ∈ {1,2,4,8}
for p in nn.state.get_parameters(model): p.shard_(GPUS)       # replicate params
x = x.shard(GPUS, axis=0); y = y.shard(GPUS, axis=0)          # split batch across GPUs
```

Run **two** scaling modes:

- **Strong scaling:** global batch fixed (e.g. 64); per-GPU batch = 64/NGPU.
- **Weak scaling:** per-GPU batch fixed (e.g. 16); global batch grows with NGPU.

```bash
for N in 1 2 4 8; do GPUS=$N python bench.py --config BASE --mode strong --steps 33; done
for N in 1 2 4 8; do GPUS=$N python bench.py --config BASE --mode weak   --steps 33; done
```

**What the results should look like.** A **speedup-vs-GPU-count plot** with the ideal `y=x` line for reference, plus a **parallel-efficiency** number at 8 GPUs. Over NVLink for a 124 M model expect **~85–95% efficiency at 8 GPUs** (strong scaling dips a little more than weak at high GPU count, because per-GPU batches shrink and overhead grows). If strong scaling falls off hard at 8, check that the per-GPU batch hasn't become too small to keep the GPU busy — note it as a finding.

---

### Experiment 7 — Inter-node data parallelism (MPI, 8→16 GPUs) — the capstone

**Concept.** To use both nodes (16 GPUs) we run **one tinygrad process per node** (each already data-parallel over its 8 GPUs via E6) and average gradients *between* the two nodes with MPI. Those averages now cross **InfiniBand (~12.5 GB/s)** instead of NVLink — ~24× slower — so communication is suddenly expensive, and the whole experiment is about clawing efficiency back.

**What to alter.** Add cross-node gradient averaging before the optimizer step, then iterate on *how* you communicate. Start naïve:

```python
# one Allreduce per parameter tensor — the slow baseline
from mpi4py import MPI; comm = MPI.COMM_WORLD; world = comm.Get_size()   # 2 ranks
def allreduce_grads(params):
  for p in params:                                  # ~150 separate messages
    out = np.empty_like(g := p.grad.numpy()); comm.Allreduce(g, out, op=MPI.SUM)
    p.grad.assign(Tensor(out / world).shard(GPUS))
```

Then alter **one communication strategy at a time** and re-measure:

1. **Bucketing:** flatten all grads into one contiguous buffer → a **single** `Allreduce` (removes ~150× per-message latency).
2. **Ring vs naïve:** compare MPI's default Allreduce against a hand-rolled ring schedule; relate the achieved bandwidth to the IB ceiling.
3. **Overlap:** start reducing early-layer gradients while later layers are still in backward.

**Run it.**

```bash
sbatch scale16.slurm     # 2 nodes, 1 task/node, gres=gpu:8, --time=01:00:00
# scale16.slurm body: srun python train_mpi.py --config BASE --comm {naive|bucket|ring|overlap} --steps 200 --measure
```

```bash
#!/bin/bash
#SBATCH --nodes=2 --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=32 --time=01:00:00
srun python train_mpi.py --config BASE --comm bucket --steps 200 --measure
```

**What the results should look like.** Two figures: (1) a **communication-strategy bar chart** of tokens/s for naïve → bucket → ring → overlap, where each step should recover throughput; (2) the **scaling curve extended to 16 GPUs** with parallel efficiency, plus a **comms-fraction** number (allreduce time ÷ step time). Expect a visible **efficiency cliff** at the 8→16 jump with naïve comms (down to perhaps **~60–75%**) that bucketing+overlap pulls back toward **~85%**. The headline sentence writes itself: *"InfiniBand is ~24× slower than NVLink, so naïve per-tensor allreduce stalls the step; bucketing + overlap hides most of it."* Also confirm the 16-GPU loss curve still matches the FP32 single-GPU curve at equal tokens-seen (correctness under scaling).

> **Risk control.** Inter-node (E7) is the only experiment that can fail to land inside the 1-hour/2-node budget. Treat the **single-node 8-GPU** result (E6) as the guaranteed headline and 16-GPU as the stretch. The new parallelism experiments below (E8–E11) are all **single-node, 8-GPU** and therefore *not* exposed to the 2-node risk. Get E1–E6 measured *before* the first 2-node job, so the talk is complete with or without E7.
> 

---

### Experiment 8 — ZeRO (sharded data parallelism)

**Concept.** Plain data parallelism (E6/E7) keeps a *full* copy of the parameters, gradients, **and Adam optimizer state** on every GPU. The optimizer state is the hidden hog: Adam keeps an fp32 master weight plus two moments (`m`, `v`) ≈ **12 bytes/param**, several times larger than the fp16 params themselves — and DP replicates all of it `N` times. ZeRO stops the replication by *sharding* this state across the `N` data-parallel GPUs: each GPU owns and updates only `1/N` of it, then the updated parameters are shared back. Stages: **ZeRO-1** shards optimizer state, **ZeRO-2** also shards gradients, **ZeRO-3** also shards parameters (≈ FSDP). We target ZeRO-1 (most memory saved per unit of complexity); 2/3 are stretch.

**What to alter.** Replace "every GPU runs the full optimizer" with three steps: reduce-scatter gradients → each GPU updates only its parameter slice → all-gather the updated parameters for the next forward.

```python
# zero1.py — shard Adam state across the DP GPUs (conceptual)
shards = partition(params, NGPU)                 # contiguous 1/N slice, one owner per GPU
m = zeros_like(shards[rank]); v = zeros_like(shards[rank])   # only MY slice's moments
def step():
  loss.backward()
  reduce_scatter_(grads, NGPU)                   # GPU r ends up with summed grad for its slice
  adam_update(params[shards[rank]], grads[shards[rank]], m, v)   # update only my slice
  all_gather_(params, NGPU)                      # rebuild full params on every GPU
```

The reduce-scatter + all-gather move the same total volume as a DP all-reduce, so **speed should be ~unchanged** while **optimizer memory drops ~N×**.

**Run it.**

```bash
ZERO=0 python bench.py --config BASE --gpus 8 --steps 33   # vanilla DP — memory reference
ZERO=1 python bench.py --config BASE --gpus 8 --steps 33   # shard optimizer state
ZERO=2 python bench.py --config BASE --gpus 8 --steps 33   # + shard grads   (stretch)
ZERO=3 python bench.py --config BASE --gpus 8 --steps 33   # + shard params  (stretch ≈ FSDP)
```

**What the results should look like.** A bar chart of **peak memory/GPU** (read straight from wandb's system metrics) that steps *down* at each stage — ZeRO-1 alone should remove the bulk, since Adam state dominates — while the **tokens/s bars stay roughly flat** (a small dip from the extra collective is expected and is the finding). Then spend the freed memory on a bigger batch and report the *effective* throughput gain. Correctness check: loss curve must still match DP at equal tokens-seen.

---

### Experiment 9 — Tensor (model) parallelism

**Concept.** Instead of replicating the model, *split each big weight matrix across GPUs* so several GPUs cooperate on one layer (Megatron-style). In the FFN, split the first linear **column-wise** (each GPU owns a slice of the hidden dimension — no communication) and the second linear **row-wise** (each GPU produces a partial sum that is combined by a single all-reduce). Attention is split by heads. Communication now lives *inside every block* — cheap over NVLink, brutal over InfiniBand — which is exactly why real systems keep TP within a node.

**What to alter.** Shard the block's weights along the right axes and let tinygrad insert the collective (sharding along the contraction axis triggers a reduce). The from-scratch craft is getting the axes consistent so a column-parallel output feeds a row-parallel input with exactly **one** all-reduce per block.

```python
# tensor-parallel FFN: column-parallel fc1, row-parallel fc2
fc1.weight.shard_(GPUS, axis=0)     # split hidden dim (output axis) — no comms here
fc2.weight.shard_(GPUS, axis=1)     # split contraction axis → tinygrad inserts the all-reduce
x = x.shard(GPUS, axis=None)        # activations replicated; each GPU computes its hidden slice
# attention: shard the qkv and output projections so each GPU owns a subset of heads
```

Confirm with `DEBUG=2` that a reduce/all-reduce kernel appears **once per block** (not per matmul, not missing).

**Run it.**

```bash
TP=1 python bench.py --config BASE --gpus 8 --steps 33                 # tensor-parallel, 1 node
# stress config that does NOT fit as data-parallel on a single GPU:
TP=1 python bench.py --config WIDE --gpus 8 --steps 33                 # e.g. d_model 2048
```

**What the results should look like.** On BASE (which already fits one GPU), TP will likely be **slower than DP** because the per-block communication isn't amortized — that is the expected, instructive result: *"don't reach for model parallelism when data parallelism fits."* TP earns its keep on the **WIDE** config that *cannot* run as DP at all: report that TP makes it runnable, plus its **communication fraction** (much higher than DP's). If time allows, show inter-node TP collapsing in throughput to make concrete *why* TP stays inside a node.

---

### Experiment 10 — Pipeline parallelism *(stretch)*

**Concept.** Split the model by **depth**: GPU0 holds the first layer-group, GPU1 the next, and so on; activations flow forward GPU→GPU and gradients flow back. A single batch leaves most stages idle (the pipeline "bubble"); splitting the batch into micro-batches keeps stages busy (GPipe).

**What to alter.** Place each layer-group's parameters on a specific device and sweep the micro-batch count.

```bash
PP=1 MICRO=1 python bench.py --config BASE --gpus 4 --steps 33    # large bubble
PP=1 MICRO=8 python bench.py --config BASE --gpus 4 --steps 33    # bubble shrinks
```

**What the results should look like.** A tokens/s-vs-micro-batch curve that rises and saturates as the bubble shrinks; compare the measured idle fraction against the theoretical bubble ≈ `(stages−1)/(stages−1+micro_batches)`. Keep this **optional** — it is the most code for the least surprising result, included only if a teammate finishes early.

---

### Experiment 11 — Strategy bake-off (the capstone figure)

**Concept.** With DP, ZeRO, and TP implemented, run them head-to-head on *identical* hardware and model so the tradeoffs land in a single chart.

**What to alter.** Only the parallelism strategy; fix BASE, 8 GPUs (1 node), FP16, best batch from E4.

```bash
for S in dp zero1 zero3 tp; do
  python bench.py --config BASE --gpus 8 --strategy $S --steps 33
done
```

**What the results should look like.** A Pareto scatter of **tokens/s (y) vs peak memory/GPU (x)**, one point per strategy, pulled directly from the wandb run group, with a small companion panel of **communication fraction** per strategy. Expected story: **DP** = fastest, most memory; **ZeRO** = ~same speed, far less memory (the best Pareto point when the model fits); **TP** = highest comms, only wins when a layer doesn't fit. This is the slide that proves the team understands *why each strategy exists* — make it the punchline.

---

## 5. Work split — five owners, one shared harness

Everyone co-owns the V100 roofline numbers and logs through the **shared wandb harness** (owned by A). Work is split by *theme*: single-GPU optimization (B, C), the data-parallel family (D), and the model-parallel family + presentation (E). Dependencies flow A → (B, C) → D → E, and the E11 bake-off is a joint D+E deliverable.

**Person A — Model, Data & wandb Harness (the foundation).**
Implements the GPT model and JIT-able step from scratch; builds the `memmap` tokenized data loader + CPU-core prefetch; **owns `bench.py` and the wandb integration** (config capture, per-step logging, summary medians, run groups, offline/sync workflow) that every other experiment calls; owns correctness (loss decreases; gradient sanity-check on TINY vs a tiny reference) and reproducibility (seeds, fixed configs). *Output:* the canonical training script + measurement harness everyone forks. *Talk slot:* model + methodology (1.5 min).

**Person B — Single-GPU Compute (E1–E2).**
Baseline B0 profiling with `DEBUG=2` + `GlobalCounters`; JIT on/off study; `BEAM` autotuning with compile-cost accounting; builds the **V100 roofline plot** and places measured kernels on it. *Output:* optimized single-GPU compute baseline + roofline. *Talk slot:* "where the time goes" + JIT/BEAM (1.5 min).

**Person C — Precision & Memory (E3–E5).**
FP16/tensor-core path with loss scaling + FP32-master-weight stability check (overlaid loss curves) and sm_70 codegen verification; batch/seq sweep → MFU curve; gradient checkpointing → memory↔throughput Pareto. *Output:* the precision+memory recipe defining B*. *Talk slot:* mixed precision + MFU curve (1.5 min).

**Person D — Data-Parallel Family (E6–E8).**
Intra-node sharding (1→8, NVLink); inter-node MPI gradient sync + bucketing/ring/overlap (8→16, IB); and **ZeRO** optimizer-state/grad/param sharding as the memory-saving extension of DP. Produces the scaling curves and the comms breakdown. *Output:* DP scaling efficiency + ZeRO memory-reduction bars. *Talk slot:* data parallelism + ZeRO (2 min).

**Person E — Model-Parallel Family + Captain (E9–E11).**
Implements **tensor parallelism** (and pipeline parallelism if time allows); owns the **E11 strategy bake-off** (DP vs ZeRO vs TP Pareto chart, built from D's and E's runs); and is the **presentation captain** — assembles the wandb dashboards into the deck, enforces the §6 protocol, times the talk, and leads QA. *Output:* TP results + the capstone bake-off figure + the deck. *Talk slot:* model parallelism + bake-off + conclusions (2.5 min).

A natural pairing if you'd rather work in twos for the hard parts: B+C on single-GPU, D+E on the parallelism families, A floating to support both and guarding correctness + the harness.

---

## 6. Measurement protocol (this is what earns the "statistics" points)

A result is only valid if it follows this protocol. The wandb harness (owned by A) enforces it.

1. **Warmup discard:** drop the first **3** steps of every run (JIT capture + autotune + allocator settling).
2. **Repeats:** ≥ **5** measured runs per configuration, fresh process each time, all sharing one wandb `group=`. Within a run, time ≥ 30 steady-state steps. wandb then draws the mean±std band across the group automatically.
3. **Timing:** **synchronize the GPU** before reading the clock (`loss.numpy()` forces completion — otherwise lazy ops make timings meaningless).
4. **Report:** **median** step time with **IQR** (or mean ± 95% CI) into `run.summary`; every bar/line in the deck carries error bars (wandb supplies them from the run group).
5. **Derived & system metrics:**
    - **MFU** = (6 · N_params · tokens_per_step / step_time) / peak_FLOP/s, vs both FP32 (15.7T) and FP16-TC (125T) ceilings.
    - **Parallel efficiency** = speedup(N) / N, for strong scaling (global batch fixed) and weak scaling (per-GPU batch fixed).
    - **Comms fraction** = collective time / step time (DP all-reduce, ZeRO reduce-scatter+all-gather, or TP per-block all-reduce).
    - **Peak memory/GPU** and **GPU utilization/power** come *for free* from wandb's system metrics — no instrumentation needed. These are the headline evidence for E5/E8.
6. **Cross-tests (交叉測試) via wandb Sweeps:**
    - *Factorial:* define a `wandb sweep` over the grid {precision} × {batch} × {strategy ∈ DP, ZeRO-1, ZeRO-3, TP} × {GPU-count} so interactions are visible at once; read it as a **parallel-coordinates plot**.
    - *Strategy bake-off (E11):* the DP/ZeRO/TP points on one tokens/s-vs-memory Pareto chart, filtered from the sweep.
    - *Placement invariance:* repeat a config on different physical GPUs / swap which node is MPI rank 0 — throughput should be statistically identical; a gap is a NUMA/topology finding.
    - *Correctness cross-check:* every parallel strategy's loss curve must overlay the single-GPU FP32 curve at equal tokens-seen.
7. **Environment capture:** `wandb.config` records the tinygrad commit, env vars (`JIT`, `BEAM`, `HALF`, `ZERO`, `TP`, `PP`), GPU/node IDs, and the Slurm job ID on every run — reproducibility is automatic.

### wandb run summary schema (one logged run per measured config)

```
# wandb.config:   exp, strategy, zero_stage, tp, pp, gpus, nodes, precision,
#                 jit, beam, batch, ctx, commit, jobid
# run.summary:    ms_median, ms_iqr, tokens_per_s, mfu_fp16, peak_mem_gb,
#                 gpu_util_mean, comms_frac, speedup, parallel_eff, n_runs
```

Export the run table to CSV for the report appendix (`wandb export` or the dashboard's download button) so the deliverable doesn't depend on a live wandb account.

---

## 7. Presentation plan (10 minutes, hard stop)

Rehearse to **~8.5 min** so the 8-minute warning is a non-event. One figure per beat; every figure is a screenshot from the wandb dashboard.

| Time | Beat | Figure (from wandb) |
| --- | --- | --- |
| 0:00–1:00 | Problem + framing: "we optimize the machine, not the loss" | the experiment-ladder table |
| 1:00–2:00 | Setup: model, wandb harness, metrics, V100 roofline | roofline with kernel markers |
| 2:00–4:00 | Single-GPU ladder, compressed: JIT → BEAM → FP16 → batch/MFU | cumulative-speedup bar + MFU-vs-batch curve |
| 4:00–6:00 | **Data parallelism + ZeRO:** 1→8→16 scaling efficiency, comms breakdown, ZeRO memory drop | scaling curve + ZeRO peak-memory bars |
| 6:00–8:00 | **Model parallelism + the bake-off:** TP concept + DP vs ZeRO vs TP Pareto | the E11 tokens/s-vs-memory scatter (the punchline) |
| 8:00–9:00 | Synthesis + correctness: cumulative speedup, where the roofline says we still leave performance, loss curves overlay | combined bar + overlaid loss curves |

QA prep (5 min): be ready to defend the MFU formula, warmup discard, FP16 stability, the intra→inter efficiency cliff, **why ZeRO saves memory without losing speed**, and **when TP beats DP** (only when a layer doesn't fit).

---

## 8. Two-week schedule

**Week 1 — build & single-GPU.**

- Days 1–2: A lands model + data loader + **wandb harness** + correctness on TINY (everyone can run *something* and see it in the dashboard).
- Days 3–4: B (JIT, BEAM, roofline) and C (FP16, batch sweep, checkpointing) optimize BASE single-GPU in parallel; D starts intra-node DP.
- Day 5: freeze B* and the wandb config/summary schema. First end-to-end dry run of the §6 protocol with a small `wandb sweep`.

**Week 2 — parallelism & present.**

- Days 6–7: D lands intra-node 1→8 (E6) and ZeRO-1 (E8); E lands tensor parallelism (E9). All single-node, low-risk.
- Day 8: one coordinated 2-node job for inter-node DP (E7, the only 2-node experiment). ZeRO-2/3 and pipeline (E10) only if ahead of schedule.
- Day 9: E builds the E11 bake-off from D's + E's runs; full factorial sweep + repeats; freeze figures. (Report order locks **5/26** — have results in hand before then.)
- Days 10–12: assemble deck from wandb screenshots, rehearse to time, QA drills.

---

## 9. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Cross-node MPI doesn't land in time | Single-node 8-GPU is the guaranteed headline; 16-GPU is stretch |
| ZeRO/TP from scratch is too much code | Target **ZeRO-1 only** and **TP via tinygrad's native weight-sharding** (collectives auto-inserted); ZeRO-2/3, pipeline, inter-node TP are all explicitly stretch — descope freely |
| TP "looks broken" (slower than DP on BASE) | That's expected when the model fits one GPU; demonstrate TP's value on the **WIDE** config that can't run as DP |
| wandb can't reach the internet from compute nodes | Run `WANDB_MODE=offline` on nodes; `wandb sync` the run dirs from a login node afterward |
| Multi-process runs spam duplicate wandb runs | Only MPI rank 0 logs aggregate metrics; tag per-rank runs with a shared `group=` |
| No Volta tensor-core codegen in tinygrad | Report FP16 memory-traffic/vector gains instead; verify via `DEBUG=4` |
| Data loader becomes the bottleneck | `memmap` + prefetch; sanity-check with a no-data-load "synthetic batch" run |
| 1-hour wall-time kills a long run | All benchmarks are fixed-step micro-runs (seconds–minutes), never convergence |
| High timing variance | Warmup discard + ≥5 repeats + wandb run groups (median/IQR); pin GPUs; capture topology |
| BEAM autotune time eats the budget | Cache autotune results; report compile cost as its own datapoint |
| One running job/person blocks the team | Develop on small single-node jobs; schedule the shared 2-node run; announce in chat |

---

## 10. Setup checklist

- [ ]  Install [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then `uv venv .venv && uv pip install tinygrad mpi4py tiktoken numpy wandb` on the cluster (match the site's CUDA module).
- [ ]  `wandb login` **on a login node** (compute nodes may have no internet); set `WANDB_MODE=offline` in the Slurm scripts and `wandb sync wandb/offline-run-*` afterward.
- [ ]  Confirm backend: `CUDA=1` (or `NV=1`) and that `Device.DEFAULT` enumerates 8 GPUs on a node.
- [ ]  Pre-tokenize corpus → `train.bin` / `val.bin` (`uint16`), commit the prep script.
- [ ]  Smoke test: TINY trains, loss drops, `DEBUG=2` prints GFLOP/s & GB/s, and the run appears in wandb (or syncs offline).
- [ ]  Lock `model.py`, `step.py`, `data.py`, `bench.py`, and the wandb config/summary schema before any measurement.
- [ ]  Verify `mpi4py` Allreduce across 2 nodes with a 1-element tensor before wiring it to gradients.
- [ ]  Verify a `wandb sweep` launches and the parallel-coordinates view populates before running the full factorial.

### Sources / further reading

tinygrad docs (multi-GPU `shard` incl. weight-axis sharding for tensor parallelism, `TinyJit`, `DEBUG`/`BEAM`/`PROFILE` env vars), the `beautiful_mnist_multigpu.py` and `gpt2.py` examples in the tinygrad repo; the NVIDIA V100 datasheet/architecture whitepaper (peak FLOP/s, HBM2 BW, NVLink); the **Megatron-LM** paper for column/row tensor-parallel splits; the **ZeRO / DeepSpeed** paper and PyTorch **FSDP** docs for optimizer/grad/param sharding stages; the **GPipe** paper for pipeline-bubble analysis; the Chinchilla-style `6ND` FLOP estimate for MFU; and the Weights & Biases docs for sweeps, run groups, and offline sync.