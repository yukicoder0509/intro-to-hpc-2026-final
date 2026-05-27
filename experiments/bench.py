"""
bench.py — GPT benchmark harness (plan §4.1).

Measures step time (ms), tokens/s, and MFU for a fixed number of steps,
discarding warmup, then writes median + IQR to wandb (or a JSON file).

B0 baseline (plan §4.0):
    python experiments/bench.py --config BASE --batch 4 --steps 33 --warmup 3

Environment variables honoured:
    JIT=1       enable torch.jit.script (E1 equivalent)
    HALF=1      FP16 autocast via torch.cuda.amp (E3)
    WANDB_MODE  online | offline | disabled
"""

import sys
import pathlib

# Ensure the project root (parent of experiments/) is on the Python path so
# that shared modules (model, data) are importable from any working directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F
import numpy as np

from model import GPT, CONFIGS
from data import DataLoader

# ── optional wandb ─────────────────────────────────────────────────────────
try:
    import wandb as _wandb
    _wandb.init  # noqa: B018  — verify import works
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

# V100-SXM2 peak FLOP/s (plan §1)
PEAK_FP32 = 15.7e12
PEAK_FP16 = 125.0e12


# ── helpers ────────────────────────────────────────────────────────────────

def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def iqr(xs):
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    return xs_sorted[3 * n // 4] - xs_sorted[n // 4]


def compute_mfu(ms_median, n_params, batch, ctx, peak):
    tokens = batch * ctx
    flops = 6 * n_params * tokens        # plan's 6ND estimate
    achieved = flops / (ms_median / 1e3)
    return achieved / peak


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GPT benchmark harness")
    p.add_argument("--config",   default="BASE",  choices=list(CONFIGS.keys()))
    p.add_argument("--data_dir", default="data")
    p.add_argument("--batch",    type=int, default=4)
    p.add_argument("--ctx",      type=int, default=None,
                   help="Override sequence length from config")
    p.add_argument("--steps",    type=int, default=33,
                   help="Total steps (warmup + measured); plan uses 33 = 3 warmup + 30")
    p.add_argument("--warmup",   type=int, default=3)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--exp",      type=str, default="B0",
                   help="Experiment tag — used as wandb group name")
    p.add_argument("--rep",      type=int, default=0,
                   help="Repetition index (0–4 for the ≥5 repeats in §6)")
    p.add_argument("--wandb_mode", type=str, default=None,
                   help="wandb mode override (online/offline/disabled)")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = CONFIGS[args.config]
    if args.ctx is not None:
        cfg.ctx = args.ctx

    # JIT / FP16 from env (plan §4.1: alter exactly one variable at a time)
    use_jit  = int(os.getenv("JIT",  "0")) == 1
    use_fp16 = int(os.getenv("HALF", "0")) == 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── model + optimizer ──────────────────────────────────────────────────
    model = GPT(cfg).to(device)
    n_params = model.n_params()

    if use_jit:
        model = torch.jit.script(model)

    # Optimizer always in FP32 (FP32 master weights for stability)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1
    )

    scaler = None
    if use_fp16 and device.type == "cuda":
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    # ── data loader ────────────────────────────────────────────────────────
    loader = DataLoader(args.data_dir, batch=args.batch, ctx=cfg.ctx)

    tokens_per_step = args.batch * cfg.ctx

    # ── wandb / logging ────────────────────────────────────────────────────
    run_cfg = {
        "config":    args.config,
        "batch":     args.batch,
        "ctx":       cfg.ctx,
        "fp16":      int(use_fp16),
        "jit":       int(use_jit),
        "beam":      int(os.getenv("BEAM", "0")),
        "steps":     args.steps - args.warmup,
        "warmup":    args.warmup,
        "lr":        args.lr,
        "n_params":  n_params,
        "exp":       args.exp,
        "rep":       args.rep,
        "commit":    git_sha(),
        "jobid":     os.getenv("SLURM_JOB_ID", "local"),
        "device":    str(device),
    }

    wandb_mode = args.wandb_mode or os.getenv("WANDB_MODE", "online")
    run = None
    if HAS_WANDB and wandb_mode != "disabled":
        run = _wandb.init(
            project="hpc-gpt-tinygrad",
            group=args.exp,
            name="{}-r{}".format(args.exp, args.rep),
            mode=wandb_mode,
            config=run_cfg,
        )

    print("=" * 60)
    print("GPT benchmark — {}".format(args.exp))
    print("  config={} | params={:.1f}M | batch={} ctx={}".format(
        args.config, n_params / 1e6, args.batch, cfg.ctx))
    print("  dtype={} | jit={} | device={}".format(
        "fp16" if use_fp16 else "fp32", use_jit, device))
    print("  tokens/step={:,} | warmup={} | measure={} steps".format(
        tokens_per_step, args.warmup, args.steps - args.warmup))
    print("=" * 60)

    # ── benchmark loop ─────────────────────────────────────────────────────
    dts = []

    for i in range(args.steps):
        x, y = loader.get_batch(device=device)

        # GPU sync before clock (plan §6 rule 3)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        if use_fp16 and scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

        # GPU sync after step (force completion before reading clock)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3

        loss_val = loss.item()

        if i < args.warmup:
            print("  warmup {:2d} | {:7.1f} ms | loss={:.4f}".format(
                i, dt, loss_val))
        else:
            dts.append(dt)
            tps = tokens_per_step / (dt / 1e3)
            step_num = i - args.warmup
            step_log = {
                "ms_per_step": dt,
                "tokens_per_s": tps,
                "loss": loss_val,
                "step": step_num,
            }
            if run is not None:
                _wandb.log(step_log)
            else:
                print("  step {:3d} | {:7.1f} ms | {:8.0f} tok/s | loss={:.4f}".format(
                    step_num, dt, tps, loss_val))

    # ── summary ────────────────────────────────────────────────────────────
    med_ms  = statistics.median(dts)
    med_tps = tokens_per_step / (med_ms / 1e3)
    mfu_fp32 = compute_mfu(med_ms, n_params, args.batch, cfg.ctx, PEAK_FP32)
    mfu_fp16 = compute_mfu(med_ms, n_params, args.batch, cfg.ctx, PEAK_FP16)

    summary = {
        "ms_median":   med_ms,
        "ms_iqr":      iqr(dts),
        "tokens_per_s": med_tps,
        "mfu_fp32":    mfu_fp32,
        "mfu_fp16":    mfu_fp16,
        "n_params":    n_params,
    }

    print("")
    print("=== Summary: {}-r{} ===".format(args.exp, args.rep))
    print("  median = {:.1f} ms/step   IQR = {:.1f} ms".format(
        med_ms, summary["ms_iqr"]))
    print("  tokens/s = {:,.0f}".format(med_tps))
    print("  MFU (FP32 @15.7T): {:.1f}%".format(mfu_fp32 * 100))
    print("  MFU (FP16 @125T):  {:.1f}%".format(mfu_fp16 * 100))

    if run is not None:
        run.summary.update(summary)
        _wandb.finish()
    else:
        out = "results_{}_r{}.json".format(args.exp, args.rep)
        with open(out, "w") as f:
            json.dump(dict(list(run_cfg.items()) + list(summary.items())), f, indent=2)
        print("  Results saved to {}".format(out))


if __name__ == "__main__":
    main()
