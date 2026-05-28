"""
train.py — nanoGPT-style GPT pre-training with tinygrad.

Quick start:
    DEV=CUDA python train.py --config TINY --data_dir data
    DEV=CUDA python train.py --config BASE --data_dir data --batch 8 --grad_accum 8

DEV=CUDA selects the CUDA backend (requires driver > 12.2; use a compute node).
Omit DEV to fall back to CPU for smoke-testing.
"""

import os
import math
import time
import argparse

from tinygrad import Tensor
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load

from model import GPT, CONFIGS
from data import DataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train GPT with tinygrad.")
    p.add_argument("--config",        choices=["TINY", "BASE", "WIDE"], default="BASE")
    p.add_argument("--data_dir",      default="data")
    p.add_argument("--out_dir",       default="out")
    p.add_argument("--batch",         type=int,   default=8,
                   help="Micro-batch size per gradient step.")
    p.add_argument("--grad_accum",    type=int,   default=4,
                   help="Gradient accumulation steps. Effective batch = batch × grad_accum.")
    p.add_argument("--max_steps",     type=int,   default=10_000)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--min_lr",        type=float, default=3e-5)
    p.add_argument("--weight_decay",  type=float, default=0.1)
    p.add_argument("--warmup_steps",  type=int,   default=200)
    p.add_argument("--eval_interval", type=int,   default=200)
    p.add_argument("--eval_steps",    type=int,   default=50)
    p.add_argument("--save_interval", type=int,   default=1_000)
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to a .safetensors checkpoint to resume from.")
    p.add_argument("--wandb",         action="store_true",
                   help="Log metrics to Weights & Biases.")
    p.add_argument("--run_name",      type=str,   default=None)
    return p.parse_args()


def cosine_lr(step: int, max_steps: int, warmup: int,
              max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


def evaluate(model: GPT, loader: DataLoader, steps: int) -> float:
    Tensor.training = False
    total = 0.0
    for _ in range(steps):
        x, y = loader.get_batch()
        loss = model(x).sparse_categorical_crossentropy(y)
        total += loss.numpy().item()
    Tensor.training = True
    return total / steps


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.init(
            project="train-gpt2",
            name=args.run_name or f"{args.config}-tinygrad",
            config=vars(args),
            dir=args.out_dir,
        )

    cfg   = CONFIGS[args.config]
    model = GPT(cfg)
    print(f"GPT-{args.config}: {model.n_params() / 1e6:.1f}M params")

    train_loader = DataLoader(args.data_dir, args.batch, cfg.ctx, "train")
    val_loader   = DataLoader(args.data_dir, args.batch, cfg.ctx, "val")

    # AdamW stores lr as a Python float; we can update opt.lr directly each step.
    opt = AdamW(get_parameters(model), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        state = safe_load(args.resume)
        load_state_dict(model, state)
        try:
            start_step = int(os.path.splitext(os.path.basename(args.resume))[0].split("_")[-1])
        except (IndexError, ValueError):
            pass
        print(f"Resumed from {args.resume} at step {start_step}")

    Tensor.training = True
    tokens_per_opt_step = args.batch * args.grad_accum * cfg.ctx
    t0 = time.perf_counter()

    for step in range(start_step, args.max_steps):
        opt.lr = cosine_lr(step, args.max_steps, args.warmup_steps, args.lr, args.min_lr)

        # Gradient accumulation over grad_accum micro-batches
        opt.zero_grad()
        step_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = train_loader.get_batch()
            loss = model(x).sparse_categorical_crossentropy(y)
            (loss / args.grad_accum).backward()
            step_loss += loss.numpy().item()
        opt.step()
        train_loss = step_loss / args.grad_accum

        if step % args.eval_interval == 0:
            t1 = time.perf_counter()
            val_loss = evaluate(model, val_loader, args.eval_steps)
            ppl      = math.exp(min(val_loss, 20))
            tok_s    = args.eval_interval * tokens_per_opt_step / max(t1 - t0, 1e-6)
            print(
                f"step {step:6d} | lr={opt.lr:.2e} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"ppl={ppl:.1f} | {tok_s:.0f} tok/s"
            )
            if args.wandb:
                import wandb
                wandb.log(
                    {"train_loss": train_loss, "val_loss": val_loss,
                     "perplexity": ppl, "lr": opt.lr, "tok_per_sec": tok_s},
                    step=step,
                )
            t0 = time.perf_counter()

        if step > 0 and step % args.save_interval == 0:
            ckpt = os.path.join(args.out_dir, f"ckpt_{step:06d}.safetensors")
            safe_save(get_state_dict(model), ckpt)
            print(f"Checkpoint saved → {ckpt}")

    ckpt = os.path.join(args.out_dir, f"ckpt_{args.max_steps:06d}.safetensors")
    safe_save(get_state_dict(model), ckpt)
    print(f"Training done. Final checkpoint: {ckpt}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
