"""
Train GPT-2 on the wikimedia/wikipedia dataset using Hugging Face Transformers.

Usage:
    python train_gpt2_wikipedia.py [--options]

Requirements:
    pip install transformers datasets accelerate torch
"""

import argparse
import logging
import math
import os
from itertools import chain

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset as TorchIterableDataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune / pretrain GPT-2 on Wikipedia.")

    # Dataset
    parser.add_argument("--dataset_name",    type=str, default="wikimedia/wikipedia")
    parser.add_argument("--dataset_config",  type=str, default="20231101.en",
                        help="Wikipedia language/date config, e.g. '20231101.en', '20231101.zh'")
    parser.add_argument("--text_column",     type=str, default="text")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap training samples (useful for quick experiments).")
    parser.add_argument("--validation_split_percentage", type=int, default=5,
                        help="Percent of training data to use as validation if no val split.")

    # Model
    parser.add_argument("--model_name_or_path", type=str, default="gpt2",
                        help="'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl', "
                             "or a local path. Pass 'scratch' to train from scratch.")
    parser.add_argument("--tokenizer_name",     type=str, default=None,
                        help="Defaults to model_name_or_path.")

    # Tokenisation
    parser.add_argument("--block_size", type=int, default=512,
                        help="Input sequence length after tokenisation.")

    # Training
    parser.add_argument("--output_dir",           type=str,   default="./gpt2-wikipedia")
    parser.add_argument("--num_train_epochs",     type=int,   default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size",  type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate",        type=float, default=5e-5)
    parser.add_argument("--weight_decay",         type=float, default=0.01)
    parser.add_argument("--warmup_ratio",         type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type",    type=str,   default="cosine")
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--checkpointing_steps",  type=str,   default="epoch",
                        help="'epoch' or an integer step count.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--streaming", action="store_true",
                        help="Stream the dataset to avoid downloading it to disk.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total optimiser steps. Required when --streaming and "
                             "--max_train_samples is not set.")

    # Misc
    parser.add_argument("--with_tracking",   action="store_true",
                        help="Enable Accelerate experiment tracking.")
    parser.add_argument("--report_to",       type=str, default="tensorboard")

    return parser.parse_args()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
class HFIterableWrapper(TorchIterableDataset):
    """Make a HuggingFace IterableDataset visible to PyTorch's DataLoader."""
    def __init__(self, hf_dataset):
        self._ds = hf_dataset

    def __iter__(self):
        yield from self._ds


def group_texts(examples, block_size: int):
    """Concatenate all texts and chunk into block_size tokens."""
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    # Drop the tail so every chunk is exactly block_size
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Accelerator ──────────────────────────
    accelerator_kwargs = {}
    if args.with_tracking:
        accelerator_kwargs["log_with"] = args.report_to
        accelerator_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(**accelerator_kwargs)
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # ── Tokenizer ────────────────────────────
    tok_name = args.tokenizer_name or args.model_name_or_path
    if tok_name == "scratch":
        tok_name = "gpt2"            # reuse GPT-2 BPE vocab when training from scratch
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token by default
    tokenizer.model_max_length = int(1e30)      # chunking is handled by group_texts; suppress length warning

    # ── Model ────────────────────────────────
    if args.model_name_or_path == "scratch":
        logger.info("Training GPT-2 from scratch.")
        config = GPT2Config()
        model = GPT2LMHeadModel(config)
    else:
        logger.info(f"Loading pretrained weights from '{args.model_name_or_path}'.")
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))

    # ── Dataset ──────────────────────────────
    logger.info(f"Loading dataset '{args.dataset_name}' config='{args.dataset_config}' …")
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config,
        streaming=args.streaming,
    )

    col = args.text_column

    def tokenize(examples):
        return tokenizer(examples[col])

    if args.streaming:
        base = raw_datasets["train"]
        # Derive a fixed validation size; can't do a percentage split on an IterableDataset
        if args.max_train_samples:
            val_samples = max(500, args.max_train_samples * args.validation_split_percentage // 100)
        else:
            val_samples = 5_000
        eval_ds  = base.take(val_samples)
        train_ds = base.skip(val_samples)
        if args.max_train_samples:
            train_ds = train_ds.take(args.max_train_samples)
        train_ds = train_ds.shuffle(buffer_size=10_000, seed=args.seed)

        orig_cols = list(next(iter(base)).keys())
        train_dataset = train_ds.map(tokenize, batched=True, remove_columns=orig_cols)
        train_dataset = train_dataset.map(lambda ex: group_texts(ex, args.block_size), batched=True)
        eval_dataset  = eval_ds.map(tokenize, batched=True, remove_columns=orig_cols)
        eval_dataset  = eval_dataset.map(lambda ex: group_texts(ex, args.block_size), batched=True)
        logger.info(f"Streaming mode — val reserved: {val_samples:,} examples")
    else:
        # Wikipedia only has a "train" split; carve out a validation set.
        if "validation" not in raw_datasets:
            split = raw_datasets["train"].train_test_split(
                test_size=args.validation_split_percentage / 100,
                seed=args.seed,
            )
            raw_datasets["train"]      = split["train"]
            raw_datasets["validation"] = split["test"]

        if args.max_train_samples:
            raw_datasets["train"] = raw_datasets["train"].select(
                range(min(args.max_train_samples, len(raw_datasets["train"])))
            )

        with accelerator.main_process_first():
            tokenized = raw_datasets.map(
                tokenize,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Tokenising",
            )
            lm_datasets = tokenized.map(
                lambda ex: group_texts(ex, args.block_size),
                batched=True,
                desc="Grouping into blocks",
            )

        train_dataset = lm_datasets["train"]
        eval_dataset  = lm_datasets["validation"]
        logger.info(f"Training samples : {len(train_dataset):,}")
        logger.info(f"Validation samples: {len(eval_dataset):,}")

    # ── DataLoaders ──────────────────────────
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if args.streaming:
        # Wrap so PyTorch's DataLoader recognises the HF IterableDataset correctly.
        train_dataset = HFIterableWrapper(train_dataset)
        eval_dataset  = HFIterableWrapper(eval_dataset)

    train_loader = DataLoader(
        train_dataset,
        shuffle=not args.streaming,  # IterableDataset is pre-shuffled via .shuffle()
        collate_fn=collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # ── Optimizer & Scheduler ────────────────
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=args.learning_rate)

    # Total training steps (after gradient accumulation)
    if args.streaming:
        if args.max_train_steps is None:
            if args.max_train_samples is None:
                raise ValueError("When using --streaming, set --max_train_steps or --max_train_samples.")
            steps_per_epoch = math.ceil(
                args.max_train_samples /
                (args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)
            )
        else:
            steps_per_epoch = math.ceil(args.max_train_steps / args.num_train_epochs)
        max_train_steps = steps_per_epoch * args.num_train_epochs
    else:
        steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        max_train_steps = args.max_train_steps or steps_per_epoch * args.num_train_epochs
    warmup_steps = int(max_train_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # ── Accelerate ───────────────────────────
    # IterableDataset doesn't support the sampler accelerate injects into DataLoader,
    # so keep the DataLoaders out of prepare() when streaming and move batches manually.
    if args.streaming:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    else:
        model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, eval_loader, scheduler
        )

    if args.with_tracking:
        accelerator.init_trackers("gpt2_wikipedia", config=vars(args))

    # ── Checkpoint resumption ────────────────
    starting_epoch   = 0
    resume_step      = 0
    checkpointing_steps = args.checkpointing_steps
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)
        if "epoch_" in path:
            starting_epoch = int(path.split("epoch_")[1]) + 1
        elif "step_" in path:
            resume_step = int(path.split("step_")[1])

    # ── Training loop ────────────────────────
    logger.info("***** Starting training *****")
    logger.info(f"  Epochs              : {args.num_train_epochs}")
    logger.info(f"  Batch/device (train): {args.per_device_train_batch_size}")
    logger.info(f"  Grad accum steps    : {args.gradient_accumulation_steps}")
    logger.info(f"  Total opt. steps    : {max_train_steps:,}")
    logger.info(f"  Warmup steps        : {warmup_steps:,}")

    global_step = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        active_dataloader = train_loader

        for step, batch in enumerate(active_dataloader):
            # Skip already-completed steps when resuming mid-epoch
            if args.resume_from_checkpoint and epoch == starting_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    scheduler.step()
                continue

            if args.streaming:
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}

            num_batches += 1
            outputs = model(**batch)
            loss    = outputs.loss / args.gradient_accumulation_steps
            total_loss += loss.detach().float() * args.gradient_accumulation_steps
            accelerator.backward(loss)

            sync_step = (step + 1) % args.gradient_accumulation_steps == 0
            if sync_step:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if isinstance(checkpointing_steps, str) and checkpointing_steps.isdigit():
                    if global_step % int(checkpointing_steps) == 0:
                        ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                        accelerator.save_state(ckpt_dir)
                        logger.info(f"Saved checkpoint → {ckpt_dir}")

        # ── Evaluation ───────────────────────
        model.eval()
        eval_losses = []
        for batch in eval_loader:
            if args.streaming:
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

        eval_losses = torch.cat(eval_losses)
        mean_loss   = eval_losses.mean().item()
        perplexity  = math.exp(mean_loss) if mean_loss < 20 else float("inf")

        logger.info(
            f"Epoch {epoch + 1}/{args.num_train_epochs} | "
            f"train_loss={total_loss.item() / num_batches:.4f} | "
            f"eval_loss={mean_loss:.4f} | perplexity={perplexity:.2f}"
        )

        if args.with_tracking:
            accelerator.log(
                {"eval_loss": mean_loss, "perplexity": perplexity, "epoch": epoch},
                step=global_step,
            )

        if checkpointing_steps == "epoch":
            ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            accelerator.save_state(ckpt_dir)

    # ── Save final model ─────────────────────
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Model saved to '{args.output_dir}'")

    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()