"""
prepare_data.py — tokenize a text corpus once to uint16 .bin files.

Uses the GPT-2 BPE tokenizer from HuggingFace (no tiktoken dependency).
Outputs data/train.bin and data/val.bin for use by DataLoader.

Supported corpora:
  --dataset roneneldan/TinyStories   (default, ~100 MB tokenized)
  --dataset openwebtext               (larger, needs more disk space)

Usage:
    python prepare_data.py --output_dir data --max_train_samples 500000
"""

import argparse
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="data")
    p.add_argument("--dataset", default="roneneldan/TinyStories",
                   help="HuggingFace dataset name")
    p.add_argument("--dataset_config", default=None,
                   help="Dataset config (e.g. '20231101.en' for Wikipedia)")
    p.add_argument("--text_column", default="text")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Cap training samples to tokenize")
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def tokenize_and_save(examples, tokenizer, col, output_path):
    """Tokenize a list of examples and write them to a uint16 .bin file."""
    all_tokens = []
    eos = tokenizer.eos_token_id
    for ex in examples:
        toks = tokenizer.encode(ex[col])
        toks.append(eos)
        all_tokens.extend(toks)
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(output_path)
    return len(arr)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Tokenizer : {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Dataset   : {args.dataset}" +
          (f" / {args.dataset_config}" if args.dataset_config else ""))
    load_kwargs = {}
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config

    ds = load_dataset(args.dataset, split="train", **load_kwargs)

    if args.max_train_samples:
        n = min(args.max_train_samples, len(ds))
        ds = ds.select(range(n))
        print(f"Capped to {n:,} examples")

    split = ds.train_test_split(test_size=args.val_fraction, seed=args.seed)
    train_data = list(split["train"])
    val_data = list(split["test"])

    print(f"Tokenizing {len(train_data):,} train / {len(val_data):,} val examples …")

    col = args.text_column
    train_path = os.path.join(args.output_dir, "train.bin")
    val_path = os.path.join(args.output_dir, "val.bin")

    n_train = tokenize_and_save(train_data, tokenizer, col, train_path)
    n_val = tokenize_and_save(val_data, tokenizer, col, val_path)

    print(f"train.bin : {n_train:,} tokens  ({n_train * 2 / 1e6:.1f} MB)")
    print(f"val.bin   : {n_val:,} tokens  ({n_val * 2 / 1e6:.1f} MB)")
    print(f"Done. Files written to {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
