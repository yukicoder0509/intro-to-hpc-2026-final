"""
data.py — nanoGPT-style memmap data loader.

Expects pre-tokenized uint16 .bin files produced by prepare_data.py:
    data/train.bin
    data/val.bin

Usage:
    loader = DataLoader("data", batch=4, ctx=512, split="train")
    x, y = loader.get_batch()
"""

import os
import numpy as np
import torch


class DataLoader:
    def __init__(self, data_dir, batch, ctx, split="train"):
        self.batch = batch
        self.ctx = ctx
        path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                "Run prepare_data.py first to create train.bin / val.bin."
            )
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        print(f"DataLoader: {split}.bin  {len(self.data):,} tokens  "
              f"({len(self.data) / 1e6:.1f}M)")

    def __len__(self):
        return len(self.data)

    def get_batch(self, device=None):
        ix = torch.randint(len(self.data) - self.ctx, (self.batch,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i + self.ctx].astype(np.int64)) for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1:i + 1 + self.ctx].astype(np.int64)) for i in ix
        ])
        if device is not None:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        return x, y
