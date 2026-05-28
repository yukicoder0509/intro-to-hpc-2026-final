"""
data.py — nanoGPT-style memmap data loader (tinygrad).

Expects pre-tokenized uint16 .bin files produced by prepare_data.py:
    data/train.bin
    data/val.bin

Usage:
    loader = DataLoader("data", batch=4, ctx=512, split="train")
    x, y = loader.get_batch()
"""

import os
import numpy as np
from tinygrad import Tensor


class DataLoader:
    def __init__(self, data_dir: str, batch: int, ctx: int, split: str = "train"):
        self.batch = batch
        self.ctx   = ctx
        path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                "Run prepare_data.py first to create train.bin / val.bin."
            )
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        print(f"DataLoader: {split}.bin  {len(self.data):,} tokens  "
              f"({len(self.data) / 1e6:.1f}M)")

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(self) -> tuple[Tensor, Tensor]:
        ix = np.random.randint(0, len(self.data) - self.ctx, size=(self.batch,))
        x  = np.stack([self.data[i     : i     + self.ctx].astype(np.int64) for i in ix])
        y  = np.stack([self.data[i + 1 : i + 1 + self.ctx].astype(np.int64) for i in ix])
        return Tensor(x), Tensor(y)
