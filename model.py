"""
model.py — decoder-only GPT in PyTorch, matching the plan's architecture spec.

Three fixed configs (plan §3):
  TINY: d=384, L=6, H=6, ctx=256  (~11M params)  — fast iteration
  BASE: d=768, L=12, H=12, ctx=512 (~124M params)  — headline throughput
  WIDE: d=2560, L=32, H=20, ctx=512 (~2.5B params)  — tensor-parallelism stress
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab: int = 50257
    ctx: int = 512
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12


CONFIGS = {
    "TINY": GPTConfig(ctx=256, dim=384, n_layers=6,  n_heads=6),
    "BASE": GPTConfig(ctx=512, dim=768, n_layers=12, n_heads=12),
    "WIDE": GPTConfig(ctx=512, dim=2560, n_layers=32, n_heads=20),
}


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.hd)
        att = (q @ k.transpose(-2, -1)) * scale
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal_mask, float('-inf'))
        att = F.softmax(att.float(), dim=-1).to(x.dtype)
        o = att @ v
        return self.proj(o.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.dim)
        self.pos = nn.Embedding(cfg.ctx, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg.dim, cfg.n_heads) for _ in range(cfg.n_layers)])
        self.lnf = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.lnf(x))

    def n_params(self):
        # Exclude position embedding from param count (matches plan's ~6ND formula)
        return sum(p.numel() for p in self.parameters()) - self.pos.weight.numel()


if __name__ == "__main__":
    for name, cfg in CONFIGS.items():
        m = GPT(cfg)
        n = m.n_params()
        print(f"{name}: {n/1e6:.1f}M params  "
              f"(d={cfg.dim}, L={cfg.n_layers}, H={cfg.n_heads}, ctx={cfg.ctx})")
