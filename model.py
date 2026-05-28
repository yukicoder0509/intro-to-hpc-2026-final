"""
model.py — decoder-only GPT in tinygrad, matching the plan's architecture spec.

Three fixed configs (plan §3):
  TINY: d=384, L=6, H=6, ctx=256  (~11M params)  — fast iteration
  BASE: d=768, L=12, H=12, ctx=512 (~124M params)  — headline throughput
  WIDE: d=2560, L=32, H=20, ctx=512 (~2.5B params)  — tensor-parallelism stress
"""

from dataclasses import dataclass
import math
from tinygrad import Tensor, nn
from tinygrad.nn.state import get_parameters


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


class CausalSelfAttention:
    def __init__(self, dim: int, n_heads: int):
        self.n_heads = n_heads
        self.hd   = dim // n_heads
        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv[..., :C], qkv[..., C:2*C], qkv[..., 2*C:]
        q = q.reshape(B, T, self.n_heads, self.hd).transpose(1, 2)  # (B, H, T, hd)
        k = k.reshape(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.hd).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.hd)
        att = q.matmul(k.transpose(-2, -1)) * scale              # (B, H, T, T)
        # Upper-triangle positions (future tokens) get a large negative bias.
        # Using -1e9 instead of -inf avoids NaN in float16 softmax.
        causal_mask = Tensor.ones(T, T).tril()
        att = att + (1 - causal_mask) * (-1e9)
        att = att.softmax(-1)
        o = att.matmul(v).transpose(1, 2).reshape(B, T, C)
        return self.proj(o)


class Block:
    def __init__(self, dim: int, n_heads: int):
        self.ln1  = nn.LayerNorm(dim)
        self.ln2  = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.fc1  = nn.Linear(dim, 4 * dim)
        self.fc2  = nn.Linear(4 * dim, dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(self.fc1(self.ln2(x)).gelu())
        return x


class GPT:
    def __init__(self, cfg: GPTConfig):
        self.cfg    = cfg
        self.tok    = nn.Embedding(cfg.vocab, cfg.dim)
        self.pos    = nn.Embedding(cfg.ctx, cfg.dim)
        self.blocks = [Block(cfg.dim, cfg.n_heads) for _ in range(cfg.n_layers)]
        self.lnf    = nn.LayerNorm(cfg.dim)
        self.head   = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying

    def __call__(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        pos = Tensor.arange(T)
        x = self.tok(idx) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.lnf(x))

    def n_params(self) -> int:
        # Exclude position embedding from count (matches plan's ~6ND formula)
        return sum(math.prod(p.shape) for p in get_parameters(self)) - math.prod(self.pos.weight.shape)


if __name__ == "__main__":
    for name, cfg in CONFIGS.items():
        m = GPT(cfg)
        n = m.n_params()
        print(f"{name}: {n/1e6:.1f}M params  "
              f"(d={cfg.dim}, L={cfg.n_layers}, H={cfg.n_heads}, ctx={cfg.ctx})")
