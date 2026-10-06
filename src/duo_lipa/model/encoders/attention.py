"""Self-attention with the two relative-mass mechanisms of encoder spec 5.1:

- RoPE keyed on m/z: queries and keys are rotated by angle m/z * theta_i, so each dot product
  depends on the m/z difference only. Tokens without a mass (conditioning) get angle 0.
- Pairwise attention bias (PAB): delta m/z Fourier-encoded, passed through an MLP, added per head
  to the attention scores. Pairs involving a massless token get bias 0.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from duo_lipa.model.fourier import geometric_wavelengths, sinusoidal


class MzRoPE(nn.Module):
    def __init__(self, head_dim: int, lam_min: float = 1e-2, lam_max: float = 1e4):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head dimension"
        self.register_buffer("lam", geometric_wavelengths(head_dim // 2, lam_min, lam_max), persistent=False)

    def angles(self, mz: Tensor, has_mass: Tensor) -> tuple[Tensor, Tensor]:
        """mz (B, T) float64 -> cos, sin (B, 1, T, head_dim/2) float32."""
        ph = 2 * math.pi * torch.where(has_mass, mz, torch.zeros_like(mz)).to(torch.float64).unsqueeze(-1) / self.lam
        return torch.cos(ph).to(torch.float32).unsqueeze(1), torch.sin(ph).to(torch.float32).unsqueeze(1)

    @staticmethod
    def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)


class PairwiseMzBias(nn.Module):
    def __init__(self, n_heads: int, n_freq: int = 16, hidden: int = 32, lam_min: float = 1e-2, lam_max: float = 1e3):
        super().__init__()
        self.register_buffer("lam", geometric_wavelengths(n_freq, lam_min, lam_max), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, hidden), nn.GELU(), nn.Linear(hidden, n_heads))

    def forward(self, mz: Tensor, has_mass: Tensor) -> Tensor:
        """-> (B, H, T, T) additive bias."""
        delta = mz.unsqueeze(2) - mz.unsqueeze(1)                       # (B, T, T) float64, signed
        pair_ok = has_mass.unsqueeze(2) & has_mass.unsqueeze(1)
        delta = torch.where(pair_ok, delta, torch.zeros_like(delta))
        b = self.mlp(sinusoidal(delta, self.lam))                      # (B, T, T, H)
        b = b * pair_ok.unsqueeze(-1)
        return b.permute(0, 3, 1, 2)


class MassAwareSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.0, rope: bool = False, pab: bool = False,
                 rope_cfg: dict | None = None, pab_cfg: dict | None = None):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.rope = MzRoPE(self.dh, **(rope_cfg or {})) if rope else None
        self.pab = PairwiseMzBias(n_heads, **(pab_cfg or {})) if pab else None

    def forward(self, x: Tensor, key_pad: Tensor, mz: Tensor, has_mass: Tensor, return_attn: bool = False):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)   # (B, H, T, dh)
        if self.rope is not None:
            cos, sin = self.rope.angles(mz, has_mass)
            q, k = MzRoPE.rotate(q, cos, sin), MzRoPE.rotate(k, cos, sin)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if self.pab is not None:
            scores = scores + self.pab(mz, has_mass)
        scores = scores.masked_fill(key_pad[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        y = (self.drop(attn) @ v).transpose(1, 2).reshape(B, T, -1)
        y = self.out(y)
        return (y, attn) if return_attn else y


class EncoderBlock(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, d: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0, **attn_kw):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MassAwareSelfAttention(d, n_heads, dropout, **attn_kw)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_pad, mz, has_mass):
        x = x + self.drop(self.attn(self.ln1(x), key_pad, mz, has_mass))
        return x + self.drop(self.ff(self.ln2(x)))
