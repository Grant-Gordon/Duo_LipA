"""Scratch encoder (encoder spec 5.1, 5.7, 6). Every arm is a config switch.

Token sequence: [precursor P ; peaks ; conditioning tokens].
- Place A, token content: `mz_encoding: binned` (LipiDetective: id = floor(m/z * 10), 16000 x d table)
  or `sinusoidal` (absolute sinusoidal m/z, Casanovo-style wavelengths 1e-3 .. 1e4).
  `rank_position_encoding` adds LipiDetective's sinusoidal PE over list index = intensity rank.
- `intensity_encoding`: discard | linear (Linear(1, d) summed into the token) | fourier.
- Place B / C: `mass_relation: {rope, pab}`.
- `precursor_token` (Q9), `conditioning_tokens` (Q18: encoder-side conditioning as extra tokens).
- `pooling`: precursor (default, Q3) | attention | mean.
- memory = precursor row + peak rows; conditioning rows are dropped (Q4a). memory_mz is filled (Q4b).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from duo_lipa.data.batch import PeakBatch
from duo_lipa.data.records import COND_FIELDS
from duo_lipa.model.encoders.attention import EncoderBlock
from duo_lipa.model.encoders.base import EncoderAdapter
from duo_lipa.model.fourier import geometric_wavelengths, sinusoidal
from duo_lipa.model.interfaces import ConditioningForEncoder, EncoderOutput

PRECURSOR_INTENSITY = 1.1  # DreaMS convention (dreams/utils/data.py:150, encoder spec 3.1)


def rank_pe(n: int, d: int) -> Tensor:
    """Fixed sinusoidal PE over list index (as in lipidetective transformer_network.py:353-371)."""
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : d // 2]
    return pe


class ScratchEncoder(EncoderAdapter):
    name = "scratch"
    accepts_conditioning = set(COND_FIELDS)

    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 4, ff_mult: int = 4,
                 dropout: float = 0.0, mz_encoding: str = "binned", rank_position_encoding: bool = True,
                 intensity_encoding: str = "discard", mass_relation: dict | None = None,
                 precursor_token: bool = True, conditioning_tokens: bool = True, pooling: str = "precursor",
                 mz_bins: int = 16000, mz_bin_width: float = 0.1, mz_lambda: tuple = (1e-3, 1e4),
                 intensity_fourier_freq: int = 8, rope: dict | None = None, pab: dict | None = None,
                 max_peaks: int = 512):
        super().__init__()
        if pooling == "precursor" and not precursor_token:
            raise ValueError("pooling: precursor needs precursor_token: true")
        if mz_encoding not in ("binned", "sinusoidal"):
            raise ValueError(f"mz_encoding {mz_encoding!r}")
        if intensity_encoding not in ("discard", "linear", "fourier"):
            raise ValueError(f"intensity_encoding {intensity_encoding!r}")
        mr = {"rope": False, "pab": False, **(mass_relation or {})}
        self.d = self.native_dim = d_model
        self.mz_encoding, self.intensity_encoding, self.pooling = mz_encoding, intensity_encoding, pooling
        self.use_rank_pe, self.use_prec, self.use_cond = rank_position_encoding, precursor_token, conditioning_tokens
        self.mz_bin_width = mz_bin_width
        if not conditioning_tokens:
            self.accepts_conditioning = set()
        if mz_encoding == "binned":
            self.mz_table = nn.Embedding(mz_bins, d_model)
        else:
            self.register_buffer("mz_lam", geometric_wavelengths(d_model // 2, *mz_lambda), persistent=False)
            self.mz_proj = nn.Linear(2 * (d_model // 2), d_model)
        if intensity_encoding == "linear":
            self.int_proj = nn.Linear(1, d_model)
        elif intensity_encoding == "fourier":
            self.register_buffer("int_lam", geometric_wavelengths(intensity_fourier_freq, 0.01, 10.0), persistent=False)
            self.int_proj = nn.Linear(2 * intensity_fourier_freq, d_model)
        if rank_position_encoding:
            self.register_buffer("rank_pe", rank_pe(max_peaks + 1, d_model), persistent=False)
        self.type_emb = nn.Embedding(3, d_model)  # 0 precursor, 1 peak, 2 conditioning
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, ff_mult, dropout, rope=mr["rope"], pab=mr["pab"],
                         rope_cfg=rope, pab_cfg=pab)
            for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        if pooling == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
            self.pool_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def _mz_tokens(self, mz: Tensor) -> Tensor:
        if self.mz_encoding == "binned":
            ids = torch.floor(mz / self.mz_bin_width).long().clamp(0, self.mz_table.num_embeddings - 1)
            return self.mz_table(ids)
        return self.mz_proj(sinusoidal(mz, self.mz_lam))

    def _int_tokens(self, it: Tensor) -> Tensor | None:
        if self.intensity_encoding == "linear":
            return self.int_proj(it.unsqueeze(-1))
        if self.intensity_encoding == "fourier":
            return self.int_proj(sinusoidal(it, self.int_lam))
        return None

    def forward(self, peaks: PeakBatch, cond: ConditioningForEncoder) -> EncoderOutput:
        B, K = peaks.mz.shape
        dev = peaks.intensity.device
        mz, it, pad = peaks.mz, peaks.intensity, peaks.pad
        has_mass = ~pad
        if self.use_prec:
            prec = peaks.precursor_mz
            prec_known = ~torch.isnan(prec)
            mz = torch.cat([torch.where(prec_known, prec, torch.zeros_like(prec)).unsqueeze(1), mz], 1)
            it = torch.cat([torch.full((B, 1), PRECURSOR_INTENSITY, device=dev), it], 1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), pad], 1)
            has_mass = torch.cat([prec_known.unsqueeze(1), has_mass], 1)
            types = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev),
                               torch.ones(B, K, dtype=torch.long, device=dev)], 1)
        else:
            types = torch.ones(B, K, dtype=torch.long, device=dev)
        x = self._mz_tokens(mz) + self.type_emb(types)
        ie = self._int_tokens(it)
        if ie is not None:
            x = x + ie
        if self.use_rank_pe:
            # peaks are sorted by decreasing intensity upstream, so list index = intensity rank;
            # the precursor takes index 0 (IMPLEMENTATION.md D-E3)
            off = 0 if self.use_prec else 1
            x = x + self.rank_pe[off: off + x.shape[1]].unsqueeze(0)
        x = x * (~pad).unsqueeze(-1)  # padded rows carry no content
        M = x.shape[1]                # memory rows = precursor + peaks
        if self.use_cond and cond.tokens is not None:
            C = cond.tokens.shape[1]
            x = torch.cat([x, cond.tokens + self.type_emb.weight[2]], 1)
            pad = torch.cat([pad, torch.zeros(B, C, dtype=torch.bool, device=dev)], 1)
            has_mass = torch.cat([has_mass, torch.zeros(B, C, dtype=torch.bool, device=dev)], 1)
            mz = torch.cat([mz, torch.zeros(B, C, dtype=mz.dtype, device=dev)], 1)
        for blk in self.blocks:
            x = blk(x, pad, mz, has_mass)
        x = self.ln_out(x)
        memory, memory_pad, memory_mz = x[:, :M], pad[:, :M], mz[:, :M]
        if self.pooling == "precursor":
            pooled = memory[:, 0]
        elif self.pooling == "mean":
            w = (~memory_pad).unsqueeze(-1).float()
            pooled = (memory * w).sum(1) / w.sum(1).clamp(min=1)
        else:
            pooled = self.pool_attn(self.pool_query.expand(B, -1, -1), memory, memory,
                                    key_padding_mask=memory_pad, need_weights=False)[0][:, 0]
        return EncoderOutput(memory=memory, memory_pad=memory_pad, pooled=pooled, memory_mz=memory_mz)
