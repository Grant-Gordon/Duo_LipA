"""MSBERT adapter (frozen). Zhang et al.; github.com/zhanghailiangcsu/MSBERT (MIT), weights from the
GitHub release 1.0 (`MSBERT.pkl`, a plain state dict).

Torch-only re-implementation of model/MSBERTModel.py (MSBERT.predict path) with matching parameter
names; the upstream module sets a global CUDA device at import (IMPLEMENTATION.md D-F1).
- tokens: index of "%.2f" % m/z in a 100000-word vocabulary 0.00..999.99, plus [PAD]=0, [MASK]=1
  (data/ProcessData.py:35-64); precursor token first;
- intensities: precursor gets raw intensity 2, then everything is divided by the max
  (ProcessData.py), peaks are max-normalized first;
- pre-norm blocks, no final norm, no positional embedding; key-padding mask;
- pooled = intensity @ H / N (N = max length 100).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from duo_lipa.model.encoders.base import FrozenFMAdapter

MAXLEN = 100


class _MHA(nn.Module):
    def __init__(self, h, d):
        super().__init__()
        self.h, self.dk = h, d // h
        self.linear_layers = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
        self.output_linear = nn.Linear(d, d)

    def forward(self, x, mask):
        B = x.size(0)
        q, k, v = [l(x).view(B, -1, self.h, self.dk).transpose(1, 2) for l in self.linear_layers]
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        s = s.masked_fill(mask, -1e9)
        y = (F.softmax(s, dim=-1) @ v).transpose(1, 2).contiguous().view(B, -1, self.h * self.dk)
        return self.output_linear(y)


class _Sub(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)


class _Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attention = _MHA(h, d)
        self.feed_forward = nn.Module()
        self.feed_forward.w_1 = nn.Linear(d, 4 * d)
        self.feed_forward.w_2 = nn.Linear(4 * d, d)
        self.input_sublayer = _Sub(d)
        self.output_sublayer = _Sub(d)

    def forward(self, x, mask):
        x = x + self.attention(self.input_sublayer.norm(x), mask)
        ff = self.feed_forward
        return x + ff.w_2(F.gelu(ff.w_1(self.output_sublayer.norm(x))))


class MSBERTBackbone(nn.Module):
    def __init__(self, vocab=100002, d=512, L=6, h=16):
        super().__init__()
        self.embedding = nn.Module()
        self.embedding.token = nn.Embedding(vocab, d, padding_idx=0)
        self.transformer_blocks = nn.ModuleList([_Block(d, h) for _ in range(L)])
        self.intensity_linear = nn.Linear(1, d)

    def forward(self, ids, intensity):
        """ids (B, 100) long; intensity (B, 1, 100). Returns per-block outputs and pooled."""
        x = self.embedding.token(ids) + self.intensity_linear(intensity.transpose(1, 2))
        mask = (ids == 0).unsqueeze(1).repeat(1, ids.size(1), 1).unsqueeze(1)
        outs = []
        for blk in self.transformer_blocks:
            x = blk(x, mask)
            outs.append(x)
        pooled = (intensity @ x).squeeze(1) / intensity.shape[-1]
        return outs, pooled


class MSBERTAdapter(FrozenFMAdapter):
    name = "msbert"
    has_per_peak = True
    accepts_conditioning = {"precursor_mz"}   # native: precursor token (IMPLEMENTATION.md D-F4)
    frozen = True

    def __init__(self, d_model: int, layer_mix: bool = True, head_mlp_layers: int = 1, weights_dir=None):
        super().__init__(d_model, layer_mix, head_mlp_layers)
        wd = Path(weights_dir or Path(__file__).resolve().parents[4] / "weights" / "msbert")
        sd = torch.load(wd / "MSBERT.pkl", map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith(("fc2", "linear."))}
        vocab, d = sd["embedding.token.weight"].shape
        L = sum(1 for k in sd if k.endswith("attention.output_linear.weight"))
        self.backbone = MSBERTBackbone(vocab, d, L, 16)
        self.backbone.load_state_dict(sd, strict=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        words = ["%.2f" % w for w in np.round(np.linspace(0, 1000, 100 * 1000, endpoint=False), 2)]
        self.word2idx = {w: i + 2 for i, w in enumerate(words)}
        self.native_dim, self.n_layers_exposed = d, L
        self._build_heads()

    def tokenize(self, peaks, cond):
        B = peaks.B
        ids = torch.zeros(B, MAXLEN, dtype=torch.long)
        inten = torch.zeros(B, 1, MAXLEN)
        mz_out = torch.zeros(B, MAXLEN, dtype=torch.float64)
        prec = cond.raw.get("precursor_mz", [None] * B)
        for b in range(B):
            keep = (~peaks.pad[b]) & (peaks.mz[b] >= 10) & (peaks.mz[b] < 1000)
            mz, it = peaks.mz[b][keep], peaks.intensity[b][keep]
            top = torch.argsort(it, descending=True)[: MAXLEN - 1]
            mz, it = mz[top], it[top]
            o = torch.argsort(mz)
            mz, it = mz[o], it[o]
            it = it / it.max() if it.numel() else it
            p = float(prec[b]) if prec[b] is not None and float(prec[b]) < 1000 else None
            toks = [self.word2idx["%.2f" % p]] if p is not None else [1]  # unknown precursor -> [MASK]
            toks += [self.word2idx["%.2f" % float(m)] for m in mz]
            vals = torch.cat([torch.tensor([2.0]), it.float()])
            vals = vals / vals.max()
            n = len(toks)
            ids[b, :n] = torch.tensor(toks)
            inten[b, 0, :n] = vals
            mz_out[b, 0] = p or 0.0
            mz_out[b, 1:n] = mz
        return ids, inten, mz_out

    def backbone_layers(self, peaks, cond):
        ids, inten, mz = self.tokenize(peaks, cond)
        outs, pooled = self.backbone(ids, inten)
        return outs, ids == 0, mz, pooled
