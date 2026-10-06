"""Spec2Vec adapter (frozen). Huber et al. 2021; github.com/iomega/spec2vec (Apache-2.0), model
"AllPositive ratio05 filtered 201101 iter 15", Zenodo 4173596.

No gensim at runtime: the vocabulary (`wv.index2word`) is read from the gensim 3.x pickle with the
restricted unpickler, the vectors with np.load (IMPLEMENTATION.md D-F1). Words are "peak@%.2f".
- per-peak memory: the word vector of every peak found in the vocabulary (others are padding);
- pooled: sum_k w_k^p v_k with w = intensity / max and p = `intensity_power` (0.5, the commonly used
  setting; vector_operations.py:61-77). Upstream's "zero vector if >10% of the weight is missing"
  rule is not applied.
Gradients reach only the projection heads: the lookup is discrete and frozen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from duo_lipa.model.encoders.base import FrozenFMAdapter
from duo_lipa.model.encoders.safe_load import RestrictedUnpickler

MODEL = "spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model"


class Spec2VecAdapter(FrozenFMAdapter):
    name = "spec2vec"
    has_per_peak = True
    accepts_conditioning: set[str] = set()
    frozen = True
    n_layers_exposed = 1

    def __init__(self, d_model: int, layer_mix: bool = True, head_mlp_layers: int = 1, weights_dir=None,
                 intensity_power: float = 0.5):
        super().__init__(d_model, layer_mix, head_mlp_layers)
        wd = Path(weights_dir or Path(__file__).resolve().parents[4] / "weights" / "spec2vec")
        with open(wd / MODEL, "rb") as f:
            m = RestrictedUnpickler(f).load()
        words = m.wv.index2word
        vecs = np.load(wd / (MODEL + ".wv.vectors.npy"))
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.backbone = nn.Embedding.from_pretrained(torch.from_numpy(vecs), freeze=True)
        self.power = intensity_power
        self.native_dim = vecs.shape[1]
        self._build_heads()

    def backbone_layers(self, peaks, cond):
        B, K = peaks.mz.shape
        idx = torch.zeros(B, K, dtype=torch.long)
        pad = torch.ones(B, K, dtype=torch.bool)
        for b in range(B):
            for j in range(K):
                if peaks.pad[b, j]:
                    continue
                i = self.word2idx.get("peak@%.2f" % float(peaks.mz[b, j]))
                if i is not None:
                    idx[b, j], pad[b, j] = i, False
        v = self.backbone(idx) * (~pad).unsqueeze(-1)
        it = peaks.intensity.clone()
        it = it / it.masked_fill(peaks.pad, 0).amax(1, keepdim=True).clamp(min=1e-12)
        w = (it.clamp(min=0) ** self.power) * (~pad)
        pooled = (w.unsqueeze(-1) * v).sum(1)
        return [v], pad, peaks.mz, pooled
