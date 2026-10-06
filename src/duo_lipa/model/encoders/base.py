"""Encoder adapter contract (encoder spec 6) and the pieces shared by FM adapters: the learned layer
mix (Q5) and the projection to the shared d_model (Q2)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from duo_lipa.data.batch import PeakBatch
from duo_lipa.model.interfaces import ConditioningForEncoder, EncoderOutput


class EncoderAdapter(nn.Module):
    name: str = ""
    native_dim: int = 0
    n_layers_exposed: int = 1       # > 1 enables the learned layer mix
    has_per_peak: bool = True
    accepts_conditioning: set[str] = set()
    frozen: bool = False

    def forward(self, peaks: PeakBatch, cond: ConditioningForEncoder) -> EncoderOutput:
        raise NotImplementedError


class ScalarMix(nn.Module):
    """h = gamma * sum_l softmax(w)_l * H_l (Peters et al. 2018). L + 1 parameters."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(()))

    def forward(self, layers: list[Tensor]) -> Tensor:
        a = torch.softmax(self.w, 0)
        return self.gamma * sum(ai * h for ai, h in zip(a, layers))


def projection_head(native_dim: int, d_model: int, n_layers: int = 1) -> nn.Module:
    """Q2 projection; `head_mlp_layers: 2` widens it to a small MLP (Q5)."""
    if n_layers <= 1:
        return nn.Linear(native_dim, d_model)
    return nn.Sequential(nn.Linear(native_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))


class FrozenFMAdapter(EncoderAdapter):
    """Common wrapper: frozen backbone -> per-layer hidden states -> layer mix -> projection.

    Subclasses implement `backbone_layers(peaks, cond) -> (layers, pad, mz, pooled_native)` where
    `layers` is a list of (B, M, native_dim) per-peak hidden states (or None for pooled-only FMs) and
    `pooled_native` is the FM's own spectrum vector (B, native_dim).
    """

    def __init__(self, d_model: int, layer_mix: bool = True, head_mlp_layers: int = 1):
        super().__init__()
        self.d_model = d_model
        self.use_layer_mix = layer_mix
        self.head_mlp_layers = head_mlp_layers

    def _build_heads(self):
        self.mix = ScalarMix(self.n_layers_exposed) if (self.use_layer_mix and self.n_layers_exposed > 1) else None
        self.proj_peaks = projection_head(self.native_dim, self.d_model, self.head_mlp_layers) if self.has_per_peak else None
        self.proj_pooled = projection_head(self.native_dim, self.d_model, self.head_mlp_layers)

    def backbone_layers(self, peaks: PeakBatch, cond: ConditioningForEncoder):
        raise NotImplementedError

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen and hasattr(self, "backbone"):
            self.backbone.eval()  # frozen FM: no dropout in the backbone
        return self

    cache_dir: str | None = None  # Q26: set from `encoder.fm_cache_dir`; off by default

    def _cache_key(self, peaks: PeakBatch, cond: ConditioningForEncoder) -> str:
        import hashlib

        h = hashlib.sha1(self.name.encode())
        for t in (peaks.mz, peaks.intensity, peaks.pad, peaks.precursor_mz):
            h.update(t.numpy().tobytes())
        h.update(repr(sorted((k, v) for k, v in cond.raw.items())).encode())
        return h.hexdigest()

    def _cached_backbone(self, peaks, cond):
        """Caches frozen backbone outputs per batch content. Incompatible with fresh per-epoch
        augmentation (encoder spec 5.5): a perturbed spectrum is simply a cache miss."""
        if not (self.cache_dir and self.frozen):
            return self.backbone_layers(peaks, cond)
        from pathlib import Path

        p = Path(self.cache_dir) / f"{self._cache_key(peaks, cond)}.pt"
        if p.exists():
            return torch.load(p, weights_only=True)
        out = self.backbone_layers(peaks, cond)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, p)
        return out

    def forward(self, peaks: PeakBatch, cond: ConditioningForEncoder) -> EncoderOutput:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            layers, pad, mz, pooled_native = self._cached_backbone(peaks, cond)
        if layers is None:
            return EncoderOutput(memory=None, memory_pad=None, pooled=self.proj_pooled(pooled_native))
        h = self.mix(layers) if self.mix is not None else layers[-1]
        memory = self.proj_peaks(h)
        return EncoderOutput(memory=memory, memory_pad=pad, pooled=self.proj_pooled(pooled_native if pooled_native is not None else h[:, 0]),
                             memory_mz=mz)
