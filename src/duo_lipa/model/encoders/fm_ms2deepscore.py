"""MS2DeepScore 2.x adapter (frozen, pooled-only). de Jonge et al.; weights Zenodo 17826815 (CC-BY-4.0),
code Apache-2.0 (github.com/matchms/ms2deepscore).

The encoder is two dense layers (SiameseSpectralModel.py: SpectralEncoder, dense_layer). It is
re-implemented here with matching parameter names; the package itself pulls onnx, tensorboard, numba
and matchms (IMPLEMENTATION.md D-F1). Input = [ionmode (1 = positive), precursor_mz / 1000] ++ 9900
bins (m/z 10-1000, width 0.1, max of intensity^0.5 per bin; tensorize_spectra.py:47-59).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from duo_lipa.model.encoders.base import FrozenFMAdapter


class _SpectralEncoder(nn.Module):
    def __init__(self, in_dim: int, base_dims: list[int], emb_dim: int):
        super().__init__()
        self.dense_layers = nn.ModuleList()
        d = in_dim
        for h in base_dims:
            self.dense_layers.append(nn.Sequential(nn.Linear(d, h), nn.ReLU()))
            d = h
        self.embedding_layer = nn.Sequential(nn.Linear(d, emb_dim), nn.Tanh())

    def forward(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return self.embedding_layer(x)


class MS2DeepScoreAdapter(FrozenFMAdapter):
    name = "ms2deepscore"
    has_per_peak = False
    accepts_conditioning = {"polarity", "precursor_mz"}   # its native metadata inputs
    frozen = True
    n_layers_exposed = 1

    def __init__(self, d_model: int, layer_mix: bool = True, head_mlp_layers: int = 1, weights_dir=None):
        super().__init__(d_model, layer_mix, head_mlp_layers)
        wd = Path(weights_dir or Path(__file__).resolve().parents[4] / "weights" / "ms2deepscore")
        ck = torch.load(wd / "ms2deepscore_model.pt", map_location="cpu", weights_only=True)
        s = json.loads(ck["settings_json"]) if "settings_json" in ck else json.loads((wd / "settings.json").read_text())
        self.min_mz, self.max_mz, self.bin_w = s["min_mz"], s["max_mz"], s["mz_bin_width"]
        self.int_scale = s["intensity_scaling"]
        self.n_bins = int((self.max_mz - self.min_mz) / self.bin_w)
        sd = {k[len("encoder."):]: v for k, v in ck["state_dict"].items() if k.startswith("encoder.")}
        n_meta = sd["dense_layers.0.0.weight"].shape[1] - self.n_bins
        self.backbone = _SpectralEncoder(self.n_bins + n_meta, s["base_dims"], s["embedding_dim"])
        self.backbone.load_state_dict(sd, strict=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.native_dim = s["embedding_dim"]
        self._build_heads()

    def tensorize(self, peaks, cond) -> torch.Tensor:
        B = peaks.B
        x = torch.zeros(B, 2 + self.n_bins)
        pol = cond.raw.get("polarity", [None] * B)
        prec = cond.raw.get("precursor_mz", [None] * B)
        for b in range(B):
            x[b, 0] = 1.0 if pol[b] == "+" else 0.0          # unknown polarity -> 0 (no native "unknown")
            x[b, 1] = float(prec[b]) / 1000.0 if prec[b] is not None else 0.0
            keep = ~peaks.pad[b]
            mz, it = peaks.mz[b][keep], peaks.intensity[b][keep].double()
            ok = (mz >= self.min_mz) & (mz < self.max_mz)
            idx = ((mz[ok] - self.min_mz) / self.bin_w).long()
            vals = it[ok] ** self.int_scale
            row = torch.zeros(self.n_bins, dtype=torch.float64)
            row.scatter_reduce_(0, idx, vals, reduce="amax")
            x[b, 2:] = row.float()
        return x

    def backbone_layers(self, peaks, cond):
        return None, None, None, self.backbone(self.tensorize(peaks, cond))
