"""Conditioning embedder, owned by the model wrapper (decoder spec 5.2, 5.4).

Categorical fields: lookup table with row 0 = "unknown". Continuous fields: Fourier features then a
linear layer; "unknown" is a learned vector. In training each field is replaced by "unknown" with
probability `field_dropout` (default 0.1).

Q40 (acquisition settings without a field) is left open: `extra_fields` lets a config add fields
without code changes, and none are added in v0 (IMPLEMENTATION.md D-C2).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from duo_lipa.data.batch import MetaBatch
from duo_lipa.model.fourier import geometric_wavelengths, sinusoidal
from duo_lipa.model.interfaces import Conditioning

DEFAULT_FIELDS = {
    "precursor_mz": {"type": "continuous", "lam_min": 1e-3, "lam_max": 1e4, "n_freq": 32},
    "polarity": {"type": "categorical", "vocab": ["+", "-"]},
    "method": {"type": "categorical", "vocab": ["CID", "HCD", "UVPD", "EAD", "OzID", "ETD", "EID"]},
    "collision_energy": {"type": "continuous", "lam_min": 0.1, "lam_max": 1e3, "n_freq": 16},
    "isotope_label": {"type": "categorical", "vocab": ["d5", "d7", "d9", "13C"]},
    "instrument": {"type": "categorical", "vocab": ["Orbitrap", "QTOF", "IonTrap", "QQQ", "FTICR"]},
}


class ConditioningEmbedder(nn.Module):
    def __init__(self, d_model: int, field_dropout: float = 0.1, extra_fields: dict | None = None):
        super().__init__()
        self.specs = {**DEFAULT_FIELDS, **(extra_fields or {})}
        self.field_names = list(self.specs)
        self.field_dropout = field_dropout
        self.cat = nn.ModuleDict()
        self.cont = nn.ModuleDict()
        self.cont_unknown = nn.ParameterDict()
        self.vocab_index: dict[str, dict] = {}
        for name, sp in self.specs.items():
            if sp["type"] == "categorical":
                self.vocab_index[name] = {v: i + 1 for i, v in enumerate(sp["vocab"])}
                self.cat[name] = nn.Embedding(len(sp["vocab"]) + 1, d_model)
            else:
                self.register_buffer(f"lam_{name}", geometric_wavelengths(sp["n_freq"], sp["lam_min"], sp["lam_max"]),
                                     persistent=False)
                self.cont[name] = nn.Linear(2 * sp["n_freq"], d_model)
                self.cont_unknown[name] = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))

    def embed_field(self, name: str, values: list, device) -> Tensor:
        B = len(values)
        sp = self.specs[name]
        drop = torch.zeros(B, dtype=torch.bool, device=device)
        if self.training and self.field_dropout > 0:
            drop = torch.rand(B, device=device) < self.field_dropout
        if sp["type"] == "categorical":
            idx = torch.tensor([self.vocab_index[name].get(v, 0) if v is not None else 0 for v in values],
                               device=device)
            idx = torch.where(drop, torch.zeros_like(idx), idx)
            return self.cat[name](idx)
        known = torch.tensor([v is not None and not (isinstance(v, float) and math.isnan(v)) for v in values],
                             device=device) & ~drop
        x = torch.tensor([float(v) if v is not None else 0.0 for v in values], dtype=torch.float64, device=device)
        e = self.cont[name](sinusoidal(x, getattr(self, f"lam_{name}")))
        return torch.where(known.unsqueeze(-1), e, self.cont_unknown[name].expand(B, -1))

    def forward(self, meta: MetaBatch, precursor_mz: Tensor) -> Conditioning:
        device = precursor_mz.device
        toks = [self.embed_field(n, meta.fields.get(n, [None] * len(meta.adduct)), device) for n in self.field_names]
        return Conditioning(
            tokens=torch.stack(toks, dim=1),
            precursor_mz=precursor_mz,
            adduct=list(meta.adduct),
            isotope_label=list(meta.isotope_label),
            field_names=list(self.field_names),
        )
