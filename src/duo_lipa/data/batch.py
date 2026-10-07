"""Batched peaks and metadata (encoder input)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from duo_lipa.data.records import COND_FIELDS, SpectrumRecord


@dataclass
class PeakBatch:
    mz: torch.Tensor            # (B, K) float64
    intensity: torch.Tensor     # (B, K) float32, max-normalized
    pad: torch.Tensor           # (B, K) bool, True = padding
    precursor_mz: torch.Tensor  # (B,) float64; NaN if unknown

    @property
    def B(self) -> int:
        return self.mz.shape[0]


@dataclass
class MetaBatch:
    fields: dict[str, list]     # field name -> list of raw values (None = unknown), length B
    adduct: list[str | None]
    isotope_label: list[str | None]
    records: list[SpectrumRecord]


def collate_peaks(recs: list[SpectrumRecord], pad_to: int | None = None) -> PeakBatch:
    K = max(max(len(r.mz) for r in recs), 1)
    if pad_to is not None:
        K = max(K, pad_to)
    B = len(recs)
    mz = np.zeros((B, K), dtype=np.float64)
    it = np.zeros((B, K), dtype=np.float32)
    pad = np.ones((B, K), dtype=bool)
    for i, r in enumerate(recs):
        n = len(r.mz)
        mz[i, :n], it[i, :n], pad[i, :n] = r.mz, r.intensity, False
    prec = np.array([np.nan if r.precursor_mz is None else r.precursor_mz for r in recs], dtype=np.float64)
    return PeakBatch(torch.from_numpy(mz), torch.from_numpy(it), torch.from_numpy(pad), torch.from_numpy(prec))


def collate_meta(recs: list[SpectrumRecord]) -> MetaBatch:
    return MetaBatch(
        fields={f: [getattr(r, f) for r in recs] for f in COND_FIELDS},
        adduct=[r.adduct for r in recs],
        isotope_label=[r.isotope_label for r in recs],
        records=list(recs),
    )
