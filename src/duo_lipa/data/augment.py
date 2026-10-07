"""Spectrum augmentation (encoder spec 5.5, Q37): perturbed input, unchanged label. Config switch."""

from __future__ import annotations

import numpy as np

from duo_lipa.data.records import SpectrumRecord


def augment(rec: SpectrumRecord, rng: np.random.Generator, peak_dropout: float = 0.1,
            intensity_jitter: float = 0.1, noise_peaks: int = 2, mz_jitter_ppm: float = 5.0, **_) -> SpectrumRecord:
    mz, it = rec.mz.copy(), rec.intensity.astype(np.float32).copy()
    keep = rng.random(len(mz)) >= peak_dropout
    if keep.sum() == 0:
        keep[np.argmax(it)] = True
    mz, it = mz[keep], it[keep]
    it = it * np.exp(rng.normal(0, intensity_jitter, len(it))).astype(np.float32)
    mz = mz * (1 + rng.normal(0, mz_jitter_ppm * 1e-6, len(mz)))
    if noise_peaks and len(mz):
        lo, hi = float(mz.min()), float(rec.precursor_mz or mz.max())
        mz = np.concatenate([mz, rng.uniform(lo, hi, noise_peaks)])
        it = np.concatenate([it, rng.uniform(0, 0.05, noise_peaks).astype(np.float32) * (it.max() if len(it) else 1)])
    order = np.argsort(-it, kind="stable")
    out = SpectrumRecord(**{**rec.__dict__})
    out.mz, out.intensity = mz[order], it[order]
    return out
