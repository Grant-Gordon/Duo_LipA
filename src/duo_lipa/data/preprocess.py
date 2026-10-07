"""Smoke-test preprocessing only. The real preprocessing spec (Q6-Q12, Q17) is deferred.

Default here is LipiDetective's setting, as proposed in the encoder diagram for the smoke test:
top-K peaks by intensity, max-normalized, sorted by decreasing intensity (so list order = intensity
rank, which LipiDetective's rank PE relies on).
"""

from __future__ import annotations

import numpy as np

from duo_lipa.data.records import SpectrumRecord


def preprocess(rec: SpectrumRecord, top_k: int = 30, mz_max: float | None = None,
               mz_min: float | None = None) -> SpectrumRecord:
    mz, it = rec.mz, rec.intensity.astype(np.float32)
    keep = np.ones_like(mz, dtype=bool)
    if mz_max is not None:
        keep &= mz <= mz_max
    if mz_min is not None:
        keep &= mz >= mz_min
    mz, it = mz[keep], it[keep]
    order = np.argsort(-it, kind="stable")[:top_k]
    mz, it = mz[order], it[order]
    if it.size and it.max() > 0:
        it = it / it.max()
    out = SpectrumRecord(**{**rec.__dict__})
    out.mz, out.intensity = mz, it
    return out
