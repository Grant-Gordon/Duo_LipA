"""Source-independent spectrum record.

Every ingestion adapter (MGF, LipidOracle, later CID libraries) produces `SpectrumRecord`s. Nothing
downstream of this module may depend on where a record came from (encoder spec 1.1: "nothing in the
encoder may hard-code LipidOracle specifics").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Conditioning fields (decoder spec 5.1). The adduct is a decoder slot, not a conditioning field.
COND_FIELDS = ("precursor_mz", "polarity", "method", "collision_energy", "isotope_label", "instrument")


@dataclass
class SpectrumRecord:
    mz: np.ndarray                      # (N,) float64, centroided
    intensity: np.ndarray               # (N,) float32, raw
    precursor_mz: float | None
    polarity: str | None = None         # "+" | "-"
    method: str | None = None           # "CID" | "HCD" | "UVPD" | "EAD" | "OzID" | ...
    collision_energy: float | None = None
    isotope_label: str | None = None    # e.g. "d7"; stripped from the label string
    instrument: str | None = None
    adduct: str | None = None           # known adduct, fixes the ADDUCT slot when given
    label: str | None = None            # Goslin-parseable label at whatever depth the source resolves
    source: str = ""
    meta: dict = field(default_factory=dict)

    def conditioning(self) -> dict:
        return {k: getattr(self, k) for k in COND_FIELDS}
