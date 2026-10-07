"""Generic MGF reader. Returns raw header dicts plus peak arrays; source-specific meaning is applied
by adapters (see `lipidoracle.py`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class MgfEntry:
    headers: dict[str, str]
    mz: np.ndarray
    intensity: np.ndarray


def read_mgf(path: str | Path) -> Iterator[MgfEntry]:
    headers: dict[str, str] = {}
    mz: list[float] = []
    it: list[float] = []
    inside = False
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                inside, headers, mz, it = True, {}, [], []
            elif line == "END IONS":
                inside = False
                yield MgfEntry(headers, np.asarray(mz, dtype=np.float64), np.asarray(it, dtype=np.float32))
            elif inside:
                if "=" in line and not line[0].isdigit():
                    k, v = line.split("=", 1)
                    headers[k.strip().upper()] = v.strip()
                else:
                    parts = line.split()
                    mz.append(float(parts[0]))
                    it.append(float(parts[1]) if len(parts) > 1 else 1.0)


def parse_charge(charge: str | None) -> tuple[int | None, str | None]:
    """'1+' -> (1, '+'); '2-' -> (2, '-'); None -> (None, None)."""
    if not charge:
        return None, None
    c = charge.strip()
    sign = "-" if c.endswith("-") or c.startswith("-") else "+"
    digits = "".join(ch for ch in c if ch.isdigit())
    return (int(digits) if digits else 1), sign
