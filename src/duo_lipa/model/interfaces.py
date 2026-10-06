"""The encoder/decoder contract (decoder spec 7.4, encoder spec 2). Every mask is boolean, True = blocked."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class EncoderOutput:
    memory: Tensor | None             # (B, M, d_model) per-peak vectors; None if no per-peak output
    memory_pad: Tensor | None         # (B, M) bool, True = padding
    pooled: Tensor                    # (B, d_model)
    memory_mz: Tensor | None = None   # (B, M) float64, m/z of each memory row (Q4b; unused in v0)


@dataclass
class Conditioning:
    tokens: Tensor                    # (B, C, d) one vector per metadata field (decoder spec 5.2)
    precursor_mz: Tensor              # (B,) float64, for the mass check
    adduct: list[str | None]
    isotope_label: list[str | None]
    field_names: list[str] = field(default_factory=list)


@dataclass
class ConditioningForEncoder:
    """The subset of conditioning tokens an encoder declared in `accepts_conditioning`."""
    tokens: Tensor | None             # (B, C', d) or None
    field_names: list[str]
    raw: dict[str, list]              # raw values of the accepted fields (FMs use native formats)


def check_bool_mask(m: Tensor | None, name: str):
    if m is not None and m.dtype != torch.bool:
        raise TypeError(f"{name} must be a bool mask (True = blocked), got {m.dtype}")
