"""Sinusoidal / Fourier features for continuous values (m/z, mass differences, energies).

Phases are computed in float64: at m/z ~ 800 and wavelengths down to 1e-3, float32 phases would
lose the sub-mDa information the encoding is meant to carry.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def geometric_wavelengths(n: int, lam_min: float, lam_max: float) -> Tensor:
    if n == 1:
        return torch.tensor([lam_min], dtype=torch.float64)
    return torch.exp(torch.linspace(math.log(lam_min), math.log(lam_max), n, dtype=torch.float64))


def sinusoidal(x: Tensor, wavelengths: Tensor) -> Tensor:
    """x (...,) any float -> (..., 2n) float32 [sin | cos] of 2*pi*x/lambda."""
    ph = 2 * math.pi * x.to(torch.float64).unsqueeze(-1) / wavelengths.to(x.device)
    return torch.cat([torch.sin(ph), torch.cos(ph)], dim=-1).to(torch.float32)
