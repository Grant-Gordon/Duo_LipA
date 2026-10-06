"""DreaMS adapter (frozen). Bushuiev et al., Nat Biotechnol 2025; weights HF roman-bushuiev/DreaMS (MIT).

The backbone below is a torch-only re-implementation of `dreams/models/dreams/layers.py`,
`dreams/models/layers/fourier_features.py`, `feed_forward.py` and `DreaMS.forward`
(dreams/models/dreams/dreams.py:138-185), from github.com/pluskal-lab/DreaMS (MIT License,
Copyright (c) Roman Bushuiev et al.). It is vendored because the upstream package pins numpy 1.25 /
torch 2.2 and imports pytorch_lightning, rdkit, matchms (IMPLEMENTATION.md D-F1). Parameter names
match the upstream state dict so checkpoints load unchanged.

Faithful-to-upstream details kept on purpose:
- the padding mask is applied to query rows, not keys (layers.py:98); padded keys stay visible, so
  every spectrum is padded to exactly `top_k` + 1 tokens, as upstream does at inference;
- the graphormer pairwise term is the unparametrized sum over Fourier-feature differences;
- Fourier phases are float32, as upstream.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from duo_lipa.data.batch import PeakBatch
from duo_lipa.model.encoders.base import FrozenFMAdapter
from duo_lipa.model.encoders.safe_load import load_checkpoint
from duo_lipa.model.interfaces import ConditioningForEncoder

MAX_MZ = 1000.0          # dformat max_mz (dreams/utils/dformats.py:97-108)
PREC_INTENSITY = 1.1     # dreams/utils/data.py:150


class _FF(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0, act_last: bool = True):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            last = i == len(dims) - 2
            if not last:
                layers.append(nn.Dropout(dropout))
            if not last or act_last:
                layers.append(nn.ReLU())
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)


class _MHA(nn.Module):
    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        self.d, self.h, self.dh, self.dropout = d, h, d // h, dropout
        self.weights = nn.Parameter(torch.empty(4 * d, d))  # q, k, v, o stacked; no bias

    def forward(self, x, mask, graphormer_bias):
        B, N, d = x.shape
        q, k, v = F.linear(x, self.weights[: 3 * d]).chunk(3, dim=-1)
        sp = lambda t: t.reshape(B, N, self.h, self.dh).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        att = (q @ k.transpose(-2, -1)) * self.dh ** -0.5
        if graphormer_bias is not None:
            att = att + graphormer_bias
        att = att.masked_fill(mask.unsqueeze(1).unsqueeze(-1), -1e9)  # query rows (upstream quirk)
        att = F.dropout(F.softmax(att, dim=-1), p=self.dropout, training=self.training)
        out = (att @ v).transpose(1, 2).reshape(B, N, d)
        return F.linear(out, self.weights[3 * d:])


class _FFN(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(d, 4 * d, bias=False)
        self.out_proj = nn.Linear(4 * d, d, bias=False)
        self.dropout = dropout

    def forward(self, x):
        return self.out_proj(F.dropout(F.relu(self.in_proj(x)), p=self.dropout, training=self.training))


class _Encoder(nn.Module):
    def __init__(self, d: int, h: int, L: int, dropout: float):
        super().__init__()
        self.atts = nn.ModuleList([_MHA(d, h, dropout) for _ in range(L)])
        self.ffs = nn.ModuleList([_FFN(d, dropout) for _ in range(L)])
        self.scales = nn.ModuleList([nn.LayerNorm(d) for _ in range(2 * L + 1)])
        self.dropout = dropout

    def forward(self, x, mask, gbias) -> list[Tensor]:
        """Returns the residual stream after every layer (pre-norm; the final norm is applied by
        the caller)."""
        outs = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(len(self.atts)):
            r = x
            x = r + F.dropout(self.atts[i](self.scales[2 * i](x), mask, gbias), p=self.dropout, training=self.training)
            r = x
            x = r + F.dropout(self.ffs[i](self.scales[2 * i + 1](x)), p=self.dropout, training=self.training)
            outs.append(x)
        return outs


class DreaMSBackbone(nn.Module):
    def __init__(self, d_model=1024, n_layers=7, n_heads=8, d_peak=44, d_fourier=980, ff_fourier_d=512,
                 ff_fourier_depth=5, ff_peak_depth=1, n_freqs=5997, dropout=0.1):
        super().__init__()
        self.register_buffer("fourier_b", torch.zeros(1, n_freqs))
        self.ff_fourier = _FF([2 * n_freqs] + [ff_fourier_d] * (ff_fourier_depth - 1) + [d_fourier], dropout)
        self.ff_peak = _FF([2] + [d_peak] * (ff_peak_depth - 1) + [d_peak], dropout)
        self.transformer_encoder = _Encoder(d_model, n_heads, n_layers, dropout)

    def forward(self, spec: Tensor) -> list[Tensor]:
        """spec (B, N, 2) float32 [m/z, intensity], token 0 = precursor, zero rows = padding.
        Returns per-layer residual streams, each passed through the trained final LayerNorm."""
        pad = spec[:, :, 0] == 0
        peak = self.ff_peak(spec / torch.tensor([MAX_MZ, 1.0], dtype=spec.dtype))
        ph = 2 * torch.pi * spec[..., [0]] @ self.fourier_b
        ff = self.ff_fourier(torch.cat([torch.cos(ph), torch.sin(ph)], dim=-1))
        x = torch.cat([peak, ff], dim=-1)
        # unparametrized graphormer term: sum_f (ff_i - ff_j) = s_i - s_j, broadcast over heads
        s = ff.sum(-1)
        gbias = (s.unsqueeze(2) - s.unsqueeze(1)).unsqueeze(1)
        outs = self.transformer_encoder(x, pad, gbias)
        final_ln = self.transformer_encoder.scales[-1]
        return [final_ln(o) for o in outs]


def load_dreams_backbone(path: str | Path) -> DreaMSBackbone:
    ck = load_checkpoint(path)
    sd = ck["state_dict"]
    prefix = "backbone." if any(k.startswith("backbone.") for k in sd) else ""
    sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    args = ck.get("hyper_parameters", {}).get("args")
    if args is not None:
        a = vars(args)
        if a.get("d_graphormer_params") or a.get("graphormer_parametrized") or a.get("vanilla_transformer") \
                or a.get("charge_feature") or a.get("scnorm") or not a.get("pre_norm", True):
            raise NotImplementedError("DreaMS variant not covered by the vendored backbone")
        n_heads, dropout = a["n_heads"], a.get("dropout", 0.1)
    else:
        # the contrastive embedding checkpoint stores no args; it fine-tunes the SSL architecture,
        # whose n_heads (8) is not recoverable from tensor shapes (IMPLEMENTATION.md D-F1)
        n_heads, dropout = 8, 0.1
    ff_idx = sorted(int(k.split(".")[2]) for k in sd if k.startswith("ff_fourier.ff.") and k.endswith(".weight"))
    m = DreaMSBackbone(
        d_model=sd["transformer_encoder.scales.0.weight"].shape[0],
        n_layers=sum(1 for k in sd if k.startswith("transformer_encoder.atts.") and k.endswith(".weights")),
        n_heads=n_heads,
        d_peak=sd["ff_peak.ff.0.weight"].shape[0],
        d_fourier=sd[f"ff_fourier.ff.{ff_idx[-1]}.weight"].shape[0],
        ff_fourier_d=sd["ff_fourier.ff.0.weight"].shape[0],
        ff_fourier_depth=len(ff_idx),
        ff_peak_depth=sum(1 for k in sd if k.startswith("ff_peak.ff.") and k.endswith(".weight")),
        n_freqs=sd["fourier_enc.b"].shape[1], dropout=dropout)
    keep = {k: v for k, v in sd.items()
            if not k.startswith(("ff_out", "ro_out", "fourier_enc.", "head.", "ff_out_intens"))}
    keep["fourier_b"] = sd["fourier_enc.b"]
    m.load_state_dict(keep, strict=True)
    return m


class DreaMSAdapter(FrozenFMAdapter):
    name = "dreams"
    has_per_peak = True
    accepts_conditioning = {"precursor_mz"}   # native: the prepended precursor token
    frozen = True

    def __init__(self, d_model: int, layer_mix: bool = True, head_mlp_layers: int = 1, weights_dir=None,
                 checkpoint: str = "ssl_model.ckpt", top_k: int = 100):
        super().__init__(d_model, layer_mix, head_mlp_layers)
        wd = Path(weights_dir or Path(__file__).resolve().parents[4] / "weights" / "dreams")
        self.backbone = load_dreams_backbone(wd / checkpoint)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.top_k = top_k
        self.native_dim = self.backbone.transformer_encoder.scales[0].normalized_shape[0]
        self.n_layers_exposed = len(self.backbone.transformer_encoder.atts)
        self._build_heads()

    def prepare(self, peaks: PeakBatch, cond: ConditioningForEncoder) -> tuple[Tensor, Tensor, Tensor]:
        """Upstream input format (dreams/utils/data.py:103-165): top-k peaks by intensity, in m/z
        order, intensities / base peak, zero-padded to k, precursor token [prec_mz, 1.1] first."""
        B = peaks.B
        k = self.top_k
        spec = torch.zeros(B, k + 1, 2, dtype=torch.float32)
        mz_out = torch.zeros(B, k + 1, dtype=torch.float64)
        prec_raw = cond.raw.get("precursor_mz") if "precursor_mz" in cond.field_names else None
        for b in range(B):
            keep = (~peaks.pad[b]) & (peaks.mz[b] <= MAX_MZ) & (peaks.mz[b] > 0)
            mz, it = peaks.mz[b][keep], peaks.intensity[b][keep]
            top = torch.argsort(it, descending=True)[:k]
            mz, it = mz[top], it[top]
            order = torch.argsort(mz)
            mz, it = mz[order], it[order]
            if it.numel():
                it = it / it.max()
            n = mz.numel()
            p = prec_raw[b] if prec_raw is not None else None
            p = float(p) if p is not None else 0.0
            spec[b, 0] = torch.tensor([p, PREC_INTENSITY])
            spec[b, 1:n + 1, 0], spec[b, 1:n + 1, 1] = mz.float(), it.float()
            mz_out[b, 0], mz_out[b, 1:n + 1] = p, mz
        pad = spec[:, :, 0] == 0
        return spec, pad, mz_out

    def backbone_layers(self, peaks, cond):
        spec, pad, mz = self.prepare(peaks, cond)
        layers = self.backbone(spec)
        return layers, pad, mz, layers[-1][:, 0]
