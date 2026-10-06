"""Model wrapper: owns the conditioning embedder, routes conditioning (decoder spec 5.4), and chains
encoder -> decoder."""

from __future__ import annotations

import torch
from torch import nn

from duo_lipa.data.batch import MetaBatch, PeakBatch
from duo_lipa.model.conditioning import ConditioningEmbedder
from duo_lipa.model.decoders.base import SlotDecoder, TargetBatch
from duo_lipa.model.encoders.base import EncoderAdapter
from duo_lipa.model.interfaces import Conditioning, ConditioningForEncoder, EncoderOutput


class DuoLipA(nn.Module):
    def __init__(self, embedder: ConditioningEmbedder, encoder: EncoderAdapter, decoder: SlotDecoder):
        super().__init__()
        if decoder.input == "peaks" and not encoder.has_per_peak:
            raise ValueError(f"decoder input 'peaks' needs per-peak output; encoder {encoder.name!r} is pooled-only")
        self.embedder, self.encoder, self.decoder = embedder, encoder, decoder
        self.last_routing: dict = {}

    def route(self, cond: Conditioning, meta: MetaBatch) -> ConditioningForEncoder:
        names = [n for n in cond.field_names if n in self.encoder.accepts_conditioning]
        idx = [cond.field_names.index(n) for n in names]
        tokens = cond.tokens[:, idx] if idx else None
        enc_cond = ConditioningForEncoder(tokens=tokens, field_names=names,
                                          raw={n: meta.fields.get(n) for n in names})
        self.last_routing = {"decoder": list(cond.field_names), "encoder": names}
        return enc_cond

    def encode(self, peaks: PeakBatch, meta: MetaBatch) -> tuple[EncoderOutput, Conditioning]:
        cond = self.embedder(meta, peaks.precursor_mz)
        enc = self.encoder(peaks, self.route(cond, meta))
        return enc, cond

    def forward(self, peaks: PeakBatch, meta: MetaBatch, tgt: TargetBatch) -> dict:
        enc, cond = self.encode(peaks, meta)
        out = self.decoder.loss(enc, cond, tgt)
        out["enc"] = enc
        return out

    def param_counts(self) -> dict:
        def count(m, trainable=None):
            return sum(p.numel() for p in m.parameters() if trainable is None or p.requires_grad == trainable)
        return {
            "embedder": count(self.embedder),
            "encoder_trainable": count(self.encoder, True),
            "encoder_frozen": count(self.encoder, False),
            "decoder": count(self.decoder),
            "total_trainable": count(self, True),
        }
