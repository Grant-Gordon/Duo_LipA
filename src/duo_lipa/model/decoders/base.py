"""Target batches and the SlotDecoder interface (decoder spec 7.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from duo_lipa.model.interfaces import Conditioning, EncoderOutput
from duo_lipa.schema.blocks import UNK
from duo_lipa.schema.schema import SlotSchema, SlotToken

CENTRE_KIND = {("glycerol",): 1, ("lcb", 2): 2, ("lcb", 3): 3, ("lcb", 4): 4}  # 5 = chain carbon
MAX_CHAINS = 4
MAX_SUB = 40


@dataclass
class TargetBatch:
    stype: Tensor        # (B, T) slot-type id
    chain: Tensor        # (B, T) 1-based chain index, 0 = none
    sub: Tensor          # (B, T) bond / modification index, 0 = none
    pos: Tensor          # (B, T) GEOM bond position / STEREO chain-carbon position, 0 = none or unknown
    centre: Tensor       # (B, T) STEREO centre kind (CENTRE_KIND, 5 = chain carbon), 0 = not stereo
    value: Tensor        # (B, T) global value id, -1 = UNK or pad
    in_ids: Tensor       # (B, T) AR input id: START, previous value, or previous type's UNK
    supervised: Tensor   # (B, T) bool
    pad: Tensor          # (B, T) bool, True = padding
    blocked: Tensor      # (B, T, V) bool, True = value not allowed (grammar + slot-type block)
    tokens: list[list[SlotToken]]

    @property
    def B(self) -> int:
        return self.stype.shape[0]


def input_vocab_size(schema: SlotSchema) -> int:
    return schema.V + len(schema.slot_types) + 1   # values, one UNK per slot type, START


def collate_targets(schema: SlotSchema, seqs: list[list[SlotToken]], grammar_mask: bool = True,
                    pad_to: int | None = None) -> TargetBatch:
    B = len(seqs)
    T = max(max((len(s) for s in seqs), default=1), 1)
    if pad_to is not None:
        T = max(T, pad_to)
    V = schema.V
    start_id = schema.V + len(schema.slot_types)
    z = lambda: torch.zeros(B, T, dtype=torch.long)
    stype, chain, sub, pos, centre = z(), z(), z(), z(), z()
    value = torch.full((B, T), -1, dtype=torch.long)
    in_ids = torch.full((B, T), start_id, dtype=torch.long)
    supervised = torch.zeros(B, T, dtype=torch.bool)
    pad = torch.ones(B, T, dtype=torch.bool)
    blocked = torch.ones(B, T, V, dtype=torch.bool)
    for b, seq in enumerate(seqs):
        prev = start_id
        for t, tok in enumerate(seq):
            sp = tok.spec
            tid = schema.type_id[sp.stype]
            stype[b, t], chain[b, t], sub[b, t], pos[b, t] = tid, sp.chain, min(sp.sub, MAX_SUB - 1), tok.pos
            if sp.stype == "STEREO":
                centre[b, t] = CENTRE_KIND.get(sp.extra, 5)
                chain[b, t] = sp.extra[1] if sp.extra and sp.extra[0] == "chain" else 0
            pad[b, t] = False
            in_ids[b, t] = prev
            if tok.value is not UNK:
                value[b, t] = schema.global_id(sp.stype, tok.value)
                prev = value[b, t].item()
            else:
                prev = schema.V + tid
            supervised[b, t] = tok.supervised
            off, n = schema.offset[sp.stype], len(schema.values[sp.stype])
            if grammar_mask:
                ids = torch.tensor(schema.legal_ids(sp.stype, tok.legal), dtype=torch.long)
                blocked[b, t, ids] = False
            else:
                blocked[b, t, off:off + n] = False
    return TargetBatch(stype, chain, sub, pos, centre, value, in_ids, supervised, pad, blocked, seqs)


def masked_log_softmax(scores: Tensor, blocked: Tensor) -> Tensor:
    scores = scores.masked_fill(blocked, float("-inf"))
    out = torch.log_softmax(scores, dim=-1)
    # rows with every value blocked (padding) -> zeros instead of NaN
    return torch.where(blocked.all(-1, keepdim=True), torch.zeros_like(out), out)


class SlotDecoder(nn.Module):
    input: Literal["pooled", "peaks"]

    def __init__(self, schema: SlotSchema):
        super().__init__()
        self.schema = schema

    def train_logprobs(self, enc: EncoderOutput, cond: Conditioning, tgt: TargetBatch) -> Tensor:
        """(B, T, V) grammar-masked log-probabilities at every target slot."""
        raise NotImplementedError

    def start(self, enc: EncoderOutput, cond: Conditioning):
        """Work that does not depend on the partial annotation."""
        return {"enc": enc, "cond": cond}

    def step_scores(self, ctx, prefix: TargetBatch, batch_index: Tensor) -> Tensor:
        """(N, V) log-probabilities for the last (unvalued) slot of each row of `prefix`; row n
        belongs to spectrum `batch_index[n]` of `ctx`. Scores are grammar-masked already, which is
        the same mask the caller would apply. Default: rerun teacher-forced scoring on the prefix
        (no KV cache; fine for v0)."""
        enc, cond = select_rows(ctx["enc"], ctx["cond"], batch_index)
        lp = self.train_logprobs(enc, cond, prefix)
        last = (~prefix.pad).sum(1) - 1
        return lp[torch.arange(lp.shape[0]), last]

    def loss(self, enc: EncoderOutput, cond: Conditioning, tgt: TargetBatch) -> dict:
        lp = self.train_logprobs(enc, cond, tgt)
        return _nll(lp, tgt, tgt.supervised)


def _nll(lp: Tensor, tgt: TargetBatch, mask: Tensor) -> dict:
    idx = tgt.value.clamp(min=0).unsqueeze(-1)
    tok_lp = lp.gather(-1, idx).squeeze(-1)
    n = mask.sum().clamp(min=1)
    return {"loss": -(tok_lp * mask).sum() / n, "n_supervised": int(mask.sum())}


def select_rows(enc: EncoderOutput, cond: Conditioning, idx: Tensor) -> tuple[EncoderOutput, Conditioning]:
    """Index the batch dimension (beam search expands one spectrum into N partial annotations)."""
    e = EncoderOutput(
        memory=enc.memory[idx] if enc.memory is not None else None,
        memory_pad=enc.memory_pad[idx] if enc.memory_pad is not None else None,
        pooled=enc.pooled[idx],
        memory_mz=enc.memory_mz[idx] if enc.memory_mz is not None else None,
    )
    c = Conditioning(tokens=cond.tokens[idx], precursor_mz=cond.precursor_mz[idx],
                     adduct=[cond.adduct[i] for i in idx.tolist()],
                     isotope_label=[cond.isotope_label[i] for i in idx.tolist()], field_names=cond.field_names)
    return e, c
