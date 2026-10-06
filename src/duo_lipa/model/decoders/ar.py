"""AR decoder (decoder spec 7.5): one transformer decoder predicting slots in sequence.

Step input = previous value (or START / UNK) + slot type at t + chain index + bond/mod index + step
position. Pre-LN blocks: causal self-attention, cross-attention to memory ++ conditioning tokens
(`input: peaks`), feed-forward. With `input: pooled`, cross-attention is off and the pooled vector
plus conditioning tokens are prefix tokens every step attends to.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from duo_lipa.model.decoders.base import (MAX_CHAINS, MAX_SUB, SlotDecoder, TargetBatch, input_vocab_size,
                                          masked_log_softmax)
from duo_lipa.model.interfaces import Conditioning, EncoderOutput, check_bool_mask
from duo_lipa.schema.schema import SlotSchema


class ARBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: int, dropout: float, cross: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross = cross
        if cross:
            self.ln2 = nn.LayerNorm(d)
            self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask, key_pad, mem=None, mem_pad=None):
        h = self.ln1(x)
        x = x + self.drop(self.self_attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_pad,
                                         need_weights=False)[0])
        if self.cross:
            h = self.ln2(x)
            x = x + self.drop(self.cross_attn(h, mem, mem, key_padding_mask=mem_pad, need_weights=False)[0])
        return x + self.drop(self.ff(self.ln3(x)))


class ARDecoder(SlotDecoder):
    def __init__(self, schema: SlotSchema, d_model: int, input: str = "peaks", n_layers: int = 2,
                 n_heads: int = 4, ff_mult: int = 4, dropout: float = 0.0, max_steps: int = 256):
        super().__init__(schema)
        self.input = input
        d = d_model
        self.in_emb = nn.Embedding(input_vocab_size(schema), d)
        self.type_emb = nn.Embedding(len(schema.slot_types), d)
        self.chain_emb = nn.Embedding(MAX_CHAINS + 1, d)
        self.sub_emb = nn.Embedding(MAX_SUB, d)
        self.step_emb = nn.Embedding(max_steps, d)
        self.prefix_type = nn.Embedding(2, d)  # pooled-mode prefix: 0 = pooled vector, 1 = conditioning
        self.blocks = nn.ModuleList([ARBlock(d, n_heads, ff_mult, dropout, cross=(input == "peaks"))
                                     for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d)
        self.out = nn.Linear(d, schema.V)  # one block of rows per slot type (decoder spec 6)

    def _step_inputs(self, tgt: TargetBatch) -> Tensor:
        T = tgt.stype.shape[1]
        steps = torch.arange(T, device=tgt.stype.device).clamp(max=self.step_emb.num_embeddings - 1)
        return (self.in_emb(tgt.in_ids) + self.type_emb(tgt.stype) + self.chain_emb(tgt.chain.clamp(max=MAX_CHAINS))
                + self.sub_emb(tgt.sub) + self.step_emb(steps)[None])

    def train_logprobs(self, enc: EncoderOutput, cond: Conditioning, tgt: TargetBatch) -> Tensor:
        check_bool_mask(tgt.pad, "tgt.pad")
        check_bool_mask(tgt.blocked, "tgt.blocked")
        x = self._step_inputs(tgt)
        B, T, _ = x.shape
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        if self.input == "peaks":
            if enc.memory is None:
                raise ValueError("input: peaks needs an encoder with per-peak output (decoder spec 13)")
            check_bool_mask(enc.memory_pad, "memory_pad")
            mem = torch.cat([enc.memory, cond.tokens], dim=1)
            mem_pad = torch.cat([enc.memory_pad, torch.zeros(cond.tokens.shape[:2], dtype=torch.bool,
                                                             device=x.device)], dim=1)
            attn_mask, key_pad, P = causal, tgt.pad, 0
        else:
            prefix = torch.cat([enc.pooled.unsqueeze(1) + self.prefix_type.weight[0],
                                cond.tokens + self.prefix_type.weight[1]], dim=1)
            P = prefix.shape[1]
            x = torch.cat([prefix, x], dim=1)
            attn_mask = torch.zeros(P + T, P + T, dtype=torch.bool, device=x.device)
            attn_mask[:P, P:] = True                 # prefix never sees the slots
            attn_mask[P:, P:] = causal
            key_pad = torch.cat([torch.zeros(B, P, dtype=torch.bool, device=x.device), tgt.pad], dim=1)
            mem = mem_pad = None
        # padded query rows would attend to nothing under key padding + causal mask; let them see
        # position 0 so softmax stays finite (their outputs are discarded)
        for blk in self.blocks:
            x = blk(x, attn_mask, _safe_pad(key_pad), mem, mem_pad)
        h = self.ln_out(x[:, P:])
        return masked_log_softmax(self.out(h), tgt.blocked)


def _safe_pad(key_pad: Tensor) -> Tensor:
    kp = key_pad.clone()
    kp[:, 0] = False
    return kp
