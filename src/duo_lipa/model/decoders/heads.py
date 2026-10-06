"""H decoder (decoder spec 7.3, 7.6): one independent classifier per slot type.

- Single-valued types: one softmax per slot. Chain-indexed types share a classifier across chains
  with a learned chain-index embedding.
- Set-valued types are predicted per (chain, position 1..39):
  DPOS -> one yes-vs-no logit per position; MOD (MOD_TYPE + MOD_POS share one classifier) -> softmax
  over {none} + modification types per position; GEOM -> Z/E per position; STEREO -> R/S per chain
  carbon position, plus one R/S output per named backbone centre.
- Classifier input: `pooled`: MLP_s(concat(pooled, mean(cond tokens), chain emb));
  `peaks`: CrossAttn(query_s + chain emb; K, V = memory ++ cond tokens) then MLP_s. No attention
  between slots.
"""

from __future__ import annotations

import torch
import torch.nn.functional as Fn
from torch import Tensor, nn

from duo_lipa.model.decoders.base import MAX_CHAINS, SlotDecoder, TargetBatch, _nll, masked_log_softmax
from duo_lipa.model.interfaces import Conditioning, EncoderOutput, check_bool_mask
from duo_lipa.schema import feasibility as F
from duo_lipa.schema.blocks import UNK
from duo_lipa.schema.schema import CHAIN_INDEXED, SlotSchema

NPOS = F.POS_MAX  # positions 1..39
N_BACKBONE = 4    # glycerol sn-2, sphingoid C2, C3, C4


class HeadsDecoder(SlotDecoder):
    def __init__(self, schema: SlotSchema, d_model: int, input: str = "pooled", hidden: int | None = None,
                 attn_layers: int = 1, n_heads: int = 4, dropout: float = 0.0):
        super().__init__(schema)
        self.input = input
        d = d_model
        hidden = hidden or 2 * d
        self.n_mod = len(schema.values.get("MOD_TYPE", []))
        # one classifier per slot type; MOD_TYPE and MOD_POS share "MOD"
        self.clf_names = []
        out_dims = {}
        for t in schema.slot_types:
            name = "MOD" if t in ("MOD_TYPE", "MOD_POS") else t
            if name in out_dims:
                continue
            self.clf_names.append(name)
            out_dims[name] = {
                "DPOS": NPOS,
                "MOD": NPOS * (1 + self.n_mod),
                "GEOM": (NPOS + 1) * 2,          # +1 row for "bond position unknown"
                "STEREO": NPOS * 2 + N_BACKBONE * 2,
            }.get(name, len(schema.values[t]) if t in schema.values else 0)
        self.out_dims = out_dims
        self.chain_emb = nn.Embedding(MAX_CHAINS + 1, d)
        in_dim = 3 * d if input == "pooled" else d
        self.mlp = nn.ModuleDict({n: nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                                   nn.Linear(hidden, out_dims[n])) for n in self.clf_names})
        if input == "peaks":
            self.query = nn.ParameterDict({n: nn.Parameter(torch.randn(d) * d ** -0.5) for n in self.clf_names})
            self.attn = nn.ModuleList([nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
                                       for _ in range(attn_layers)])
            self.attn_ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(attn_layers)])

    def _chain_indexed(self, name: str) -> bool:
        return name in CHAIN_INDEXED or name in ("MOD", "STEREO")

    def head_outputs(self, enc: EncoderOutput, cond: Conditioning) -> dict[str, Tensor]:
        """name -> (B, MAX_CHAINS+1, out_dim). Chain index 0 = not chain-indexed."""
        B = enc.pooled.shape[0]
        out = {}
        if self.input == "peaks":
            if enc.memory is None:
                raise ValueError("input: peaks needs an encoder with per-peak output (decoder spec 13)")
            check_bool_mask(enc.memory_pad, "memory_pad")
            mem = torch.cat([enc.memory, cond.tokens], dim=1)
            mem_pad = torch.cat([enc.memory_pad, torch.zeros(cond.tokens.shape[:2], dtype=torch.bool,
                                                             device=mem.device)], dim=1)
        for name in self.clf_names:
            n_ch = MAX_CHAINS + 1 if self._chain_indexed(name) else 1
            ce = self.chain_emb.weight[:n_ch]                                    # (n_ch, d)
            if self.input == "pooled":
                ctx = torch.cat([enc.pooled, cond.tokens.mean(1)], dim=-1)        # (B, 2d)
                x = torch.cat([ctx.unsqueeze(1).expand(-1, n_ch, -1), ce.unsqueeze(0).expand(B, -1, -1)], -1)
            else:
                q = (self.query[name] + ce).unsqueeze(0).expand(B, -1, -1)       # (B, n_ch, d)
                for ln, att in zip(self.attn_ln, self.attn):
                    q = q + att(ln(q), mem, mem, key_padding_mask=mem_pad, need_weights=False)[0]
                x = q
            h = self.mlp[name](x)
            if n_ch == 1:
                h = h.expand(-1, MAX_CHAINS + 1, -1)
            out[name] = h
        return out

    # ------------------------------------------------------------------------------- scoring
    def _scores(self, heads: dict[str, Tensor], tgt: TargetBatch) -> Tensor:
        sch = self.schema
        B, T = tgt.stype.shape
        scores = torch.full((B, T, sch.V), float("-inf"), device=tgt.stype.device)
        bidx = torch.arange(B, device=scores.device).unsqueeze(1).expand(B, T)
        chain = tgt.chain.clamp(max=MAX_CHAINS)
        prev_val = torch.cat([torch.full_like(tgt.value[:, :1], -1), tgt.value[:, :-1]], dim=1)
        for t in sch.slot_types:
            sel = (tgt.stype == sch.type_id[t]) & ~tgt.pad
            if not sel.any():
                continue
            b, c = bidx[sel], chain[sel]
            off, n = sch.offset[t], len(sch.values[t])
            if t == "DPOS":
                s = heads["DPOS"][b, c]                                             # (N, 39) yes-no logit
            elif t in ("MOD_TYPE", "MOD_POS"):
                m = heads["MOD"][b, c].view(-1, NPOS, 1 + self.n_mod)
                rel = m[..., 1:] - m[..., :1]                                       # (N, 39, n_mod) type - none
                if t == "MOD_TYPE":
                    s = torch.logsumexp(rel, dim=1)
                else:
                    ty = (prev_val[sel] - sch.offset["MOD_TYPE"]).clamp(0, self.n_mod - 1)
                    s = rel[torch.arange(rel.shape[0]), :, ty]
            elif t == "GEOM":
                g = heads["GEOM"][b, c].view(-1, NPOS + 1, 2)
                p = tgt.pos[sel].clamp(0, NPOS)                                    # 0 = unknown row
                s = g[torch.arange(g.shape[0]), p]
            elif t == "STEREO":
                st = heads["STEREO"][b, c]
                chain_part = st[:, : NPOS * 2].view(-1, NPOS, 2)
                bb = st[:, NPOS * 2:].view(-1, N_BACKBONE, 2)
                kind = tgt.centre[sel]
                p = (tgt.pos[sel] - 1).clamp(0, NPOS - 1)
                rows = torch.arange(st.shape[0])
                s = torch.where((kind == 5).unsqueeze(-1), chain_part[rows, p],
                                bb[rows, (kind - 1).clamp(0, N_BACKBONE - 1)])
            else:
                s = heads[t][b, c]
            sc = scores[sel]
            sc[:, off:off + n] = s
            scores[sel] = sc
        return scores

    def train_logprobs(self, enc, cond, tgt):
        check_bool_mask(tgt.blocked, "tgt.blocked")
        return masked_log_softmax(self._scores(self.head_outputs(enc, cond), tgt), tgt.blocked)

    def start(self, enc, cond):
        return {"enc": enc, "cond": cond, "heads": self.head_outputs(enc, cond)}

    def step_scores(self, ctx, prefix, batch_index):
        heads = {k: v[batch_index] for k, v in ctx["heads"].items()}
        lp = masked_log_softmax(self._scores(heads, prefix), prefix.blocked)
        last = (~prefix.pad).sum(1) - 1
        return lp[torch.arange(lp.shape[0]), last]

    # --------------------------------------------------------------------------------- loss
    def loss(self, enc, cond, tgt):
        """Single-valued slots, GEOM and STEREO: cross-entropy at supervised slots (grammar-masked).
        DPOS and MOD: per-position cross-entropy over every position of each chain whose set the
        label resolves (decoder spec 7.3)."""
        heads = self.head_outputs(enc, cond)
        lp = masked_log_softmax(self._scores(heads, tgt), tgt.blocked)
        sch = self.schema
        set_types = {sch.type_id[t] for t in ("DPOS", "MOD_TYPE", "MOD_POS") if t in sch.type_id}
        is_set = torch.zeros_like(tgt.supervised)
        for tid in set_types:
            is_set |= tgt.stype == tid
        out = _nll(lp, tgt, tgt.supervised & ~is_set)
        set_terms = []
        for b, toks in enumerate(tgt.tokens):
            chains = {}
            for tok in toks:
                sp = tok.spec
                if sp.stype in ("DPOS", "MOD_TYPE", "MOD_POS"):
                    chains.setdefault((sp.stype == "DPOS", sp.chain), []).append(tok)
            for (is_dpos, k), ts in chains.items():
                if any(t.value is UNK for t in ts):
                    continue
                c_len = _chain_len(toks, k)
                if is_dpos:
                    n = min(c_len - 1, NPOS)
                    y = torch.zeros(NPOS)
                    for t in ts:
                        y[t.value - 1] = 1.0
                    logit = heads["DPOS"][b, min(k, MAX_CHAINS)]
                    set_terms.append(Fn.binary_cross_entropy_with_logits(logit[:n], y[:n], reduction="sum") / n)
                else:
                    n = min(c_len, NPOS)
                    y = torch.zeros(NPOS, dtype=torch.long)
                    for a in range(0, len(ts), 2):
                        ty, p = ts[a].value, ts[a + 1].value
                        y[p - 1] = sch.index["MOD_TYPE"][ty] + 1
                    logit = heads["MOD"][b, min(k, MAX_CHAINS)].view(NPOS, 1 + self.n_mod)
                    set_terms.append(Fn.cross_entropy(logit[:n], y[:n], reduction="mean"))
        if set_terms:
            out["loss"] = out["loss"] + torch.stack(set_terms).mean()
            out["n_set_chains"] = len(set_terms)
        return out


def _chain_len(toks, k: int) -> int:
    for t in toks:
        if t.spec.stype == "C" and t.spec.chain == k:
            return t.value
    return NPOS + 1
