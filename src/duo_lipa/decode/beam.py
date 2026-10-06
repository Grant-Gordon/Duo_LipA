"""Grammar-masked beam search shared by every decoder (decoder spec 9.1, 9.3).

v0 simplifications (IMPLEMENTATION.md D-I1, D-I2):
- the beam is pruned to width k after every slot, not only at block boundaries;
- the precursor-mass check (9.2) is not implemented;
- reporting depth uses tau on the level's best probability, as in 9.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from duo_lipa.labels.goslin import render
from duo_lipa.model.decoders.base import SlotDecoder, collate_targets
from duo_lipa.schema.blocks import UNK, State
from duo_lipa.schema.schema import SlotSchema, SlotToken


@dataclass
class Hyp:
    state: State
    tokens: list[SlotToken]
    score: float = 0.0
    done: bool = False


@dataclass
class DecodeResult:
    levels: dict[str, list[tuple[str, float]]] = field(default_factory=dict)  # block -> [(goslin, prob)]
    best: str = ""
    best_tokens: list[SlotToken] = field(default_factory=list)
    best_score: float = 0.0


def _expand_state(schema: SlotSchema, h: Hyp, fixed: dict) -> tuple[SlotToken | None, list]:
    spec = schema.next_slot(h.state)
    if spec is None:
        return None, []
    legal = schema.legal(h.state, spec)
    if spec.stype in fixed:
        v = fixed[spec.stype]
        legal = [v] if (v is UNK or v in legal) else legal
    pos = 0
    if spec.stype == "GEOM":
        p = h.state.v("DPOS", "dpos", spec.chain, spec.sub)
        pos = p if h.state.known(p) else 0
    elif spec.stype == "STEREO" and spec.extra and spec.extra[0] == "chain":
        pos = spec.extra[2]
    tok = SlotToken(spec, UNK, supervised=False, forced=len(legal) == 1, legal=legal if legal != [UNK] else [], pos=pos)
    return tok, legal


@torch.no_grad()
def beam_search(decoder: SlotDecoder, enc, cond, b: int, k: int = 10, fixed: dict | None = None,
                max_steps: int = 256) -> DecodeResult:
    """Decode spectrum `b` of the batch. `fixed` maps a slot type to a value that is given (e.g. the
    adduct from metadata) or to UNK (decoder spec 9.3: "C=C unknown")."""
    schema = decoder.schema
    fixed = dict(fixed or {})
    ctx = decoder.start(enc, cond)
    beam = [Hyp(State(), [])]
    res = DecodeResult()
    boundary_seen: dict[str, list[Hyp]] = {}
    for _ in range(max_steps):
        live = [h for h in beam if not h.done]
        if not live:
            break
        cands: list[Hyp] = [h for h in beam if h.done]
        to_score, info = [], []
        for h in live:
            tok, legal = _expand_state(schema, h, fixed)
            if tok is None:
                h.done = True
                cands.append(h)
                continue
            if legal == [UNK]:                       # slot fed as UNK, no branching
                st = h.state.copy(); st.set(tok.spec, UNK)
                cands.append(Hyp(st, h.tokens + [tok], h.score))
                continue
            if len(legal) == 1:                      # forced: fill without branching
                st = h.state.copy(); st.set(tok.spec, legal[0])
                tok.value = legal[0]
                cands.append(Hyp(st, h.tokens + [tok], h.score))
                continue
            to_score.append(h.tokens + [tok])
            info.append((h, tok, legal))
        if to_score:
            batch = collate_targets(schema, to_score)
            lp = decoder.step_scores(ctx, batch, torch.full((len(to_score),), b, dtype=torch.long))
            for (h, tok, legal), row in zip(info, lp):
                ids = schema.legal_ids(tok.spec.stype, legal)
                vals = row[ids]
                top = torch.topk(vals, min(k, len(ids)))
                for sc, j in zip(top.values.tolist(), top.indices.tolist()):
                    st = h.state.copy(); st.set(tok.spec, legal[j])
                    t2 = SlotToken(tok.spec, legal[j], False, False, legal, tok.pos)
                    cands.append(Hyp(st, h.tokens + [t2], h.score + sc))
        cands.sort(key=lambda h: -h.score)
        beam = cands[:k]
        # record level answers at block boundaries
        for h in beam:
            nxt = schema.next_slot(h.state)
            last_block = h.tokens[-1].spec.block if h.tokens else None
            if last_block and last_block != "adduct" and (nxt is None or nxt.block != last_block):
                key = render(schema.detokenize(h.state))
                lst = boundary_seen.setdefault(last_block, [])
                if all(render(schema.detokenize(x.state)) != key for x in lst):
                    lst.append(h)
    for blk, hs in boundary_seen.items():
        hs = sorted(hs, key=lambda h: -h.score)[:k]
        res.levels[blk] = [(render(schema.detokenize(h.state)), float(torch.tensor(h.score).exp())) for h in hs]
    best = max(beam, key=lambda h: h.score)
    res.best, res.best_tokens, res.best_score = render(schema.detokenize(best.state)), best.tokens, best.score
    return res
