"""SlotSchema: block order as config, slot vocabularies, tokenizer and grammar (decoder spec 3, 4).

Nothing outside the blocks refers to a block position: tokenizer, grammar, output layer, decoding
and rendering all loop over `schema.blocks`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from duo_lipa.labels.goslin import class_table, render
from duo_lipa.labels.structure import MOD_TYPE_INDEX, MOD_TYPES, LipidStructure
from duo_lipa.schema import feasibility as F
from duo_lipa.schema.blocks import ADDUCTS, BLOCK_REGISTRY, UNK, SlotSpec, State

DEFAULT_BLOCKS = ["adduct", "class", "species", "chains", "dpos", "mods", "sn", "geom", "stereo"]
FIXED_PREFIX = ["adduct", "class", "species"]  # the mass check works on exactly these (spec 3.7)


class SchemaError(ValueError):
    pass


class LabelGrammarError(ValueError):
    """A label breaks a rule that no real lipid breaks (label audit, decoder spec 13)."""


def slot_values() -> dict[str, list]:
    return {
        "ADDUCT": list(ADDUCTS),
        "CLASS": list(class_table()),
        "LINK": ["none", "O-", "P-"],
        "SUM_C": list(range(F.SUM_C_MAX + 1)),
        "SUM_DB": list(range(F.SUM_DB_MAX + 1)),
        "SUM_OX": list(range(F.SUM_OX_MAX + 1)),
        "C": list(range(F.C_MAX + 1)),
        "DB": list(range(F.DB_MAX + 1)),
        "OX": list(range(F.OX_MAX + 1)),
        "DPOS": list(range(1, F.POS_MAX + 1)),
        "MOD_TYPE": list(MOD_TYPES),
        "MOD_POS": list(range(1, F.POS_MAX + 1)),
        "SN": [1, 2, 3, 4],
        "GEOM": ["Z", "E"],
        "STEREO": ["R", "S"],
    }


# Slot types H predicts per (chain, position) instead of per slot (decoder spec 7.3).
SET_VALUED = {"DPOS", "MOD_TYPE", "MOD_POS", "GEOM", "STEREO"}
CHAIN_INDEXED = {"C", "DB", "OX", "SN", "DPOS", "MOD_TYPE", "MOD_POS", "GEOM"}


@dataclass
class SlotToken:
    spec: SlotSpec
    value: object             # slot value, or UNK
    supervised: bool          # known and not forced
    forced: bool              # the grammar allows exactly one value
    legal: list               # allowed values (grammar mask)
    pos: int = 0              # GEOM: bond position; STEREO: chain carbon position (H indexing)


class SlotSchema:
    def __init__(self, blocks: list[str] | None = None):
        blocks = list(blocks or DEFAULT_BLOCKS)
        self._validate(blocks)
        self.block_names = blocks
        self.blocks = [BLOCK_REGISTRY[b]() for b in blocks]
        self.block_by_name = {b.name: b for b in self.blocks}
        vals = slot_values()
        self.slot_types: list[str] = []
        for b in self.blocks:
            for t in b.slot_types:
                if t not in self.slot_types:
                    self.slot_types.append(t)
        self.values = {t: vals[t] for t in self.slot_types}
        self.index = {t: {v: i for i, v in enumerate(vs)} for t, vs in self.values.items()}
        self.offset, off = {}, 0
        for t in self.slot_types:
            self.offset[t] = off
            off += len(self.values[t])
        self.V = off
        self.type_id = {t: i for i, t in enumerate(self.slot_types)}

    @staticmethod
    def _validate(blocks: list[str]):
        for b in blocks:
            if b not in BLOCK_REGISTRY:
                raise SchemaError(f"unknown block {b!r}; known: {sorted(BLOCK_REGISTRY)}")
        if len(set(blocks)) != len(blocks):
            raise SchemaError("duplicate block in schema")
        if blocks[: len(FIXED_PREFIX)] != FIXED_PREFIX:
            raise SchemaError(f"schema must start with {FIXED_PREFIX} (mass check, decoder spec 3.7)")
        seen: set[str] = set()
        for b in blocks:
            missing = BLOCK_REGISTRY[b].requires - seen
            if missing:
                raise SchemaError(f"block {b!r} must come after {sorted(missing)}; order given: {blocks}")
            seen.add(b)

    @property
    def hash(self) -> str:
        payload = {"blocks": self.block_names, "values": {t: [str(v) for v in vs] for t, vs in self.values.items()}}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    # ---------------------------------------------------------------------------------- grammar
    def next_slot(self, st: State) -> SlotSpec | None:
        for b in self.blocks:
            sp = b.next_slot(st)
            if sp is not None:
                return sp
        return None

    def legal(self, st: State, spec: SlotSpec) -> list:
        return self.block_by_name[spec.block].legal(st, spec)

    def global_id(self, stype: str, value) -> int:
        return self.offset[stype] + self.index[stype][value]

    def legal_ids(self, stype: str, legal: list) -> list[int]:
        return [self.global_id(stype, v) for v in legal]

    # -------------------------------------------------------------------------------- tokenizer
    def chain_sort_key(self, ch) -> tuple:
        """Canonical chain order under this block order (D-G3)."""
        rank = {"lcb": 0, "ether": 1, "acyl": 2}
        if ch.kind != "acyl":
            return (rank[ch.kind],)
        key: list = [rank[ch.kind], (ch.c, ch.db, ch.ox)]
        for b in self.block_names:
            if b == "dpos":
                key.append(tuple(p if p is not None else -1 for p in (ch.dpos or [])))
            elif b == "mods":
                key.append(tuple(x for t, p in (ch.mods or []) for x in (MOD_TYPE_INDEX[t], p)))
            elif b == "sn":
                key.append(ch.sn if ch.sn is not None else 99)
        return tuple(key)

    def tokenize(self, s: LipidStructure, adduct_given: bool = False) -> list[SlotToken]:
        """Slots for one label, truncated after its last known slot (decoder spec 3.5)."""
        if s.chains is not None:
            s = LipidStructure(**{**s.__dict__})
            old = list(s.chains)
            s.chains = sorted(old, key=self.chain_sort_key)
            new_ids = [id(ch) for ch in s.chains]
            remap = {i + 1: new_ids.index(id(ch)) + 1 for i, ch in enumerate(old)}
            s.stereo = {(("chain", remap[k[1]], k[2]) if k[0] == "chain" else k): v for k, v in s.stereo.items()}
        st = State()
        toks: list[SlotToken] = []
        last_known = -1
        while (spec := self.next_slot(st)) is not None:
            block = self.block_by_name[spec.block]
            val = block.label_value(s, st, spec)
            legal = block.legal(st, spec)
            if spec.stype == "ADDUCT" and adduct_given and val is not UNK:
                legal = [val]
            forced = len(legal) == 1
            if val is not UNK and val not in legal:
                raise LabelGrammarError(f"{render(s)}: slot {spec} value {val!r} not in grammar {legal[:20]}")
            pos = 0
            if spec.stype == "GEOM":
                p = st.v("DPOS", "dpos", spec.chain, spec.sub)
                pos = p if st.known(p) else 0
            elif spec.stype == "STEREO" and spec.extra and spec.extra[0] == "chain":
                pos = spec.extra[2]
            toks.append(SlotToken(spec, val, supervised=(val is not UNK and not forced), forced=forced,
                                  legal=legal, pos=pos))
            st.set(spec, val)
            if val is not UNK:
                last_known = len(toks) - 1
        return toks[: last_known + 1]

    def detokenize(self, toks_or_state) -> LipidStructure:
        st = toks_or_state if isinstance(toks_or_state, State) else self._state_from(toks_or_state)
        cls = st.v("CLASS", "class")
        s = LipidStructure(cls=cls)
        s.link = None
        for b in self.blocks:
            b.write(st, s)
        if s.link is None:
            s.link = "none"
        return s

    def _state_from(self, toks: list[SlotToken]) -> State:
        st = State()
        for t in toks:
            st.set(t.spec, t.value)
        return st

    def block_boundaries(self, toks: list[SlotToken]) -> dict[str, int]:
        """block name -> number of tokens up to and including that block."""
        out, n = {}, 0
        for b in self.block_names:
            n += sum(1 for t in toks if t.spec.block == b)
            out[b] = n
        return out
