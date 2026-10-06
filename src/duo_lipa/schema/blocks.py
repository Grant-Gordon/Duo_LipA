"""Slot blocks (decoder spec 3.2, 3.7). Each block is self-contained: its slot types, which slots it
has given what is decided so far, its grammar (`legal`), how it reads a label, and how it writes
its values back into a `LipidStructure` for rendering.

The grammar encodes only rules a real lipid's correct annotation never breaks (decoder spec 8.2),
and every `legal` set keeps at least one valid completion (no dead ends, decoder spec 4.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from duo_lipa.labels.goslin import ClassInfo, class_table, supported_classes
from duo_lipa.labels.structure import CHIRAL_MODS, MOD_TYPE_INDEX, MOD_TYPES, Chain, LipidStructure
from duo_lipa.schema import feasibility as F


class _Unk:
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self):
        return "UNK"


UNK = _Unk()

ADDUCTS = [
    "[M+H]+", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+H-H2O]+", "[M]+", "[M+2H]2+", "[M+Li]+",
    "[M-H]-", "[M+HCOO]-", "[M+CH3COO]-", "[M+Cl]-", "[M-CH3]-", "[M-2H]2-", "[M-H2O-H]-",
]


@dataclass(frozen=True)
class SlotSpec:
    stype: str
    block: str
    chain: int = 0          # 1-based chain index, 0 = not chain-indexed
    sub: int = 0            # 1-based bond / modification index, 0 = none
    extra: tuple = ()       # stereo centre key


class State:
    """Values decided so far, keyed by slot. A value may be UNK (label does not say)."""

    def __init__(self):
        self.vals: dict[SlotSpec, Any] = {}
        self.order: list[SlotSpec] = []

    def set(self, spec: SlotSpec, v: Any):
        self.vals[spec] = v
        self.order.append(spec)

    def has(self, spec: SlotSpec) -> bool:
        return spec in self.vals

    def get(self, spec: SlotSpec, default=None):
        return self.vals.get(spec, default)

    def copy(self) -> "State":
        s = State()
        s.vals = dict(self.vals)
        s.order = list(self.order)
        return s

    # convenience accessors -----------------------------------------------------------------
    def v(self, stype: str, block: str, chain: int = 0, sub: int = 0, extra: tuple = ()):
        return self.vals.get(SlotSpec(stype, block, chain, sub, extra))

    def known(self, x) -> bool:
        return x is not None and x is not UNK

    def class_info(self) -> ClassInfo | None:
        c = self.v("CLASS", "class")
        return class_table()[c] if self.known(c) else None

    def chain_kinds(self) -> list[str]:
        info = self.class_info()
        if info is None:
            return []
        n = info.n_chains
        if n == 0:
            return []
        link = self.v("LINK", "species")
        if info.category == "SP":
            first = "lcb"
        elif link in ("O-", "P-"):
            first = "ether"
        else:
            first = "acyl"
        return [first] + ["acyl"] * (n - 1)

    def chain_cdo(self, k: int):
        return tuple(self.v(t, "chains", k) for t in ("C", "DB", "OX"))


def _cdo_known(st: State, k: int) -> tuple[int, int, int] | None:
    x = st.chain_cdo(k)
    return x if all(st.known(a) for a in x) else None


class Block:
    name: str = ""
    slot_types: list[str] = []
    requires: set[str] = set()

    def next_slot(self, st: State) -> SlotSpec | None:
        raise NotImplementedError

    def legal(self, st: State, spec: SlotSpec) -> list:
        raise NotImplementedError

    def label_value(self, s: LipidStructure, st: State, spec: SlotSpec):
        raise NotImplementedError

    def write(self, st: State, s: LipidStructure):
        pass

    def _first_unset(self, st: State, specs) -> SlotSpec | None:
        for sp in specs:
            if not st.has(sp):
                return sp
        return None


# --------------------------------------------------------------------------------------- blocks

class AdductBlock(Block):
    name, slot_types = "adduct", ["ADDUCT"]

    def next_slot(self, st):
        return self._first_unset(st, [SlotSpec("ADDUCT", self.name)])

    def legal(self, st, spec):
        return list(ADDUCTS)

    def label_value(self, s, st, spec):
        return s.adduct if s.adduct is not None else UNK

    def write(self, st, s):
        a = st.v("ADDUCT", self.name)
        s.adduct = a if st.known(a) else None


class ClassBlock(Block):
    name, slot_types = "class", ["CLASS"]

    def next_slot(self, st):
        return self._first_unset(st, [SlotSpec("CLASS", self.name)])

    def legal(self, st, spec):
        sup = supported_classes()
        return [c for c in class_table() if c in sup]

    def label_value(self, s, st, spec):
        return s.cls


class SpeciesBlock(Block):
    name, slot_types, requires = "species", ["LINK", "SUM_C", "SUM_DB", "SUM_OX"], {"class"}

    def next_slot(self, st):
        if st.class_info() is None:
            return None
        return self._first_unset(st, [SlotSpec(t, self.name) for t in self.slot_types])

    def legal(self, st, spec):
        info = st.class_info()
        if spec.stype == "LINK":
            return ["none", "O-", "P-"] if info.ether_ok and info.n_chains >= 1 else ["none"]
        Fz = F.sums_feasible(tuple(st.chain_kinds()))
        c = st.v("SUM_C", self.name)
        db = st.v("SUM_DB", self.name)
        if spec.stype == "SUM_C":
            return [i for i in range(F.SUM_C_MAX + 1) if Fz[i].any()]
        if not st.known(c):
            return list(range(F.SUM_DB_MAX + 1)) if spec.stype == "SUM_DB" else list(range(F.SUM_OX_MAX + 1))
        if spec.stype == "SUM_DB":
            return [i for i in range(F.SUM_DB_MAX + 1) if Fz[c, i].any()]
        if not st.known(db):
            return list(range(F.SUM_OX_MAX + 1))
        return [i for i in range(F.SUM_OX_MAX + 1) if Fz[c, db, i]]

    def label_value(self, s, st, spec):
        v = {"LINK": s.link, "SUM_C": s.sum_c, "SUM_DB": s.sum_db, "SUM_OX": s.sum_ox}[spec.stype]
        return UNK if v is None else v

    def write(self, st, s):
        for t, attr in (("LINK", "link"), ("SUM_C", "sum_c"), ("SUM_DB", "sum_db"), ("SUM_OX", "sum_ox")):
            v = st.v(t, self.name)
            setattr(s, attr, v if st.known(v) else None)


@lru_cache(maxsize=200_000)
def _chain_legal(stype: str, kind: str, typed_after: tuple, m_after: int, lo: tuple, R: tuple,
                 c: int | None, db: int | None) -> tuple:
    acyl_lo = (F.MIN_C["acyl"], 0, 0)

    def ok(x):
        if not F.in_domain(kind, *x):
            return False
        if kind == "acyl":
            if x < lo:
                return False
            return F.acyl_feasible(m_after, x, (R[0] - x[0], R[1] - x[1], R[2] - x[2]))
        return F.rest_feasible(typed_after, m_after, acyl_lo, (R[0] - x[0], R[1] - x[1], R[2] - x[2]))

    olo = F.OX_RANGE[kind][0]
    if stype == "C":
        out = []
        for cc in range(0, F.C_MAX + 1):
            if any(ok((cc, d, o)) for d in range(0, F.db_max(cc) + 1) for o in range(olo, F.ox_max(kind, cc) + 1)):
                out.append(cc)
        return tuple(out)
    if stype == "DB":
        return tuple(d for d in range(0, F.DB_MAX + 1)
                     if any(ok((c, d, o)) for o in range(olo, F.OX_MAX + 1)))
    return tuple(o for o in range(0, F.OX_MAX + 1) if ok((c, db, o)))


class ChainsBlock(Block):
    name, slot_types, requires = "chains", ["C", "DB", "OX"], {"species"}

    def _specs(self, st):
        return [SlotSpec(t, self.name, k) for k in range(1, len(st.chain_kinds()) + 1) for t in self.slot_types]

    def next_slot(self, st):
        sums = [st.v(t, "species") for t in ("SUM_C", "SUM_DB", "SUM_OX")]
        if not all(st.known(x) for x in sums):
            return None
        return self._first_unset(st, self._specs(st))

    def legal(self, st, spec):
        kinds = st.chain_kinds()
        k = spec.chain
        R = [st.v(t, "species") for t in ("SUM_C", "SUM_DB", "SUM_OX")]
        for j in range(1, k):
            x = _cdo_known(st, j)
            if x is None:  # an earlier chain is unknown: no constraint can be computed
                return list(range({"C": F.C_MAX, "DB": F.DB_MAX, "OX": F.OX_MAX}[spec.stype] + 1))
            R = [R[0] - x[0], R[1] - x[1], R[2] - x[2]]
        kind = kinds[k - 1]
        after = kinds[k:]
        typed_after = tuple(a for a in after if a != "acyl")
        m_after = sum(1 for a in after if a == "acyl")
        lo = (F.MIN_C["acyl"], 0, 0)
        if kind == "acyl" and k > 1 and kinds[k - 2] == "acyl":
            lo = _cdo_known(st, k - 1)
        c = st.v("C", self.name, k)
        db = st.v("DB", self.name, k)
        if (spec.stype == "DB" and not st.known(c)) or (spec.stype == "OX" and not (st.known(c) and st.known(db))):
            return list(range({"DB": F.DB_MAX, "OX": F.OX_MAX}[spec.stype] + 1))
        return list(_chain_legal(spec.stype, kind, typed_after, m_after, tuple(lo), tuple(R),
                                 c if st.known(c) else None, db if st.known(db) else None))

    def label_value(self, s, st, spec):
        if s.chains is None:
            return UNK
        ch = s.chains[spec.chain - 1]
        return {"C": ch.c, "DB": ch.db, "OX": ch.ox}[spec.stype]

    def write(self, st, s):
        kinds = st.chain_kinds()
        if not kinds or not all(st.has(sp) for sp in self._specs(st)):
            return
        chains = []
        for k, kind in enumerate(kinds, start=1):
            c, db, ox = st.chain_cdo(k)
            if not all(st.known(a) for a in (c, db, ox)):
                return
            chains.append(Chain(kind, c, db, ox, dpos=[] if db == 0 else None, geom=[] if db == 0 else None,
                                mods=[] if (ox == 0 or kind == "lcb") else None,
                                mod_stereo=[] if (ox == 0 or kind == "lcb") else None))
        s.chains = chains


def _acyl_tied(st: State, k: int, upto_block: str, upto: tuple | None = None) -> bool:
    """Are chains k-1 and k still tied in canonical order, given everything decided so far?
    Chains already distinguished by sn are never tied (alternative block orders, D-G3)."""
    kinds = st.chain_kinds()
    if k < 2 or kinds[k - 1] != "acyl" or kinds[k - 2] != "acyl":
        return False
    a, b = _cdo_known(st, k - 1), _cdo_known(st, k)
    if a is None or a != b:
        return False
    sa, sb = st.v("SN", "sn", k - 1), st.v("SN", "sn", k)
    if st.known(sa) and st.known(sb):
        return False
    if upto_block in ("mods",):  # mods tie also needs equal double-bond positions
        for j in range(1, a[1] + 1):
            if st.v("DPOS", "dpos", k - 1, j) != st.v("DPOS", "dpos", k, j):
                return False
    return True


class DposBlock(Block):
    name, slot_types, requires = "dpos", ["DPOS"], {"chains"}

    def _specs(self, st):
        out = []
        for k in range(1, len(st.chain_kinds()) + 1):
            db = st.v("DB", "chains", k)
            if st.known(db):
                out += [SlotSpec("DPOS", self.name, k, j) for j in range(1, db + 1)]
        return out

    def next_slot(self, st):
        return self._first_unset(st, self._specs(st))

    def legal(self, st, spec):
        k, j = spec.chain, spec.sub
        c, db = st.v("C", "chains", k), st.v("DB", "chains", k)
        maxp = min(c - 1, F.POS_MAX)
        prev = st.v("DPOS", self.name, k, j - 1) if j > 1 else 0
        low = (prev + 1) if st.known(prev) else j
        high = maxp - (db - j)
        if _acyl_tied(st, k, "dpos"):
            same_prefix = all(st.v("DPOS", self.name, k - 1, i) == st.v("DPOS", self.name, k, i)
                              and st.known(st.v("DPOS", self.name, k, i)) for i in range(1, j))
            ref = st.v("DPOS", self.name, k - 1, j)
            if same_prefix and st.known(ref):
                low = max(low, ref)
        return list(range(low, high + 1))

    def label_value(self, s, st, spec):
        ch = s.chains[spec.chain - 1] if s.chains else None
        if ch is None or ch.dpos is None:
            return UNK
        v = ch.dpos[spec.sub - 1]
        return UNK if v is None else v

    def write(self, st, s):
        if not s.chains:
            return
        for k, ch in enumerate(s.chains, start=1):
            if ch.db == 0:
                continue
            vals = [st.v("DPOS", self.name, k, j) for j in range(1, ch.db + 1)]
            if all(st.known(v) for v in vals):
                ch.dpos = vals


def _mod_cost(t) -> int:
    return MOD_TYPES[t] if t in MOD_TYPES else 1  # UNK type: assume one oxygen (D-G5)


class ModsBlock(Block):
    name, slot_types, requires = "mods", ["MOD_TYPE", "MOD_POS"], {"chains"}

    def _chain_specs(self, st, k):
        """Pairs (MOD_TYPE, MOD_POS) until chain k's oxygen budget is used."""
        ox = st.v("OX", "chains", k)
        if not st.known(ox) or ox == 0:
            return []
        out, used, i = [], 0, 1
        while used < ox:
            t_spec = SlotSpec("MOD_TYPE", self.name, k, i)
            out.append(t_spec)
            out.append(SlotSpec("MOD_POS", self.name, k, i))
            if not st.has(t_spec):
                break
            used += _mod_cost(st.get(t_spec))
            i += 1
        return out

    def next_slot(self, st):
        kinds = st.chain_kinds()
        for k in range(1, len(kinds) + 1):
            if kinds[k - 1] == "lcb":
                continue  # sphingoid-base oxygens are implicit (D-L3)
            sp = self._first_unset(st, self._chain_specs(st, k))
            if sp is not None:
                return sp
        return None

    def _budget(self, st, k, i):
        ox = st.v("OX", "chains", k)
        used = sum(_mod_cost(st.v("MOD_TYPE", self.name, k, a)) for a in range(1, i))
        prev = st.v("MOD_POS", self.name, k, i - 1) if i > 1 else 0
        q = prev if st.known(prev) else i - 1
        P = min(st.v("C", "chains", k), F.POS_MAX)
        return ox - used, q, P

    def _tie_ref(self, st, spec):
        k, i = spec.chain, spec.sub
        if not _acyl_tied(st, k, "mods"):
            return None
        for a in range(1, i):
            for t in ("MOD_TYPE", "MOD_POS"):
                if st.v(t, self.name, k - 1, a) != st.v(t, self.name, k, a):
                    return None
        if spec.stype == "MOD_POS" and st.v("MOD_TYPE", self.name, k - 1, i) != st.v("MOD_TYPE", self.name, k, i):
            return None
        ref = st.v(spec.stype, self.name, k - 1, i)
        return ref if st.known(ref) else None

    def legal(self, st, spec):
        k, i = spec.chain, spec.sub
        rem, q, P = self._budget(st, k, i)
        ref = self._tie_ref(st, spec)
        if spec.stype == "MOD_TYPE":
            out = [t for t in MOD_TYPES if MOD_TYPES[t] <= rem and q + 1 <= P - -(-(rem - MOD_TYPES[t]) // 2)]
            if ref is not None:
                out = [t for t in out if MOD_TYPE_INDEX[t] >= MOD_TYPE_INDEX[ref]]
            return out
        t = st.v("MOD_TYPE", self.name, k, i)
        after = rem - _mod_cost(t)
        high = P - (-(-after // 2))
        low = q + 1
        if ref is not None:
            low = max(low, ref)
        return list(range(low, high + 1))

    def label_value(self, s, st, spec):
        ch = s.chains[spec.chain - 1] if s.chains else None
        if ch is None or ch.mods is None or spec.sub > len(ch.mods):
            return UNK
        t, p = ch.mods[spec.sub - 1]
        return t if spec.stype == "MOD_TYPE" else p

    def write(self, st, s):
        if not s.chains:
            return
        for k, ch in enumerate(s.chains, start=1):
            if ch.kind == "lcb" or ch.ox == 0:
                continue
            specs = self._chain_specs(st, k)
            if not specs or not all(st.has(sp) for sp in specs):
                continue
            pairs = [(st.get(specs[a]), st.get(specs[a + 1])) for a in range(0, len(specs), 2)]
            if all(st.known(t) and st.known(p) for t, p in pairs):
                ch.mods = pairs
                ch.mod_stereo = [None] * len(pairs)


def _chain_identity(st: State, k: int):
    """Everything known about chain k that canonical order compares (UNK compares equal)."""
    x = _cdo_known(st, k)
    if x is None:
        return None
    dpos = tuple(st.v("DPOS", "dpos", k, j) for j in range(1, x[1] + 1))
    mods = tuple(st.v(t, "mods", k, i) for i in range(1, x[2] + 1) for t in ("MOD_TYPE", "MOD_POS"))
    return x, dpos, mods


def _identical(st: State, a: int, b: int) -> bool:
    kinds = st.chain_kinds()
    if kinds[a - 1] != "acyl" or kinds[b - 1] != "acyl":
        return False
    ia, ib = _chain_identity(st, a), _chain_identity(st, b)
    if ia is None or ib is None or ia[0] != ib[0]:
        return False
    for xa, xb in zip(ia[1] + ia[2], ib[1] + ib[2]):
        if st.known(xa) and st.known(xb) and xa != xb:
            return False
    return True


class SnBlock(Block):
    name, slot_types, requires = "sn", ["SN"], {"chains"}

    def next_slot(self, st):
        n = len(st.chain_kinds())
        if n == 0:
            return None
        return self._first_unset(st, [SlotSpec("SN", self.name, k) for k in range(1, n + 1)])

    def legal(self, st, spec):
        info = st.class_info()
        k = spec.chain
        n = len(st.chain_kinds())
        if info.category == "SP":
            return [k]  # sphingoid base sn-1, N-acyl sn-2 (forced)
        used = {st.v("SN", self.name, j) for j in range(1, k)}
        free = [p for p in range(1, max(info.n_positions, n) + 1) if p not in used]
        low = 0
        if _identical(st, k - 1, k) if k > 1 else False:
            prev = st.v("SN", self.name, k - 1)
            low = prev if st.known(prev) else 0
        s_after = 0
        for j in range(k + 1, n + 1):
            if _identical(st, k, j):
                s_after += 1
            else:
                break
        return [p for p in free if p > low and sum(1 for f in free if f > p) >= s_after]

    def label_value(self, s, st, spec):
        info = class_table()[s.cls]
        if info.category == "SP":
            return spec.chain
        if not s.sn_known or s.chains is None:
            return UNK
        return s.chains[spec.chain - 1].sn

    def write(self, st, s):
        if not s.chains:
            return
        vals = [st.v("SN", self.name, k) for k in range(1, len(s.chains) + 1)]
        if all(st.known(v) for v in vals):
            for ch, v in zip(s.chains, vals):
                ch.sn = v
            s.sn_known = class_table()[s.cls].category != "SP"


class GeomBlock(Block):
    name, slot_types, requires = "geom", ["GEOM"], {"dpos"}

    def next_slot(self, st):
        specs = []
        for k in range(1, len(st.chain_kinds()) + 1):
            db = st.v("DB", "chains", k)
            if st.known(db):
                specs += [SlotSpec("GEOM", self.name, k, j) for j in range(1, db + 1)]
        return self._first_unset(st, specs)

    def legal(self, st, spec):
        return ["Z", "E"]

    def label_value(self, s, st, spec):
        ch = s.chains[spec.chain - 1] if s.chains else None
        if ch is None or ch.geom is None:
            return UNK
        v = ch.geom[spec.sub - 1]
        return UNK if v is None else v

    def write(self, st, s):
        if not s.chains:
            return
        for k, ch in enumerate(s.chains, start=1):
            if ch.db == 0 or ch.dpos is None:
                continue
            vals = [st.v("GEOM", self.name, k, j) for j in range(1, ch.db + 1)]
            ch.geom = [v if st.known(v) else None for v in vals]


BACKBONE_CENTRES = (("glycerol",), ("lcb", 2), ("lcb", 3), ("lcb", 4))


def stereo_centres(st: State) -> list[tuple]:
    """Stereocentres implied by earlier blocks, in a fixed order (decoder spec 3.2). Rule table keyed
    on class category and modification type. Not cross-checked against RDKit (D-G6)."""
    info = st.class_info()
    if info is None:
        return []
    kinds = st.chain_kinds()
    out: list[tuple] = []
    if info.category == "GP" and info.n_positions >= 2:
        out.append(("glycerol",))
    elif info.category == "GL" and info.n_positions >= 3:
        sn = {st.v("SN", "sn", k): k for k in range(1, len(kinds) + 1)}
        if all(st.known(p) for p in sn):
            a, b = sn.get(1), sn.get(3)
            ia = _chain_identity(st, a) if a else None
            ib = _chain_identity(st, b) if b else None
            if ia != ib:
                out.append(("glycerol",))
    if info.category == "SP" and kinds and kinds[0] == "lcb":
        out += [("lcb", 2), ("lcb", 3)]
        ox = st.v("OX", "chains", 1)
        if st.known(ox) and ox >= 3:
            out.append(("lcb", 4))
    for k in range(1, len(kinds) + 1):
        ox = st.v("OX", "chains", k)
        if not st.known(ox):
            continue
        i = 1
        while st.has(SlotSpec("MOD_TYPE", "mods", k, i)):
            t, p = st.v("MOD_TYPE", "mods", k, i), st.v("MOD_POS", "mods", k, i)
            if st.known(t) and st.known(p) and t in CHIRAL_MODS:
                out.append(("chain", k, p))
            i += 1
    return out


class StereoBlock(Block):
    name, slot_types, requires = "stereo", ["STEREO"], {"chains", "mods", "sn"}

    def next_slot(self, st):
        return self._first_unset(st, [SlotSpec("STEREO", self.name, extra=c) for c in stereo_centres(st)])

    def legal(self, st, spec):
        return ["R", "S"]

    def label_value(self, s, st, spec):
        v = s.stereo.get(spec.extra)
        return UNK if v is None else v

    def write(self, st, s):
        for c in stereo_centres(st):
            v = st.v("STEREO", self.name, extra=c)
            if st.known(v):
                s.stereo[c] = v
        if s.chains:
            for k, ch in enumerate(s.chains, start=1):
                if ch.mods:
                    ch.mod_stereo = [s.stereo.get(("chain", k, p)) for _, p in ch.mods]


BLOCK_REGISTRY: dict[str, type[Block]] = {
    b.name: b for b in (AdductBlock, ClassBlock, SpeciesBlock, ChainsBlock, DposBlock, ModsBlock,
                        SnBlock, GeomBlock, StereoBlock)
}
