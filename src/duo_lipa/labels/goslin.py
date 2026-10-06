"""Goslin label parsing (via pygoslin) into `LipidStructure`, and rendering back to shorthand.

pygoslin is used to parse labels and to validate rendered strings. Rendering is our own, because
pygoslin's string renderer drops double-bond positions at molecular-species level
(`PC 16:0_18:1(9)` -> `PC 16:0_18:1`, checked 2026-10-06 with pygoslin 2.2.5), and "C=C known, sn
unknown" is exactly the label form UVPD produces (decoder spec 3.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from duo_lipa.labels.structure import CHIRAL_MODS, MOD_TYPES, Chain, LipidStructure, canonical_order


class UnsupportedLabel(ValueError):
    pass


@dataclass(frozen=True)
class ClassInfo:
    name: str
    category: str        # "GL" | "GP" | "SP" | "ST" | "FA" | ...
    n_chains: int        # pygoslin poss_fa: chains a label of this class carries
    n_positions: int     # pygoslin max_fa: sn positions available
    ether_ok: bool


@lru_cache(maxsize=1)
def class_table() -> dict[str, ClassInfo]:
    """Every class pygoslin knows, keyed by canonical name (decoder spec 3.6: enumerated from
    pygoslin's class table, not from our data)."""
    from pygoslin.domain.LipidClass import all_lipids

    out: dict[str, ClassInfo] = {}
    for v in all_lipids:
        if not v or v["name"] in out:
            continue
        cat = v["category"].name
        out[v["name"]] = ClassInfo(
            name=v["name"], category=cat, n_chains=int(v["poss_fa"]), n_positions=int(v["max_fa"]),
            ether_ok=cat in ("GL", "GP"),
        )
    return out


@lru_cache(maxsize=1)
def supported_classes() -> frozenset[str]:
    """Classes whose rendered shorthand pygoslin parses at species, molecular-species and sn level.

    pygoslin's class table includes classes with their own syntax (most sterols other than SE, some
    N-acyl lysolipids) that our renderer does not produce. The grammar allows only supported classes
    so that every decoded string parses (decoder spec 9.5); the CLASS output rows stay fixed at the
    full table (IMPLEMENTATION.md D-G7).
    """
    out = set()
    for name, info in class_table().items():
        if info.n_chains == 0:
            if is_valid_goslin(name):
                out.add(name)
            continue
        chains = []
        for k in range(info.n_chains):
            kind = "lcb" if (k == 0 and info.category == "SP") else "acyl"
            chains.append(Chain(kind, 18 if kind == "lcb" else 16 + 2 * k, 1, 2 if kind == "lcb" else 0,
                                dpos=[9], geom=["Z"], mods=[], mod_stereo=[], sn=k + 1))
        s = LipidStructure(cls=name, sum_c=sum(c.c for c in chains), sum_db=len(chains),
                           sum_ox=sum(c.ox for c in chains))
        forms = [render(s)]
        s.chains = chains
        s.sn_known = info.category != "SP"
        forms.append(render(s))
        s.sn_known = False
        if info.category != "SP":
            for c in s.chains:
                c.sn = None
            forms.append(render(s))
        # modifications and (where the grammar allows it) an ether chain, at sn level
        last = chains[-1]
        if last.kind == "acyl":
            for c, k in zip(chains, range(1, len(chains) + 1)):
                c.sn = k
            s.sn_known = info.category != "SP"
            last.ox, last.mods, last.mod_stereo = 1, [("Ep", last.c)], [None]
            forms.append(render(s))
            last.mods = [("OH", 2)]
            s.stereo = {("chain", len(chains), 2): "R"}
            forms.append(render(s))
            s.stereo = {}
            last.ox, last.mods, last.mod_stereo = 2, [("COOH", 3)], [None]
            forms.append(render(s))
            last.ox, last.mods, last.mod_stereo = 0, [], []
        if info.ether_ok and chains[0].kind == "acyl":
            for link in ("O-", "P-"):
                chains[0].kind = "ether"
                s.link = link
                forms.append(render(s))
            chains[0].kind, s.link = "acyl", "none"
        if all(is_valid_goslin(f) for f in forms):
            out.add(name)
    return frozenset(out)


@lru_cache(maxsize=1)
def _parser():
    from pygoslin.parser.Parser import LipidParser

    return LipidParser()


_ISO_RE = re.compile(r";(d\d+)(?=[\s/_]|$)|\((d\d+)\)")


def strip_isotope(label: str) -> tuple[str, str | None]:
    m = _ISO_RE.search(label)
    if not m:
        return label, None
    return _ISO_RE.sub("", label).strip(), (m.group(1) or m.group(2))


def is_valid_goslin(s: str) -> bool:
    try:
        _parser().parse(s)
        return True
    except Exception:
        return False


def _db_info(db) -> tuple[int, list[int] | None, list[str | None] | None]:
    if isinstance(db, dict):
        pos = sorted(db)
        return len(pos), pos, [db[p] or None for p in pos]
    n = int(db)
    return n, ([] if n == 0 else None), ([] if n == 0 else None)


def parse_label(label: str, adduct: str | None = None) -> LipidStructure:
    from pygoslin.domain.Element import Element
    from pygoslin.domain.LipidFaBondType import LipidFaBondType as BT

    label, _iso = strip_isotope(label)
    l = _parser().parse(label)
    info = l.lipid.info
    table = class_table()
    from pygoslin.domain.LipidClass import all_lipids

    cls = all_lipids[l.lipid.headgroup.lipid_class]["name"]
    if cls not in table:
        raise UnsupportedLabel(f"class {cls!r} not in class table")
    s = LipidStructure(cls=cls, adduct=adduct)

    fa_list = [f for f in l.lipid.fa_list]
    if not fa_list:
        bt = info.lipid_FA_bond_type
        s.link = "O-" if bt == BT.ETHER_PLASMANYL else "P-" if bt == BT.ETHER_PLASMENYL else "none"
        s.sum_c = int(info.num_carbon)
        s.sum_db = _db_info(info.double_bonds)[0]
        s.sum_ox = int(info.num_oxygens()) if callable(getattr(info, "num_oxygens", None)) else 0
        return s

    chains: list[Chain] = []
    link = "none"
    for f in fa_list:
        if f.num_carbon == 0:  # 0:0 placeholder for an empty sn position
            continue
        bt = f.lipid_FA_bond_type
        if bt in (BT.LCB_REGULAR, BT.LCB_EXCEPTION):
            kind = "lcb"
        elif bt in (BT.ETHER_PLASMANYL, BT.ETHER_PLASMENYL):
            kind = "ether"
            link = "O-" if bt == BT.ETHER_PLASMANYL else "P-"
        else:
            kind = "acyl"
        n_db, dpos, geom = _db_info(f.double_bonds)
        n_o = int(f.get_elements()[Element.O])
        ox = n_o - 1 if kind == "acyl" else n_o
        mods: list[tuple[str, int]] | None = []
        mod_st: list[str | None] | None = []
        if kind == "lcb":
            mods, mod_st = [], []  # sphingoid-base oxygens are implicit (IMPLEMENTATION.md D-L3)
        else:
            for t, groups in f.functional_groups.items():
                for g in groups:
                    if t not in MOD_TYPES:
                        if t == "O":
                            mods = None
                            continue
                        raise UnsupportedLabel(f"functional group {t!r} not supported")
                    if g.position < 1:
                        mods = None
                        continue
                    if mods is not None:
                        mods.append((t, g.position))
                        mod_st.append(g.stereochemistry or None)
            if mods is not None:
                order = sorted(range(len(mods)), key=lambda i: (mods[i][1], mods[i][0]))
                mods = [mods[i] for i in order]
                mod_st = [mod_st[i] for i in order]
            else:
                mod_st = None
        sn = f.position if f.position and f.position > 0 else None
        chains.append(Chain(kind, int(f.num_carbon), n_db, ox, dpos, geom, mods, mod_st, sn))

    info_cls = table[cls]
    sn_known = all(ch.sn is not None for ch in chains) and info_cls.category != "SP"
    if not sn_known:
        for ch in chains:
            ch.sn = None
    s.chains = canonical_order(chains)
    if info_cls.category == "SP":
        # Sphingolipid sn is fixed by convention (base = 1, N-acyl = 2): forced slots, and pygoslin
        # only parses SP shorthand with "/" once the base is written as 18:1;O2.
        for k, ch in enumerate(s.chains):
            ch.sn = k + 1
    s.sn_known = sn_known
    s.link = link
    s.sum_c = sum(ch.c for ch in chains)
    s.sum_db = sum(ch.db for ch in chains)
    s.sum_ox = sum(ch.ox for ch in chains)
    for k, ch in enumerate(s.chains, start=1):
        for (t, p), st in zip(ch.mods or [], ch.mod_stereo or []):
            if st and t in CHIRAL_MODS:
                s.stereo[("chain", k, p)] = st
    return s


# ----------------------------------------------------------------------------------------- render

def _ox_suffix(n: int) -> str:
    return "" if n == 0 else ";O" if n == 1 else f";O{n}"


def render_chain(ch: Chain, link: str, stereo: dict, k: int) -> str:
    prefix = (link if link in ("O-", "P-") else "") if ch.kind == "ether" else ""
    s = f"{prefix}{ch.c}:{ch.db}"
    if ch.db > 0 and ch.dpos is not None and all(p is not None for p in ch.dpos):
        geom = ch.geom or [None] * ch.db
        s += "(" + ",".join(f"{p}{g or ''}" for p, g in zip(ch.dpos, geom)) + ")"
    if ch.kind == "lcb":
        return s + _ox_suffix(ch.ox)
    if ch.ox > 0:
        if ch.mods is None or len(ch.mods) == 0:
            s += _ox_suffix(ch.ox)
        else:
            parts = []
            for t, p in ch.mods:
                st = stereo.get(("chain", k, p))
                parts.append(f"{p}{t}" + (f"[{st}]" if st else ""))
            s += ";" + ",".join(parts)
    return s


def render(s: LipidStructure) -> str:
    """Render whatever the structure resolves. Backbone stereocentres (glycerol sn-2, sphingoid
    C2/C3/C4) are not rendered (IMPLEMENTATION.md D-L5)."""
    info = class_table()[s.cls]
    sep = "/" if info.category == "ST" else " "
    if info.n_chains == 0:
        return s.cls  # fixed-structure classes (e.g. eicosanoids): the name is the structure
    if s.chains is None:
        link = s.link if s.link in ("O-", "P-") else ""
        return f"{s.cls}{sep}{link}{s.sum_c}:{s.sum_db}{_ox_suffix(s.sum_ox or 0)}"
    rendered = [(ch, render_chain(ch, s.link or "none", s.stereo, k)) for k, ch in enumerate(s.chains, start=1)]
    if s.sn_known or info.category == "SP" and all(ch.sn is not None for ch in s.chains):
        by_pos = {ch.sn: r for ch, r in rendered}
        n_pos = max(info.n_positions, max(by_pos))
        body = "/".join(by_pos.get(p, "0:0") for p in range(1, n_pos + 1))
    else:
        body = "_".join(r for _, r in rendered)
    return f"{s.cls}{sep}{body}"
