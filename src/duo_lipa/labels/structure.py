"""Internal lipid structure: the decoder's view of a label, at whatever depth the label resolves.

`None` in any field means "the label does not say". The tokenizer turns `None` into UNK slots or
truncation (decoder spec 3.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Oxygen-containing modification types and their oxygen cost (decoder spec 3.6, MOD_TYPE).
MOD_TYPES: dict[str, int] = {"OH": 1, "oxo": 1, "Ep": 1, "OOH": 2, "COOH": 2}
# Modification types whose carbon is a stereocentre (decoder spec 3.2, stereo block).
CHIRAL_MODS = {"OH", "OOH", "Ep"}


@dataclass
class Chain:
    kind: str                                   # "lcb" | "ether" | "acyl"
    c: int
    db: int
    ox: int
    dpos: list[int | None] | None = None        # length db; None = positions unknown
    geom: list[str | None] | None = None        # aligned with dpos; "Z" | "E" | None
    mods: list[tuple[str, int]] | None = None   # (type, position); None = positions unknown
    mod_stereo: list[str | None] | None = None  # aligned with mods
    sn: int | None = None

    def key(self) -> tuple:
        return (self.c, self.db, self.ox)


@dataclass
class LipidStructure:
    cls: str                                    # pygoslin canonical class name
    adduct: str | None = None
    link: str | None = "none"                   # "none" | "O-" | "P-"
    sum_c: int | None = None
    sum_db: int | None = None
    sum_ox: int | None = None
    chains: list[Chain] | None = None           # canonical order; None = molecular species unknown
    sn_known: bool = False
    stereo: dict[tuple, str] = field(default_factory=dict)  # centre key -> "R" | "S"


MOD_TYPE_INDEX = {t: i for i, t in enumerate(MOD_TYPES)}


def full_key(ch: Chain) -> tuple:
    """Total order used for canonical chain order: (C, DB, OX), then double-bond positions, then the
    emitted modification tokens. Unknown parts compare as empty. The grammar enforces the same order
    slot by slot (IMPLEMENTATION.md D-G3)."""
    dpos = tuple(p for p in (ch.dpos or []) if p is not None)
    mods = tuple(x for t, p in (ch.mods or []) for x in (MOD_TYPE_INDEX[t], p))
    return (ch.c, ch.db, ch.ox, dpos, mods)


def canonical_order(chains: list[Chain]) -> list[Chain]:
    """Typed chains first (sphingoid base, then ether), then acyl chains sorted by `full_key`; ties
    broken by sn position so identical chains take sn in ascending order (decoder spec 3.2)."""
    rank = {"lcb": 0, "ether": 1, "acyl": 2}

    def k(ch: Chain):
        body = full_key(ch) if ch.kind == "acyl" else ()
        return (rank[ch.kind], body, ch.sn if ch.sn is not None else 99)

    return sorted(chains, key=k)
