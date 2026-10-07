"""Adapter for the LipidOracle Zenodo deposit (record 22974483), `04_uvpd`.

All LipidOracle-specific parsing lives here: the MGF title format, the kit names and the join to
`splash_ground_truth.csv`. The output is ordinary `SpectrumRecord`s.

Title format (DATA_lipidoracle_notes.md):
    compound:<kit> | <name>;<isotope label> <adduct> | src:<file> | id:<n>, rt:<s>, mz:<m/z>, energy:<v>
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from duo_lipa.data.mgf import parse_charge, read_mgf
from duo_lipa.data.records import SpectrumRecord

KIT_NAMES = {"Equi": "EquiSPLASH", "Ultimate": "UltimateSPLASH", "Light": "LightSPLASH"}
_ISO_RE = re.compile(r";(d\d+)\b")
_ADDUCT_RE = re.compile(r"(\[M[^\]]*\]\d*[+-])\s*$")


def parse_title(title: str) -> dict:
    parts = [p.strip() for p in title.split("|")]
    out: dict = {"kit": parts[0].split(":", 1)[1].strip()}
    name_adduct = parts[1]
    m = _ADDUCT_RE.search(name_adduct)
    out["adduct"] = m.group(1) if m else None
    name = name_adduct[: m.start()].strip() if m else name_adduct
    iso = _ISO_RE.search(name)
    out["isotope_label"] = iso.group(1) if iso else None
    name = _ISO_RE.sub("", name).strip()
    # LipidOracle writes sphingolipids as "Cer d18:1_15:0", which pygoslin rejects; sphingolipid sn is
    # fixed by convention (base = sn-1), so "/" asserts nothing extra.
    if name.split()[0] in ("Cer", "SM", "HexCer", "Hex2Cer", "SPB") and "_" in name:
        name = name.replace("_", "/")
    out["name"] = name
    for p in parts[2:]:
        if p.startswith("src:"):
            out["src"] = p[4:].strip()
        else:
            for kv in p.split(","):
                if ":" in kv:
                    k, v = kv.split(":", 1)
                    out[k.strip()] = v.strip()
    return out


def _composition_key(cls: str, label: str) -> tuple:
    """(class, sorted chain (C, DB) pairs) used to join title names to the ground-truth table."""
    from duo_lipa.labels.goslin import parse_label

    s = parse_label(label)
    chains = tuple(sorted((c.c, c.db) for c in s.chains)) if s.chains is not None else None
    return (cls, chains, s.sum_c, s.sum_db)


def load_ground_truth(csv_path: str | Path) -> dict:
    table = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["Kit"],) + _composition_key(row["Class"], row["Species"])
            table[key] = row["Species"]
    return table


def load_uvpd(root: str | Path, join_ground_truth: bool = True) -> list[SpectrumRecord]:
    """root = .../extracted/04_uvpd"""
    root = Path(root)
    gt = load_ground_truth(root / "input" / "splash_ground_truth.csv") if join_ground_truth else {}
    records = []
    for e in read_mgf(root / "input" / "uvpd-gt-pos.mgf"):
        t = parse_title(e.headers["TITLE"])
        _, polarity = parse_charge(e.headers.get("CHARGE"))
        label = t["name"]
        kit = KIT_NAMES.get(t["kit"])
        if kit is not None:
            cls = label.split()[0]
            full = gt.get((kit,) + _composition_key(cls, label))
            if full is not None:
                label = full
        records.append(
            SpectrumRecord(
                mz=e.mz,
                intensity=e.intensity,
                precursor_mz=float(e.headers["PEPMASS"].split()[0]),
                polarity=polarity,
                method="UVPD",
                collision_energy=float(e.headers["ENERGY"]) if "ENERGY" in e.headers else None,
                isotope_label=t["isotope_label"],
                instrument=None,  # not stated per spectrum in the deposit
                adduct=t["adduct"],
                label=label,
                source=f"lipidoracle/04_uvpd/{t.get('src', '')}",
                meta={"title": e.headers["TITLE"], "feature_id": e.headers.get("FEATURE_ID"),
                      "title_name": t["name"], "kit": t["kit"]},
            )
        )
    return records
