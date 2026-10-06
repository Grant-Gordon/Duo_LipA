"""Decoder spec 14: tests 1 (round trip), 2 (no dead ends), 7 (block order is config), plus label
parsing facts the spec asked to verify in code."""

import random

import pytest

from duo_lipa.labels.goslin import (is_valid_goslin, parse_label, render, supported_classes)
from duo_lipa.labels.structure import Chain, LipidStructure
from duo_lipa.schema.blocks import State
from duo_lipa.schema.schema import LabelGrammarError, SchemaError, SlotSchema

SN_FIRST = ["adduct", "class", "species", "chains", "sn", "dpos", "mods", "geom", "stereo"]
SN_ONLY = ["adduct", "class", "species", "chains", "sn"]


def test_pygoslin_keeps_positions_in_underscore_notation():
    s = parse_label("PC 16:0_18:1(9)")
    assert not s.sn_known
    assert [c.dpos for c in s.chains] == [[], [9]]
    assert render(s) == "PC 16:0_18:1(9)"
    assert is_valid_goslin(render(s))


def test_isotope_label_stripped():
    s = parse_label("PC 15:0_18:1;d7")
    assert render(s) == "PC 15:0_18:1"


@pytest.mark.parametrize("blocks", [None, SN_FIRST, SN_ONLY], ids=["default", "sn_first", "sn_only"])
def test_round_trip_dummy_labels(uvpd_records, blocks):
    sch = SlotSchema(blocks)
    labels = sorted({(r.label, r.adduct) for r in uvpd_records})
    for lab, add in labels:
        s = parse_label(lab, add)
        out = render(sch.detokenize(sch.tokenize(s, adduct_given=True)))
        assert is_valid_goslin(out), out
        if blocks in (None, SN_FIRST):
            assert out == render(s), (lab, out)
        else:  # labels deeper than the schema are cut to it; cutting is idempotent
            again = render(sch.detokenize(sch.tokenize(parse_label(out, add), adduct_given=True)))
            assert again == out
            assert "(" not in out  # no C=C positions in the sn-only schema


def test_sn_before_cc_allows_non_canonical_dpos_order():
    """Under sn-first, two same-composition chains are ordered by sn, so the sn-1 chain may carry the
    larger double-bond position."""
    lab = "PC 18:1(11Z)/18:1(9Z)"
    for blocks in (None, SN_FIRST):
        sch = SlotSchema(blocks)
        out = render(sch.detokenize(sch.tokenize(parse_label(lab, "[M+H]+"), True)))
        assert out == lab


@pytest.mark.parametrize("blocks", [None, SN_FIRST, SN_ONLY], ids=["default", "sn_first", "sn_only"])
def test_no_dead_ends_random_walks(blocks):
    sch = SlotSchema(blocks)
    rng = random.Random(0)
    for cls in sorted(supported_classes()):
        for _ in range(2):
            st = State()
            while (sp := sch.next_slot(st)) is not None:
                legal = [cls] if sp.stype == "CLASS" else sch.legal(st, sp)
                assert legal, f"dead end at {sp} for {cls}"
                st.set(sp, rng.choice(legal))
            out = render(sch.detokenize(st))
            assert is_valid_goslin(out), out


def test_bad_block_order_fails_at_construction():
    with pytest.raises(SchemaError, match="must come after"):
        SlotSchema(["adduct", "class", "species", "dpos", "chains"])
    with pytest.raises(SchemaError, match="must start with"):
        SlotSchema(["class", "adduct", "species", "chains"])
    with pytest.raises(SchemaError, match="must come after"):
        SlotSchema(["adduct", "class", "species", "chains", "sn", "stereo"])  # stereo needs mods


def test_label_breaking_a_never_rule_is_rejected(schema):
    # chains do not add up to the sum composition
    s = LipidStructure(cls="PC", adduct="[M+H]+", link="none", sum_c=34, sum_db=1, sum_ox=0,
                       chains=[Chain("acyl", 16, 0, 0, [], [], [], []), Chain("acyl", 20, 1, 0, None, None, [], [])])
    with pytest.raises(LabelGrammarError):
        schema.tokenize(s, adduct_given=True)


def test_truncation_and_unk(schema):
    # species-level label stops after the species block
    toks = schema.tokenize(parse_label("PE 38:4", "[M+H]+"), True)
    assert toks[-1].spec.block == "species"
    # sn known, C=C unknown: DPOS slots are fed as UNK and sn is supervised
    toks = schema.tokenize(parse_label("PC 16:0/18:1", "[M+H]+"), True)
    dpos = [t for t in toks if t.spec.stype == "DPOS"]
    assert dpos and all(not t.supervised for t in dpos)
    assert any(t.spec.stype == "SN" and t.supervised for t in toks)


def test_forced_slots(schema):
    toks = schema.tokenize(parse_label("SM d18:1/18:1(9Z)", "[M+H]+"), True)
    by = {(t.spec.stype, t.spec.chain): t for t in toks}
    assert by[("ADDUCT", 0)].forced
    assert by[("SN", 1)].forced and by[("SN", 2)].forced     # sphingolipid sn
    assert by[("C", 2)].forced                               # last chain follows from the sums


def test_schema_hash_depends_on_block_order():
    assert SlotSchema().hash != SlotSchema(SN_FIRST).hash
