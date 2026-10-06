"""Decoder spec 14: tests 3 (validity before training), 4 (padding invariance, targets), 5 (exact
scores), 11 (embedding separation), on all four decoder variants with a stand-in encoder output.
The full encoder + decoder path is in test_forward_backward.py."""

import pytest
import torch

from duo_lipa.data.batch import collate_meta, collate_peaks
from duo_lipa.decode.beam import beam_search
from duo_lipa.labels.goslin import is_valid_goslin, parse_label
from duo_lipa.model.conditioning import ConditioningEmbedder
from duo_lipa.model.decoders.ar import ARDecoder
from duo_lipa.model.decoders.base import collate_targets
from duo_lipa.model.decoders.heads import HeadsDecoder
from duo_lipa.model.interfaces import EncoderOutput

D = 32
VARIANTS = [(HeadsDecoder, "pooled"), (HeadsDecoder, "peaks"), (ARDecoder, "pooled"), (ARDecoder, "peaks")]
IDS = ["H-pooled", "H-peaks", "AR-pooled", "AR-peaks"]


def _inputs(recs, seed=0, M=6):
    torch.manual_seed(seed)
    emb = ConditioningEmbedder(D).eval()
    cond = emb(collate_meta(recs), collate_peaks(recs).precursor_mz)
    B = len(recs)
    enc = EncoderOutput(memory=torch.randn(B, M, D), memory_pad=torch.zeros(B, M, dtype=torch.bool),
                        pooled=torch.randn(B, D))
    return enc, cond


@pytest.mark.parametrize("cls,inp", VARIANTS, ids=IDS)
def test_untrained_outputs_are_valid_goslin(cls, inp, schema, test_record):
    enc, cond = _inputs([test_record])
    for seed in range(3):
        torch.manual_seed(seed)
        dec = cls(schema, D, input=inp).eval()
        res = beam_search(dec, enc, cond, 0, k=3, fixed={"ADDUCT": test_record.adduct})
        assert is_valid_goslin(res.best), res.best
        for name, lvl in res.levels.items():
            if name == "class":
                continue  # a bare class name is not a parseable Goslin level (IMPLEMENTATION.md D-I3)
            for s, _ in lvl:
                assert is_valid_goslin(s), s


@pytest.mark.parametrize("cls,inp", VARIANTS, ids=IDS)
def test_beam_score_equals_teacher_forced(cls, inp, schema, test_record):
    enc, cond = _inputs([test_record])
    torch.manual_seed(1)
    dec = cls(schema, D, input=inp).eval()
    res = beam_search(dec, enc, cond, 0, k=4, fixed={"ADDUCT": test_record.adduct})
    bt = collate_targets(schema, [res.best_tokens])
    lp = dec.train_logprobs(enc, cond, bt)
    tf = lp.gather(-1, bt.value.clamp(min=0).unsqueeze(-1)).squeeze(-1)[~bt.pad].sum().item()
    assert abs(tf - res.best_score) < 1e-4


@pytest.mark.parametrize("cls,inp", VARIANTS, ids=IDS)
def test_target_padding_invariance(cls, inp, schema, test_record):
    torch.manual_seed(2)
    dec = cls(schema, D, input=inp).eval()
    enc, cond = _inputs([test_record])
    toks = schema.tokenize(parse_label(test_record.label, test_record.adduct), True)
    a = collate_targets(schema, [toks])
    b = collate_targets(schema, [toks], pad_to=len(toks) + 7)
    la, lb = dec.train_logprobs(enc, cond, a), dec.train_logprobs(enc, cond, b)
    T = len(toks)
    assert torch.allclose(la[:, :T], lb[:, :T], atol=1e-5)


@pytest.mark.parametrize("cls,inp", VARIANTS, ids=IDS)
def test_memory_padding_invariance(cls, inp, schema, test_record):
    if inp == "pooled":
        pytest.skip("pooled decoders do not read memory")
    torch.manual_seed(3)
    dec = cls(schema, D, input=inp).eval()
    enc, cond = _inputs([test_record])
    toks = schema.tokenize(parse_label(test_record.label, test_record.adduct), True)
    tgt = collate_targets(schema, [toks])
    padded = EncoderOutput(memory=torch.cat([enc.memory, torch.randn(1, 4, D)], 1),
                           memory_pad=torch.cat([enc.memory_pad, torch.ones(1, 4, dtype=torch.bool)], 1),
                           pooled=enc.pooled)
    assert torch.allclose(dec.train_logprobs(enc, cond, tgt), dec.train_logprobs(padded, cond, tgt), atol=1e-5)


def test_ar_output_rows_separated_by_slot_type(schema, test_record):
    """Decoder spec 6 / test 11: one output layer, one block of rows per slot type. The loss of one
    slot type sends gradient only into that type's rows."""
    dec = ARDecoder(schema, D, input="peaks")
    enc, cond = _inputs([test_record])
    toks = schema.tokenize(parse_label(test_record.label, test_record.adduct), True)
    tgt = collate_targets(schema, [toks])
    lp = dec.train_logprobs(enc, cond, tgt)
    t_class = next(i for i, t in enumerate(toks) if t.spec.stype == "CLASS")
    lp[0, t_class, tgt.value[0, t_class]].backward()
    g = dec.out.weight.grad.abs().sum(1)
    off, n = schema.offset["CLASS"], len(schema.values["CLASS"])
    assert g[off:off + n].sum() > 0
    assert g[:off].sum() == 0 and g[off + n:].sum() == 0


def test_heads_classifiers_share_no_parameters(schema):
    dec = HeadsDecoder(schema, D, input="pooled")
    ptrs = {}
    for name, mod in dec.mlp.items():
        for p in mod.parameters():
            assert p.data_ptr() not in ptrs, (name, ptrs.get(p.data_ptr()))
            ptrs[p.data_ptr()] = name
