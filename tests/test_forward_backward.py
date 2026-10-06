"""Full pipeline on ONE UVPD spectrum: every scratch-encoder arm x every decoder variant runs forward
and backward, with finite loss and gradients reaching every trainable encoder parameter.
Also: conditioning routing (decoder spec test 12), padding invariance (test 4, peaks side),
memory layout (Q4a/Q4b), and properties of RoPE / PAB."""

from pathlib import Path

import numpy as np
import pytest
import torch

from duo_lipa.config import deep_merge, load_config
from duo_lipa.data.augment import augment
from duo_lipa.data.batch import collate_meta, collate_peaks
from duo_lipa.data.preprocess import preprocess
from duo_lipa.data.records import COND_FIELDS
from duo_lipa.labels.goslin import parse_label
from duo_lipa.model.build import build_model
from duo_lipa.model.decoders.base import collate_targets
from duo_lipa.model.encoders.attention import MassAwareSelfAttention

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")

ARMS = {
    "s0_lipidetective": {"mz_encoding": "binned", "rank_position_encoding": True},
    "s1_absolute": {},
    "s1_rope": {"mass_relation": {"rope": True, "pab": False}},
    "s1_pab": {"mass_relation": {"rope": False, "pab": True}},
    "s1_rope_pab": {"mass_relation": {"rope": True, "pab": True}},
    "s2_int_linear": {"intensity_encoding": "linear"},
    "s2_int_fourier": {"intensity_encoding": "fourier"},
    "s3_no_encoder_conditioning": {"conditioning_tokens": False},
    "pool_attention": {"pooling": "attention"},
    "pool_mean": {"pooling": "mean"},
    "no_precursor_token": {"precursor_token": False, "pooling": "attention"},
}
DECODERS = {"H-pooled": ("heads", "pooled"), "H-peaks": ("heads", "peaks"),
            "AR-pooled": ("ar", "pooled"), "AR-peaks": ("ar", "peaks")}

# Trainable parameters allowed to receive no gradient on this spectrum, with the reason:
ALLOWED_NO_GRAD = (
    "embedder.cont_unknown.",  # "unknown" vectors of continuous fields: both fields are known here
    "decoder.mlp.MOD.",        # H: the label has no oxygenated chains, so no MOD slots
    "decoder.mlp.STEREO.",     # H: the label has no stereo slots
    "decoder.query.MOD", "decoder.query.STEREO",
    "decoder.prefix_type.",    # AR-peaks: prefix embeddings are used only by AR-pooled
    "encoder.pool_",           # attention pooling under input: peaks decoders (pooled unused)
)


def _cfg(arm: dict, dec: tuple, **extra) -> dict:
    cfg = deep_merge(BASE, {"encoder": arm, "decoder": {"type": dec[0], "input": dec[1]}})
    return deep_merge(cfg, extra)


def _batch(recs, schema, pad_to=None):
    recs = [preprocess(r, 30) for r in recs]
    toks = [schema.tokenize(parse_label(r.label, r.adduct), True) for r in recs]
    return collate_peaks(recs, pad_to=pad_to), collate_meta(recs), collate_targets(schema, toks)


@pytest.mark.parametrize("dec", list(DECODERS), ids=list(DECODERS))
@pytest.mark.parametrize("arm", list(ARMS), ids=list(ARMS))
def test_single_spectrum_forward_backward(arm, dec, test_record):
    torch.manual_seed(0)
    model, schema = build_model(_cfg(ARMS[arm], DECODERS[dec]))
    peaks, meta, tgt = _batch([test_record], schema)
    model.train()
    out = model(peaks, meta, tgt)
    assert torch.isfinite(out["loss"]) and out["n_supervised"] > 0
    out["loss"].backward()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            assert n.startswith(ALLOWED_NO_GRAD), f"{n} received no gradient"
        else:
            assert torch.isfinite(p.grad).all(), n
    # attention pooling feeds only `pooled`, which input: peaks decoders never read
    skip = ("pool_",) if DECODERS[dec][1] == "peaks" else ()
    enc_grads = [p.grad for n, p in model.encoder.named_parameters() if p.requires_grad and not n.startswith(skip)]
    assert all(g is not None for g in enc_grads), "every encoder parameter must get a gradient"
    assert sum(float(g.abs().sum()) for g in enc_grads) > 0


def test_conditioning_routing(test_record):
    for arm, expect in (({}, list(COND_FIELDS)), ({"conditioning_tokens": False}, [])):
        model, schema = build_model(_cfg(arm, ("ar", "peaks")))
        peaks, meta, tgt = _batch([test_record], schema)
        model(peaks, meta, tgt)
        assert model.last_routing["decoder"] == list(COND_FIELDS)
        assert model.last_routing["encoder"] == expect


def test_memory_layout(test_record):
    model, schema = build_model(_cfg({}, ("ar", "peaks")))
    peaks, meta, _ = _batch([test_record], schema)
    enc, _ = model.encode(peaks, meta)
    K = peaks.mz.shape[1]
    assert enc.memory.shape == (1, K + 1, BASE["d_model"])       # precursor + peaks, no conditioning rows
    assert enc.memory_mz.dtype == torch.float64
    assert float(enc.memory_mz[0, 0]) == pytest.approx(test_record.precursor_mz)
    assert torch.equal(enc.memory_mz[0, 1:], peaks.mz[0])


@pytest.mark.parametrize("arm", list(ARMS), ids=list(ARMS))
def test_peak_padding_invariance(arm, test_record):
    torch.manual_seed(0)
    model, schema = build_model(_cfg(ARMS[arm], ("ar", "peaks")))
    model.eval()
    p1, meta, _ = _batch([test_record], schema)
    p2, _, _ = _batch([test_record], schema, pad_to=p1.mz.shape[1] + 9)
    e1, _ = model.encode(p1, meta)
    e2, _ = model.encode(p2, meta)
    M = e1.memory.shape[1]
    assert torch.allclose(e1.pooled, e2.pooled, atol=1e-5)
    assert torch.allclose(e1.memory, e2.memory[:, :M], atol=1e-5)
    assert e2.memory_pad[:, M:].all()


def test_batch_of_two_spectra(uvpd_records, test_record):
    other = next(r for r in uvpd_records if r.meta["title_name"].startswith("TG") and r.adduct == "[M+NH4]+")
    model, schema = build_model(_cfg({"mass_relation": {"rope": True, "pab": True}}, ("ar", "peaks")))
    peaks, meta, tgt = _batch([test_record, other], schema)
    out = model(peaks, meta, tgt)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])


@pytest.mark.parametrize("arm", [a for a in ARMS if a not in ("s1_absolute",)])
def test_each_switch_changes_the_encoder(arm, test_record):
    """Guards against a config switch being silently ignored."""
    def run(a):
        torch.manual_seed(0)
        model, schema = build_model(_cfg(a, ("ar", "peaks")))
        model.eval()
        peaks, meta, _ = _batch([test_record], schema)
        return model.encode(peaks, meta)[0].pooled
    assert not torch.allclose(run(ARMS[arm]), run({}), atol=1e-6)


def test_augmentation_changes_input_not_label(test_record):
    a = augment(test_record, np.random.default_rng(0), **BASE["augmentation"])
    assert a.label == test_record.label
    assert not np.array_equal(np.sort(a.mz), np.sort(test_record.mz))


@pytest.mark.parametrize("rope,pab,massless", [(True, False, False), (False, True, False), (True, True, False),
                                               (False, True, True)])
def test_relative_mass_mechanisms_are_shift_invariant(rope, pab, massless):
    """RoPE and PAB see only m/z differences: shifting every mass by a constant leaves the attention
    output unchanged (token content held fixed). With massless (conditioning) tokens present this
    holds for PAB only: an unrotated token's dot product with a rotated key depends on that key's
    absolute m/z (IMPLEMENTATION.md D-E5)."""
    torch.manual_seed(0)
    att = MassAwareSelfAttention(32, 4, rope=rope, pab=pab).eval()
    x = torch.randn(1, 6, 32)
    mz = torch.tensor([[760.585, 184.073, 577.519, 301.2, 0.0, 0.0]], dtype=torch.float64)
    has_mass = torch.tensor([[True, True, True, True, not massless, not massless]])
    if not massless:
        mz[0, 4:] = torch.tensor([420.1, 95.05], dtype=torch.float64)
    pad = torch.zeros(1, 6, dtype=torch.bool)
    y1 = att(x, pad, mz, has_mass)
    y2 = att(x, pad, torch.where(has_mass, mz + 123.456, mz), has_mass)
    assert torch.allclose(y1, y2, atol=1e-4)
    # and they are not trivially constant: a change in one mass difference changes the output
    mz3 = mz.clone(); mz3[0, 2] += 18.011
    assert not torch.allclose(y1, att(x, pad, mz3, has_mass), atol=1e-5)


def test_peaks_decoder_refuses_pooled_only_encoder():
    from duo_lipa.model.conditioning import ConditioningEmbedder
    from duo_lipa.model.decoders.ar import ARDecoder
    from duo_lipa.model.encoders.base import EncoderAdapter
    from duo_lipa.model.wrapper import DuoLipA
    from duo_lipa.schema.schema import SlotSchema

    class PooledOnly(EncoderAdapter):
        name, has_per_peak = "pooled_only", False

    with pytest.raises(ValueError, match="pooled-only"):
        DuoLipA(ConditioningEmbedder(16), PooledOnly(), ARDecoder(SlotSchema(), 16, input="peaks"))
