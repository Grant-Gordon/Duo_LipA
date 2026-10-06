"""Frozen foundation-model adapters on one spectrum. Skipped when weights are not downloaded."""

from pathlib import Path

import pytest
import torch

from duo_lipa.config import deep_merge, load_config
from duo_lipa.data.batch import collate_meta, collate_peaks
from duo_lipa.data.preprocess import preprocess
from duo_lipa.labels.goslin import parse_label
from duo_lipa.model.build import build_model
from duo_lipa.model.decoders.base import collate_targets

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "weights"
BASE = load_config(ROOT / "configs" / "base.yaml")

FMS = {
    "dreams": (W / "dreams" / "ssl_model.ckpt", {"precursor_mz"}, True),
    "msbert": (W / "msbert" / "MSBERT.pkl", {"precursor_mz"}, True),
    "spec2vec": (W / "spec2vec", set(), True),
    "ms2deepscore": (W / "ms2deepscore" / "ms2deepscore_model.pt", {"polarity", "precursor_mz"}, False),
}

pytestmark = pytest.mark.fm


def _need(path):
    if not Path(path).exists():
        pytest.skip(f"weights not downloaded: {path}")


def _run(fm, dec, rec):
    cfg = deep_merge(BASE, {"data": {"top_k": 100}, "encoder": {"type": fm},
                            "decoder": {"type": dec[0], "input": dec[1]}})
    torch.manual_seed(0)
    model, schema = build_model(cfg)
    r = preprocess(rec, 100)
    tgt = collate_targets(schema, [schema.tokenize(parse_label(r.label, r.adduct), True)])
    return model, model(collate_peaks([r]), collate_meta([r]), tgt)


@pytest.mark.parametrize("fm", list(FMS))
def test_fm_forward_backward_frozen(fm, test_record):
    path, accepts, per_peak = FMS[fm]
    _need(path)
    decs = [("ar", "peaks"), ("heads", "pooled")] if per_peak else [("ar", "pooled"), ("heads", "pooled")]
    for dec in decs:
        model, out = _run(fm, dec, test_record)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        bb = list(model.encoder.backbone.parameters())
        assert bb and all(not p.requires_grad and p.grad is None for p in bb), "backbone must stay frozen"
        heads = [(n, p) for n, p in model.encoder.named_parameters() if p.requires_grad]
        assert heads
        for n, p in heads:
            if dec[1] == "peaks" and n.startswith("proj_pooled"):
                continue  # peaks decoders do not read `pooled`
            if dec[1] == "pooled" and (n.startswith("proj_peaks") or n.startswith("mix")):
                continue  # pooled decoders do not read memory
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
        assert model.last_routing["encoder"] == sorted(accepts, key=list(model.last_routing["decoder"]).index)
        if per_peak:
            enc = out["enc"]
            assert enc.memory is not None and enc.memory_mz is not None
            assert (~enc.memory_pad).sum() > 0


def test_fm_cache_round_trip(test_record, tmp_path):
    _need(FMS["msbert"][0])
    cfg = deep_merge(BASE, {"data": {"top_k": 100}, "encoder": {"type": "msbert", "fm_cache_dir": str(tmp_path)},
                            "decoder": {"type": "ar", "input": "peaks"}})
    torch.manual_seed(0)
    model, _ = build_model(cfg)
    model.eval()
    r = preprocess(test_record, 100)
    p, m = collate_peaks([r]), collate_meta([r])
    e1, _ = model.encode(p, m)
    assert len(list(tmp_path.glob("*.pt"))) == 1
    e2, _ = model.encode(p, m)        # served from the cache
    assert torch.allclose(e1.memory, e2.memory) and torch.allclose(e1.pooled, e2.pooled)


def test_ms2deepscore_refuses_peaks_decoder():
    _need(FMS["ms2deepscore"][0])
    cfg = deep_merge(BASE, {"encoder": {"type": "ms2deepscore"}, "decoder": {"type": "ar", "input": "peaks"}})
    with pytest.raises(ValueError, match="pooled-only"):
        build_model(cfg)


def test_dreams_vendored_backbone_matches_upstream_torchscript():
    """The vendored backbone + the checkpoint's head reproduces upstream's TorchScript embedding."""
    ts_path, ck_path = W / "dreams" / "DreaMS_embedding_model_torchscript.pt", W / "dreams" / "embedding_model.ckpt"
    _need(ts_path); _need(ck_path)
    from duo_lipa.model.encoders.fm_dreams import load_dreams_backbone
    from duo_lipa.model.encoders.safe_load import load_checkpoint

    bb = load_dreams_backbone(ck_path).eval()
    sd = load_checkpoint(ck_path)["state_dict"]
    ts = torch.jit.load(str(ts_path)).eval()
    torch.manual_seed(0)
    spec = torch.zeros(1, 101, 2)
    mz = torch.sort(torch.rand(40) * 700 + 100).values
    it = torch.rand(40)
    spec[0, 0] = torch.tensor([760.585, 1.1])
    spec[0, 1:41, 0], spec[0, 1:41, 1] = mz, it / it.max()
    with torch.no_grad():
        ours = bb(spec)[-1][:, 0] @ sd["head.weight"].T + sd["head.bias"]
        ref = ts(spec)
    assert torch.allclose(ours, ref, atol=1e-4)


def test_dreams_input_format(test_record):
    _need(FMS["dreams"][0])
    from duo_lipa.model.encoders.fm_dreams import DreaMSAdapter
    from duo_lipa.model.interfaces import ConditioningForEncoder

    ad = DreaMSAdapter(d_model=16)
    r = preprocess(test_record, 100)
    spec, pad, mz = ad.prepare(collate_peaks([r]), ConditioningForEncoder(None, ["precursor_mz"],
                                                                           {"precursor_mz": [r.precursor_mz]}))
    assert spec.shape == (1, 101, 2)
    assert spec[0, 0, 0] == pytest.approx(r.precursor_mz) and spec[0, 0, 1] == pytest.approx(1.1)
    real = spec[0, 1:][~pad[0, 1:]]
    assert torch.all(real[1:, 0] >= real[:-1, 0])          # m/z order
    assert real[:, 1].max() == pytest.approx(1.0)


def test_spec2vec_finds_peaks_in_vocabulary(test_record):
    _need(FMS["spec2vec"][0])
    from duo_lipa.model.encoders.fm_spec2vec import Spec2VecAdapter
    from duo_lipa.model.interfaces import ConditioningForEncoder

    ad = Spec2VecAdapter(d_model=16)
    r = preprocess(test_record, 100)
    layers, pad, _, pooled = ad.backbone_layers(collate_peaks([r]), ConditioningForEncoder(None, [], {}))
    assert (~pad).sum() >= 0.5 * len(r.mz)
    assert pooled.abs().sum() > 0
