"""Build a DuoLipA model from a config dict (configs/*.yaml)."""

from __future__ import annotations

from duo_lipa.model.conditioning import ConditioningEmbedder
from duo_lipa.model.decoders.ar import ARDecoder
from duo_lipa.model.decoders.heads import HeadsDecoder
from duo_lipa.model.wrapper import DuoLipA
from duo_lipa.schema.schema import SlotSchema

SCRATCH_KEYS = ("n_layers", "n_heads", "ff_mult", "dropout", "mz_encoding", "rank_position_encoding",
                "intensity_encoding", "mass_relation", "precursor_token", "conditioning_tokens", "pooling",
                "mz_bins", "mz_bin_width", "mz_lambda", "intensity_fourier_freq", "rope", "pab", "max_peaks")


def build_encoder(cfg: dict, d_model: int):
    e = cfg["encoder"]
    t = e.get("type", "scratch")
    if t == "scratch":
        from duo_lipa.model.encoders.scratch import ScratchEncoder

        return ScratchEncoder(d_model=d_model, **{k: e[k] for k in SCRATCH_KEYS if k in e})
    fm_kw = dict(d_model=d_model, layer_mix=e.get("layer_mix", True), head_mlp_layers=e.get("head_mlp_layers", 1),
                 weights_dir=e.get("weights_dir"))
    if t == "dreams":
        from duo_lipa.model.encoders.fm_dreams import DreaMSAdapter
        return DreaMSAdapter(**fm_kw, checkpoint=e.get("checkpoint", "ssl_model.ckpt"), top_k=e.get("top_k", 100))
    if t == "msbert":
        from duo_lipa.model.encoders.fm_msbert import MSBERTAdapter
        return MSBERTAdapter(**fm_kw)
    if t == "ms2deepscore":
        from duo_lipa.model.encoders.fm_ms2deepscore import MS2DeepScoreAdapter
        return MS2DeepScoreAdapter(**fm_kw)
    if t == "spec2vec":
        from duo_lipa.model.encoders.fm_spec2vec import Spec2VecAdapter
        return Spec2VecAdapter(**fm_kw, intensity_power=e.get("intensity_power", 0.5))
    raise ValueError(f"unknown encoder type {t!r}")


def build_model(cfg: dict, schema: SlotSchema | None = None) -> tuple[DuoLipA, SlotSchema]:
    schema = schema or SlotSchema(cfg.get("schema", {}).get("blocks"))
    d = cfg["d_model"]
    c = cfg.get("conditioning", {})
    embedder = ConditioningEmbedder(d, field_dropout=c.get("field_dropout", 0.1), extra_fields=c.get("extra_fields"))
    encoder = build_encoder(cfg, d)
    dc = cfg["decoder"]
    if dc["type"] in ("heads", "H"):
        decoder = HeadsDecoder(schema, d, input=dc["input"], hidden=dc.get("hidden"),
                               attn_layers=dc.get("attn_layers", 1), n_heads=dc.get("n_heads", 4),
                               dropout=dc.get("dropout", 0.0))
    elif dc["type"] in ("ar", "AR"):
        decoder = ARDecoder(schema, d, input=dc["input"], n_layers=dc.get("n_layers", 2),
                            n_heads=dc.get("n_heads", 4), ff_mult=dc.get("ff_mult", 4), dropout=dc.get("dropout", 0.0))
    else:
        raise ValueError(f"unknown decoder type {dc['type']!r}")
    return DuoLipA(embedder, encoder, decoder), schema
