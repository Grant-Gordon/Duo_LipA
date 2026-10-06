"""v0 harness: run any list of configs on one spectrum, forward + backward (no optimizer step unless
`training.optimizer_step`), then an eval-mode beam decode. Writes one table (decoder spec 10).

Reported per run: loss, supervised slot count, parameter counts, gradient checks (non-finite grads,
trainable parameters that got no gradient, grad norm per module), the decoded annotation per level,
its Goslin validity, wall time.
"""

from __future__ import annotations

import json
import math
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from duo_lipa.config import config_hash, deep_merge, load_config
from duo_lipa.data.augment import augment
from duo_lipa.data.batch import collate_meta, collate_peaks
from duo_lipa.data.preprocess import preprocess
from duo_lipa.decode.beam import beam_search
from duo_lipa.labels.goslin import is_valid_goslin, parse_label
from duo_lipa.model.build import build_model
from duo_lipa.model.decoders.base import collate_targets

_DATA_CACHE: dict = {}


def load_test_spectrum(cfg: dict, root_dir: Path):
    d = cfg["data"]
    key = (d["root"], d["test_spectrum"]["title_name_prefix"], d["test_spectrum"]["adduct"])
    if key not in _DATA_CACHE:
        from duo_lipa.data.lipidoracle import load_uvpd

        recs = load_uvpd(root_dir / d["root"])
        ts = d["test_spectrum"]
        rec = next(r for r in recs if r.meta["title_name"].startswith(ts["title_name_prefix"]) and r.adduct == ts["adduct"])
        _DATA_CACHE[key] = rec
    return _DATA_CACHE[key]


def grad_report(model: torch.nn.Module) -> dict:
    no_grad, nonfinite = [], []
    norms: dict[str, float] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            no_grad.append(n)
            continue
        if not torch.isfinite(p.grad).all():
            nonfinite.append(n)
        top = n.split(".")[0]
        norms[top] = norms.get(top, 0.0) + float(p.grad.pow(2).sum())
    return {"no_grad": no_grad, "nonfinite": nonfinite, "grad_norm": {k: math.sqrt(v) for k, v in norms.items()}}


def run_one(name: str, cfg: dict, root_dir: Path, decode: bool = True) -> dict:
    t0 = time.time()
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    rec = load_test_spectrum(cfg, root_dir)
    if cfg.get("augmentation", {}).get("enabled"):
        rec = augment(rec, np.random.default_rng(seed), **cfg["augmentation"])
    rec = preprocess(rec, top_k=cfg["data"].get("top_k", 30))
    model, schema = build_model(cfg)
    struct = parse_label(rec.label, rec.adduct)
    toks = schema.tokenize(struct, adduct_given=cfg["training"].get("adduct_given", True))
    tgt = collate_targets(schema, [toks], grammar_mask=cfg["training"].get("grammar_mask_train", True))
    peaks, meta = collate_peaks([rec]), collate_meta([rec])

    model.train()
    out = model(peaks, meta, tgt)
    loss = out["loss"]
    loss.backward()
    if cfg["training"].get("optimizer_step"):
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3).step()
    gr = grad_report(model)
    enc = out["enc"]
    res = {
        "name": name, "config_hash": config_hash(cfg), "schema_hash": schema.hash,
        "encoder": cfg["encoder"].get("type", "scratch"), "decoder": f'{cfg["decoder"]["type"]}-{cfg["decoder"]["input"]}',
        "label": rec.label, "adduct": rec.adduct, "n_peaks": int(len(rec.mz)), "n_slots": len(toks),
        "loss": float(loss), "loss_finite": bool(torch.isfinite(loss)), "n_supervised": out["n_supervised"],
        "params": model.param_counts(), "routing": model.last_routing,
        "memory_shape": list(enc.memory.shape) if enc.memory is not None else None,
        "pooled_shape": list(enc.pooled.shape),
        "memory_mz": enc.memory_mz is not None,
        **gr,
    }
    if decode:
        model.eval()
        with torch.no_grad():
            enc_e, cond_e = model.encode(peaks, meta)
            dres = beam_search(model.decoder, enc_e, cond_e, 0, k=3, fixed={"ADDUCT": rec.adduct})
        res["decoded"] = dres.best
        res["decoded_valid"] = is_valid_goslin(dres.best) or dres.best in schema.values["CLASS"]
        res["decoded_levels"] = {k: v[0][0] for k, v in dres.levels.items()}
    res["seconds"] = round(time.time() - t0, 2)
    res["ok"] = res["loss_finite"] and not res["nonfinite"] and res.get("decoded_valid", True)
    return res


def expand_runs(spec: dict, base_dir: Path) -> list[tuple[str, dict]]:
    """spec: {base: path, encoders: {name: overrides}, decoders: {name: overrides},
    cross: [[enc names] or 'all', [dec names] or 'all'], extra: {name: overrides}}"""
    base = load_config(base_dir / spec["base"])
    runs = []
    encs, decs = spec.get("encoders", {}), spec.get("decoders", {})
    for group in spec.get("cross", []):
        en = list(encs) if group[0] == "all" else group[0]
        dn = list(decs) if group[1] == "all" else group[1]
        for e in en:
            for d in dn:
                runs.append((f"{e}|{d}", deep_merge(deep_merge(base, encs[e]), decs[d])))
    for name, ov in (spec.get("extra") or {}).items():
        runs.append((name, deep_merge(base, ov)))
    return runs


def write_table(results: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    lines = ["| run | ok | loss | sup | enc train / frozen | dec | no-grad params | memory | decoded (best) | s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | ERROR | | | | | | | `{r['error'][:80]}` | |")
            continue
        p = r["params"]
        ng = ", ".join(sorted({n.rsplit('.', 1)[0] for n in r["no_grad"]}))[:120] or "-"
        lines.append(f"| {r['name']} | {'yes' if r['ok'] else 'NO'} | {r['loss']:.3f} | {r['n_supervised']} | "
                     f"{p['encoder_trainable']:,} / {p['encoder_frozen']:,} | {p['decoder']:,} | {ng} | "
                     f"{r['memory_shape']} | `{r.get('decoded', '')[:60]}` | {r['seconds']} |")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def run_all(runs: list[tuple[str, dict]], root_dir: Path, out_dir: Path, decode: bool = True) -> list[dict]:
    results = []
    for name, cfg in runs:
        try:
            r = run_one(name, cfg, root_dir, decode=decode)
        except Exception as e:  # recorded, not hidden
            r = {"name": name, "ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        results.append(r)
        status = "ok" if r.get("ok") else ("ERROR " + r.get("error", "")[:100] if "error" in r else "FAILED")
        print(f"[{len(results)}/{len(runs)}] {name}: {status} loss={r.get('loss', float('nan')):.3f} "
              f"t={r.get('seconds', '-')}s decoded={r.get('decoded', '')[:50]}", flush=True)
        write_table(results, out_dir)
    return results
