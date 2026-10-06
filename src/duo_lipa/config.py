"""YAML config loading with `base:` inheritance and dotted overrides.

A config file may name a parent with `base: <relative path>`; the child is deep-merged over it.
Configs stay plain nested dicts so that every value can be changed between runs without code changes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("base", None)
    if parent is not None:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def set_dotted(cfg: dict, dotted: str, value: Any) -> dict:
    """Return a copy of cfg with `a.b.c = value`."""
    out = copy.deepcopy(cfg)
    node = out
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return out


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
