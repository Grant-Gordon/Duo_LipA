"""Restricted checkpoint loading.

Some FM checkpoints are pickles that reference classes from packages we do not install (DreaMS
Lightning checkpoints reference `msml.*`). The unpickler below resolves only an allowlist of
globals and replaces everything else with inert stub classes, so loading a checkpoint never imports
or runs third-party code.
"""

from __future__ import annotations

import pickle
import types

_ALLOWED_PREFIXES = ("torch", "collections", "numpy", "_codecs")
_ALLOWED_EXACT = {("argparse", "Namespace"), ("builtins", "set"), ("builtins", "frozenset"),
                  ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"), ("builtins", "slice"),
                  ("builtins", "complex"), ("builtins", "bytearray"), ("builtins", "object")}


class Stub:
    """Inert stand-in for a class we refuse to import; keeps pickled attributes in __dict__."""

    def __init__(self, *a, **k):
        self._args = a

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


_stub_cache: dict = {}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(_ALLOWED_PREFIXES) or (module, name) in _ALLOWED_EXACT:
            return super().find_class(module, name)
        key = f"{module}.{name}"
        if key not in _stub_cache:
            _stub_cache[key] = type(name, (Stub,), {"__module__": "stub." + module})
        return _stub_cache[key]


restricted_pickle = types.ModuleType("restricted_pickle")
restricted_pickle.Unpickler = RestrictedUnpickler
restricted_pickle.load = lambda f, **kw: RestrictedUnpickler(f, **kw).load()
restricted_pickle.loads = lambda b, **kw: RestrictedUnpickler(__import__("io").BytesIO(b), **kw).load()
restricted_pickle.__name__ = "pickle"


def load_checkpoint(path, map_location="cpu"):
    import torch

    return torch.load(path, map_location=map_location, weights_only=False, pickle_module=restricted_pickle)
