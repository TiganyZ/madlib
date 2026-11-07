# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""A per-atom property from a function you wrote.

    config "backend=callable fn=my_module:my_property types=Si,O rcut=5.0"

`my_property(graph)` is handed the Graph of geometry.py -- positions on the
autograd tape, the cell, the edge list, the atomic numbers -- and returns
either a tensor of per-atom values or a dict of them. Nothing else: the
caching, the reduction and the backward pass are the engine's.

This is the escape hatch for a model no listed backend covers, and it is what
the tests use, since a property whose value is known in closed form is the
only kind that can prove the plumbing rather than agree with it.
"""

import importlib

import torch

from .base import Backend, pick_device, pick_dtype

KEYS = {"fn", "rcut"}
DEFAULT_OUTPUT = "value"
DEFAULT_QUANTITY = "scalar"


def identity(cfg):
    return (cfg.get("fn", ""), cfg.get("device", ""), cfg.get("dtype", ""), cfg.get("types", ""),
            cfg.get("rcut", ""))


def build(cfg):
    if "fn" not in cfg:
        raise ValueError("backend=callable needs fn=<module>:<function>")
    if ":" not in cfg["fn"]:
        raise ValueError(f"fn must be module:function, not '{cfg['fn']}'")
    mod_name, fn_name = cfg["fn"].split(":", 1)
    fn = getattr(importlib.import_module(mod_name), fn_name)
    return _Callable(cfg, fn)


class _Callable(Backend):
    def __init__(self, cfg, fn):
        super().__init__(cfg, pick_dtype(cfg, torch.float64), pick_device(cfg), rcut=None)
        self.fn = fn

    def forward(self, graph):
        out = self.fn(graph)
        return out if isinstance(out, dict) else {DEFAULT_OUTPUT: out}
