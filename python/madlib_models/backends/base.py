# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""What every backend shares: the atom-type map, the device, and the graph."""

import torch

from .. import config as cfgmod
from .. import geometry
from ..elements import numbers_of_types

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def pick_device(cfg):
    name = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def pick_dtype(cfg, default):
    if "dtype" not in cfg:
        return default
    if cfg["dtype"] not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    return DTYPES[cfg["dtype"]]


class Backend:
    """Holds the loaded model; subclasses supply forward()."""

    # True for a model whose own data pipeline builds the periodic graph, so
    # that the edge convention is the one its weights were trained with.
    builds_own_graph = False

    def __init__(self, cfg, dtype, device, rcut=None):
        self.z_of_type = numbers_of_types(cfgmod.species(cfg))
        self.dtype = dtype
        self.device = device
        self.rcut = cfgmod.number(cfg, "rcut", rcut)

    def graph(self, pos, typ, ei, ej, nlocal, cell):
        if typ.max() >= len(self.z_of_type):
            raise ValueError(f"the deck has at least {typ.max()} atom types but types= named "
                             f"{len(self.z_of_type) - 1}")
        return geometry.build(pos, self.z_of_type[typ], cell, ei, ej, nlocal, self.rcut,
                              self.dtype, self.device, edges=not self.builds_own_graph)

    def forward(self, graph):
        raise NotImplementedError
