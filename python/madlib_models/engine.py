# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""The forward pass, its cache, and the one backward pass MADLIB asks for.

A backend's job is to turn a Graph into a dict of per-atom outputs and nothing
else. Everything that is the same whatever the model -- holding the loaded
model, not evaluating it twice for one geometry, reducing its output, taking
the weighted gradient, and handing back the three extra rows a periodic run's
virial needs -- is here.

THE ADJOINT IS A GRADIENT, AND THE FORCE IS MINUS IT. What the compute wants
back is

    f_a = - d/dr_a sum_i lambda_i . q_i

with lambda held fixed, which is one backward pass over the forward the
property already ran. In 'pbc yes' mode three rows follow the forces: -dE/dh,
the gradient with respect to the cell vectors, without which the virial is
missing every contribution carried by a periodic image.

CACHED BY GEOMETRY, NOT BY TIMESTEP. An MC fix re-prices a trial move inside
one timestep, so a property cached on the step number would report the old
structure's spectrum for the new one and accept the move on it. The key is the
positions, the cell, and the edge list -- the last because two computes may
share one model with different cutoffs, and the shorter one must not be served
the longer one's forward pass.
"""

import numpy as np
import torch

from . import reduce

_MODELS = {}


class Model:
    """A loaded backend and its last forward pass, graph included."""

    def __init__(self, backend):
        self.backend = backend
        self._key = None
        self._graph = None
        self._out = None

    def evaluate(self, pos, typ, ei, ej, nlocal, cell):
        key = (hash(pos.tobytes()), hash(cell.tobytes()), nlocal,
               hash(ei.tobytes()), hash(ej.tobytes()))
        if key != self._key:
            graph = self.backend.graph(pos, typ, ei, ej, nlocal, cell)
            self._key = key
            self._graph = graph
            self._out = self.backend.forward(graph)
        return self._graph, self._out


def model_for(cfg, backend_module):
    """One loaded model per distinct model, shared by every compute naming it."""
    key = (backend_module.__name__,) + tuple(backend_module.identity(cfg))
    if key not in _MODELS:
        _MODELS[key] = Model(backend_module.build(cfg))
    return _MODELS[key]


class Property:
    """One compute's view of a model: which output, reduced how."""

    def __init__(self, model, output, quantity, scale=1.0, shift=0.0):
        self.model = model
        self.output = output
        self.quantity = quantity
        self.scale = scale
        self.shift = shift
        self.width = reduce.width(quantity)

    def _values(self, positions, types, edge_i, edge_j, nlocal, ncomp, cell):
        from .geometry import unpack
        if ncomp != self.width:
            raise ValueError(f"quantity '{self.quantity}' gives {self.width} component(s) per "
                             f"atom but the compute asked for ncomp {ncomp}")
        pos, typ, ei, ej, h = unpack(positions, types, edge_i, edge_j, cell)
        graph, out = self.model.evaluate(pos, typ, ei, ej, nlocal, h)
        if self.output not in out:
            raise ValueError(f"the model has no output '{self.output}'; it produces "
                             f"{sorted(k for k, v in out.items() if v is not None)}")
        raw = out[self.output]
        if raw is None:
            raise ValueError(f"the model returned nothing for output '{self.output}'")
        if raw.shape[0] != graph.pos.shape[0]:
            raise ValueError(f"output '{self.output}' has {raw.shape[0]} row(s) for "
                             f"{graph.pos.shape[0]} atoms, so it is not a per-atom quantity")
        q = reduce.apply(self.quantity, raw[:nlocal])
        return graph, self.scale * q + self.shift

    def compute(self, positions, types, edge_i, edge_j, nlocal, ncomp, cell):
        _, q = self._values(positions, types, edge_i, edge_j, nlocal, ncomp, cell)
        return np.ascontiguousarray(q.detach().cpu().numpy(), dtype=np.float64)

    def adjoint(self, positions, types, edge_i, edge_j, nlocal, ncomp, lam, cell):
        graph, q = self._values(positions, types, edge_i, edge_j, nlocal, ncomp, cell)
        weights = np.array(np.frombuffer(lam, dtype=np.float64))    # a read-only view
        w = torch.as_tensor(weights.reshape(q.shape), dtype=q.dtype, device=q.device)

        gx, gh = torch.autograd.grad((q * w).sum(), [graph.pos, graph.cell],
                                     retain_graph=True, allow_unused=True)
        f = np.zeros(tuple(graph.pos.shape)) if gx is None else -gx.detach().cpu().numpy()
        if not graph.periodic:
            return np.ascontiguousarray(f, dtype=np.float64)
        dh = np.zeros((3, 3)) if gh is None else -gh.detach().cpu().numpy()
        return np.ascontiguousarray(np.vstack([f, dh]), dtype=np.float64)
