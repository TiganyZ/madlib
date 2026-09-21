# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""A smooth coordination number in torch, as a MADLIB python-backend model.

Deliberately the same quantity as compute mad/property/coord, so the C++ and
the torch autograd answer can be compared directly -- which is what makes this
a test of the binding rather than of a model nobody can check.

The adjoint is one backward pass: nothing here differentiates anything by hand.
"""

import numpy as np
import torch

R1 = 1.7
R2 = 2.1


def _unpack(positions, types, edge_i, edge_j, nlocal):
    pos = np.frombuffer(positions, dtype=np.float64).reshape(-1, 3)
    ei = np.frombuffer(edge_i, dtype=np.int32)
    ej = np.frombuffer(edge_j, dtype=np.int32)
    return pos, ei, ej


def _coordination(pos, ei, ej, nlocal):
    """q_i = sum_j fc(r_ij), with fc a raised cosine between R1 and R2."""
    d = pos[ei] - pos[ej]
    r = torch.linalg.norm(d, dim=1)
    fc = torch.where(
        r <= R1,
        torch.ones_like(r),
        0.5 * (1.0 + torch.cos(np.pi * (r - R1) / (R2 - R1))),
    )
    fc = torch.where(r < R2, fc, torch.zeros_like(r))
    q = torch.zeros(pos.shape[0], dtype=pos.dtype)
    q.index_add_(0, ei, fc)
    return q[:nlocal]


def mad_compute(positions, types, edge_i, edge_j, nlocal, ncomp, cell):
    pos, ei, ej = _unpack(positions, types, edge_i, edge_j, nlocal)
    q = _coordination(torch.from_numpy(pos.copy()),
                      torch.from_numpy(ei.astype(np.int64)),
                      torch.from_numpy(ej.astype(np.int64)), nlocal)
    return np.ascontiguousarray(q.detach().numpy(), dtype=np.float64)


def mad_adjoint(positions, types, edge_i, edge_j, nlocal, ncomp, lam, cell):
    pos, ei, ej = _unpack(positions, types, edge_i, edge_j, nlocal)
    weights = np.frombuffer(lam, dtype=np.float64)

    x = torch.from_numpy(pos.copy()).requires_grad_(True)
    q = _coordination(x, torch.from_numpy(ei.astype(np.int64)),
                      torch.from_numpy(ej.astype(np.int64)), nlocal)
    loss = (q * torch.from_numpy(weights.copy())).sum()
    loss.backward()

    # MADLIB wants a force, which is minus the gradient.
    return np.ascontiguousarray(-x.grad.numpy(), dtype=np.float64)
