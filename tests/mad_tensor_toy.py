# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""A per-atom tensor with a closed form, for testing the anisotropic path.

    T_i = c(Z_i) I + sum_j fc(r_ij) ( rhat_ij (x) rhat_ij )

with fc the same raised cosine compute mad/property/coord uses. It is a real
quantity -- the bond-orientational tensor -- but what matters here is that it
is symmetric, anisotropic, differentiable in closed form and owes nothing to a
model: a gradient checked through it is a test of MADLIB, not an agreement
between two copies of the same network.

The per-species diagonal term makes two sites with the same neighbour
directions different tensors, so a spectrum built from it has more than one
feature to fit.

    compute t all mad/property/python madlib_models ncomp 6 cutoff 3.0 &
            config "backend=callable fn=mad_tensor_toy:tensor types=C,O &
                    quantity=tensor6"
"""

import torch

R1 = 1.7
R2 = 2.1
DIAGONAL = {6: 10.0, 8: 16.0}    # per element, in whatever unit the deck uses


def tensor(graph):
    """(natoms, 3, 3), as a dict so the engine can name it."""
    pos = graph.pos
    src, dst = graph.sender, graph.receiver
    d = pos[dst] - pos[src] + graph.shifts
    r = torch.linalg.norm(d, dim=1, keepdim=True)
    fc = torch.where(r <= R1, torch.ones_like(r),
                     0.5 * (1.0 + torch.cos(torch.pi * (r - R1) / (R2 - R1))))
    fc = torch.where(r < R2, fc, torch.zeros_like(r))

    u = d / r
    outer = fc[:, :, None] * u[:, :, None] * u[:, None, :]
    t = torch.zeros(pos.shape[0], 3, 3, dtype=pos.dtype, device=pos.device)
    t.index_add_(0, dst, outer)

    eye = torch.eye(3, dtype=pos.dtype, device=pos.device)
    diag = torch.tensor([DIAGONAL.get(int(z), 1.0) for z in graph.numbers],
                        dtype=pos.dtype, device=pos.device)
    return {"value": t + diag[:, None, None] * eye}
