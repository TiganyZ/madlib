# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""A per-atom tensor with a closed form, for testing the anisotropic path.

    T_i = c(Z_i) I + sum_j fc(r_ij) ( rhat_ij (x) rhat_ij )

fc is the raised cosine of compute mad/property/coord. Symmetric, anisotropic,
differentiable by hand and owing nothing to a model, so a gradient checked
through it tests MADLIB rather than two copies of one network. The per-species
diagonal term gives the spectrum more than one feature.
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
