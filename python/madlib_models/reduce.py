# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""Model output -> the quantity an observable reads, differentiably.

A model predicts what it was trained on; an observable reads what it needs.
These are that map, as torch expressions, so the adjoint stays one backward
pass.

VOIGT ORDER IS (xx, yy, zz, xy, xz, yz) throughout, with the off-diagonal
components the tensor elements themselves: the factor of two in omega = n.T.n
lives in the fix, not here.

Only the symmetric part of a tensor reaches a reduction that treats it as one,
since an NMR experiment cannot see the rest to first order.

'span' is the largest principal value minus the smallest, which can only
coincide when all three do -- no ordering convention, no branch. 'vzz' and
'etaq' use the quadrupolar convention |Vzz| >= |Vyy| >= |Vxx|, and etaq
divides by Vzz, so its gradient is unbounded where the field gradient
vanishes: a quantity to plot, not to bias.
"""

import torch

VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
NAMES = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "xz": 4, "yz": 5}

# quantity -> number of components it produces
WIDTH = {"scalar": 1, "vector": 3, "tensor6": 6, "tensor9": 9, "iso": 1, "span": 1,
         "vzz": 1, "etaq": 1}


def width(quantity):
    """Components the named quantity produces, for the ncomp cross-check."""
    if quantity.startswith("component:"):
        return 1
    if quantity not in WIDTH:
        raise ValueError(f"unknown quantity '{quantity}'; one of "
                         f"{sorted(WIDTH) + ['component:<xx|yy|zz|xy|xz|yz|index>']}")
    return WIDTH[quantity]


def _as_3x3(v):
    if v.dim() == 3 and v.shape[1:] == (3, 3):
        return v
    if v.dim() == 2 and v.shape[1] == 9:
        return v.reshape(-1, 3, 3)
    if v.dim() == 2 and v.shape[1] == 6:
        t = v.new_zeros(v.shape[0], 3, 3)
        for c, (a, b) in enumerate(VOIGT):
            t[:, a, b] = v[:, c]
            t[:, b, a] = v[:, c]
        return t
    raise ValueError(f"a tensor quantity needs an output of 6, 9 or (3,3) components "
                     f"per atom, not {tuple(v.shape[1:])}")


def _symmetric(v):
    t = _as_3x3(v)
    return 0.5 * (t + t.transpose(1, 2))


def _voigt(t):
    return torch.stack([t[:, a, b] for a, b in VOIGT], dim=1)


def _flat(v):
    """(n,) or (n,1) -> (n,); anything wider is an error, not a column taken."""
    if v.dim() == 1:
        return v
    if v.dim() == 2 and v.shape[1] == 1:
        return v[:, 0]
    raise ValueError(f"a scalar quantity needs one component per atom, not {v.shape[1]}; "
                     f"use quantity=component:<name> to pick one")


def _principal_q(v):
    """Principal values of the traceless symmetric part, |Vzz| >= |Vyy| >= |Vxx|."""
    t = _symmetric(v)
    trace = t.diagonal(dim1=1, dim2=2).sum(dim=1) / 3.0
    eye = torch.eye(3, dtype=t.dtype, device=t.device)
    lam = torch.linalg.eigvalsh(t - trace[:, None, None] * eye)
    order = torch.argsort(lam.abs(), dim=1)
    return torch.gather(lam, 1, order)    # (n,3) as (Vxx, Vyy, Vzz)


def apply(quantity, v):
    """Reduce a model's per-atom output v to the named quantity."""
    if quantity == "scalar":
        return _flat(v)
    if quantity == "vector":
        if v.dim() != 2 or v.shape[1] != 3:
            raise ValueError(f"a vector quantity needs three components per atom, "
                             f"not {tuple(v.shape[1:])}")
        return v
    if quantity == "tensor6":
        return _voigt(_symmetric(v))
    if quantity == "tensor9":
        return _as_3x3(v).reshape(-1, 9)
    if quantity == "iso":
        return _symmetric(v).diagonal(dim1=1, dim2=2).sum(dim=1) / 3.0
    if quantity == "span":
        lam = torch.linalg.eigvalsh(_symmetric(v))
        return lam[:, 2] - lam[:, 0]
    if quantity == "vzz":
        return _principal_q(v)[:, 2]
    if quantity == "etaq":
        p = _principal_q(v)
        return (p[:, 0] - p[:, 1]) / p[:, 2]
    if quantity.startswith("component:"):
        name = quantity.split(":", 1)[1]
        if name in NAMES:
            return _voigt(_symmetric(v))[:, NAMES[name]]
        try:
            idx = int(name)
        except ValueError:
            raise ValueError(f"component '{name}' is neither xx..yz nor an integer") from None
        flat = v.reshape(v.shape[0], -1)
        if not 0 <= idx < flat.shape[1]:
            raise ValueError(f"component {idx} is outside the model's {flat.shape[1]} outputs")
        return flat[:, idx]
    raise ValueError(f"unknown quantity '{quantity}'")
