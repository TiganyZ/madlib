# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""ShiftML as a MADLIB python-backend model: NMR chemical shieldings, in ppm.

ShiftML (lab-cosmo/shiftml) predicts the shielding TENSOR of every atom in a
molecular crystal. It is a metatomic torch model, so the adjoint MADLIB needs
is one backward pass -- nothing here differentiates anything by hand.

TWO QUANTITIES, ONE FORWARD PASS. This module serves the isotropic shielding;
its sibling mad_shiftml_span serves the span. Both read the same cached graph,
so a deck biasing on both pays for one model evaluation a step:

    compute iso all mad/property/python mad_shiftml      ncomp 1 pbc yes
    compute spn all mad/property/python mad_shiftml_span ncomp 1 pbc yes

WHY THE CELL IS NEVER PASSED. The backend hands over the positions of the owned
atoms AND their ghosts, each ghost carrying the real position of the image it
stands for, so the cluster of the two is geometrically complete for every owned
atom. The model is therefore run on a NON-PERIODIC system and only the owned
atoms' values are read. This is exact as long as the ghost shell reaches the
model's interaction range: ShiftML asks for a 5 A neighbour list and passes two
messages over it, so the range is 10 A and the compute needs

    compute q all mad/property/python mad_shiftml ncomp 1 cutoff 10.0

with a communication cutoff to match. A shorter one truncates the environment
of the atoms nearest the subdomain boundary and changes their shift by a little,
silently -- so the cutoff is checked here rather than trusted.

SHIELDING, NOT SHIFT. The values are shieldings sigma, and an experiment reports
shifts delta = a - b sigma. fix mad/nmr is what applies that, and it refuses to
run until told which it has been given.

Configured by environment, since a LAMMPS compute passes no arguments to its
module:

    MADLIB_SHIFTML_TYPES    element of each LAMMPS atom type, in order: "H C O"
    MADLIB_SHIFTML_MODEL    ShiftML4 (the 7-member ensemble), ShiftML3, or one
                            member such as ShiftML41 -- the default, because an
                            ensemble costs seven forward passes a step and the
                            spread it buys is a diagnostic, not a force
    MADLIB_SHIFTML_DEVICE   cpu or cuda; default cuda when torch sees one
"""

import os

import numpy as np
import torch
from metatomic.torch import ModelEvaluationOptions, ModelOutput, System
from shiftml.ase import ShiftML
from shiftml.utils.tensorial import T_sym_np_inv

Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12,
     "P": 15, "S": 16, "Cl": 17, "K": 19, "Ca": 20}

OUTPUT = {"mtt::cs_iso": ModelOutput(quantity="", unit="", sample_kind="atom")}

# The model's nine outputs are a general rank-2 tensor in an irreducible basis
# -- one l=0, three l=1, five l=2 -- and the Cartesian tensor is the fixed
# linear map below, exactly as shiftml's own get_cs_tensor applies it. Keeping
# the whole tensor rather than only its trace costs one 9x9 matmul and leaves
# the autograd graph the model's own.
_T = torch.tensor(T_sym_np_inv.T)

_state = {}

# One forward pass per geometry, not per caller. A 1H and a 13C spectrum are two
# fixes over one model, and MADLIB deliberately does not cache the property on
# the timestep (an MC fix re-evaluates within one), so the geometry itself is
# the key. The graph is kept, not just the values, so each fix's adjoint is a
# backward pass over the forward the other one already paid for.
_cache = {"key": None}


def _setup():
    if _state:
        return _state
    syms = os.environ.get("MADLIB_SHIFTML_TYPES", "").split()
    if not syms:
        raise RuntimeError("set MADLIB_SHIFTML_TYPES to the element of each LAMMPS "
                           "atom type, in order, e.g. 'H C O'")
    device = os.environ.get("MADLIB_SHIFTML_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu")
    name = os.environ.get("MADLIB_SHIFTML_MODEL", "ShiftML41")
    model = ShiftML(name, device=device)
    if hasattr(model, "models"):    # an ensemble: average the members
        members = model.models
    else:
        members = [model]
    _state.update(z=np.array([0] + [Z[s] for s in syms], dtype=np.int32),
                  members=members, dtype=members[0]._dtype,
                  device=members[0]._device, name=name)
    _state["t"] = _T.to(dtype=_state["dtype"], device=_state["device"])
    return _state


def _predict(pos, types, cell, nlocal, grad):
    """The owned atoms' shielding quantities, as torch tensors on the graph."""
    st = _setup()
    # a copy: the array is a read-only view of LAMMPS's own storage
    x = torch.tensor(pos, dtype=st["dtype"], device=st["device"], requires_grad=grad)
    h = torch.tensor(cell, dtype=st["dtype"], device=st["device"], requires_grad=grad)
    z = torch.as_tensor(st["z"][types], dtype=torch.int32, device=st["device"])
    periodic = bool(cell.any())
    pbc = torch.full((3,), periodic, dtype=torch.bool, device=st["device"])
    opts = ModelEvaluationOptions(length_unit="angstrom", outputs=OUTPUT)

    total = None
    for m in st["members"]:
        systems = m._nl_calculators.compute(systems=[System(z, x, h, pbc)])
        out = m._model(systems=systems, options=opts, check_consistency=False)
        v = torch.cat([b.values.reshape(b.values.shape[0], -1)
                       for b in out["mtt::cs_iso"].blocks()], dim=1)
        total = v if total is None else total + v
    sigma = (total[:nlocal] / len(st["members"])) @ st["t"]
    return x, h, _quantities(sigma.reshape(-1, 3, 3))


def _quantities(sigma):
    """The scalars a spectrum can be built from, given the Cartesian tensors.

    An NMR experiment sees only the symmetric part, so the l=1 block is dropped
    here as shiftml's own get_cs_tensor does.

    iso   trace/3, the isotropic shielding
    span  the Herzfeld-Berger span, largest principal value minus smallest.
          Only the extreme eigenvalues are used, and those two can coincide
          only when all three do, so unlike the Haeberlen anisotropy or the
          skew this needs no ordering convention and has no branch to cross.
          torch's eigenvalue-only backward is V diag(g) V^T with no eigenvalue
          differences in a denominator, so it is safe to differentiate.
    """
    s = 0.5 * (sigma + sigma.transpose(1, 2))
    lam = torch.linalg.eigvalsh(s)
    return {"iso": s.diagonal(dim1=1, dim2=2).sum(dim=1) / 3.0,
            "span": lam[:, 2] - lam[:, 0]}


def _cached(pos, types, cell, nlocal, which):
    key = (hash(pos.tobytes()), hash(cell.tobytes()), nlocal)
    if _cache["key"] != key:
        x, h, q = _predict(pos, types, cell, nlocal, grad=True)
        _cache.update(key=key, x=x, h=h, q=q)
    return _cache["x"], _cache["h"], _cache["q"][which]


def _unpack(positions, types, cell):
    pos = np.frombuffer(positions, dtype=np.float64).reshape(-1, 3)
    t = np.frombuffer(types, dtype=np.int32)
    h = np.frombuffer(cell, dtype=np.float64).reshape(3, 3)
    return pos, t, h


def compute_of(which, positions, types, edge_i, edge_j, nlocal, ncomp, cell):
    pos, t, h = _unpack(positions, types, cell)
    _, _, q = _cached(pos, t, h, nlocal, which)
    return np.ascontiguousarray(q.detach().cpu().numpy(), dtype=np.float64)


def adjoint_of(which, positions, types, edge_i, edge_j, nlocal, ncomp, lam, cell):
    pos, t, h = _unpack(positions, types, cell)
    weights = np.array(np.frombuffer(lam, dtype=np.float64))    # read-only view
    x, hh, q = _cached(pos, t, h, nlocal, which)
    w = torch.as_tensor(weights, dtype=q.dtype, device=q.device)
    gx, gh = torch.autograd.grad((q * w).sum(), [x, hh], retain_graph=True,
                                 allow_unused=True)
    # MADLIB wants a force, which is minus the gradient.
    f = -gx.cpu().numpy()
    if not h.any():
        return np.ascontiguousarray(f, dtype=np.float64)
    # periodic: three more rows, -dE/dh, which carry the rest of the virial
    dh = np.zeros((3, 3)) if gh is None else -gh.cpu().numpy()
    return np.ascontiguousarray(np.vstack([f, dh]), dtype=np.float64)


def mad_compute(positions, types, edge_i, edge_j, nlocal, ncomp, cell):
    return compute_of("iso", positions, types, edge_i, edge_j, nlocal, ncomp, cell)


def mad_adjoint(positions, types, edge_i, edge_j, nlocal, ncomp, lam, cell):
    return adjoint_of("iso", positions, types, edge_i, edge_j, nlocal, ncomp,
                      lam, cell)
