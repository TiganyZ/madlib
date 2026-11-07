# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""A MACE model's per-atom outputs as a MADLIB property.

    compute q all mad/property/python madlib_models ncomp 1 cutoff 12.0 &
            config "backend=mace model=cebe.model types=C,O"

WHICH OUTPUT. A MACE model returns a dict, and the per-atom entries of it are
what can be a property: `node_energy` from any energy model, `charges` and
`atomic_dipoles` from a dipole model, `node_feats` if you want descriptors.
`output=` names one and the default is node_energy, because that is where a
model trained on a per-atom target puts it -- fitting core-electron binding
energies as if they were atomic energy contributions is the usual way to get
them out of an energy-shaped architecture, and `fix mad/xps` then reads them
with no further ceremony. A per-graph output such as `energy` or a total
`dipole` has one row where a property needs one per atom, and is refused on
that ground rather than reshaped.

    XPS            output=node_energy  quantity=scalar    ncomp 1
    IR             output=atomic_dipoles quantity=vector  ncomp 3
    NMR, isotropic output=<tensor head> quantity=iso      ncomp 1
    NMR, powder    output=<tensor head> quantity=tensor6  ncomp 6

THE COMPUTE'S CUTOFF HAS TO REACH r_max, and reaching further buys nothing.
This is not what one would guess for a model that passes several messages, and
the reason is in how LAMMPS lists neighbours: the list covers OWNED atoms, so
every edge handed over ends on an owned atom and a ghost receives no messages
of its own. A second layer then reads an un-updated feature for every
neighbour that is a ghost, and no depth of ghost shell changes that -- a 4 A
shell and a 2.5 A shell give the same numbers to the last bit. Measured
against MACE's own pipeline on a 91-atom cell, a two-layer model at r_max 2 A
is 6e-4 eV per atom out; the same model with one layer agrees to 4e-16.

So: in 'pbc yes' the model builds its own graph and the answer is exact. On a
ghost cloud -- the MPI route -- a one-pass model is exact and a deeper one is
approximate near every subdomain and periodic boundary. The error is reported
here once rather than left to be discovered, and `ghost_check=no` silences it.
A cutoff SHORTER than r_max is a different thing and an error: the model is
then missing edges it was trained to have.

Edges longer than r_max are dropped, since MACE's radial envelope is exactly
zero beyond it. That changes no value; it only saves the work.
"""

import numpy as np
import torch

from .base import Backend, pick_device, pick_dtype

KEYS = {"head", "rcut", "ghost_check"}
DEFAULT_OUTPUT = "node_energy"
DEFAULT_QUANTITY = "scalar"


def identity(cfg):
    return (cfg.get("model", ""), cfg.get("device", ""), cfg.get("dtype", ""),
            cfg.get("head", ""), cfg.get("types", ""), cfg.get("rcut", ""))


def build(cfg):
    if "model" not in cfg:
        raise ValueError("backend=mace needs model=<file>")
    device = pick_device(cfg)
    model = torch.load(cfg["model"], map_location=device, weights_only=False)
    return _Mace(cfg, model, device)


class _Mace(Backend):
    def __init__(self, cfg, model, device):
        dtype = pick_dtype(cfg, next(model.parameters()).dtype)
        self.model = model.to(device=device, dtype=dtype)
        self.model.eval()
        self.r_max = float(model.r_max)
        self.layers = len(model.interactions)
        super().__init__(cfg, dtype, device, rcut=self.r_max)

        zs = [int(z) for z in model.atomic_numbers]
        self.index_of_z = {z: i for i, z in enumerate(zs)}
        self.nelem = len(zs)

        heads = list(getattr(model, "heads", []) or [])
        if "head" in cfg:
            if cfg["head"] not in heads:
                raise ValueError(f"the model has no head '{cfg['head']}'; it has {heads}")
            self.head = heads.index(cfg["head"])
        else:
            self.head = 0
        self.have_heads = bool(heads)
        self.ghost_check = cfg.get("ghost_check", "yes") != "no"
        self.checked = False

    def _node_attrs(self, numbers):
        try:
            idx = np.array([self.index_of_z[int(z)] for z in numbers])
        except KeyError as exc:
            raise ValueError(f"the model was not fitted for element Z={exc.args[0]}; it covers "
                             f"{sorted(self.index_of_z)}") from None
        oh = torch.zeros(len(numbers), self.nelem, dtype=self.dtype, device=self.device)
        oh[torch.arange(len(numbers)), torch.as_tensor(idx, device=self.device)] = 1.0
        return oh

    def _check_ghosts(self, graph, lengths):
        """Once, at the first forward pass: a missing edge, and a missing message."""
        if self.checked or graph.periodic:
            return
        self.checked = True
        pos = graph.pos.detach()
        extent = float(torch.linalg.norm(pos.max(dim=0).values - pos.min(dim=0).values))
        reach = float(lengths.max()) if lengths.numel() else 0.0

        # A cluster smaller than r_max is complete whatever the cutoff was.
        if extent > self.r_max and reach < 0.999 * self.r_max:
            raise ValueError(
                f"the longest edge supplied is {reach:.3g} A but this model's graph is "
                f"{self.r_max:.3g} A, so it is missing edges it was fitted with. Raise the "
                f"compute's cutoff to at least {self.r_max:.3g} (and the comm_modify cutoff "
                f"with it).")
        if self.ghost_check and self.layers > 1:
            print(f"madlib_models: this MACE model passes {self.layers} messages, and a ghost "
                  f"atom receives none, so atoms with ghost neighbours read incomplete "
                  f"features. Use 'pbc yes' for an exact answer on one rank, or "
                  f"ghost_check=no to silence this.")

    def forward(self, graph):
        shifts = graph.shifts
        vec = graph.pos[graph.receiver] - graph.pos[graph.sender] + shifts
        lengths = torch.linalg.norm(vec.detach(), dim=1)
        self._check_ghosts(graph, lengths)

        keep = lengths < self.r_max
        sender, receiver = graph.sender[keep], graph.receiver[keep]
        n = graph.pos.shape[0]
        data = {
            "positions": graph.pos,
            "node_attrs": self._node_attrs(graph.numbers),
            "edge_index": torch.stack([sender, receiver]),
            "shifts": shifts[keep],
            "unit_shifts": torch.as_tensor(graph.unit_shifts[keep.cpu().numpy()],
                                           dtype=self.dtype, device=self.device),
            "cell": graph.cell,
            "batch": torch.zeros(n, dtype=torch.long, device=self.device),
            "ptr": torch.tensor([0, n], dtype=torch.long, device=self.device),
        }
        if self.have_heads:
            data["head"] = torch.tensor([self.head], dtype=torch.long, device=self.device)

        return self.model(data, training=False, compute_force=False)
