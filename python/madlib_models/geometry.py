# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""The geometry LAMMPS hands over, as a graph a message-passing model can eat.

compute mad/property/python sends one of two things, and which one is decided
by the cell being all zeros or not.

GHOST CLOUD (the default, and the only one that works under MPI). Positions
cover the owned atoms and their ghosts, each ghost already carrying the
position of the image it stands for, and the edge list is the compute's own
neighbour list over them. The model therefore runs on a finite cluster with no
periodicity at all, and only the owned atoms' values are read. It is exact as
long as the ghost shell reaches the model's whole interaction range -- for a
message-passing model that is num_layers * r_max, not r_max -- which is what
the compute's 'cutoff' and comm_modify cutoff have to cover.

ONE PERIODIC CELL ('pbc yes'). Positions are the owned atoms only and the cell
is real, so the neighbour list has to be built here, with periodic images. One
rank only, by construction.

EDGE DIRECTION. Both MACE and NequIP build their lists with ASE's 'ijS'
convention, D = pos[j] - pos[i] + S.cell, store edge_index = (i, j), and
aggregate messages onto j. So the CENTRE of an environment is edge_index[1],
and the edge vector points from the neighbour to the centre. LAMMPS gives the
opposite pairing -- (centre, neighbour) -- so the two are swapped here. Get it
backwards and every edge vector flips: a model with odd-parity features then
returns a different number, with nothing to show that anything is wrong.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Graph:
    pos: torch.Tensor        # (nsent, 3), requires_grad
    cell: torch.Tensor       # (3, 3), requires_grad; zeros when aperiodic
    numbers: np.ndarray      # (nsent,) atomic numbers
    sender: torch.Tensor     # (nedge,) neighbour index
    receiver: torch.Tensor   # (nedge,) centre index
    unit_shifts: np.ndarray  # (nedge, 3) integer image offsets
    nlocal: int
    periodic: bool

    @property
    def shifts(self):
        """S . h, kept as a torch expression so the cell gradient flows."""
        if not self.periodic:
            return self.pos.new_zeros(self.sender.shape[0], 3)
        s = torch.as_tensor(self.unit_shifts, dtype=self.pos.dtype, device=self.pos.device)
        return s @ self.cell


def unpack(positions, types, edge_i, edge_j, cell):
    """The memoryviews from LAMMPS as numpy arrays. They are read-only views."""
    pos = np.frombuffer(positions, dtype=np.float64).reshape(-1, 3)
    typ = np.frombuffer(types, dtype=np.int32)
    ei = np.frombuffer(edge_i, dtype=np.int32)
    ej = np.frombuffer(edge_j, dtype=np.int32)
    h = np.frombuffer(cell, dtype=np.float64).reshape(3, 3)
    return pos, typ, ei, ej, h


def _periodic_edges(pos, cell, rcut):
    from ase.neighborlist import primitive_neighbor_list
    i, j, s = primitive_neighbor_list("ijS", (True, True, True), cell, pos, rcut,
                                      self_interaction=False)
    return i.astype(np.int64), j.astype(np.int64), s.astype(np.int64)


def build(pos, numbers, cell, edge_i, edge_j, nlocal, rcut, dtype, device, edges=True):
    """A Graph on the autograd tape, from either of the two geometries."""
    periodic = bool(cell.any())
    x = torch.tensor(np.array(pos), dtype=dtype, device=device, requires_grad=True)
    h = torch.tensor(np.array(cell), dtype=dtype, device=device, requires_grad=True)

    if not edges:
        empty = np.zeros(0, dtype=np.int64)
        return Graph(pos=x, cell=h, numbers=numbers,
                     sender=torch.as_tensor(empty, dtype=torch.long, device=device),
                     receiver=torch.as_tensor(empty, dtype=torch.long, device=device),
                     unit_shifts=np.zeros((0, 3), dtype=np.int64), nlocal=nlocal,
                     periodic=periodic)

    if periodic:
        if rcut is None:
            raise ValueError("pbc yes needs a cutoff for the model's own neighbour list: "
                             "give rcut=<r> in the config, or a model that reports its r_max")
        i, j, s = _periodic_edges(np.array(pos), np.array(cell), rcut)
    else:
        # LAMMPS pairs are (centre, neighbour); the frameworks' are (sender, receiver)
        i, j = edge_j.astype(np.int64), edge_i.astype(np.int64)
        s = np.zeros((i.shape[0], 3), dtype=np.int64)

    return Graph(pos=x, cell=h, numbers=numbers,
                 sender=torch.as_tensor(i, dtype=torch.long, device=device),
                 receiver=torch.as_tensor(j, dtype=torch.long, device=device),
                 unit_shifts=s, nlocal=nlocal, periodic=periodic)
