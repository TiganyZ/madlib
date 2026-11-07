# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""The geometry LAMMPS hands over, as a graph a message-passing model can eat.

Which of the two arrives is decided by the cell being all zeros or not.

GHOST CLOUD (the default, and the only MPI-capable one). Positions cover owned
atoms and their ghosts, each ghost carrying the position of the image it
stands for, with the compute's own edge list over them: a finite cluster, no
periodicity, only the owned atoms' values read. The ghost shell must reach the
model's whole range, and see compute_mad_property_python.h for why that is
still approximate beyond one message.

ONE PERIODIC CELL ('pbc yes'). Owned atoms and a real cell, so the neighbour
list is built here with periodic images. One rank.

EDGE DIRECTION. MACE and NequIP both build their lists with ASE's 'ijS'
convention -- D = pos[j] - pos[i] + S.cell, edge_index = (i, j) -- and
aggregate onto j, so the CENTRE of an environment is edge_index[1] and the
edge vector points from neighbour to centre. LAMMPS pairs the other way
round, (centre, neighbour), so the two are swapped here. Backwards, every edge
vector flips and a model with odd-parity features returns a different number
with nothing to show for it.
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
