# Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
"""The NMR shielding span from ShiftML, as a MADLIB python-backend model.

Span is the Herzfeld-Berger anisotropy: the largest principal value of the
symmetric shielding tensor minus the smallest, in ppm. Unlike the isotropic
shielding it distinguishes sites by the SHAPE of their electronic environment,
so a carbonyl and a hydroxyl oxygen that happened to share a centre of gravity
are still two different numbers.

Everything is in mad_shiftml; this module only selects which of the quantities
that one forward pass produced is handed to the compute. A deck using both pays
for one model evaluation a step.
"""

from mad_shiftml import adjoint_of, compute_of


def mad_compute(positions, types, edge_i, edge_j, nlocal, ncomp, cell):
    return compute_of("span", positions, types, edge_i, edge_j, nlocal, ncomp, cell)


def mad_adjoint(positions, types, edge_i, edge_j, nlocal, ncomp, lam, cell):
    return adjoint_of("span", positions, types, edge_i, edge_j, nlocal, ncomp,
                      lam, cell)
