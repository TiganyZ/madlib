/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(mad/property/python,ComputeMADPropertyPython);
// clang-format on
#else

#ifndef LMP_COMPUTE_MAD_PROPERTY_PYTHON_H
#define LMP_COMPUTE_MAD_PROPERTY_PYTHON_H

#include "compute_mad_property.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

/* A per-atom property from a model written in Python.

   Any framework reachable from Python is a backend here: MACE, GRACE, a torch
   module, a plain numpy expression. The module supplies two functions,

     mad_compute(positions, types, edge_i, edge_j, nlocal, ncomp, cell) -> q
     mad_adjoint(positions, types, edge_i, edge_j, nlocal, ncomp, lam, cell) -> f

   with q of shape (nlocal,) or (nlocal, ncomp), and f equal to
   -d/dr sum_i lam_i . q_i -- one backward pass for a torch model.

   TWO WAYS TO SAY WHERE THE ATOMS ARE, and the model picks by looking at cell.

   By default cell is nine zeros and positions cover owned atoms AND ghosts,
   each ghost carrying the position of the periodic image it stands for. The
   edge list plus those coordinates fix the geometry completely, the model
   needs to know nothing about domain decomposition or boundaries, f is
   (nall, 3), and forces returned for ghosts are communicated back to their
   owners. This is the general case and the only one that works under MPI.

   With 'pbc yes', cell holds the three cell vectors, positions are the owned
   atoms only, and a model that handles periodicity itself is left to. This is
   for a model with a long interaction range on a small cell, where the ghost
   route is not merely slower but absurd: ShiftML passes two messages over a
   5 A neighbour list, so a 28-atom molecular crystal needs a 10 A ghost shell,
   which is 3304 atoms -- 118 copies of the cell, and 8x the model time of just
   handing over the 28 and the cell. It is one rank only, by construction.

   The virial cannot then come from sum f (x) r over owned atoms, because the
   images that carry the rest of it are inside the model rather than in the
   atom arrays. So in this mode f is (nlocal + 3, 3): the first nlocal rows are
   the forces and the last three are -dE/dh, the gradient with respect to the
   three cell vectors, which is the missing half of

       W_ab = sum_i r_ia f_ib + sum_c h_ca (-dE/dh_c)_b

   and is one more output of the same backward pass.

   Arrays cross as memoryviews over LAMMPS's own storage, so nothing is copied
   and the package needs libpython but not numpy's C API. */

class ComputeMADPropertyPython : public ComputeMADProperty {
 public:
  ComputeMADPropertyPython(class LAMMPS *, int, char **);
  ~ComputeMADPropertyPython() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  void compute_property() override;
  void adjoint(double **lambda, double *virial) override;
  double memory_usage() override;

 private:
  std::string module_name;
  double rcut;
  int nmax_q, pbc_mode;
  double cellbuf[9];
  class NeighList *list;

  void *py_module, *py_compute, *py_adjoint;    // PyObject*, hidden from the header

  std::vector<int> edge_i, edge_j;
  std::vector<double> pos, lam_flat;
  std::vector<int> type_flat;

  void build_edges();
  void call_python(void *fn, const double *lam, double *out, int nout);
  void fail(const char *what);
};

}    // namespace LAMMPS_NS

#endif
#endif
