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

     mad_compute(positions, types, edge_i, edge_j, nlocal, ncomp) -> q
     mad_adjoint(positions, types, edge_i, edge_j, nlocal, ncomp, lam) -> f

   with q of shape (nlocal,) or (nlocal, ncomp), and f of shape (nall, 3) equal
   to -d/dr sum_i lam_i . q_i -- one backward pass for a torch model.

   Positions cover owned atoms AND ghosts, and a ghost carries the position of
   the periodic image it stands for, so the edge list plus these coordinates
   fix the geometry completely: no cell and no boundary conditions are passed,
   and the model needs to know nothing about domain decomposition. Forces
   returned for ghosts are communicated back to their owners.

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
  int nmax_q;
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
