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

#ifndef LMP_COMPUTE_MAD_PROPERTY_H
#define LMP_COMPUTE_MAD_PROPERTY_H

#include "compute.h"

namespace LAMMPS_NS {

/* A per-atom quantity predicted by some model, and its adjoint.

   Observables like XPS depend on the positions only through a per-atom scalar
   q_i -- a core-electron binding energy, a charge -- or a per-atom vector,
   such as a local dipole. What the bias needs back is not the full Jacobian
   dq_i/dr_j but its contraction with a weight:

       f_j -= sum_i,c  lambda_(i,c)  d q_(i,c) / d r_j

   and that is exactly the force a model already computes from a per-atom
   energy sum_i lambda_i q_i with the weights held fixed. Every ML potential
   framework can do it: a GAP by scaling its kernel derivative row, MLIAP by
   scaling betas, a torch model by one backward pass. So a new backend is this
   one method plus a forward pass, and MADLIB is model-agnostic for the price
   of a two-method interface.

   dq_i/dr_j reaches the neighbours of i, ghosts included, and Verlet reverse-
   communicates before post_force() -- so a force written to a ghost there is
   silently lost. Derived classes therefore accumulate into fadj, which covers
   nlocal+nghost, and call scatter_adjoint() to reverse-communicate it and add
   the result to atom->f. */

class ComputeMADProperty : public Compute {
 public:
  ComputeMADProperty(class LAMMPS *, int, char **);
  ~ComputeMADProperty() override;

  int n_components() const { return ncomp; }

  // q for every local atom. Not cached on the timestep: an MC fix re-evaluates
  // within one, and a stale property would hide every trial move.
  virtual void compute_property() = 0;

  // lambda is (nlocal x ncomp). Adds -dE/dr to atom->f and the stress to
  // virial[6]; each rank contributes its own share.
  virtual void adjoint(double **lambda, double *virial) = 0;

  double *scalar_property() { return vector_atom; }
  double **vector_property() { return array_atom; }

  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage() override;

 protected:
  int ncomp, nmax_adj;
  double **fadj;

  void grow_adjoint();
  void zero_adjoint();
  void scatter_adjoint();
};

}    // namespace LAMMPS_NS

#endif
