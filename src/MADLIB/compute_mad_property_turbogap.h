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
ComputeStyle(mad/property/turbogap,ComputeMADPropertyTurbogap);
// clang-format on
#else

#ifndef LMP_COMPUTE_MAD_PROPERTY_TURBOGAP_H
#define LMP_COMPUTE_MAD_PROPERTY_TURBOGAP_H

#include "compute_mad_property.h"
#include "mad_gap_file.h"

#include <vector>

namespace LAMMPS_NS {

/* A per-atom property from a TurboGAP soap_turbo GPR.

     q_i = V0 + delta^2 sum_s alpha_s ( soap_i . Q_s )^zeta

   An existing .gap file works unchanged, which is the point: a fitted
   core-electron-binding-energy model is months of work and there is no reason
   to refit one to move it.

   A model is normally one block per central species -- C 1s and O 1s have
   different V0 and often a different zeta from the energy GAP sharing their
   block -- so every block is evaluated over the atoms whose species it claims,
   and the results go into one array.

   Needs the Fortran backend in lib/madlib. Without it the style is still
   registered, so the error says what to build rather than "unknown compute". */

class ComputeMADPropertyTurbogap : public ComputeMADProperty {
 public:
  ComputeMADPropertyTurbogap(class LAMMPS *, int, char **);
  ~ComputeMADPropertyTurbogap() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  void compute_property() override;
  void adjoint(double **lambda, double *virial) override;
  double memory_usage() override;

 private:
  struct Batch {
    std::vector<int> n_neigh, species_idx, jatom, site_atom;
    std::vector<double> rjs, thetas, phis, xyz;
    std::vector<double> q, qder;
  };

  std::vector<MADGapBlock> blocks;
  std::vector<Batch> batch;
  std::vector<int> type2species;    // LAMMPS type -> index in the block's species list
  double rcut, rcutsq;
  int nmax_q, built;
  class NeighList *list;

  void build_batches(bool derivatives);
};

}    // namespace LAMMPS_NS

#endif
#endif
