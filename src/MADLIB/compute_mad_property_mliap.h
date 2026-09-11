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
ComputeStyle(mad/property/mliap,ComputeMADPropertyMliap);
// clang-format on
#else

#ifndef LMP_COMPUTE_MAD_PROPERTY_MLIAP_H
#define LMP_COMPUTE_MAD_PROPERTY_MLIAP_H

#include "compute_mad_property.h"

namespace LAMMPS_NS {
class MLIAPData;
class MLIAPModel;
class MLIAPDescriptor;

/* A per-atom property from an ML-IAP model.

   An ML-IAP model maps descriptors to a per-atom scalar. That scalar is
   normally an energy, but nothing in the machinery requires it to be: fitted
   against core-electron binding energies instead, the same linear, quadratic or
   neural-network model is a binding-energy model, and every descriptor ML-IAP
   supports -- SNAP, SO3, ACE -- comes with it.

   The adjoint costs nothing extra, which is the point. ML-IAP already computes
   beta_ik = dq_i/dB_ik and contracts it with dB/dr to make forces; scaling the
   betas by lambda_i before that contraction gives sum_i lambda_i dq_i/dr_j
   directly. */

class ComputeMADPropertyMliap : public ComputeMADProperty {
 public:
  ComputeMADPropertyMliap(class LAMMPS *, int, char **);
  ~ComputeMADPropertyMliap() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  void compute_property() override;
  void adjoint(double **lambda, double *virial) override;
  double memory_usage() override;

 private:
  MLIAPData *data;
  MLIAPModel *model;
  MLIAPDescriptor *descriptor;
  int *map;
  int nmax_q, built;
  class NeighList *list;

  void run_model();
};

}    // namespace LAMMPS_NS

#endif
#endif
