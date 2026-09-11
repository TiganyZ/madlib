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
ComputeStyle(mad/property/dipole,ComputeMADPropertyDipole);
// clang-format on
#else

#ifndef LMP_COMPUTE_MAD_PROPERTY_DIPOLE_H
#define LMP_COMPUTE_MAD_PROPERTY_DIPOLE_H

#include "compute_mad_property.h"

namespace LAMMPS_NS {

/* Per-atom dipole from partial charges, mu_i = q_i r_i.

   The charges come from per-type values or from atom->q, so any model that
   writes atom->q -- a charge-equilibration fix, an ML charge model -- is a
   dipole model here without further work.

   SUM q_i r_i IS NOT DEFINED FOR A PERIODIC SYSTEM. It depends on the origin
   and on which images the atoms happen to be wrapped into, and an atom
   crossing the boundary steps it. The honest treatments are the Berry phase,
   or -- for a molecular liquid, which is what an IR spectrum is usually of --
   the sum of molecular dipoles built from intramolecular vectors, which is
   origin-independent for neutral molecules and is what this does.

   So: a non-periodic box is summed directly; a periodic box needs molecule
   IDs and every molecule to be neutral; a periodic box without them is
   refused, because the alternative is a spectrum that looks entirely
   plausible and means nothing. */

class ComputeMADPropertyDipole : public ComputeMADProperty {
 public:
  ComputeMADPropertyDipole(class LAMMPS *, int, char **);
  ~ComputeMADPropertyDipole() override;

  void init() override;
  void compute_peratom() override;
  void compute_property() override;
  void adjoint(double **lambda, double *virial) override;

 private:
  double *bytype;
  int use_atom_q, molecular_mode, nmax_q;
  double charge_tol;

  double charge(int i) const;
  void check_neutrality();
};

}    // namespace LAMMPS_NS

#endif
#endif
