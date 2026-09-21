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

#ifndef LMP_MAD_FORM_FACTORS_H
#define LMP_MAD_FORM_FACTORS_H

#include <string>
#include <vector>

namespace LAMMPS_NS {

/* Scattering power per species on a fixed abscissa.

   Built once per fix from the species labels, then indexed by species rather
   than searched, because the pdf route evaluates f_a(q) inside the q loop for
   every partial channel. */

class MADFormFactors {
 public:
  MADFormFactors(bool neutron, bool compat);

  // Resolves a label against the tables. Returns false if it is absent.
  bool add_species(const std::string &label);

  // f_a(x_k) on the grid x = 2 sin(theta)/lambda = q/2pi, for every species
  // added so far. Row-major (nspecies, nsamples).
  void tabulate(const std::vector<double> &x);

  double f(int ispecies, int k) const { return table[ispecies * nsamples + k]; }
  const double *row(int ispecies) const { return &table[ispecies * nsamples]; }
  int n_species() const { return (int) species.size(); }
  bool is_neutron() const { return neutron; }

  static bool known(const std::string &label, bool neutron);

 private:
  bool neutron, compat;
  int nsamples;
  std::vector<int> species;    // index into the static table
  std::vector<std::string> names;
  std::vector<double> table;

  static int lookup(const std::string &label, bool neutron);
  double evaluate(int entry, double s) const;
};

}    // namespace LAMMPS_NS

#endif
