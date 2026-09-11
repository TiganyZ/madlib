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

#ifndef LMP_FIX_MAD_H
#define LMP_FIX_MAD_H

#include "fix.h"

#include <vector>

namespace LAMMPS_NS {
class MADTarget;

/* Base for every experimentally biased observable.

   E = 1/2 gamma sum_k w_k ( s y_k({r}) + b - yexp_k )^2
   f_i = - dE/dr_i

   This class owns the target, the gamma ramp, the loss, the diagnostics and
   the LAMMPS energy/virial plumbing. A derived observable supplies the
   spectrum and its adjoint, and nothing else.

   The energy and virial contribute by default (thermo_energy = thermo_virial
   = 1, as fix plumed does), so the bias is priced by compute pe and therefore
   by every MC fix without further ceremony. */

class FixMAD : public Fix {
 public:
  FixMAD(class LAMMPS *, int, char **);
  ~FixMAD() override;

  int setmask() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void min_post_force(int) override;

  double compute_scalar() override;
  double compute_vector(int) override;

 protected:
  MADTarget *target;
  double gamma_beg, gamma_end, gamma_now;
  double e_bias;
  int nk, nevery_bias;
  bool compat, wrote_header;
  std::vector<double> xgrid, y_pred, dEdy;
  char *outfile;
  int outevery;

  // Fill y_pred on every rank identically -- the residual drives the force, so
  // a rank holding a partial spectrum would build a wrong one.
  virtual void compute_spectrum() = 0;

  // Add -dE/dr to atom->f and the stress to virial[6]. Each rank contributes
  // its own share; LAMMPS all-reduces both.
  virtual void accumulate_bias(const double *dEdy) = 0;

  // Consumes the keywords every observable shares. Returns 1 if arg[iarg] was
  // one of them.
  int parse_common(int narg, char **arg, int &iarg);
  void finish_setup(const std::vector<double> &grid);
  void write_spectrum();

 private:
  void apply(int vflag);
};

}    // namespace LAMMPS_NS

#endif
