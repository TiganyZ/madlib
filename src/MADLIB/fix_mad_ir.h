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

#ifdef FIX_CLASS
// clang-format off
FixStyle(mad/ir,FixMADIr);
// clang-format on
#else

#ifndef LMP_FIX_MAD_IR_H
#define LMP_FIX_MAD_IR_H

#include "fix.h"
#include "mad_ir_spectrum.h"

#include <vector>

namespace LAMMPS_NS {
class ComputeMADProperty;

/* An infrared spectrum from the total dipole of a running trajectory.

   Prediction only. The bias is a much larger job -- the spectrum is a
   functional of the whole trajectory while only the newest frame can still be
   moved, which is what forces TurboGAP's frozen-past adjoint and its three
   alternative estimators -- and none of it is needed to report a spectrum.

   A fix rather than a compute because the sampling is the point: this has to
   see every stride-th step whether anything asks for its output or not, which
   is why fix ave/correlate is a fix too.

   Two things that surprise people, both written into the output header rather
   than left to be discovered:

   RESOLUTION IS THE LONGEST LAG, NOT THE RUN LENGTH. acf_ratio is the fraction
   of the buffer kept as lags; a long run with a short lag window gives a
   coarse spectrum, and no amount of extra sampling changes that.

   THE STORED-FRAME INTERVAL IS IN FEMTOSECONDS. LAMMPS metal units are
   picoseconds, so the conversion is real, and getting it wrong rescales the
   whole wavenumber axis while producing a spectrum that looks plausible. */

class FixMADIr : public Fix {
 public:
  FixMADIr(class LAMMPS *, int, char **);
  ~FixMADIr() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void end_of_step() override;
  void post_run() override;

  double compute_scalar() override;
  double compute_vector(int) override;
  double compute_array(int, int) override;
  double memory_usage() override;

 private:
  char *propid;
  ComputeMADProperty *prop;
  MADIrConfig cfg;
  MADIrResult res;

  std::vector<double> mu;    // (nframes x 3), chronological
  int nframes, outevery, wrote_header, have_spectrum;
  double dt_fs_user;
  char *outfile;
  bigint last_transform;

  void collect();
  bool transform_now();
  void write();
  double frame_dt_fs() const;
};

}    // namespace LAMMPS_NS

#endif
#endif
