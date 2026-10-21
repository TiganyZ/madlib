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
class MADTarget;

/* An infrared spectrum from the total dipole of a running trajectory.

   Prediction only. The bias is a much larger job -- the spectrum is a
   functional of the whole trajectory while only the newest frame can still be
   moved, which is what forces TurboGAP's frozen-past adjoint and its three
   alternative estimators -- and none of it is needed to report a spectrum.

   A fix rather than a compute because the sampling is the point: this has to
   see every stride-th step whether anything asks for its output or not, which
   is why fix ave/correlate is a fix too.

   THE DIPOLE IS WHATEVER THE NAMED COMPUTE PRODUCES, and the only style that
   produces one today is mad/property/dipole, from partial charges. TurboGAP
   drives its IR from a fitted dipole model instead -- see the note in
   compute_mad_property_dipole.h. The estimator below matches TurboGAP's
   ir_fft to machine precision, but that test hands both codes the same stored
   dipoles: it verifies the spectrum, not the dipole feeding it.

   Two things that surprise people, both written into the output header rather
   than left to be discovered:

   WITH A TARGET AND A GAMMA IT BIASES, AND WHAT IT APPLIES IS NOT A GRADIENT.
   A spectrum is a functional of the whole trajectory while only the newest
   frame can still be moved, so what the force comes from is the FROZEN-PAST
   partial derivative: differentiate holding every earlier frame fixed. The
   reported bias energy is therefore NOT the generator of the reported force,
   which is why this fix contributes nothing to the potential energy -- adding
   a non-generating term to pe makes the total energy's drift unreadable as an
   integration diagnostic, and that has already cost one spectrum its
   interpretation elsewhere. The loss is on the global vector instead.

   WHAT IS CORRELATED DECIDES WHETHER THE BIAS WORKS AT ALL, and the
   difference is not subtle. Under correlate "dipole" the multiplier lands in
   phase with each mode's DISPLACEMENT, so it retunes bands rather than doing
   work on them; measured against the Downing water spectrum that scored worse
   than no bias at all. Under "rate" it lands in phase with each mode's
   VELOCITY, pumping a band the experiment says is too weak and draining one
   it says is too strong -- a frequency-resolved friction, which is to say a
   generalized Langevin bias with the target spectrum in the part of the bath.
   Same comparison, better by more than an order of magnitude. So the bias
   wants correlate rate and qc none; nothing here enforces that pairing,
   because "which estimator" is a question the prediction path asks too.

   WITH A TARGET IT SCORES, AND WITHOUT A GAMMA THAT IS ALL IT DOES. Giving this fix an
   experimental spectrum buys a misfit and nothing else: no force, for the
   reason above. That is enough to make the spectrum an objective -- which is
   what a search over structures needs -- without pretending the gradient
   exists.

   The comparison window is not the prediction grid. The grid starts at nu = 0
   and runs to nu_max; a measured curve covers neither end, and MADTarget
   refuses to extrapolate. So the overlap is taken, and it is recomputed every
   transform because n_freq grows while the buffer fills.

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
  void post_force(int) override;
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

  MADTarget *target;
  double fit_nu_min, fit_nu_max;
  double rel_err, dissim;
  int n_fit;
  std::vector<double> fit_nu, fit_y;
  std::vector<int> fit_idx;    // where each fit point sits on the spectrum grid

  int biasing, nevery_kernel, bias_live, trace;
  bigint warmup, kernel_step;
  double gamma_beg, gamma_end, gamma_now, e_bias;
  double dLdmu[3];             // per unit of the newest total dipole
  std::vector<double> kern;    // S(tau) = dL/dC(tau)
  std::vector<double> dLdI;    // on the spectrum's own grid
  double **lam;                // (nmax x 3) handed to the compute's adjoint
  int nmax_lam;

  void collect();
  bool transform_now();
  void write();
  double frame_dt_fs() const;

  // Resample the target onto the overlap and leave the misfit in rel_err.
  void score();

  void ramp_gamma();
  void build_kernel();    // the expensive half, every nevery_kernel steps
  void update_bias();     // the cheap half, every time a frame is stored
  void grow_lambda();
};

}    // namespace LAMMPS_NS

#endif
#endif
