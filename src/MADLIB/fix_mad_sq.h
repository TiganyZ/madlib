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
FixStyle(mad/sq,FixMADSq);
FixStyle(mad/xrd,FixMADSq);
// clang-format on
#else

#ifndef LMP_FIX_MAD_SQ_H
#define LMP_FIX_MAD_SQ_H

#include "fix_mad.h"

namespace LAMMPS_NS {
class ComputeMADPdf;
class MADFormFactors;

/* Structure factor, X-ray and neutron diffraction, from the partial pair
   distributions.

   S_ab(q) - d_ab = 4 pi rho sqrt(c_a c_b) sum_l dr r_l^2 (g_ab(r_l) - 1)
                                            sinc(q r_l) w(r_l)

   The three observables differ only in the weight applied to the interference
   part, so they are one class:

     sq       W_ab(q) = f_ab sqrt(N_a N_b)/N            const 1
     xray     W_ab(q) = f_ab sqrt(c_a c_b) f_a(q) f_b(q)  const sum_a c_a f_a^2
     neutron  the same with bound scattering lengths

   and the output conventions are affine maps of the result with
   position-independent coefficients, so one gradient channel per q carries
   all of them.

   The abscissa held internally is x = 2 sin(theta)/lambda = q/2pi, not q, and
   form factors are sampled at x/2. This is TurboGAP's convention and the
   easiest thing in the whole calculation to get wrong by 2 pi. */

class FixMADSq : public FixMAD {
 public:
  FixMADSq(class LAMMPS *, int, char **);
  ~FixMADSq() override;
  void init() override;

 protected:
  void compute_spectrum() override;
  void accumulate_bias(const double *dEdy) override;

 private:
  enum { W_SQ, W_XRAY, W_NEUTRON };
  enum { O_RAW, O_FQ, O_QFQ };

  char *pdfid;
  ComputeMADPdf *pdf;
  MADFormFactors *ff;

  int nq, weighting, output, window, lorentz;
  double qmin, qmax, wavelength, polarization, sin_theta_min, window_cut;
  int q_in_angle;

  double *xg;        // 2 sin(theta)/lambda
  double *qg;        // the user's abscissa, for output
  double *mat;       // (nq x nbins) sinc kernel
  double *interf;    // interference part per q
  double *chanf;     // lp * channel: dy/d(interf)
  double *constf;    // lp * const
  double *wab;       // (nchan x nq) per-channel weight
  double *sab;       // (nchan x nq) S_ab - delta_ab
  double *dEdg;      // (nchan x nbins)

  void build_grid(const std::vector<double> &user_q);
  void build_kernel();
  void build_weights();
};

}    // namespace LAMMPS_NS

#endif
#endif
