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
FixStyle(mad/pdf,FixMADPdf);
// clang-format on
#else

#ifndef LMP_FIX_MAD_PDF_H
#define LMP_FIX_MAD_PDF_H

#include "fix_mad.h"

namespace LAMMPS_NS {
class ComputeMADPdf;

/* Bias towards a measured pair distribution.

   g(r) = sum_(a<=b) f_ab (N_a N_b / N^2) g_ab(r)
   D(r) = 4 pi rho r ( g(r) - 1 )

   Both are affine in the partials with position-independent coefficients, so
   one weight per channel per bin carries the adjoint through either. */

class FixMADPdf : public FixMAD {
 public:
  FixMADPdf(class LAMMPS *, int, char **);
  ~FixMADPdf() override;
  void init() override;

 protected:
  void compute_spectrum() override;
  void accumulate_bias(const double *dEdy) override;

 private:
  char *pdfid;
  ComputeMADPdf *pdf;
  int reduced;      // 1 for D(r)
  double *chanw;    // f_ab N_a N_b / N^2 per channel
  double *dEdg;     // (nchan x nbins)

  void channel_weights();
};

}    // namespace LAMMPS_NS

#endif
#endif
