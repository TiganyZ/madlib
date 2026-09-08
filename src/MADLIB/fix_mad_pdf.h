/* -*- c++ -*- ----------------------------------------------------------
   MADLIB - Molecular Augmented Dynamics for LAMMPS
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
  int reduced;          // 1 for D(r)
  double *chanw;        // f_ab N_a N_b / N^2 per channel
  double *dEdg;         // (nchan x nbins)

  void channel_weights();
};

}    // namespace LAMMPS_NS

#endif
#endif
