/* -*- c++ -*- ----------------------------------------------------------
   MADLIB - Molecular Augmented Dynamics for LAMMPS
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(mad/property/coord,ComputeMADPropertyCoord);
// clang-format on
#else

#ifndef LMP_COMPUTE_MAD_PROPERTY_COORD_H
#define LMP_COMPUTE_MAD_PROPERTY_COORD_H

#include "compute_mad_property.h"

namespace LAMMPS_NS {

/* A smooth coordination number, as a per-atom property.

   q_i = sum_j fc(r_ij),   fc = 1 below r1, a raised cosine to r2, 0 beyond.

   No model, no data files, an analytic gradient in six lines -- which is the
   point of it. It lets the whole XPS chain (spectrum, normalisation, loss,
   lambda, adjoint, reverse communication, virial) be finite-differenced before
   any ML backend is involved, so a failure there is unambiguous. It is also a
   real descriptor: a coordination-resolved spectrum is a reasonable toy XPS. */

class ComputeMADPropertyCoord : public ComputeMADProperty {
 public:
  ComputeMADPropertyCoord(class LAMMPS *, int, char **);
  ~ComputeMADPropertyCoord() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  void compute_property() override;
  void adjoint(double **lambda, double *virial) override;

 private:
  double r1, r2, invdr;
  int nmax_q;
  class NeighList *list;
};

}    // namespace LAMMPS_NS

#endif
#endif
