/* -*- c++ -*- ----------------------------------------------------------
   MADLIB - Molecular Augmented Dynamics for LAMMPS
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(mad/property/table,ComputeMADPropertyTable);
// clang-format on
#else

#ifndef LMP_COMPUTE_MAD_PROPERTY_TABLE_H
#define LMP_COMPUTE_MAD_PROPERTY_TABLE_H

#include "compute_mad_property.h"

namespace LAMMPS_NS {

/* A per-atom property read from a file or given per type.

   The gradient is identically zero, so this predicts a spectrum but exerts no
   force. It is what to use when the per-atom quantities came from somewhere
   else -- a DFT calculation, an external model -- and the question is what
   spectrum they imply, not how to bias towards one. Asking it for forces is an
   error rather than a silent no-op. */

class ComputeMADPropertyTable : public ComputeMADProperty {
 public:
  ComputeMADPropertyTable(class LAMMPS *, int, char **);
  ~ComputeMADPropertyTable() override;

  void init() override {}
  void compute_peratom() override;
  void compute_property() override;
  void adjoint(double **lambda, double *virial) override;

 private:
  double *bytype;
  double *byid;          // indexed by atom tag, 1..natoms
  tagint maxtag;
  int nmax_q;

  void read_file(const char *);
};

}    // namespace LAMMPS_NS

#endif
#endif
