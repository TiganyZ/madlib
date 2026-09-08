/* -*- c++ -*- ----------------------------------------------------------
   MADLIB - Molecular Augmented Dynamics for LAMMPS
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(mad/pdf,ComputeMADPdf);
// clang-format on
#else

#ifndef LMP_COMPUTE_MAD_PDF_H
#define LMP_COMPUTE_MAD_PDF_H

#include "compute.h"

namespace LAMMPS_NS {

/* Partial pair distribution functions g_ab(r), differentiably.

   g_ab(r_l) = V/(N_a N_b f_ab) * 1/(sqrt(2 pi) sigma 4 pi r_l^2)
               * sum_(i,j) exp( -(r_l - r_ij)^2 / 2 sigma^2 )

   over ORDERED pairs of the channel, f_ab = 1 if a == b else 2. The Gaussian
   is what makes this differentiable at all: a histogram has zero gradient
   almost everywhere, so a bias built on one applies no force.

   Channels run over the upper triangle (a <= b), row-major.

   The list is full rather than half, so that the ordered pair (i,j) is seen on
   the rank owning i and (j,i) on the rank owning j. Every atom's force is then
   complete without communication, which matters because Verlet reverse-
   communicates before post_force() and a ghost force added there is lost. */

class ComputeMADPdf : public Compute {
 public:
  ComputeMADPdf(class LAMMPS *, int, char **);
  ~ComputeMADPdf() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_array() override;

  // Given dL/dg (nbins x nchan, row-major), add -dL/dr to f and the pair and
  // volume terms to virial[6]. Assumes compute_array() ran this step.
  void adjoint(const double *dEdg, double *virial);

  int n_channels() const { return nchan; }
  int n_bins() const { return nbins; }
  int channel(int itype, int jtype) const { return chan[itype * (ntypes + 1) + jtype]; }
  double bin_r(int l) const { return rgrid[l]; }
  double concentration(int a) const { return conc[a]; }
  int n_of_type(int a) const { return count[a]; }
  double cutoff() const { return rcut; }
  double r_max() const { return rmax; }
  double kde_sigma() const { return sigma; }
  const double *g() const { return gpart; }

  double memory_usage() override;

 private:
  int nbins, nchan, ntypes, ntrunc;
  double rmin, rmax, rcut, sigma, trunc, delr, delrinv;
  bool compat;
  bigint invoked_types;

  int *chan;         // (ntypes+1)^2 lookup, -1 for an unused type
  int *count;        // atoms of each type
  double *conc;      // count / natoms
  double *rgrid;     // bin centres
  double *shellinv;  // 1 / (sqrt(2pi) sigma 4 pi r_l^2)
  double *cnorm;     // per-channel V/(N_a N_b f_ab)
  double *gpart;     // (nbins x nchan) the reduced g_ab
  double *glocal;    // this rank's partial accumulation

  class NeighList *list;

  void count_types();
  void setup_grid();
};

}    // namespace LAMMPS_NS

#endif
#endif
