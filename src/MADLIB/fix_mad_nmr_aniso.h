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
FixStyle(mad/nmr/aniso,FixMADNmrAniso);
// clang-format on
#else

#ifndef LMP_FIX_MAD_NMR_ANISO_H
#define LMP_FIX_MAD_NMR_ANISO_H

#include "fix_mad.h"

#include <vector>

namespace LAMMPS_NS {
class ComputeMADProperty;

/* A static powder NMR lineshape, from per-atom shielding TENSORS.

   fix mad/nmr broadens one number per nucleus, so every site is a symmetric
   peak and the anisotropy is gone before the comparison. Here a crystallite
   whose field direction is n resonates at omega_i(n) = n . T_i . n, and the
   spectrum is that over orientations: a skewed lineshape with edges at the
   principal values.

     u_k = sum_i sum_m w_m exp( -(x_k - x_i(n_m))^2 / 2 sigma^2 )
     y   = u / ||u||

   x_i = a - b omega_i is the shielding-to-shift map, which this fix demands be
   stated for the reason fix mad/nmr does: broadening one against the other
   mirrors the spectrum, and the scale fit then hides it.

   The property is the SYMMETRIC tensor in Voigt order (xx yy zz xy xz yz),
   ncomp 6, with no factor of two in the off-diagonal components -- the two in
   omega = n.T.n lives here, in v(n). Nothing diagonalises anything, so the
   average needs no principal values, no ordering convention and no eigenvalue
   derivatives.

   omega is linear in the tensor, omega_i(n) = v(n) . T_i with

     v(n) = (n_x^2, n_y^2, n_z^2, 2 n_x n_y, 2 n_x n_z, 2 n_y n_z)

   so the gradient is the mad/xps chain times v_c(n_m).

   ORIENTATIONS IS A CONVERGENCE PARAMETER. Too coarse a grid resolves the
   pattern's edges into a comb of Gaussians smooth enough to look like a
   spectrum. An exact powder average is rotation-independent and so exerts no
   torque; what a finite grid leaves is an antisymmetric part of sum r (x) f,
   falling as 1/N, which is the sharpest check that N is large enough. */

class FixMADNmrAniso : public FixMAD {
 public:
  FixMADNmrAniso(class LAMMPS *, int, char **);
  ~FixMADNmrAniso() override;
  void init() override;

 protected:
  void compute_spectrum() override;
  void accumulate_bias(const double *dEdy) override;

 protected:
  enum { N_L2, N_AREA, N_NONE };

  char *propid;
  ComputeMADProperty *prop;
  double sigma, emin, emax, trunc;
  double qscale, qoffset;
  int normalise, nmax_lam, norient;
  double unorm;
  double *u, *ulocal;
  double **lambda;
  std::vector<double> nvec;    // (norient x 6), the v(n_m) rows
  double wquad;                // equal quadrature weight

  void build_orientations();
  void grow_lambda();

  // omega_i(n_m) mapped onto the target's axis
  double shifted(const double *t6, int m) const
  {
    const double *v = &nvec[(size_t) m * 6];
    double w = 0.0;
    for (int c = 0; c < 6; c++) w += v[c] * t6[c];
    return qscale * w + qoffset;
  }
};

}    // namespace LAMMPS_NS

#endif
#endif
