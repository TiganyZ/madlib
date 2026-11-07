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

/* Bias towards a static powder NMR lineshape, from per-atom shielding
   TENSORS rather than from isotropic shieldings.

   fix mad/nmr broadens one number per nucleus, so every site enters the
   spectrum as a symmetric peak and the anisotropy of its electronic
   environment is thrown away before the comparison. A static powder spectrum
   is the anisotropy: a crystallite whose field direction is n resonates at

     omega_i(n) = n . T_i . n

   and what is measured is that over all orientations, which is a skewed
   lineshape with edges at the principal values. Two sites with the same
   isotropic shielding and different tensors are then two different patterns,
   which is the whole reason for predicting a tensor.

     u_k = sum_i sum_m w_m exp( -(x_k - x_i(n_m))^2 / 2 sigma^2 )
     y   = u / ||u||

   with {n_m, w_m} a quadrature over orientations and x_i = a - b omega_i the
   shielding-to-shift map that fix mad/nmr also applies, and demands be stated
   for the same reason: broadening shieldings against a shift target mirrors
   the spectrum, and the scale/offset fit then turns the mirror image into
   something that looks like a fit.

   The property is a SYMMETRIC tensor in Voigt order, ncomp 6:

     T_xx  T_yy  T_zz  T_xy  T_xz  T_yz

   An NMR experiment sees only the symmetric part, so a model predicting a
   general rank-2 tensor symmetrises it on the way out. Only the orientation
   dependence is kept here, and nothing in this fix diagonalises anything: the
   quadrature needs no principal values, no ordering convention and no
   eigenvalue derivatives, and so has no branch for degenerate sites to cross.

   The gradient is the mad/xps chain with one factor more. With

     v(n) = (n_x^2, n_y^2, n_z^2, 2 n_x n_y, 2 n_x n_z, 2 n_y n_z)

   the orientation average is linear in the tensor, omega_i(n) = v(n) . T_i, so

     lambda_(i,c) = b_scale sum_m w_m v_c(n_m)
                    sum_k dE/du_k (x_k - x_i(n_m)) / sigma^2 exp(...)

   and the force is the one weighted backward pass any property backend
   already supplies.

   ORIENTATIONS ARE A CONVERGENCE PARAMETER, not a cosmetic one. A powder
   pattern has edges at the principal values, and too coarse a quadrature
   resolves them into a comb of Gaussians that is smooth enough to look like a
   spectrum. The default is a golden-spiral grid over the hemisphere -- a
   hemisphere because omega(n) = omega(-n) -- and the check is to raise
   'orientations' until the spectrum stops moving, at the sigma being used. */

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
