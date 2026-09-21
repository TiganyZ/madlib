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
FixStyle(mad/xps,FixMADXps);
FixStyle(mad/nmr,FixMADXps);
FixStyle(mad/spectrum,FixMADXps);
// clang-format on
#else

#ifndef LMP_FIX_MAD_XPS_H
#define LMP_FIX_MAD_XPS_H

#include "fix_mad.h"

namespace LAMMPS_NS {
class ComputeMADProperty;

/* Bias towards a measured X-ray photoelectron spectrum, or any other spectrum
   built by broadening a per-atom scalar.

     u_l = sum_i exp( -(x_l - q_i)^2 / 2 sigma^2 )      q_i a binding energy
     y   = u / ||u||

   The chain to the atoms is short, and its shape is the reason MADLIB can be
   model-agnostic. Differentiating the loss to the per-atom property gives one
   weight per atom,

     dE/du_m = ( dE/dy_m - y_m sum_l dE/dy_l y_l ) / ||u||
     lambda_i = sum_m dE/du_m (x_m - q_i)/sigma^2 exp( -(x_m - q_i)^2/2 sigma^2 )

   and the force is then sum_i lambda_i dq_i/dr_j, which is the force any model
   already computes from a weighted per-atom energy. Nothing here knows what
   produced q_i.

   Contracting to lambda first also removes the (3, npairs, nsamples) array
   that dominates TurboGAP's memory for this observable: the work is O(N nk)
   and the storage O(N).

   mad/nmr is the same operation on chemical shifts, with one addition that
   earns it a name of its own. A model predicts a SHIELDING and an experiment
   reports a SHIFT, related by delta = a - b sigma; broadening shieldings and
   fitting them against a shift target gives a MIRRORED spectrum, which the
   scale/offset fit then happily rescales into something that looks like a fit
   and is backwards. So mad/nmr insists the convention be stated: 'reference a
   [slope b]' to convert, or 'convention shift' to say the property already is
   one. Which nucleus a spectrum is of is the fix's GROUP, since a 1H and a
   13C spectrum are two fixes over two groups of the same model.

   The conversion is a pre-map q_i -> qscale q_i + qoffset applied wherever the
   property is read, and its adjoint is one factor of qscale. */

class FixMADXps : public FixMAD {
 public:
  FixMADXps(class LAMMPS *, int, char **);
  ~FixMADXps() override;
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
  int normalise, nmax_lam;
  double unorm;
  double *u, *ulocal;
  double **lambda;

  void grow_lambda();
  double shifted(const double *q, int i) const { return qscale * q[i] + qoffset; }
};

}    // namespace LAMMPS_NS

#endif
#endif
