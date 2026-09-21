/* ----------------------------------------------------------------------
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

#include "fix_mad.h"

#include "atom_masks.h"
#include "atom.h"
#include "comm.h"

#ifdef LMP_KOKKOS
#include "atom_kokkos.h"
#include "kokkos.h"
#endif
#include "error.h"
#include "mad_target.h"
#include "update.h"

#include <cstring>
#include <stdexcept>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixMAD::FixMAD(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), target(nullptr), gamma_beg(0.0), gamma_end(0.0), gamma_now(0.0),
    e_bias(0.0), nk(0), nevery_bias(1), compat(false), wrote_header(false), outfile(nullptr),
    outevery(0)
{
  scalar_flag = 1;
  extscalar = 0;    // a global collective quantity, not extensive in N
  vector_flag = 1;
  size_vector = 6;
  extvector = 0;
  global_freq = 1;
  energy_global_flag = virial_global_flag = 1;
  thermo_energy = thermo_virial = 1;

  // What this fix actually touches. Fix's own defaults are ALL_MASK for both,
  // which claims it writes positions -- and under KOKKOS that claim is acted
  // on: LAMMPS marks atom:x modified on the host after every invocation while
  // the minimiser holds x on the device, so the next access finds both flags
  // set and Kokkos aborts the run with
  //   Concurrent modification of host and device views in DualView "atom:x"
  // The bias reads coordinates and adds to forces. It has never written a
  // coordinate.
  datamask_read = X_MASK | F_MASK | TYPE_MASK | MASK_MASK;
  // EMPTY rather than F_MASK, and apply() marks f itself. ModifyKokkos applies
  // datamask_modify AFTER post_force returns, so declaring F_MASK here leaves
  // the host holding the only claim on f at the moment the fix hands back --
  // and MinFireKokkos::run_iterate ends its loop with a bare
  //   atomKK->modified(Device, X_MASK | V_MASK | F_MASK);
  // and no sync before it. Both sides then claim f and Kokkos aborts. Doing
  // the marking inside apply() lets the sync that follows it close that
  // window; see the end of apply().
  datamask_modify = EMPTY_MASK;

  target = new MADTarget(error);
}

/* ---------------------------------------------------------------------- */

FixMAD::~FixMAD()
{
  delete target;
  delete[] outfile;
}

/* ---------------------------------------------------------------------- */

int FixMAD::parse_common(int narg, char **arg, int &iarg)
{
  if (strcmp(arg[iarg], "target") == 0) {
    if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix mad target", error);
    try {
      target->read(arg[iarg + 1]);
    } catch (std::exception &e) {
      error->all(FLERR, e.what());
    }
    iarg += 2;
  } else if (strcmp(arg[iarg], "gamma") == 0) {
    if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix mad gamma", error);
    gamma_beg = gamma_end = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
    iarg += 2;
    // A second number makes it a ramp, which is how a restraint is switched on
    // without kicking the structure apart on step one.
    if (iarg < narg && utils::is_double(arg[iarg])) {
      gamma_end = utils::numeric(FLERR, arg[iarg], false, lmp);
      iarg++;
    }
  } else if (strcmp(arg[iarg], "every") == 0) {
    if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix mad every", error);
    nevery_bias = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
    if (nevery_bias < 1) error->all(FLERR, "fix mad every must be positive");
    iarg += 2;
  } else if (strcmp(arg[iarg], "fit") == 0) {
    if (iarg + 2 >= narg) utils::missing_cmd_args(FLERR, "fix mad fit", error);
    const bool s = (strcmp(arg[iarg + 1], "yes") == 0);
    const bool b = (strcmp(arg[iarg + 2], "yes") == 0);
    target->set_fit(s, b);
    iarg += 3;
  } else if (strcmp(arg[iarg], "file") == 0) {
    if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix mad file", error);
    delete[] outfile;
    outfile = utils::strdup(arg[iarg + 1]);
    if (outevery == 0) outevery = 1;
    iarg += 2;
  } else if (strcmp(arg[iarg], "write_every") == 0) {
    if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix mad write_every", error);
    outevery = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
    iarg += 2;
  } else if (strcmp(arg[iarg], "compat") == 0) {
    if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix mad compat", error);
    compat = (strcmp(arg[iarg + 1], "turbogap") == 0);
    iarg += 2;
  } else {
    return 0;
  }
  return 1;
}

/* ---------------------------------------------------------------------- */

void FixMAD::finish_setup(const std::vector<double> &grid)
{
  xgrid = grid;
  nk = (int) grid.size();
  try {
    target->set_grid(xgrid, !compat);
  } catch (std::exception &e) {
    error->all(FLERR, e.what());
  }
  y_pred.assign(nk, 0.0);
  dEdy.assign(nk, 0.0);
}

/* ---------------------------------------------------------------------- */

int FixMAD::setmask()
{
  return POST_FORCE | MIN_POST_FORCE;
}

void FixMAD::setup(int vflag)
{
  post_force(vflag);
}

void FixMAD::min_setup(int vflag)
{
  post_force(vflag);
}

void FixMAD::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixMAD::post_force(int vflag)
{
  if (nevery_bias > 1 && (update->ntimestep % nevery_bias)) return;
  apply(vflag);
}

void FixMAD::apply(int vflag)
{
  v_init(vflag);
  for (int i = 0; i < 6; i++) virial[i] = 0.0;

  // Linear in the fraction of the run elapsed, exactly as TurboGAP ramps
  // exp_energy_scales towards exp_energy_scales_final.
  if (gamma_end == gamma_beg || update->endstep <= update->beginstep) {
    gamma_now = gamma_end;
  } else {
    double t = (double) (update->ntimestep - update->beginstep) /
        (double) (update->endstep - update->beginstep);
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    gamma_now = (1.0 - t) * gamma_beg + t * gamma_end;
  }

  compute_spectrum();
  e_bias = target->loss(y_pred.data(), gamma_now, dEdy.data());
  accumulate_bias(dEdy.data());

  if (outfile && outevery > 0 && (update->ntimestep % outevery) == 0) write_spectrum();

#ifdef LMP_KOKKOS
  // Hand the forces back coherent rather than host-authoritative.
  //
  // The bias adds to f on the host, so the host's copy is the new one and
  // something has to say so. Doing it here instead of through datamask_modify
  // is what makes the pair atomic: mark, then immediately push across, so the
  // fix never returns with the host holding the only claim. LAMMPS's Kokkos
  // FIRE minimiser closes its iteration loop with a bare
  //   atomKK->modified(Device, X_MASK | V_MASK | F_MASK);
  // and no sync ahead of it, which aborts the run the moment a non-Kokkos fix
  // has written forces:
  //   Concurrent modification of host and device views in DualView "atom:f"
  // Costs one 3*nall copy per bias evaluation, and only when Kokkos is on.
  if (lmp->kokkos) {
    auto *akk = (AtomKokkos *) atom;
    akk->modified(Host, F_MASK);
    akk->sync(Device, F_MASK);
  }
#endif
}

/* ---------------------------------------------------------------------- */

double FixMAD::compute_scalar()
{
  return e_bias;
}

double FixMAD::compute_vector(int n)
{
  switch (n) {
    case 0:
      return e_bias;
    case 1:
      return target->rel_error();
    case 2:
      return target->dissimilarity();
    case 3:
      return gamma_now;
    case 4:
      return target->scale();
    case 5:
      return target->offset();
  }
  return 0.0;
}

/* ---------------------------------------------------------------------- */

void FixMAD::write_spectrum()
{
  if (comm->me) return;
  FILE *fp = fopen(outfile, wrote_header ? "a" : "w");
  if (!fp) error->one(FLERR, "Cannot open fix mad output file {}", outfile);
  if (!wrote_header) {
    fprintf(fp, "# MADLIB %s: x  y_pred  y_exp  (scale and offset applied)\n", style);
    wrote_header = true;
  }
  fprintf(
      fp, "# step " BIGINT_FORMAT " gamma %.10g E %.10g rel_error %.10g scale %.10g offset %.10g\n",
      update->ntimestep, gamma_now, e_bias, target->rel_error(), target->scale(), target->offset());
  const double s = target->scale(), b = target->offset();
  for (int k = 0; k < nk; k++)
    // Enough digits that a reference check is limited by the calculation and
    // not by the file. TurboGAP's F20.8 puts a 5e-9 floor under its own tests.
    fprintf(fp, "%24.16e %24.16e %24.16e\n", xgrid[k], s * y_pred[k] + b, target->y_exp()[k]);
  fclose(fp);
}
