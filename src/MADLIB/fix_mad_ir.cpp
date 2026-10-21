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

#include "atom_masks.h"
#include "fix_mad_ir.h"

#include "atom.h"
#include "comm.h"
#include "compute_mad_property.h"
#include "error.h"
#include "force.h"
#include "mad_target.h"
#include "modify.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixMADIr::FixMADIr(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), propid(nullptr), prop(nullptr), nframes(0), outevery(0), wrote_header(0),
    have_spectrum(0), dt_fs_user(-1.0), outfile(nullptr), last_transform(-1), target(nullptr),
    fit_nu_min(0.0), fit_nu_max(1.0e30), rel_err(0.0), dissim(0.0), n_fit(0)
{
  if (narg < 5) utils::missing_cmd_args(FLERR, "fix mad/ir", error);

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery < 1) error->all(FLERR, "fix mad/ir stride must be positive");

  const char *c = arg[4];
  if (strncmp(c, "c_", 2) == 0) c += 2;
  propid = utils::strdup(c);

  std::string targetfile;
  bool fit_s = true, fit_b = false;
  int saw_fit = 0;

  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "nu_max") == 0) {
      cfg.max_freq_cm = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "acf_ratio") == 0) {
      cfg.acf_ratio = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "window") == 0) {
      cfg.window = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "correlate") == 0) {
      cfg.correlate = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "qc") == 0) {
      cfg.quantum_correction = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "temp") == 0) {
      cfg.temperature = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "smooth") == 0) {
      if (iarg + 2 >= narg) utils::missing_cmd_args(FLERR, "fix mad/ir smooth", error);
      cfg.smooth_k = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      cfg.smooth_kind = arg[iarg + 2];
      iarg += 3;
    } else if (strcmp(arg[iarg], "subtract_mean") == 0) {
      cfg.subtract_mean = (strcmp(arg[iarg + 1], "yes") == 0);
      iarg += 2;
    } else if (strcmp(arg[iarg], "normalise") == 0 || strcmp(arg[iarg], "normalize") == 0) {
      cfg.normalise = (strcmp(arg[iarg + 1], "yes") == 0);
      iarg += 2;
    } else if (strcmp(arg[iarg], "dc_cutoff") == 0) {
      cfg.power_dc_cutoff_cm = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "frame_dt") == 0) {
      // For a trajectory replayed through rerun, where update->dt is not the
      // interval between the frames being read.
      dt_fs_user = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "file") == 0) {
      delete[] outfile;
      outfile = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "write_every") == 0) {
      outevery = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "target") == 0) {
      if (iarg + 1 >= narg) utils::missing_cmd_args(FLERR, "fix mad/ir target", error);
      targetfile = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "fit") == 0) {
      if (iarg + 2 >= narg) utils::missing_cmd_args(FLERR, "fix mad/ir fit", error);
      fit_s = (strcmp(arg[iarg + 1], "yes") == 0);
      fit_b = (strcmp(arg[iarg + 2], "yes") == 0);
      saw_fit = 1;
      iarg += 3;
    } else if (strcmp(arg[iarg], "fit_range") == 0) {
      if (iarg + 2 >= narg) utils::missing_cmd_args(FLERR, "fix mad/ir fit_range", error);
      fit_nu_min = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      fit_nu_max = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      saw_fit = 1;
      iarg += 3;
    } else {
      error->all(FLERR, "Unknown fix mad/ir keyword {}", arg[iarg]);
    }
  }

  if (!targetfile.empty()) {
    target = new MADTarget(error);
    target->read(targetfile);
    // Unlike the biasing fixes, which default to no fit at all: a predicted IR
    // intensity is in arbitrary units and is normalised to a peak of one by
    // default, so without a fitted scale the misfit would measure the units.
    target->set_fit(fit_s, fit_b);
  } else if (saw_fit) {
    error->all(FLERR, "fix mad/ir fit and fit_range need a target to compare against");
  }

  auto *cmp = modify->get_compute_by_id(propid);
  if (!cmp) error->all(FLERR, "fix mad/ir could not find compute {}", propid);
  prop = dynamic_cast<ComputeMADProperty *>(cmp);
  if (!prop) error->all(FLERR, "fix mad/ir compute {} is not a mad/property style", propid);
  if (prop->n_components() != 3)
    error->all(FLERR,
               "fix mad/ir needs a per-atom dipole, a property with three components, "
               "not {}",
               prop->n_components());

  scalar_flag = 1;

  // Reads coordinates, adds to forces, and writes neither a coordinate nor a
  // velocity. Fix defaults both masks to ALL_MASK, which under KOKKOS makes
  // LAMMPS mark atom:x modified on the host after every call and aborts the
  // run the moment the device also holds x. See fix_mad.cpp.
  datamask_read = X_MASK | F_MASK | TYPE_MASK | MASK_MASK;
  datamask_modify = F_MASK;
  extscalar = 0;
  vector_flag = 1;
  size_vector = 9;
  extvector = 0;
  array_flag = 1;
  size_array_rows = 0;
  size_array_cols = 3;
  extarray = 0;
  size_array_rows_variable = 1;
  global_freq = nevery;
}

/* ---------------------------------------------------------------------- */

FixMADIr::~FixMADIr()
{
  delete[] propid;
  delete[] outfile;
  delete target;
}

int FixMADIr::setmask()
{
  return END_OF_STEP | POST_RUN;
}

/* ---------------------------------------------------------------------- */

double FixMADIr::frame_dt_fs() const
{
  if (dt_fs_user > 0.0) return dt_fs_user;
  // LAMMPS metal time is ps; real is fs. Anything else has no business here,
  // because the wavenumber conversion is 33356.40952 cm^-1 per inverse fs.
  const double to_fs = (strcmp(update->unit_style, "metal") == 0) ? 1000.0 : 1.0;
  return update->dt * nevery * to_fs;
}

void FixMADIr::init()
{
  prop = dynamic_cast<ComputeMADProperty *>(modify->get_compute_by_id(propid));
  if (!prop) error->all(FLERR, "fix mad/ir lost its compute {}", propid);

  if (strcmp(update->unit_style, "metal") != 0 && strcmp(update->unit_style, "real") != 0)
    error->all(FLERR,
               "fix mad/ir needs metal or real units; a wavenumber axis has no meaning without a "
               "known time unit");

  cfg.dt_fs = frame_dt_fs();
  const double nyq = 33356.40952 / (2.0 * cfg.dt_fs);
  if (cfg.max_freq_cm > nyq)
    error->all(FLERR,
               "fix mad/ir: nu_max {:.1f} cm^-1 is above the Nyquist limit {:.1f} cm^-1 set by a "
               "stored-frame interval of {:.4g} fs; sample more often",
               cfg.max_freq_cm, nyq, cfg.dt_fs);

  if (comm->me == 0)
    utils::logmesg(lmp,
                   "fix {}: sampling every {} steps = {:.4g} fs, Nyquist {:.1f} cm^-1, "
                   "weighting '{}'\n",
                   id, nevery, cfg.dt_fs, nyq, cfg.quantum_correction);
}

void FixMADIr::setup(int)
{
  collect();
}

/* ---------------------------------------------------------------------- */

void FixMADIr::collect()
{
  prop->compute_property();
  double **m = prop->vector_property();
  const int *mask = atom->mask;
  const int nlocal = atom->nlocal;

  double local[3] = {0.0, 0.0, 0.0}, total[3];
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    local[0] += m[i][0];
    local[1] += m[i][1];
    local[2] += m[i][2];
  }
  MPI_Allreduce(local, total, 3, MPI_DOUBLE, MPI_SUM, world);

  mu.push_back(total[0]);
  mu.push_back(total[1]);
  mu.push_back(total[2]);
  nframes++;
  have_spectrum = 0;
}

void FixMADIr::end_of_step()
{
  collect();
  if (outfile && outevery > 0 && (update->ntimestep % outevery) == 0)
    if (transform_now()) write();
}

void FixMADIr::post_run()
{
  if (outfile && transform_now()) write();
}

/* ---------------------------------------------------------------------- */

bool FixMADIr::transform_now()
{
  if (have_spectrum && last_transform == update->ntimestep) return true;
  cfg.dt_fs = frame_dt_fs();

  std::string err;
  if (!mad_ir_spectrum(mu.data(), nframes, cfg, res, err)) {
    // Early in a run there are simply not enough frames yet; that is not a
    // reason to stop, only a reason to have nothing to say.
    if (nframes < 8) return false;
    error->all(FLERR, err);
  }
  have_spectrum = 1;
  last_transform = update->ntimestep;
  size_array_rows = res.n_freq;
  score();
  return true;
}

/* ---------------------------------------------------------------------- */

void FixMADIr::score()
{
  rel_err = dissim = 0.0;
  n_fit = 0;
  if (!target) return;

  // MADTarget refuses to extrapolate, and rightly: the prediction grid starts
  // at nu = 0 and ends at nu_max, while a measured curve covers neither end.
  // Rebuilt every transform because n_freq grows as the buffer fills.
  const double lo = std::max(fit_nu_min, target->range_min());
  const double hi = std::min(fit_nu_max, target->range_max());

  fit_nu.clear();
  fit_y.clear();
  for (int k = 0; k < res.n_freq; k++) {
    if (res.freq[k] < lo || res.freq[k] > hi) continue;
    fit_nu.push_back(res.freq[k]);
    fit_y.push_back(res.intensity[k]);
  }
  if ((int) fit_nu.size() < 2) return;

  target->set_grid(fit_nu, true);
  target->loss(fit_y.data(), 0.0, nullptr);
  rel_err = target->rel_error();
  dissim = target->dissimilarity();
  n_fit = (int) fit_nu.size();
}

/* ---------------------------------------------------------------------- */

double FixMADIr::compute_scalar()
{
  return (double) nframes;
}

double FixMADIr::compute_vector(int n)
{
  if (!transform_now()) return 0.0;
  switch (n) {
    case 0:
      return (double) res.n_frames;
    case 1:
      return (double) res.n_lag;
    case 2:
      return res.d_nu;
    case 3:
      return res.nyquist;
    case 4:
      return res.peak;
    case 5:
      return rel_err;
    case 6:
      return target ? target->scale() : 0.0;
    case 7:
      return target ? target->offset() : 0.0;
    case 8:
      return (double) n_fit;
  }
  return 0.0;
}

double FixMADIr::compute_array(int i, int j)
{
  if (!transform_now()) return 0.0;
  if (i < 0 || i >= res.n_freq) return 0.0;
  if (j == 0) return res.freq[i];
  if (j == 1) return res.intensity[i];
  return res.power[i];
}

/* ---------------------------------------------------------------------- */

void FixMADIr::write()
{
  if (comm->me) return;
  FILE *fp = fopen(outfile, "w");
  if (!fp) error->one(FLERR, "Cannot open fix mad/ir output file {}", outfile);

  // Every limit of the calculation, in the file, because none of them is a
  // property of the sample and all of them are easy to assume wrongly.
  fprintf(fp, "# MADLIB IR spectrum, step " BIGINT_FORMAT "\n", update->ntimestep);
  fprintf(fp, "# frames %d   stride %d   frame interval %.6g fs\n", res.n_frames, nevery,
          cfg.dt_fs);
  fprintf(fp, "# lags %d of %d (acf_ratio %.4g)   bin %.6g cm^-1   resolution %.6g cm^-1\n",
          res.n_lag, res.n_frames, cfg.acf_ratio, res.d_nu, res.resolution);
  fprintf(fp, "# correlating %s\n", cfg.correlate.c_str());
  fprintf(fp, "# Nyquist %.6g cm^-1   window %s   weighting %s", res.nyquist, cfg.window.c_str(),
          cfg.quantum_correction.c_str());
  if (cfg.quantum_correction == "harmonic") fprintf(fp, " at %.4g K", cfg.temperature);
  fprintf(fp, "\n");
  fprintf(fp, "# smoothing %d bins %s   mean dipole %.6g %.6g %.6g\n", cfg.smooth_k,
          cfg.smooth_kind.c_str(), res.mu_mean[0], res.mu_mean[1], res.mu_mean[2]);
  fprintf(fp, "# peak %.10e   peak_power %.10e   (divide out to recover the raw transform)\n",
          res.peak, res.peak_power);
  if (target) {
    // Overlay the experiment as scale*intensity + offset; those two numbers
    // are here so the overlay cannot be drawn with a different fit than the
    // one the misfit was computed from.
    fprintf(fp, "# target rel_error %.10f   dissimilarity %.10e   over %d points of [%g, %g]\n",
            rel_err, dissim, n_fit, fit_nu.empty() ? 0.0 : fit_nu.front(),
            fit_nu.empty() ? 0.0 : fit_nu.back());
    fprintf(fp, "# target fit scale %.10e   offset %.10e\n", target->scale(), target->offset());
  }
  fprintf(fp, "# nu[cm^-1]  intensity  power\n");
  for (int k = 0; k < res.n_freq; k++)
    fprintf(fp, "%20.10f %24.16e %24.16e\n", res.freq[k], res.intensity[k], res.power[k]);
  fclose(fp);
  wrote_header = 1;
}

double FixMADIr::memory_usage()
{
  return (double) mu.capacity() * sizeof(double);
}
