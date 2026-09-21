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

#include "mad_target.h"

#include "fmt/format.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace LAMMPS_NS;

MADTarget::MADTarget(Error *error) :
    error(error), nk(0), fit_scale(false), fit_offset(false), s_fit(1.0), b_fit(0.0), dissim(0.0),
    dissim_ref(0.0)
{
}

static void fail(const std::string &msg)
{
  throw std::runtime_error(msg);
}

void MADTarget::read(const std::string &filename)
{
  fname = filename;
  std::ifstream in(filename);
  if (!in) fail("MADLIB: cannot open target file " + filename);

  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    lineno++;
    const size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) continue;
    if (line[first] == '#' || line[first] == '!') continue;

    std::istringstream ss(line);
    double a, b;
    if (!(ss >> a >> b))
      fail("MADLIB: target file " + filename + " line " + std::to_string(lineno) +
           " is not two numbers");
    xraw.push_back(a);
    yraw.push_back(b);
  }
  if (xraw.size() < 2) fail("MADLIB: target file " + filename + " holds fewer than two points");

  for (size_t i = 1; i < xraw.size(); i++)
    if (xraw[i] <= xraw[i - 1])
      fail("MADLIB: target file " + filename + " abscissa is not strictly increasing at line " +
           std::to_string(i + 1));
}

void MADTarget::set_grid(const std::vector<double> &x, bool weight_by_spacing)
{
  nk = (int) x.size();
  if (nk < 2) fail("MADLIB: prediction grid needs at least two points");

  // Extrapolating a measured curve is never what the user meant.
  const double span = xraw.back() - xraw.front();
  const double tol = 1.0e-8 * span;
  if ((x.front() < xraw.front() - tol) || (x.back() > xraw.back() + tol))
    fail(fmt::format(
        "MADLIB: prediction grid [{}, {}] is not inside the range [{}, {}] of target file {}",
        x.front(), x.back(), xraw.front(), xraw.back(), fname));

  yexp.assign(nk, 0.0);
  size_t j = 0;
  for (int k = 0; k < nk; k++) {
    while (j + 2 < xraw.size() && xraw[j + 1] < x[k]) j++;
    const double t = (x[k] - xraw[j]) / (xraw[j + 1] - xraw[j]);
    yexp[k] = (1.0 - t) * yraw[j] + t * yraw[j + 1];
  }

  wgt.assign(nk, 1.0);
  if (weight_by_spacing) {
    // Trapezoidal, so the loss is an integral over the abscissa rather than a
    // sum over however the file happened to be sampled. Normalised to mean one
    // so that gamma means the same thing on any grid.
    wgt[0] = 0.5 * (x[1] - x[0]);
    wgt[nk - 1] = 0.5 * (x[nk - 1] - x[nk - 2]);
    for (int k = 1; k < nk - 1; k++) wgt[k] = 0.5 * (x[k + 1] - x[k - 1]);
    double sum = 0.0;
    for (int k = 0; k < nk; k++) sum += wgt[k];
    for (int k = 0; k < nk; k++) wgt[k] *= nk / sum;
  }

  dissim_ref = 0.0;
  for (int k = 0; k < nk; k++) dissim_ref += wgt[k] * yexp[k] * yexp[k];
}

void MADTarget::solve_fit(const double *y)
{
  s_fit = 1.0;
  b_fit = 0.0;
  if (!fit_scale) return;

  double sw = 0.0, syy = 0.0, sy = 0.0, sye = 0.0, se = 0.0;
  for (int k = 0; k < nk; k++) {
    const double w = wgt[k];
    sw += w;
    sy += w * y[k];
    syy += w * y[k] * y[k];
    sye += w * y[k] * yexp[k];
    se += w * yexp[k];
  }

  if (fit_offset) {
    const double det = syy * sw - sy * sy;
    if (fabs(det) > 1.0e-300) {
      const double s = (sye * sw - sy * se) / det;
      const double b = (syy * se - sy * sye) / det;
      // A negative scale reverses the bias, because dE/dy carries a factor of
      // s. That means the shapes disagree, not the baselines; fall back rather
      // than push the prediction away from the data.
      if (s > 0.0) {
        s_fit = s;
        b_fit = b;
        return;
      }
    }
  }

  if (syy > 1.0e-300) s_fit = sye / syy;
  if (s_fit <= 0.0) s_fit = 1.0;
  b_fit = 0.0;
}

double MADTarget::loss(const double *y, double gamma, double *dEdy)
{
  solve_fit(y);

  double e = 0.0;
  dissim = 0.0;
  for (int k = 0; k < nk; k++) {
    const double c = s_fit * y[k] + b_fit - yexp[k];
    e += 0.5 * gamma * wgt[k] * c * c;
    dissim += wgt[k] * c * c;
    if (dEdy) dEdy[k] = gamma * wgt[k] * s_fit * c;
  }
  return e;
}

double MADTarget::rel_error() const
{
  if (dissim_ref <= 0.0) return 0.0;
  return sqrt(dissim / dissim_ref);
}
