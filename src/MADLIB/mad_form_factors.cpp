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

#include "mad_form_factors.h"

#include "mad_scattering_data.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace LAMMPS_NS::MADScatteringData;

static inline double f32(double v)
{
  return (double) (float) v;
}

MADFormFactors::MADFormFactors(bool neutron, bool compat) :
    neutron(neutron), compat(compat), nsamples(0)
{
}

int MADFormFactors::lookup(const std::string &label, bool neutron)
{
  if (neutron) {
    for (int i = 0; i < n_neutron; i++)
      if (label == MADScatteringData::neutron[i].name) return i;
  } else {
    for (int i = 0; i < n_xray; i++)
      if (label == xray[i].name) return i;
  }
  return -1;
}

bool MADFormFactors::known(const std::string &label, bool neutron)
{
  return lookup(label, neutron) >= 0;
}

bool MADFormFactors::add_species(const std::string &label)
{
  int entry = lookup(label, neutron);
  if (entry < 0) return false;
  species.push_back(entry);
  names.push_back(label);
  return true;
}

double MADFormFactors::evaluate(int entry, double s) const
{
  if (neutron) return MADScatteringData::neutron[entry].b;

  const XrayEntry &e = xray[entry];
  const double s2 = s * s;
  double f = compat ? f32(e.c) : e.c;
  for (int j = 0; j < 4; j++) {
    const double a = compat ? f32(e.a[j]) : e.a[j];
    const double b = compat ? f32(e.b[j]) : e.b[j];
    f += a * exp(-b * s2);
  }
  return f;
}

void MADFormFactors::tabulate(const std::vector<double> &x)
{
  nsamples = (int) x.size();
  table.assign(species.size() * nsamples, 0.0);
  for (size_t a = 0; a < species.size(); a++)
    for (int k = 0; k < nsamples; k++) table[a * nsamples + k] = evaluate(species[a], 0.5 * x[k]);
}
