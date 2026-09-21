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

#include "mad_gap_file.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace LAMMPS_NS;

#ifdef MADLIB_TURBOGAP
extern "C" {
void madlib_soap_dim(int n_species, const int *alpha_max, int l_max, int compress, const char *mode,
                     int *dim, int *nonzero);
void madlib_compress_indices(int n_species, const int *alpha_max, int l_max, const char *mode,
                             int nonzero, int *p_i, int *p_j, double *p_el);
}
#else
// The compression recipes live in soap_turbo, so without the backend there is
// nothing to parse a potential for.
static void madlib_soap_dim(int, const int *, int, int, const char *, int *, int *)
{
  throw std::runtime_error(
      "MADLIB: reading a .gap potential needs the Fortran backend: build lib/madlib and "
      "configure with -D MADLIB_ENABLE_TURBOGAP=on");
}
static void madlib_compress_indices(int, const int *, int, const char *, int, int *, int *,
                                    double *)
{
}
#endif

double MADGapBlock::rcut_max() const
{
  double r = 0.0;
  for (double v : rcut_hard) r = std::max(r, v);
  return r;
}

/* ---------------------------------------------------------------------- */

static std::string strip(std::string s)
{
  const size_t a = s.find_first_not_of(" \t\r\n'\"");
  if (a == std::string::npos) return "";
  const size_t b = s.find_last_not_of(" \t\r\n'\"");
  return s.substr(a, b - a + 1);
}

static std::vector<std::string> words(const std::string &s)
{
  std::vector<std::string> w;
  std::istringstream ss(s);
  std::string t;
  while (ss >> t) w.push_back(strip(t));
  return w;
}

// Fortran logicals, as TurboGAP writes them: .true. / .false.
static bool fortran_true(const std::string &v)
{
  const std::string t = strip(v);
  return !t.empty() && (t[0] == 'T' || t[0] == 't' ||
                        (t[0] == '.' && t.size() > 1 && (t[1] == 'T' || t[1] == 't')));
}

static std::string dirname_of(const std::string &p)
{
  const size_t s = p.find_last_of('/');
  return (s == std::string::npos) ? std::string(".") : p.substr(0, s);
}

static std::vector<double> read_column(const std::string &path, const std::string &what)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("MADLIB: cannot open " + what + " file " + path);
  std::vector<double> v;
  std::string line;
  while (std::getline(in, line)) {
    const size_t p = line.find_first_not_of(" \t\r\n");
    if (p == std::string::npos || line[p] == '#' || line[p] == '!') continue;
    std::istringstream ss(line);
    double x;
    // ONE value per line, not every value on it. TurboGAP sizes a sparse set
    // by COUNTING LINES and then reads `alphas(i)` per line, so a trailing
    // column is ignored -- and the soap alphas files do carry one in some
    // potentials (CO.gap) and not others (carbon.gap). Taking every number
    // would double n_sparse for the first kind, which then fails the
    // dimension check with a confusing message about the descriptor file.
    if (ss >> x) v.push_back(x);
  }
  if (v.empty()) throw std::runtime_error("MADLIB: " + what + " file " + path + " is empty");
  return v;
}

/* ---------------------------------------------------------------------- */

// TurboGAP writes these paths relative to the directory it is RUN from, not to
// the .gap file, so "gap_files/alphas_1.dat" appears inside gap_files/CO.gap.
// Accept either, since a LAMMPS run need not share a working directory with
// whatever produced the potential.
static std::string resolve_beside(const std::string &f, const std::string &dir)
{
  if (f.empty() || f[0] == '/') return f;
  if (std::ifstream(f)) return f;
  const std::string beside = dir + "/" + f;
  if (std::ifstream(beside)) return beside;
  const size_t sl = f.find_last_of('/');
  if (sl != std::string::npos) {
    const std::string flat = dir + "/" + f.substr(sl + 1);
    if (std::ifstream(flat)) return flat;
  }
  return beside;
}

void MADGapFile::load_sparse(MADGapBlock &b, const std::string &dir)
{
  // TurboGAP writes these paths relative to the directory it is RUN from,
  // not to the .gap file, so "gap_files/alphas_1.dat" appears inside
  // gap_files/CO.gap. Accept either, since a LAMMPS run need not share a
  // working directory with whatever produced the potential.
  auto resolve = [&](const std::string &f) { return resolve_beside(f, dir); };

  b.alphas = read_column(resolve(b.file_alphas), "local_property_alphas");
  b.n_sparse = (int) b.alphas.size();

  // The descriptor file is one real per line, n_sparse * dim of them, laid out
  // so that Qs(1:dim, 1:n_sparse) reads column-major -- the QUIP sparseX
  // layout, which is what Fortran wants without a transpose.
  b.qs = read_column(resolve(b.file_qs), "local_property_qs");
  if ((int) b.qs.size() % b.n_sparse != 0)
    throw std::runtime_error("MADLIB: " + b.file_qs + " holds " + std::to_string(b.qs.size()) +
                             " numbers, which is not a multiple of the " +
                             std::to_string(b.n_sparse) + " sparse points in " + b.file_alphas);

  const int dim_from_file = (int) b.qs.size() / b.n_sparse;
  if (b.dim != dim_from_file)
    throw std::runtime_error("MADLIB: the descriptor in " + b.file_qs + " has dimension " +
                             std::to_string(dim_from_file) + " but the hyperparameters give " +
                             std::to_string(b.dim) +
                             "; the local property must use the same descriptor as its block");
}

/* ---------------------------------------------------------------------- */

// The soap_turbo hyperparameters, which are the same keys whether the block
// carries a local property or the energy GAP. One copy, so a key added for one
// reader cannot go missing from the other.
//
// Returns false for a key it does not know, leaving the caller to handle the
// keys that ARE specific to its kind of block: local_property_* on one side,
// desc_sparse/alphas_sparse/zeta/delta on the other.
static bool apply_soap_hyper(MADGapBlock &b, const std::string &key, const std::string &val)
{
  auto num = [&](const std::string &v) { return std::stod(strip(v)); };
  auto inum = [&](const std::string &v) { return (int) std::lround(std::stod(strip(v))); };
  auto vec = [&](const std::string &v) {
    std::vector<double> r;
    for (auto &w : words(v)) r.push_back(std::stod(w));
    return r;
  };
  auto ivec = [&](const std::string &v) {
    std::vector<int> r;
    for (auto &w : words(v)) r.push_back((int) std::lround(std::stod(w)));
    return r;
  };
  (void) num;

  if (key == "n_species")
    b.n_species = inum(val);
  else if (key == "species")
    b.species = words(val);
  else if (key == "central_species")
    b.central_species = inum(val);
  else if (key == "rcut" || key == "rcut_hard")
    b.rcut_hard = vec(val);
  else if (key == "buffer")
    b.buffer = vec(val);
  else if (key == "atom_sigma_r")
    b.atom_sigma_r = vec(val);
  else if (key == "atom_sigma_t")
    b.atom_sigma_t = vec(val);
  else if (key == "atom_sigma_r_scaling")
    b.atom_sigma_r_scaling = vec(val);
  else if (key == "atom_sigma_t_scaling")
    b.atom_sigma_t_scaling = vec(val);
  else if (key == "amplitude_scaling")
    b.amplitude_scaling = vec(val);
  else if (key == "central_weight")
    b.central_weight = vec(val);
  else if (key == "global_scaling")
    b.global_scaling = vec(val);
  else if (key == "nf")
    b.nf = vec(val);
  else if (key == "n_max" || key == "alpha_max")
    b.alpha_max = ivec(val);
  else if (key == "l_max")
    b.l_max = inum(val);
  else if (key == "radial_enhancement")
    b.radial_enhancement = inum(val);
  else if (key == "basis")
    b.basis = strip(val);
  else if (key == "scaling_mode")
    b.scaling_mode = strip(val);
  else if (key == "compress_soap")
    b.compress = (strip(val) == ".true." || strip(val) == "T");
  else if (key == "compress_mode")
    b.compress_mode = strip(val);
  else if (key == "file_compress_soap" || key == "file_compress" || key == "compress_file") {
    // file_compress_soap is the spelling in a .gap file; TurboGAP reads it
    // into a field it calls file_compress, and both short forms appear in
    // hand-written potentials. Quotes are part of the value here.
    std::string f = strip(val);
    if (f.size() > 1 && (f.front() == '\'' || f.front() == '"')) f = f.substr(1, f.size() - 2);
    b.compress_file = f;
  }
  else
    return false;
  return true;
}

std::vector<MADGapBlock> MADGapFile::read(const std::string &path, const std::string &label)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("MADLIB: cannot open gap file " + path);
  const std::string dir = dirname_of(path);

  std::vector<MADGapBlock> out;
  std::string line;
  int lineno = 0;
  bool inblock = false;
  MADGapBlock b;
  bool has_lp = false;
  bool any_dipole = false, any_label = false;

  auto num = [&](const std::string &v) {
    return std::stod(strip(v));
  };
  auto inum = [&](const std::string &v) {
    return (int) std::lround(std::stod(strip(v)));
  };
  auto vec = [&](const std::string &v) {
    std::vector<double> r;
    for (auto &w : words(v)) r.push_back(std::stod(w));
    return r;
  };
  auto ivec = [&](const std::string &v) {
    std::vector<int> r;
    for (auto &w : words(v)) r.push_back((int) std::lround(std::stod(w)));
    return r;
  };

  while (std::getline(in, line)) {
    lineno++;
    const size_t c = line.find_first_of("!#");
    if (c != std::string::npos) line = line.substr(0, c);
    const size_t p = line.find_first_not_of(" \t\r\n");
    if (p == std::string::npos) continue;

    const auto w = words(line);
    if (!w.empty() && w[0] == "gap_beg") {
      inblock = (w.size() > 1 && w[1] == "soap_turbo");
      b = MADGapBlock();
      has_lp = false;
      continue;
    }
    if (!w.empty() && w[0] == "gap_end") {
      if (inblock && has_lp) out.push_back(b);
      inblock = false;
      continue;
    }
    if (!inblock) continue;

    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = strip(line.substr(0, eq));
    const std::string val = line.substr(eq + 1);
    std::replace(key.begin(), key.end(), ' ', '_');

    try {
      if (apply_soap_hyper(b, key, val)) {
        ;
      } else if (key == "dipole_model") {
        any_dipole = any_dipole || fortran_true(val);
      } else if (key == "local_property_labels") {
        b.label = strip(val);
        any_label = true;
        has_lp = (b.label == label);
      } else if (key == "local_property_qs")
        b.file_qs = strip(val);
      else if (key == "local_property_alphas")
        b.file_alphas = strip(val);
      else if (key == "local_property_zetas")
        b.zeta = num(val);
      else if (key == "local_property_deltas")
        b.delta = num(val);
      else if (key == "local_property_v0s")
        b.v0 = num(val);
      else if (key == "n_local_properties" && inum(val) != 1)
        throw std::runtime_error("MADLIB: this block declares " + strip(val) +
                                 " local properties; only one per block is supported so far");
    } catch (std::invalid_argument &) {
      throw std::runtime_error("MADLIB: " + path + " line " + std::to_string(lineno) +
                               ": could not read a number from '" + strip(val) + "'");
    }
  }

  // Only when the file carries no local property at all: a potential can hold
  // a dipole block beside local-property ones, and there the missing label is
  // the missing label.
  if (out.empty() && any_dipole && !any_label)
    throw std::runtime_error(
        "MADLIB: " + path +
        " is a dipole model, not a local-property model. Its dipole is the gradient of a fitted "
        "local energy with respect to the central atom's own position, which is a different "
        "contraction of the same descriptor and is not implemented yet; a local property is the "
        "kernel value itself and is what this style evaluates");
  if (out.empty())
    throw std::runtime_error("MADLIB: no soap_turbo block in " + path +
                             " carries a local property labelled '" + label + "'");

  for (auto &blk : out) {
    if ((int) blk.alpha_max.size() != blk.n_species || blk.rcut_hard.size() != blk.alpha_max.size())
      throw std::runtime_error("MADLIB: " + path +
                               ": a soap_turbo block has per-species lists of inconsistent length");
    if (blk.compress && blk.compress_file != "none")
      throw std::runtime_error(
          "MADLIB: " + path +
          " uses a compression file, which is not supported yet; use compress_mode");

    // soap_turbo takes the soft cutoff, not the buffer the file gives.
    blk.rcut_soft.resize(blk.rcut_hard.size());
    for (size_t i = 0; i < blk.rcut_hard.size(); i++)
      blk.rcut_soft[i] = blk.rcut_hard[i] - (i < blk.buffer.size() ? blk.buffer[i] : 0.0);
    if (blk.nf.empty()) blk.nf.assign(blk.n_species, 4.0);
    if (blk.global_scaling.empty()) blk.global_scaling.assign(blk.n_species, 1.0);
    if (blk.central_weight.empty()) blk.central_weight.assign(blk.n_species, 1.0);

    madlib_soap_dim(blk.n_species, blk.alpha_max.data(), blk.l_max, blk.compress ? 1 : 0,
                    blk.compress_mode.c_str(), &blk.dim, &blk.n_nonzero);
    blk.p_i.assign(blk.n_nonzero, 0);
    blk.p_j.assign(blk.n_nonzero, 0);
    blk.p_el.assign(blk.n_nonzero, 1.0);
    if (blk.compress)
      madlib_compress_indices(blk.n_species, blk.alpha_max.data(), blk.l_max,
                              blk.compress_mode.c_str(), blk.n_nonzero, blk.p_i.data(),
                              blk.p_j.data(), blk.p_el.data());
    else
      for (int i = 0; i < blk.n_nonzero; i++) {
        blk.p_i[i] = i + 1;
        blk.p_j[i] = i + 1;
      }

    load_sparse(blk, dir);
  }
  return out;
}

/* ----------------------------------------------------------------------
   Reading a whole potential: soap_turbo + distance_2b + angle_3b + core_pot
------------------------------------------------------------------------- */

double MADGapPotential::rcut_max() const
{
  double r = 0.0;
  for (const auto &b : soap) r = std::max(r, b.rcut_max());
  for (const auto &b : twob) r = std::max(r, b.rcut);
  for (const auto &b : threeb) r = std::max(r, b.rcut);
  for (const auto &b : core)
    if (!b.x.empty()) r = std::max(r, b.x.back());
  return r;
}

// Two columns of reals: alpha and its cutoff, which is how TurboGAP writes the
// sparse weights for the 2b and 3b terms.
static void read_two_columns(const std::string &path, const std::string &what,
                             std::vector<double> &c1, std::vector<double> &c2)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("MADLIB: cannot open " + what + " file " + path);
  c1.clear();
  c2.clear();
  std::string line;
  while (std::getline(in, line)) {
    const auto w = words(line);
    if (w.empty()) continue;
    c1.push_back(std::stod(w[0]));
    c2.push_back(w.size() > 1 ? std::stod(w[1]) : 1.0);
  }
  if (c1.empty()) throw std::runtime_error("MADLIB: " + path + " holds no numbers");
}

// Second derivatives for a cubic spline through (x, y).
//
// This is the tridiagonal solve from Numerical Recipes, which is what QUIP
// uses and therefore what TurboGAP copied, so the interpolation here and the
// interpolation there agree to the last digit rather than merely closely. A
// first derivative above 0.99e30 means "natural" at that end.
static std::vector<double> spline_second_derivatives(const std::vector<double> &x,
                                                     const std::vector<double> &y, double yp1,
                                                     double ypn)
{
  const int n = (int) x.size();
  std::vector<double> y2(n, 0.0), u(n, 0.0);

  if (yp1 > 0.99e30) {
    y2[0] = 0.0;
    u[0] = 0.0;
  } else {
    y2[0] = -0.5;
    u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
  }

  for (int i = 1; i < n - 1; i++) {
    const double sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
    const double p = sig * y2[i - 1] + 2.0;
    y2[i] = (sig - 1.0) / p;
    u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
  }

  double qn, un;
  if (ypn > 0.99e30) {
    qn = 0.0;
    un = 0.0;
  } else {
    qn = 0.5;
    un = (3.0 / (x[n - 1] - x[n - 2])) *
        (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
  }
  y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);
  for (int i = n - 2; i >= 0; i--) y2[i] = y2[i] * y2[i + 1] + u[i];
  return y2;
}

// The tabulated repulsive potential, truncated and tapered exactly as TurboGAP
// truncates it on read. core_pot_cutoff and core_pot_buffer are run parameters
// rather than properties of the file, so the same table gives a different
// potential under different settings -- which is why they are arguments here
// instead of constants.
static MADGapCorePot read_core_pot(const std::string &path, double cutoff, double buffer)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("MADLIB: cannot open core_pot file " + path);

  MADGapCorePot c;
  int n = 0;
  in >> n >> c.yp1 >> c.ypn;
  if (!in || n < 2)
    throw std::runtime_error("MADLIB: " + path +
                             " does not start with 'n yp1 ypn' and at least two points");

  const double pi = std::acos(-1.0);
  for (int i = 0; i < n; i++) {
    double xi = 0.0, vi = 0.0;
    in >> xi >> vi;
    if (!in)
      throw std::runtime_error("MADLIB: " + path + " ends after " + std::to_string(i) +
                               " points but declares " + std::to_string(n));
    if (xi > cutoff) continue;
    if (xi > cutoff - buffer)
      vi *= 0.5 * (std::cos(pi / buffer * (xi - cutoff + buffer)) + 1.0);
    c.x.push_back(xi);
    c.V.push_back(vi);
  }
  if (c.x.size() < 2)
    throw std::runtime_error("MADLIB: " + path + " has fewer than two points inside core_pot_cutoff");

  c.dVdx2 = spline_second_derivatives(c.x, c.V, c.yp1, c.ypn);
  return c;
}

// A compression given as a file rather than a named recipe.
//
// Two forms exist and both are in circulation. The header is n_species ints,
// then one more, then the compressed dimension. If the next line says
// "P_transformation" the body is (i, j, element) triples up to
// "end_transformation"; otherwise it is the older form, one kept index per
// line, which is the same thing with an identity row index and unit elements.
static void read_compress_file(const std::string &path, MADGapBlock &b)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("MADLIB: cannot open compression file " + path);

  for (int i = 0; i < b.n_species + 1; i++) {
    int junk = 0;
    in >> junk;
  }
  in >> b.dim;
  if (!in || b.dim <= 0)
    throw std::runtime_error("MADLIB: " + path + " does not begin with a compressed dimension");
  std::string rest;
  std::getline(in, rest);

  std::streampos mark = in.tellg();
  std::string tag;
  std::getline(in, tag);
  tag = strip(tag);

  b.p_i.clear();
  b.p_j.clear();
  b.p_el.clear();

  if (tag == "P_transformation") {
    std::string line;
    while (std::getline(in, line)) {
      if (strip(line) == "end_transformation") break;
      const auto w = words(line);
      if (w.size() < 3) continue;
      b.p_i.push_back((int) std::lround(std::stod(w[0])));
      b.p_j.push_back((int) std::lround(std::stod(w[1])));
      b.p_el.push_back(std::stod(w[2]));
    }
  } else {
    in.clear();
    in.seekg(mark);
    for (int i = 0; i < b.dim; i++) {
      int j = 0;
      in >> j;
      if (!in)
        throw std::runtime_error("MADLIB: " + path + " declares " + std::to_string(b.dim) +
                                 " kept components but lists " + std::to_string(i));
      b.p_i.push_back(i + 1);
      b.p_j.push_back(j);
      b.p_el.push_back(1.0);
    }
  }
  b.n_nonzero = (int) b.p_i.size();
  if (b.n_nonzero == 0)
    throw std::runtime_error("MADLIB: " + path + " lists no compression entries");
}

// Finish an energy soap_turbo block: defaults the file leaves out, the
// compression, and the sparse set.
//
// Mirrors what read() does for a local-property block, with two differences.
// The weights come from desc_sparse/alphas_sparse rather than the
// local_property_* keys -- a block can carry both, as carbon.gap does, and
// they are different fits. And a compression FILE is honoured here: the
// local-property path rejects one, but carbon.gap and the CLi potentials all
// use file_compress_soap, so refusing it would refuse the most obvious
// potentials anyone would try.
static void finalise_soap_energy(MADGapBlock &b, const std::string &dir)
{
  if ((int) b.alpha_max.size() != b.n_species || b.rcut_hard.size() != b.alpha_max.size())
    throw std::runtime_error("MADLIB: a soap_turbo block has per-species lists of "
                             "inconsistent length");

  b.rcut_soft.resize(b.rcut_hard.size());
  for (size_t i = 0; i < b.rcut_hard.size(); i++)
    b.rcut_soft[i] = b.rcut_hard[i] - (i < b.buffer.size() ? b.buffer[i] : 0.0);
  if (b.nf.empty()) b.nf.assign(b.n_species, 4.0);
  if (b.global_scaling.empty()) b.global_scaling.assign(b.n_species, 1.0);
  if (b.central_weight.empty()) b.central_weight.assign(b.n_species, 1.0);

  if (b.compress && b.compress_file != "none" && !b.compress_file.empty()) {
    read_compress_file(resolve_beside(b.compress_file, dir), b);
  } else {
    madlib_soap_dim(b.n_species, b.alpha_max.data(), b.l_max, b.compress ? 1 : 0,
                    b.compress_mode.c_str(), &b.dim, &b.n_nonzero);
    b.p_i.assign(b.n_nonzero, 0);
    b.p_j.assign(b.n_nonzero, 0);
    b.p_el.assign(b.n_nonzero, 1.0);
    if (b.compress)
      madlib_compress_indices(b.n_species, b.alpha_max.data(), b.l_max, b.compress_mode.c_str(),
                              b.n_nonzero, b.p_i.data(), b.p_j.data(), b.p_el.data());
    else
      for (int i = 0; i < b.n_nonzero; i++) {
        b.p_i[i] = i + 1;
        b.p_j[i] = i + 1;
      }
  }

  b.alphas = read_column(resolve_beside(b.file_alphas, dir), "alphas_sparse");
  b.n_sparse = (int) b.alphas.size();
  b.qs = read_column(resolve_beside(b.file_qs, dir), "desc_sparse");
  if (b.n_sparse == 0 || (int) b.qs.size() % b.n_sparse != 0)
    throw std::runtime_error("MADLIB: " + b.file_qs + " holds " + std::to_string(b.qs.size()) +
                             " numbers, which is not a multiple of the " +
                             std::to_string(b.n_sparse) + " sparse points in " + b.file_alphas);

  const int dim_from_file = (int) b.qs.size() / b.n_sparse;
  if (b.dim != dim_from_file)
    throw std::runtime_error("MADLIB: the descriptor in " + b.file_qs + " has dimension " +
                             std::to_string(dim_from_file) + " but the hyperparameters give " +
                             std::to_string(b.dim) + "; check compress_soap and compress_mode");
}

MADGapPotential MADGapFile::read_potential(const std::string &path,
                                           const std::vector<std::string> &species_types,
                                           double core_pot_cutoff, double core_pot_buffer)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("MADLIB: cannot open gap file " + path);
  const std::string dir = dirname_of(path);

  MADGapPotential pot;
  pot.species_types = species_types;
  pot.e0.assign(species_types.size(), 0.0);

  std::string line, kind;
  int lineno = 0;
  MADGapBlock soap;
  MADGap2b twob;
  MADGap3b threeb;
  MADGapCorePot core;
  std::string core_file;

  auto num = [&](const std::string &v) { return std::stod(strip(v)); };
  auto vec = [&](const std::string &v) {
    std::vector<double> r;
    for (auto &w : words(v)) r.push_back(std::stod(w));
    return r;
  };
  auto unquote = [&](const std::string &v) {
    std::string t = strip(v);
    if (t.size() > 1 && (t.front() == '\'' || t.front() == '"')) t = t.substr(1, t.size() - 2);
    return t;
  };

  while (std::getline(in, line)) {
    lineno++;
    const size_t c = line.find_first_of("!#");
    if (c != std::string::npos) line = line.substr(0, c);
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    const auto w = words(line);

    if (!w.empty() && w[0] == "gap_beg") {
      kind = (w.size() > 1) ? w[1] : "";
      soap = MADGapBlock();
      twob = MADGap2b();
      threeb = MADGap3b();
      core = MADGapCorePot();
      core_file.clear();
      continue;
    }

    if (!w.empty() && w[0] == "gap_end") {
      // A dipole model's block looks exactly like an energy block -- same
      // hyperparameters, same desc_sparse, same alphas_sparse -- and TurboGAP
      // tells them apart by this one flag. Reading it as a potential gives a
      // perfectly ordinary-looking energy that is a fitting artefact, so it is
      // refused here rather than reported.
      if (kind == "soap_turbo" && soap.dipole_model)
        throw std::runtime_error(
            "MADLIB: " + path +
            " is a dipole model (dipole_model = .true.). Its local energy is a fitted scalar "
            "whose gradient with respect to the central atom's own position is the local dipole; "
            "TurboGAP never adds it to the energy or its gradient to the forces. There is no "
            "potential here to evaluate");

      if (kind == "soap_turbo" && !soap.file_qs.empty()) {
        finalise_soap_energy(soap, dir);
        pot.soap.push_back(soap);
      } else if (kind == "distance_2b") {
        read_two_columns(resolve_beside(twob.file_alphas, dir), "2b alphas", twob.alphas,
                         twob.cutoff);
        twob.qs = read_column(resolve_beside(twob.file_qs, dir), "2b desc_sparse");
        twob.n_sparse = (int) twob.alphas.size();
        if ((int) twob.qs.size() != twob.n_sparse)
          throw std::runtime_error("MADLIB: " + twob.file_qs + " holds " +
                                   std::to_string(twob.qs.size()) + " descriptors but " +
                                   twob.file_alphas + " holds " + std::to_string(twob.n_sparse) +
                                   " weights");
        pot.twob.push_back(twob);
      } else if (kind == "angle_3b") {
        read_two_columns(resolve_beside(threeb.file_alphas, dir), "3b alphas", threeb.alphas,
                         threeb.cutoff);
        threeb.qs = read_column(resolve_beside(threeb.file_qs, dir), "3b desc_sparse");
        threeb.n_sparse = (int) threeb.alphas.size();
        if ((int) threeb.qs.size() != 3 * threeb.n_sparse)
          throw std::runtime_error("MADLIB: the 3b descriptor " + threeb.file_qs + " holds " +
                                   std::to_string(threeb.qs.size()) +
                                   " numbers, not three per each of the " +
                                   std::to_string(threeb.n_sparse) + " sparse points");
        if (threeb.sigma.size() != 3) threeb.sigma.assign(3, 1.0);
        pot.threeb.push_back(threeb);
      } else if (kind == "core_pot" && !core_file.empty()) {
        MADGapCorePot t = read_core_pot(resolve_beside(core_file, dir), core_pot_cutoff,
                                        core_pot_buffer);
        t.species1 = core.species1;
        t.species2 = core.species2;
        t.file = core_file;
        pot.core.push_back(t);
      }
      kind.clear();
      continue;
    }

    if (kind.empty()) continue;
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = strip(line.substr(0, eq));
    const std::string val = line.substr(eq + 1);
    std::replace(key.begin(), key.end(), ' ', '_');

    try {
      if (kind == "soap_turbo") {
        if (apply_soap_hyper(soap, key, val)) {
          ;
        } else if (key == "dipole_model")
          soap.dipole_model = fortran_true(val);
        else if (key == "zeta")
          soap.zeta = num(val);
        else if (key == "delta")
          soap.delta = num(val);
        else if (key == "desc_sparse")
          soap.file_qs = unquote(val);
        else if (key == "alphas_sparse")
          soap.file_alphas = unquote(val);
      } else if (kind == "distance_2b") {
        if (key == "species1")
          twob.species1 = strip(val);
        else if (key == "species2")
          twob.species2 = strip(val);
        else if (key == "delta")
          twob.delta = num(val);
        else if (key == "sigma")
          twob.sigma = num(val);
        else if (key == "rcut")
          twob.rcut = num(val);
        else if (key == "buffer")
          twob.buffer = num(val);
        else if (key == "desc_sparse")
          twob.file_qs = unquote(val);
        else if (key == "alphas_sparse")
          twob.file_alphas = unquote(val);
      } else if (kind == "angle_3b") {
        if (key == "species_center")
          threeb.species_center = strip(val);
        else if (key == "species1")
          threeb.species1 = strip(val);
        else if (key == "species2")
          threeb.species2 = strip(val);
        else if (key == "delta")
          threeb.delta = num(val);
        else if (key == "sigma")
          threeb.sigma = vec(val);
        else if (key == "rcut")
          threeb.rcut = num(val);
        else if (key == "buffer")
          threeb.buffer = num(val);
        else if (key == "kernel_type")
          threeb.kernel_type = unquote(val);
        else if (key == "desc_sparse")
          threeb.file_qs = unquote(val);
        else if (key == "alphas_sparse")
          threeb.file_alphas = unquote(val);
      } else if (kind == "core_pot") {
        if (key == "species1")
          core.species1 = strip(val);
        else if (key == "species2")
          core.species2 = strip(val);
        else if (key == "core_pot_file")
          core_file = unquote(val);
      }
    } catch (std::invalid_argument &) {
      throw std::runtime_error("MADLIB: " + path + " line " + std::to_string(lineno) +
                               ": could not read a number from '" + strip(val) + "'");
    }
  }

  if (pot.soap.empty() && pot.twob.empty() && pot.threeb.empty() && pot.core.empty())
    throw std::runtime_error("MADLIB: " + path + " holds no readable potential blocks");

  return pot;
}
