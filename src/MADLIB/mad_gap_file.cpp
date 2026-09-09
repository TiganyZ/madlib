/* ----------------------------------------------------------------------
   MADLIB - Molecular Augmented Dynamics for LAMMPS
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
void madlib_soap_dim(int n_species, const int *alpha_max, int l_max, int compress,
                     const char *mode, int *dim, int *nonzero);
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
static void madlib_compress_indices(int, const int *, int, const char *, int, int *, int *, double *)
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
    while (ss >> x) v.push_back(x);
  }
  if (v.empty()) throw std::runtime_error("MADLIB: " + what + " file " + path + " is empty");
  return v;
}

/* ---------------------------------------------------------------------- */

void MADGapFile::load_sparse(MADGapBlock &b, const std::string &dir)
{
  // TurboGAP writes these paths relative to the directory it is RUN from,
  // not to the .gap file, so "gap_files/alphas_1.dat" appears inside
  // gap_files/CO.gap. Accept either, since a LAMMPS run need not share a
  // working directory with whatever produced the potential.
  auto resolve = [&](const std::string &f) {
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
  };

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
      if (key == "n_species") b.n_species = inum(val);
      else if (key == "species") b.species = words(val);
      else if (key == "central_species") b.central_species = inum(val);
      else if (key == "rcut" || key == "rcut_hard") b.rcut_hard = vec(val);
      else if (key == "buffer") b.buffer = vec(val);
      else if (key == "atom_sigma_r") b.atom_sigma_r = vec(val);
      else if (key == "atom_sigma_t") b.atom_sigma_t = vec(val);
      else if (key == "atom_sigma_r_scaling") b.atom_sigma_r_scaling = vec(val);
      else if (key == "atom_sigma_t_scaling") b.atom_sigma_t_scaling = vec(val);
      else if (key == "amplitude_scaling") b.amplitude_scaling = vec(val);
      else if (key == "central_weight") b.central_weight = vec(val);
      else if (key == "global_scaling") b.global_scaling = vec(val);
      else if (key == "nf") b.nf = vec(val);
      else if (key == "n_max" || key == "alpha_max") b.alpha_max = ivec(val);
      else if (key == "l_max") b.l_max = inum(val);
      else if (key == "radial_enhancement") b.radial_enhancement = inum(val);
      else if (key == "basis") b.basis = strip(val);
      else if (key == "scaling_mode") b.scaling_mode = strip(val);
      else if (key == "compress_soap") b.compress = (strip(val) == ".true." || strip(val) == "T");
      else if (key == "compress_mode") b.compress_mode = strip(val);
      else if (key == "file_compress" || key == "compress_file") b.compress_file = strip(val);
      else if (key == "local_property_labels") { b.label = strip(val); has_lp = (b.label == label); }
      else if (key == "local_property_qs") b.file_qs = strip(val);
      else if (key == "local_property_alphas") b.file_alphas = strip(val);
      else if (key == "local_property_zetas") b.zeta = num(val);
      else if (key == "local_property_deltas") b.delta = num(val);
      else if (key == "local_property_v0s") b.v0 = num(val);
      else if (key == "n_local_properties" && inum(val) != 1)
        throw std::runtime_error(
            "MADLIB: this block declares " + strip(val) +
            " local properties; only one per block is supported so far");
    } catch (std::invalid_argument &) {
      throw std::runtime_error("MADLIB: " + path + " line " + std::to_string(lineno) +
                               ": could not read a number from '" + strip(val) + "'");
    }
  }

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
      for (int i = 0; i < blk.n_nonzero; i++) { blk.p_i[i] = i + 1; blk.p_j[i] = i + 1; }

    load_sparse(blk, dir);
  }
  return out;
}
