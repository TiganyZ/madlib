// Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
/* Drives the soap_turbo backend directly from an extended-xyz frame, with no
   LAMMPS anywhere, and prints one core-electron binding energy per atom.
   The frame carries TurboGAP's own values, so the comparison is exact. */
#include "../src/MADLIB/mad_gap_file.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

using namespace LAMMPS_NS;

extern "C" void madlib_local_property(
    int n_sites, int n_pairs, int n_species, const int *n_neigh, const int *species_idx,
    const double *rjs, const double *thetas, const double *phis, const int *alpha_max, int l_max,
    const double *rcut_hard, const double *rcut_soft, const double *nf, const double *global_scaling,
    const double *asr, const double *asrs, const double *ast, const double *asts,
    const double *amp, int radial_enhancement, const double *cw, const char *basis,
    const char *scaling_mode, int compress, int n_nonzero, const int *p_i, const int *p_j,
    const double *p_el, int dim, int n_sparse, const double *qs, const double *alphas, double v0,
    double delta, double zeta, int do_derivatives, double *v, double *v_der);

struct Frame { std::vector<std::string> sp; std::vector<double> x; double cell[9]; std::vector<double> ref; };

static Frame read_xyz(const char *p, int refcol)
{
  std::ifstream in(p); std::string line; Frame f;
  std::getline(in, line); int n = std::stoi(line);
  std::getline(in, line);
  const size_t a = line.find("Lattice=\"") + 9, b = line.find('"', a);
  std::istringstream cs(line.substr(a, b - a));
  for (int i = 0; i < 9; i++) cs >> f.cell[i];
  for (int i = 0; i < n; i++) {
    std::getline(in, line); std::istringstream ss(line);
    std::vector<std::string> w; std::string t;
    while (ss >> t) w.push_back(t);
    f.sp.push_back(w[0]);
    for (int k = 1; k <= 3; k++) f.x.push_back(std::stod(w[k]));
    f.ref.push_back(std::stod(w[refcol]));
  }
  return f;
}

int main(int argc, char **argv)
{
  const Frame f = read_xyz(argv[1], 11);
  const int n = (int) f.sp.size();
  auto blocks = MADGapFile::read(argv[2], "core_electron_be");
  fprintf(stderr, "blocks: %zu\n", blocks.size());

  std::vector<double> q(n, 0.0);

  for (auto &b : blocks) {
    // species index within this block's own species list
    auto sidx = [&](const std::string &s) {
      for (size_t i = 0; i < b.species.size(); i++) if (b.species[i] == s) return (int) i + 1;
      return 0;
    };
    const double rc = b.rcut_max();

    // periodic images out to the cutoff
    std::vector<double> sh;
    for (int i = -2; i <= 2; i++) for (int j = -2; j <= 2; j++) for (int k = -2; k <= 2; k++)
      for (int d = 0; d < 3; d++)
        sh.push_back(i * f.cell[d] + j * f.cell[3 + d] + k * f.cell[6 + d]);
    const int nsh = (int) sh.size() / 3;

    std::vector<int> nn, sidxv;
    std::vector<double> rjs, th, ph;
    std::vector<int> owner;
    for (int i = 0; i < n; i++) {
      if (sidx(f.sp[i]) != b.central_species) continue;
      owner.push_back(i);
      int cnt = 1;
      sidxv.push_back(sidx(f.sp[i])); rjs.push_back(0.0); th.push_back(0.0); ph.push_back(0.0);
      for (int j = 0; j < n; j++) for (int s = 0; s < nsh; s++) {
        const double dx = f.x[3*j+0] + sh[3*s+0] - f.x[3*i+0];
        const double dy = f.x[3*j+1] + sh[3*s+1] - f.x[3*i+1];
        const double dz = f.x[3*j+2] + sh[3*s+2] - f.x[3*i+2];
        const double r = sqrt(dx*dx + dy*dy + dz*dz);
        if (r >= rc || r < 1e-10) continue;
        sidxv.push_back(sidx(f.sp[j]));
        rjs.push_back(r); th.push_back(acos(dz / r)); ph.push_back(atan2(dy, dx));
        cnt++;
      }
      nn.push_back(cnt);
    }
    const int ns = (int) nn.size(), np = (int) rjs.size();
    fprintf(stderr, "  central_species %d: %d sites, %d pairs, dim %d, n_sparse %d\n",
            b.central_species, ns, np, b.dim, b.n_sparse);
    if (ns == 0) continue;

    std::vector<double> v(ns, 0.0), vder(3, 0.0);
    madlib_local_property(ns, np, b.n_species, nn.data(), sidxv.data(), rjs.data(), th.data(),
        ph.data(), b.alpha_max.data(), b.l_max, b.rcut_hard.data(), b.rcut_soft.data(),
        b.nf.data(), b.global_scaling.data(), b.atom_sigma_r.data(), b.atom_sigma_r_scaling.data(),
        b.atom_sigma_t.data(), b.atom_sigma_t_scaling.data(), b.amplitude_scaling.data(),
        b.radial_enhancement, b.central_weight.data(), b.basis.c_str(), b.scaling_mode.c_str(),
        b.compress ? 1 : 0, b.n_nonzero, b.p_i.data(), b.p_j.data(), b.p_el.data(), b.dim,
        b.n_sparse, b.qs.data(), b.alphas.data(), b.v0, b.delta, b.zeta, 0, v.data(), vder.data());
    for (int i = 0; i < ns; i++) q[owner[i]] = v[i];
  }

  double worst = 0.0;
  for (int i = 0; i < n; i++) {
    printf("%4d %-3s  %20.12f  %20.12f\n", i + 1, f.sp[i].c_str(), q[i], f.ref[i]);
    worst = std::max(worst, std::fabs(q[i] - f.ref[i]));
  }
  fprintf(stderr, "max |madlib - turbogap| = %.3e eV\n", worst);
  return worst < 1e-6 ? 0 : 1;
}
