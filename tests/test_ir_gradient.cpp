// Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
/* The frozen-past IR gradient against a finite difference of its own loss.

   The loss is linear in the spectrum, L = sum_k c_k I_k with c fixed, so
   dL/dI is known exactly and the only thing under test is the chain from
   there back to the newest stored dipole -- the transpose of the cosine
   transform, the lag window, the biased ACF estimator and, under correlate
   rate, the backward difference.

   The finite difference perturbs the newest frame and rebuilds the whole
   spectrum from scratch, so it shares no code with the gradient beyond
   mad_ir_spectrum itself. Freezing the past is not an approximation being
   tested here: moving only the newest frame is exactly what the gradient
   claims to describe. */
#include "../src/MADLIB/mad_ir_spectrum.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace LAMMPS_NS;

static std::vector<double> read_dipoles(const char *path)
{
  std::vector<double> mu;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    double a, b, c;
    if (ss >> a >> b >> c) {
      mu.push_back(a);
      mu.push_back(b);
      mu.push_back(c);
    }
  }
  return mu;
}

// Deterministic and not smooth in k, so a gradient that happened to be right
// only for a flat weighting cannot pass.
static std::vector<double> weights(int n)
{
  std::vector<double> c(n);
  for (int k = 0; k < n; k++) c[k] = sin(0.7 * k + 0.3) + 0.5 * cos(0.11 * k);
  return c;
}

static bool loss(const std::vector<double> &mu, int n, const MADIrConfig &cfg,
                 const std::vector<double> &c, double &out, MADIrResult &res)
{
  std::string err;
  if (!mad_ir_spectrum(mu.data(), n, cfg, res, err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return false;
  }
  out = 0.0;
  const int nk = (int) std::min(c.size(), res.intensity_raw.size());
  for (int k = 0; k < nk; k++) out += c[k] * res.intensity_raw[k];
  return true;
}

int main(int argc, char **argv)
{
  if (argc < 9) {
    fprintf(stderr, "usage: %s dip dt window ratio numax qc correlate h [tol]\n", argv[0]);
    return 2;
  }

  std::vector<double> mu = read_dipoles(argv[1]);
  const int n = (int) mu.size() / 3;
  if (n < 16) {
    fprintf(stderr, "need at least 16 frames, got %d\n", n);
    return 2;
  }

  MADIrConfig cfg;
  cfg.dt_fs = atof(argv[2]);
  cfg.window = argv[3];
  cfg.acf_ratio = atof(argv[4]);
  cfg.max_freq_cm = atof(argv[5]);
  cfg.quantum_correction = argv[6];
  cfg.correlate = argv[7];
  cfg.smooth_k = 1;    // the gradient refuses smoothing, and says why
  const double h = atof(argv[8]);
  const double tol = (argc > 9) ? atof(argv[9]) : 1.0e-6;

  MADIrResult res;
  double base = 0.0;
  std::vector<double> c = weights(4096);
  if (!loss(mu, n, cfg, c, base, res)) return 1;
  c.resize(res.intensity_raw.size());

  std::vector<double> dLdI = c;
  std::vector<double> kern;
  std::string err;
  if (!mad_ir_acf_gradient(cfg, res, dLdI.data(), kern, err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }

  double mean[3], ana[3];
  mad_ir_series_mean(mu.data(), n, cfg, mean);
  mad_ir_newest_gradient(mu.data(), n, cfg, mean, kern, ana);

  printf("# frames %d   lags %d   freqs %d   correlate %s   qc %s   h %g\n", res.n_frames,
         res.n_lag, res.n_freq, cfg.correlate.c_str(), cfg.quantum_correction.c_str(), h);
  printf("# %-3s %22s %22s %14s\n", "dim", "analytic", "finite difference", "rel. error");

  double worst = 0.0;
  for (int d = 0; d < 3; d++) {
    const int at = 3 * (n - 1) + d;
    const double keep = mu[at];
    double plus = 0.0, minus = 0.0;
    MADIrResult tmp;

    mu[at] = keep + h;
    if (!loss(mu, n, cfg, c, plus, tmp)) return 1;
    mu[at] = keep - h;
    if (!loss(mu, n, cfg, c, minus, tmp)) return 1;
    mu[at] = keep;

    const double fd = (plus - minus) / (2.0 * h);
    const double scale = std::max(fabs(fd), fabs(ana[d]));
    const double rel = (scale > 0.0) ? fabs(fd - ana[d]) / scale : 0.0;
    worst = std::max(worst, rel);
    printf("  %-3d %22.14e %22.14e %14.3e\n", d, ana[d], fd, rel);
  }

  printf("# worst relative error %.3e against tolerance %.3e\n", worst, tol);
  return (worst <= tol) ? 0 : 1;
}
