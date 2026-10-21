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

#include "mad_ir_spectrum.h"

#include <cmath>
#include <complex>

using namespace LAMMPS_NS;

static const double CM_PER_INV_FS = 33356.40952;
static const double HBAR_C_EV_CM = 1.23984e-4;
static const double KB_EV = 8.617333e-5;

/* ---------------------------------------------------------------------- */

static int next_pow2(int n)
{
  int m = 1;
  while (m < n) m *= 2;
  return m;
}

// In-place radix-2 Cooley-Tukey. sign = -1 forward, +1 inverse (unnormalised).
static void transform(std::complex<double> *z, int n, int sign)
{
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(z[i], z[j]);
  }
  for (int len = 2; len <= n; len <<= 1) {
    const double ang = sign * 2.0 * M_PI / len;
    const std::complex<double> wl(cos(ang), sin(ang));
    for (int i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (int k = 0; k < len / 2; k++) {
        const std::complex<double> u = z[i + k];
        const std::complex<double> v = z[i + k + len / 2] * w;
        z[i + k] = u + v;
        z[i + k + len / 2] = u - v;
        w *= wl;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void LAMMPS_NS::mad_ir_autocorrelation(const double *mu, int n_frames, bool subtract_mean,
                                       std::vector<double> &acf, double mu_mean[3])
{
  acf.assign(n_frames > 0 ? n_frames : 1, 0.0);
  mu_mean[0] = mu_mean[1] = mu_mean[2] = 0.0;
  if (n_frames < 1) return;

  const int n = next_pow2(2 * n_frames);
  std::vector<std::complex<double>> z(n);

  for (int d = 0; d < 3; d++) {
    double mean = 0.0;
    if (subtract_mean) {
      for (int t = 0; t < n_frames; t++) mean += mu[3 * t + d];
      mean /= (double) n_frames;
    }
    mu_mean[d] = mean;

    std::fill(z.begin(), z.end(), std::complex<double>(0.0, 0.0));
    for (int t = 0; t < n_frames; t++) z[t] = std::complex<double>(mu[3 * t + d] - mean, 0.0);

    transform(z.data(), n, -1);
    for (int t = 0; t < n; t++) {
      const double re = z[t].real(), im = z[t].imag();
      z[t] = std::complex<double>(re * re + im * im, 0.0);
    }
    transform(z.data(), n, +1);

    for (int t = 0; t < n_frames; t++) acf[t] += z[t].real() / (double) n;
  }

  // Biased: the divisor is T at every lag, not T - tau. The unwindowed
  // transform of a biased estimate cannot go negative; an unbiased one can,
  // and an absorption spectrum that does is meaningless.
  for (int t = 0; t < n_frames; t++) acf[t] /= (double) n_frames;
}

/* ---------------------------------------------------------------------- */

static void lag_window(const std::string &kind, int L, std::vector<double> &w)
{
  w.assign(L, 1.0);
  if (kind == "none" || kind == "boxcar") return;

  if (kind == "blackman") {
    // The HALF window, 1 at lag zero. np.blackman is symmetric, so as a lag
    // window it would delete C(0) -- the largest and best determined term,
    // the one carrying the total intensity -- and weight the middle most.
    for (int t = 0; t < L; t++) {
      const double x = M_PI * t / (double) L;
      w[t] = 0.42 + 0.5 * cos(x) + 0.08 * cos(2.0 * x);
    }
  } else if (kind == "bartlett") {
    for (int t = 0; t < L; t++) w[t] = 1.0 - t / (double) L;
  } else if (kind == "welch") {
    for (int t = 0; t < L; t++) {
      const double x = t / (double) L;
      w[t] = 1.0 - x * x;
    }
  } else if (kind == "lorch") {
    w[0] = 1.0;
    for (int t = 1; t < L; t++) {
      const double x = M_PI * t / (double) L;
      w[t] = sin(x) / x;
    }
  } else {    // hann
    for (int t = 0; t < L; t++) w[t] = 0.5 * (cos(M_PI * t / (double) L) + 1.0);
  }
}

static bool prefactor(const std::string &kind, double temperature, const std::vector<double> &nu,
                      std::vector<double> &pref, std::string &err)
{
  const int n = (int) nu.size();
  pref.assign(n, 1.0);
  if (kind == "classical" || kind == "quadratic") {
    for (int k = 0; k < n; k++) pref[k] = nu[k] * nu[k];
  } else if (kind == "linear") {
    pref = nu;
  } else if (kind == "none") {
    return true;
  } else if (kind == "harmonic") {
    double kT = KB_EV * temperature;
    if (kT < 1.0e-30) kT = 1.0e-30;
    for (int k = 0; k < n; k++) {
      const double x = HBAR_C_EV_CM * nu[k] / kT;
      pref[k] = nu[k] * (1.0 - exp(-x));
    }
  } else {
    err =
        "MADLIB: quantum correction must be harmonic, classical (alias quadratic), linear or none";
    return false;
  }
  return true;
}

static void smooth_gaussian(const std::vector<double> &x, int k, std::vector<double> &y)
{
  const int n = (int) x.size();
  const double sigma = k / 2.355;
  const int radius = (int) (4.0 * sigma + 0.5);
  if (radius < 1) {
    y = x;
    return;
  }

  std::vector<double> wk(2 * radius + 1);
  double s = 0.0;
  for (int i = -radius; i <= radius; i++) {
    wk[i + radius] = exp(-0.5 * (i / sigma) * (i / sigma));
    s += wk[i + radius];
  }
  for (auto &v : wk) v /= s;

  y.assign(n, 0.0);
  for (int j = 0; j < n; j++) {
    double acc = 0.0;
    for (int i = -radius; i <= radius; i++) {
      int m = j + i;
      if (m < 0) m = 0;
      if (m >= n) m = n - 1;
      acc += wk[i + radius] * x[m];
    }
    y[j] = acc;
  }
}

// GPUMD's, via np.convolve(mode="valid"): it shortens the axis and shifts it.
static int smooth_box(const std::vector<double> &x, int k, std::vector<double> &y)
{
  const int n = (int) x.size();
  const int n_out = n - k + 1;
  if (n_out < 1) {
    y = x;
    return n;
  }
  y.assign(n_out, 0.0);
  for (int j = 0; j < n_out; j++) {
    double acc = 0.0;
    for (int m = 0; m < k; m++) acc += x[j + m];
    y[j] = acc / (double) k;
  }
  return n_out;
}

/* ---------------------------------------------------------------------- */

bool LAMMPS_NS::mad_ir_spectrum(const double *mu, int n_frames, const MADIrConfig &cfg,
                                MADIrResult &res, std::string &err)
{
  err.clear();
  if (n_frames < 2) {
    err = "MADLIB: an IR spectrum needs at least two stored frames";
    return false;
  }
  if (cfg.dt_fs <= 0.0) {
    err = "MADLIB: the stored-frame interval must be positive";
    return false;
  }
  if (cfg.acf_ratio <= 0.0 || cfg.acf_ratio > 1.0) {
    err = "MADLIB: ir acf_ratio must be in (0, 1]";
    return false;
  }

  // The backward difference costs one frame and a constant factor; the
  // factor matters only to the absolute units, which a fitted scale absorbs.
  std::vector<double> rate;
  if (cfg.correlate == "rate") {
    rate.resize(3 * (n_frames - 1));
    for (int t = 1; t < n_frames; t++)
      for (int d = 0; d < 3; d++)
        rate[3 * (t - 1) + d] = (mu[3 * t + d] - mu[3 * (t - 1) + d]) / cfg.dt_fs;
    mu = rate.data();
    n_frames -= 1;
  } else if (cfg.correlate != "dipole") {
    err = "MADLIB: ir correlate must be dipole or rate, not " + cfg.correlate;
    return false;
  }

  int L = (int) (n_frames * cfg.acf_ratio);
  if (L < 1) L = 1;
  if (L > n_frames) L = n_frames;
  const int N2 = 2 * L - 1;
  const double dnu = CM_PER_INV_FS / ((double) N2 * cfg.dt_fs);

  std::vector<double> acf_full;
  mad_ir_autocorrelation(mu, n_frames, cfg.subtract_mean, acf_full, res.mu_mean);
  res.acf.assign(acf_full.begin(), acf_full.begin() + L);

  std::vector<double> w;
  lag_window(cfg.window, L, w);
  std::vector<double> a(L);
  a[0] = res.acf[0] * w[0];
  for (int t = 1; t < L; t++) a[t] = 2.0 * res.acf[t] * w[t];

  // The cut is applied before the transform, so it is only evaluated where it
  // will be kept. k_max is found by scanning with the same expression that
  // builds nu, not by int(max_freq/dnu): the two differ in the last bit at a
  // bin landing exactly on max_freq, and that is a silent one-bin
  // disagreement at some grids and not others.
  int k_max = -1;
  for (int k = 0; k < L; k++) {
    if (k * dnu <= cfg.max_freq_cm)
      k_max = k;
    else
      break;
  }
  if (k_max < 1) {
    err = "MADLIB: the bin spacing is " + std::to_string(dnu) +
        " cm^-1, so no bin lands at or below nu_max = " + std::to_string(cfg.max_freq_cm) +
        "; keep more lags or run longer";
    return false;
  }
  const int n_out = k_max + 1;

  std::vector<double> nu(n_out), M(n_out);
  for (int k = 0; k < n_out; k++) nu[k] = k * dnu;
  for (int k = 0; k < n_out; k++) {
    double acc = 0.0;
    for (int t = 0; t < L; t++) acc += a[t] * cos(2.0 * M_PI * k * t / (double) N2);
    M[k] = acc;
  }

  std::vector<double> pref;
  if (!prefactor(cfg.quantum_correction, cfg.temperature, nu, pref, err)) return false;

  std::vector<double> inten(n_out), pw(n_out);
  for (int k = 0; k < n_out; k++) {
    inten[k] = pref[k] * M[k];
    pw[k] = M[k];
  }

  int n_box = n_out;
  if (cfg.smooth_k > 1 && n_out > cfg.smooth_k) {
    std::vector<double> tmp;
    if (cfg.smooth_kind == "box") {
      n_box = smooth_box(inten, cfg.smooth_k, tmp);
      inten = tmp;
      smooth_box(pw, cfg.smooth_k, tmp);
      pw = tmp;
      for (int k = 0; k < n_box; k++) nu[k] = k * dnu + 0.5 * (cfg.smooth_k - 1) * dnu;
    } else {
      smooth_gaussian(inten, cfg.smooth_k, tmp);
      inten = tmp;
      smooth_gaussian(pw, cfg.smooth_k, tmp);
      pw = tmp;
    }
  }

  res.n_frames = n_frames;
  res.n_lag = L;
  res.n_freq = n_box;
  res.d_nu = dnu;
  res.resolution = 2.0 * dnu;
  res.nyquist = CM_PER_INV_FS / (2.0 * cfg.dt_fs);

  res.freq.assign(nu.begin(), nu.begin() + n_box);
  res.intensity_raw.assign(inten.begin(), inten.begin() + n_box);
  res.power_raw.assign(pw.begin(), pw.begin() + n_box);

  // The intensity is normalised over everything returned; the power only over
  // the vibrational range, because M(w) carries a DC peak that is not a
  // vibration and would otherwise set the scale for the whole plot.
  double peak = 0.0, peakp = 0.0;
  for (int k = 0; k < n_box; k++) peak = std::max(peak, std::fabs(inten[k]));
  for (int k = 0; k < n_box; k++)
    if (res.freq[k] >= cfg.power_dc_cutoff_cm) peakp = std::max(peakp, std::fabs(pw[k]));
  if (peakp <= 0.0)
    for (int k = 0; k < n_box; k++) peakp = std::max(peakp, std::fabs(pw[k]));
  res.peak = peak;
  res.peak_power = peakp;

  res.intensity = res.intensity_raw;
  res.power = res.power_raw;
  if (cfg.normalise && peak > 0.0)
    for (auto &v : res.intensity) v /= peak;
  if (cfg.normalise && peakp > 0.0)
    for (auto &v : res.power) v /= peakp;

  return true;
}
