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

#include <algorithm>
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

// What is actually correlated. The backward difference costs one frame and a
// constant factor; the factor matters only to the absolute units, which a
// fitted scale absorbs. Shared by the spectrum and its gradient so the two
// cannot disagree about which series they are talking about.
static bool apply_correlate(const MADIrConfig &cfg, const double *&mu, int &n_frames,
                            std::vector<double> &rate, std::string &err)
{
  if (cfg.correlate == "dipole") return true;
  if (cfg.correlate != "rate") {
    err = "MADLIB: ir correlate must be dipole or rate, not " + cfg.correlate;
    return false;
  }
  if (n_frames < 2) {
    err = "MADLIB: correlate rate needs at least two stored frames";
    return false;
  }
  rate.resize(3 * (n_frames - 1));
  for (int t = 1; t < n_frames; t++)
    for (int d = 0; d < 3; d++)
      rate[3 * (t - 1) + d] = (mu[3 * t + d] - mu[3 * (t - 1) + d]) / cfg.dt_fs;
  mu = rate.data();
  n_frames -= 1;
  return true;
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

  std::vector<double> rate;
  if (!apply_correlate(cfg, mu, n_frames, rate, err)) return false;

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

/* ---------------------------------------------------------------------- */

bool LAMMPS_NS::mad_ir_acf_gradient(const MADIrConfig &cfg, const MADIrResult &res,
                                    const double *dLdI, std::vector<double> &acf_grad,
                                    std::string &err)
{
  err.clear();
  if (cfg.smooth_k > 1) {
    err =
        "MADLIB: an IR bias needs smooth 1. Smoothing is a second linear map and its "
        "transpose is not applied here; the lag window already sets the resolution, so "
        "smoothing on top of it is weighting the same thing twice";
    return false;
  }

  const int L = res.n_lag, nf = res.n_freq;
  if (L < 1 || nf < 1) {
    err = "MADLIB: no spectrum to take a gradient of yet";
    return false;
  }
  const int N2 = 2 * L - 1;

  std::vector<double> pref;
  if (!prefactor(cfg.quantum_correction, cfg.temperature, res.freq, pref, err)) return false;

  std::vector<double> w;
  lag_window(cfg.window, L, w);

  // The transpose of the forward transform: I_k = pref_k sum_tau a(tau)
  // cos(theta_k tau) with a(0) = C(0)w(0) and a(tau) = 2C(tau)w(tau), so
  // dI_k/dC(tau) carries the same window and the same factor of two back.
  acf_grad.assign(L, 0.0);
  for (int t = 0; t < L; t++) {
    double acc = 0.0;
    for (int k = 0; k < nf; k++) {
      if (dLdI[k] == 0.0) continue;
      acc += dLdI[k] * pref[k] * cos(2.0 * M_PI * k * t / (double) N2);
    }
    acf_grad[t] = (t == 0 ? 1.0 : 2.0) * w[t] * acc;
  }
  return true;
}

/* ---------------------------------------------------------------------- */

void LAMMPS_NS::mad_ir_series_mean(const double *mu, int n_frames, const MADIrConfig &cfg,
                                   double mean[3])
{
  mean[0] = mean[1] = mean[2] = 0.0;
  if (!cfg.subtract_mean) return;

  if (cfg.correlate == "rate") {
    // sum of (mu(t+1) - mu(t)) telescopes to the endpoints.
    if (n_frames < 2) return;
    const double s = 1.0 / ((double) (n_frames - 1) * cfg.dt_fs);
    for (int d = 0; d < 3; d++) mean[d] = (mu[3 * (n_frames - 1) + d] - mu[d]) * s;
    return;
  }
  if (n_frames < 1) return;
  for (int t = 0; t < n_frames; t++)
    for (int d = 0; d < 3; d++) mean[d] += mu[3 * t + d];
  for (int d = 0; d < 3; d++) mean[d] /= (double) n_frames;
}

/* ---------------------------------------------------------------------- */

void LAMMPS_NS::mad_ir_newest_gradient(const double *mu, int n_frames, const MADIrConfig &cfg,
                                       const double mean[3], const std::vector<double> &acf_grad,
                                       double dLdmu[3])
{
  dLdmu[0] = dLdmu[1] = dLdmu[2] = 0.0;
  const bool rate = (cfg.correlate == "rate");
  const int M = rate ? n_frames - 1 : n_frames;
  const int L = std::min((int) acf_grad.size(), M);
  if (M < 2 || L < 1) return;

  // The correlated series, read rather than built.
  auto dev = [&](int t, int d) {
    const double v =
        rate ? (mu[3 * (t + 1) + d] - mu[3 * t + d]) / cfg.dt_fs : mu[3 * t + d];
    return v - mean[d];
  };

  // C(tau) = (1/M) sum_a d(a).d(a+tau), so the newest frame enters twice at
  // tau = 0 and once at every other lag -- and also through <mu>, which
  // depends on it. That second route contributes partial sums of d at both
  // ends of the buffer; because the whole series sums to zero each is a sum
  // over at most L terms, so carrying it exactly costs O(L) and the gradient
  // is finite-difference exact instead of right to O(M^-1/2).
  std::vector<double> front(3 * L, 0.0), back(3 * L, 0.0);
  if (cfg.subtract_mean)
    for (int t = 1; t < L; t++)
      for (int d = 0; d < 3; d++) {
        front[3 * t + d] = front[3 * (t - 1) + d] + dev(t - 1, d);
        back[3 * t + d] = back[3 * (t - 1) + d] + dev(M - t, d);
      }

  const double invM = 1.0 / (double) M;
  for (int d = 0; d < 3; d++) {
    double acc = 2.0 * acf_grad[0] * dev(M - 1, d) * invM;
    for (int t = 1; t < L; t++)
      acc += acf_grad[t] *
          (dev(M - 1 - t, d) * invM + (front[3 * t + d] + back[3 * t + d]) * invM * invM);
    dLdmu[d] = acc;
  }

  // One more link when the rate was correlated: eta = (mu - mu_prev)/dt, and
  // mu_prev is part of the frozen past.
  if (rate)
    for (int d = 0; d < 3; d++) dLdmu[d] /= cfg.dt_fs;
}
