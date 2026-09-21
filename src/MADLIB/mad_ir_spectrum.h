/* -*- c++ -*- ----------------------------------------------------------
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

#ifndef LMP_MAD_IR_SPECTRUM_H
#define LMP_MAD_IR_SPECTRUM_H

#include <string>
#include <vector>

namespace LAMMPS_NS {

/* An infrared spectrum from a total-dipole trajectory, by Wiener-Khinchin.

     C(tau) = < dmu(t) . dmu(t+tau) >,   dmu = mu - <mu>
     I(nu)  = P(nu) sum_tau a(tau) cos( 2 pi k tau / N2 )
     a(0) = C(0) w(0),  a(tau) = 2 C(tau) w(tau)

   A translation of TurboGAP's ir_fft.f90, which is itself a translation of
   GPUMD's TNEP/spectroscopy.py (Xu et al., JCTC 20, 3273 (2024)) and is
   verified against it to 3.5e-14 over eleven configurations. Following it
   exactly is the point: it means this file inherits an external reference,
   which no estimator written from the definition would have.

   The correlation is by FFT over the whole buffer, O(T log T), with a radix-2
   transform of its own so that the package needs no FFT library.

   Two things about it are chosen, not incidental:

   THE LONGEST LAG SETS THE RESOLUTION, NOT THE RUN LENGTH. acf_ratio is the
   fraction of the buffer kept as lags, and d_nu = CM_PER_INV_FS/(N2 dt). A
   long run with a short lag window gives a coarse spectrum.

   THE WEIGHTING IS A PHYSICAL CHOICE. "harmonic" is nu(1-exp(-hbar c nu/kT)),
   "classical" is nu^2. They agree only for hbar w << kT -- below about 208
   cm^-1 at 300 K -- and at 3400 cm^-1 the harmonic form is 16 times smaller
   relative to the librational region. That is a tilt across the range, so no
   overall scale absorbs it, and two spectra of one trajectory disagreeing
   about band heights is this before it is anything else. */

struct MADIrConfig {
  double dt_fs = 1.0;    // between STORED frames, in fs
  double acf_ratio = 0.1;
  double max_freq_cm = 4000.0;
  double temperature = 300.0;
  double power_dc_cutoff_cm = 100.0;
  int smooth_k = 10;    // in BINS, not cm^-1
  std::string smooth_kind = "gaussian";
  std::string window = "hann";
  std::string quantum_correction = "harmonic";
  bool subtract_mean = true;
  bool normalise = true;
};

struct MADIrResult {
  int n_frames = 0, n_lag = 0, n_freq = 0;
  double d_nu = 0.0, resolution = 0.0, nyquist = 0.0;
  double peak = 0.0, peak_power = 0.0;
  double mu_mean[3] = {0.0, 0.0, 0.0};
  std::vector<double> freq, intensity, power, intensity_raw, power_raw, acf;
};

// mu is (n_frames x 3), row-major and chronological. Returns false and fills
// err when the configuration cannot produce a spectrum.
bool mad_ir_spectrum(const double *mu, int n_frames, const MADIrConfig &cfg, MADIrResult &res,
                     std::string &err);

// Exposed for the tests: the mean-subtracted autocorrelation at every lag.
void mad_ir_autocorrelation(const double *mu, int n_frames, bool subtract_mean,
                            std::vector<double> &acf, double mu_mean[3]);

}    // namespace LAMMPS_NS

#endif
