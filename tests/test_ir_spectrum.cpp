// Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)
/* Drives the MADLIB IR pipeline on a dipole file, in the same CLI shape as
   TurboGAP's irfftverify, so the two can be compared directly. */
#include "../src/MADLIB/mad_ir_spectrum.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

using namespace LAMMPS_NS;

int main(int argc, char **argv)
{
  if (argc < 12) { fprintf(stderr, "usage: %s dip dt win ratio numax smk smkind qc T dc out\n", argv[0]); return 2; }

  std::vector<double> mu;
  std::ifstream in(argv[1]);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    double a, b, c;
    if (ss >> a >> b >> c) { mu.push_back(a); mu.push_back(b); mu.push_back(c); }
  }
  const int n = (int) mu.size() / 3;

  MADIrConfig cfg;
  cfg.dt_fs = atof(argv[2]);
  cfg.window = argv[3];
  cfg.acf_ratio = atof(argv[4]);
  cfg.max_freq_cm = atof(argv[5]);
  cfg.smooth_k = atoi(argv[6]);
  cfg.smooth_kind = argv[7];
  cfg.quantum_correction = argv[8];
  cfg.temperature = atof(argv[9]);
  cfg.power_dc_cutoff_cm = atof(argv[10]);

  MADIrResult res;
  std::string err;
  if (!mad_ir_spectrum(mu.data(), n, cfg, res, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }

  FILE *f = fopen(argv[11], "w");
  fprintf(f, "# n_frames = %d   n_lag = %d   n_freq = %d\n", res.n_frames, res.n_lag, res.n_freq);
  fprintf(f, "# d_nu = %14.6f   resolution = %14.6f   nyquist = %14.6f\n", res.d_nu, res.resolution, res.nyquist);
  fprintf(f, "# peak = %22.14E   peak_power = %22.14E\n", res.peak, res.peak_power);
  fprintf(f, "# nu   intensity   power   intensity_raw   power_raw\n");
  for (int k = 0; k < res.n_freq; k++)
    fprintf(f, "%24.15E%24.15E%24.15E%24.15E%24.15E\n", res.freq[k], res.intensity[k],
            res.power[k], res.intensity_raw[k], res.power_raw[k]);
  fclose(f);
  printf("n_lag = %d  n_freq = %d  d_nu = %10.4f\n", res.n_lag, res.n_freq, res.d_nu);
  return 0;
}
