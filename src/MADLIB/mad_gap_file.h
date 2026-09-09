/* -*- c++ -*- ----------------------------------------------------------
   MADLIB - Molecular Augmented Dynamics for LAMMPS
------------------------------------------------------------------------- */

#ifndef LMP_MAD_GAP_FILE_H
#define LMP_MAD_GAP_FILE_H

#include <string>
#include <vector>

namespace LAMMPS_NS {

/* The subset of a TurboGAP .gap file that a local-property model needs.

   Only soap_turbo blocks carrying local_property_* keys are kept; the 2b, 3b
   and core_pot blocks, and the energy GAP's own sparse set, are skipped. That
   is the whole reason this is 300 lines of C++ rather than a vendored
   read_files.f90, which is 6.7k lines entangled with the rest of TurboGAP.

   One block per central species is the normal arrangement -- a CEBE model fits
   C 1s and O 1s separately, with different V0 and often a different zeta from
   the energy GAP in the same block -- so a model is a list of blocks and each
   atom is handled by the block whose central_species it matches. */

struct MADGapBlock {
  int central_species = 0;
  int n_species = 0;
  std::vector<std::string> species;

  std::vector<double> rcut_hard, rcut_soft, buffer, nf, global_scaling;
  std::vector<double> atom_sigma_r, atom_sigma_t;
  std::vector<double> atom_sigma_r_scaling, atom_sigma_t_scaling;
  std::vector<double> amplitude_scaling, central_weight;
  std::vector<int> alpha_max;
  int l_max = 0, radial_enhancement = 0;
  std::string basis = "poly3gauss", scaling_mode = "polynomial";

  bool compress = false;
  std::string compress_mode = "none";
  std::string compress_file = "none";

  // the local-property model itself
  std::string label;
  std::string file_qs, file_alphas;
  double zeta = 1.0, delta = 1.0, v0 = 0.0;
  int n_sparse = 0, dim = 0, n_nonzero = 0;
  std::vector<double> qs;        // (dim x n_sparse), column-major as Fortran wants
  std::vector<double> alphas;
  std::vector<int> p_i, p_j;
  std::vector<double> p_el;

  double rcut_max() const;
};

class MADGapFile {
 public:
  // Reads every soap_turbo block carrying a local property with this label.
  // Throws std::runtime_error with a message naming the file and line.
  static std::vector<MADGapBlock> read(const std::string &path, const std::string &label);

 private:
  static void load_sparse(MADGapBlock &b, const std::string &dir);
};

}    // namespace LAMMPS_NS

#endif
