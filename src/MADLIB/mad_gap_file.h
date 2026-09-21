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

  // dipole_model = .true.: the block's "energy" is a fitting artefact whose
  // gradient is the local dipole, not an energy. Read so it can be refused.
  bool dipole_model = false;

  // the local-property model itself
  std::string label;
  std::string file_qs, file_alphas;
  double zeta = 1.0, delta = 1.0, v0 = 0.0;
  int n_sparse = 0, dim = 0, n_nonzero = 0;
  std::vector<double> qs;    // (dim x n_sparse), column-major as Fortran wants
  std::vector<double> alphas;
  std::vector<int> p_i, p_j;
  std::vector<double> p_el;

  double rcut_max() const;
};

/* A whole TurboGAP potential, which is FOUR kinds of term summed:

       E = E_soap_turbo + E_distance_2b + E_angle_3b + E_core_pot

   Not one. Of the potentials in the test data exactly one is soap_turbo alone
   -- water_dipole.gap -- and that is a dipole model with no energy. Every
   fitted potential anyone runs carries 2b, 3b and core_pot blocks beside its
   SOAP block, so a reader that returned only the SOAP part would let a caller
   compute a plausible-looking energy missing three terms, silently. */

struct MADGap2b {
  std::string species1, species2;
  double delta = 1.0, sigma = 1.0, rcut = 0.0, buffer = 0.5, e0 = 0.0;
  std::string file_qs, file_alphas;
  int n_sparse = 0;
  std::vector<double> qs;        // one descriptor value per sparse point
  std::vector<double> alphas;    // column 1 of alphas_sparse
  std::vector<double> cutoff;    // column 2 of the same file
};

struct MADGap3b {
  std::string species_center, species1, species2;
  double delta = 1.0, rcut = 0.0, buffer = 0.5, e0 = 0.0;
  std::vector<double> sigma;    // three of them
  std::string kernel_type = "pol";
  std::string file_qs, file_alphas;
  int n_sparse = 0, dim = 3;
  std::vector<double> qs;    // (3 x n_sparse), column-major
  std::vector<double> alphas, cutoff;
};

struct MADGapCorePot {
  std::string species1, species2, file;
  double yp1 = 1.0e30, ypn = 1.0e30;
  std::vector<double> x, V, dVdx2;
};

struct MADGapPotential {
  std::vector<std::string> species_types;    // the order pair_coeff gave
  std::vector<MADGapBlock> soap;
  std::vector<MADGap2b> twob;
  std::vector<MADGap3b> threeb;
  std::vector<MADGapCorePot> core;
  std::vector<double> e0;    // per species, zero unless the caller sets it

  double rcut_max() const;
};

class MADGapFile {
 public:
  // Reads every soap_turbo block carrying a local property with this label.
  // Throws std::runtime_error with a message naming the file and line.
  static std::vector<MADGapBlock> read(const std::string &path, const std::string &label);

  // Reads the whole potential: the energy soap_turbo blocks and every 2b, 3b
  // and core_pot block. species_types fixes the species numbering, because
  // TurboGAP's routines take species as indices into exactly that list.
  //
  // core_pot_cutoff and core_pot_buffer are TurboGAP input parameters, not
  // properties of the .gap file, and they change the tabulated potential on
  // read. Defaults match TurboGAP's own.
  static MADGapPotential read_potential(const std::string &path,
                                        const std::vector<std::string> &species_types,
                                        double core_pot_cutoff = 1.0e10,
                                        double core_pot_buffer = 1.0);

 private:
  static void load_sparse(MADGapBlock &b, const std::string &dir);
};

}    // namespace LAMMPS_NS

#endif
