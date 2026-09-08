/* -*- c++ -*- ----------------------------------------------------------
   MADLIB - Molecular Augmented Dynamics for LAMMPS
------------------------------------------------------------------------- */

#ifndef LMP_MAD_TARGET_H
#define LMP_MAD_TARGET_H

#include <string>
#include <vector>

namespace LAMMPS_NS {
class Error;

/* An experimental curve resampled onto the prediction grid, plus the loss it
   defines.

   E = 1/2 gamma sum_k w_k ( s y_k + b - yexp_k )^2

   The weights w_k are trapezoidal in the abscissa and normalised to mean one,
   so gamma means the same thing whatever the grid spacing. s and b sit at
   their own least-squares optimum, so by the envelope theorem dE/dy picks up
   no term from them. */

class MADTarget {
 public:
  MADTarget(Error *error);

  void read(const std::string &filename);
  void set_grid(const std::vector<double> &x, bool weight_by_spacing);

  // Both default off; scale is what makes an arbitrary-units prediction
  // comparable, offset adds a free baseline.
  void set_fit(bool scale, bool offset) { fit_scale = scale; fit_offset = offset; }

  // Residual c_k = s*y_k + b - yexp_k, the loss, and dE/dy_k. Returns the loss.
  double loss(const double *y, double gamma, double *dEdy);

  double dissimilarity() const { return dissim; }
  double rel_error() const;
  double scale() const { return s_fit; }
  double offset() const { return b_fit; }

  int size() const { return nk; }
  const double *y_exp() const { return yexp.data(); }
  const double *weights() const { return wgt.data(); }
  double range_min() const { return xraw.front(); }
  double range_max() const { return xraw.back(); }

 private:
  Error *error;
  std::string fname;
  std::vector<double> xraw, yraw;    // the file, as read
  std::vector<double> yexp, wgt;     // on the prediction grid
  int nk;
  bool fit_scale, fit_offset;
  double s_fit, b_fit, dissim, dissim_ref;

  void solve_fit(const double *y);
};

}    // namespace LAMMPS_NS

#endif
