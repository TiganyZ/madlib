#include "../src/MADLIB/mad_target.h"
#include <cmath>
#include <cstdio>
#include <vector>
using namespace LAMMPS_NS;

int main(int argc, char **argv)
{
  std::vector<double> x;
  for (int i = 0; i < 41; i++) x.push_back(1.2 + (9.5 - 1.2) * i / 40.0);

  for (int mode = 0; mode < 4; mode++) {
    MADTarget t(nullptr);
    t.read(argv[1]);
    t.set_fit(mode & 1, mode & 2);
    t.set_grid(x, true);

    std::vector<double> y(x.size()), d(x.size());
    for (size_t k = 0; k < x.size(); k++) y[k] = 0.7 * sin(1.3 * x[k]) + 0.11 * x[k];

    const double gamma = 3.5;
    const double e = t.loss(y.data(), gamma, d.data());
    printf("mode %d E %.17g D %.17g rel %.17g s %.17g b %.17g\n", mode, e, t.dissimilarity(),
           t.rel_error(), t.scale(), t.offset());
    for (size_t k = 0; k < x.size(); k++) printf("  g %zu %.17g %.17g %.17g\n", k, x[k], d[k], t.y_exp()[k]);
  }
  return 0;
}
