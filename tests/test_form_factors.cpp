#include "../src/MADLIB/mad_form_factors.h"
#include <cmath>
#include <cstdio>
#include <vector>
using namespace LAMMPS_NS;

int main()
{
  std::vector<double> q;
  for (int i = 0; i < 51; i++) q.push_back((1.0 + (10.0 - 1.0) * i / 50.0) / (2.0 * M_PI));

  for (int compat = 0; compat < 2; compat++) {
    MADFormFactors ff(false, compat);
    ff.add_species("C");
    ff.add_species("O");
    ff.tabulate(q);
    for (int k = 0; k < 51; k++) printf("%d xray %.17g %.17g %.17g\n", compat, q[k], ff.f(0, k), ff.f(1, k));
  }
  MADFormFactors nf(true, false);
  nf.add_species("C");
  nf.add_species("O");
  nf.tabulate(q);
  printf("neutron %.17g %.17g\n", nf.f(0, 0), nf.f(1, 0));
  return 0;
}
