# MADLIB -- Molecular Augmented Dynamics for LAMMPS

Bias a LAMMPS simulation towards experimental data. MADLIB predicts an
observable from the current configuration, penalises the mismatch, and adds the
gradient of that penalty to the forces:

```
V_tot({r}) = V_pot({r}) + sum_o (gamma_o/2) sum_k w_k ( y_pred,k({r}) - y_exp,k )^2
```

It works with **any** LAMMPS potential, and the per-atom properties some
observables need (core-electron binding energies, charges, dipoles) come from
**any** ML model behind one two-method interface.

Ported from the experimental-observables machinery in
[TurboGAP](https://turbogap.fi), which is welded to its own driver and its own
SOAP/GAP potential.

## Status

| | |
|---|---|
| PDF `g(r)` / `D(r)` | working, validated |
| `S(q)`, XRD, ND | working, validated |
| **XPS**, and any spectrum from a broadened per-atom scalar | working, validated |
| **The per-atom property layer** | working -- `turbogap`, `coord` and `table` backends |
| **TurboGAP `.gap` models, unchanged** | working -- reproduces TurboGAP's own binding energies to 1.2e-9 relative |
| Experimentally biased MC | working -- via LAMMPS's own MC fixes, no new MC code |
| **IR prediction** | working -- agrees with GPUMD's pipeline to 1.5e-13, via TurboGAP |
| IR bias | not planned here; see the note under `fix mad/ir` |
| **Python / torch property backend** | working -- MACE, GRACE, any torch model, differentiated by autograd |
| **MLIAP property backend** | working -- SNAP, SO3, ACE descriptors with linear, quadratic or NN models |

Built against LAMMPS `stable_22Jul2025_update6`.

## Quick start

```lammps
units      metal                                   # eV, Angstrom

compute gr  all mad/pdf r 0.1 6.0 201 kde 0.1
fix     xrd all mad/xrd c_gr target measured.fq q 1.05 9.97 201 &
                        output qFq species C O gamma 0.0 100.0

thermo_style custom step temp pe f_xrd f_xrd[2]
#                                E_exp  rel_error
```

`f_xrd` is the bias energy; `f_xrd[2]` is the dimensionless relative error,
which is the number to watch. The energy and virial contribute to `pe` and to
the pressure by default; `fix_modify xrd energy no` turns that off.

`gamma` with two values ramps linearly over the run, which is how a restraint
is switched on without kicking the structure apart on step one.

**Size gamma from the unbiased dissimilarity.** Run once with `gamma 0.0`, read
`f_xrd[3]`, and choose gamma so that `E = gamma*D/2` is a small fraction of the
cohesive energy. Getting this wrong by two orders of magnitude is the commonest
way to destroy a run, and it does not look like a failure -- the dissimilarity
falls obediently while the structure comes apart.

## Experimentally biased Monte Carlo

There is no MADLIB MC code, and none is needed. `fix gcmc`, `fix atom/swap`,
`fix widom` and `fix tfmc` all price a trial move with `compute pe` after
calling `modify->post_force()`, and a fix with `energy_global_flag` set and
`fix_modify energy yes` contributes to that. So

```lammps
fix mad  all mad/xrd c_gr target measured.fq q 1.05 9.97 201 species C O gamma 5.0
fix swap all atom/swap 5 20 29494 300.0 ke no types 1 2
```

is a reverse Monte Carlo run. Measured on the 91-atom glassy-carbon test case:
the bias takes the swap acceptance from 1200/1200 to 335/1200 and the relative
error from 0.371 to 0.109.

## Commands

### `compute ID group mad/pdf r <rmin> <rmax> <nbins> [keyword value ...]`

Partial pair distributions `g_ab(r)`, differentiably. Output is an array of
`nbins` rows and `nchannels+1` columns: `r`, then one column per species pair
in upper-triangle row-major order.

| keyword | default | meaning |
|---|---|---|
| `kde` | **required** | Gaussian width in Angstrom. A histogram has zero gradient, so a bias built on one applies no force; a missing or zero width is an error rather than a silent no-op. |
| `rcut` | `rmax + trunc*kde` | neighbour cutoff. The default puts the hard cut far enough out that a pair crossing it changes nothing -- see *Numerics* below. |
| `trunc` | `8.0` | Gaussian tails past this many sigma are dropped, an error of `exp(-32)`. |
| `compat` | | `turbogap` reproduces TurboGAP's bin-lower-edge shell volume. |

### `fix ID group mad/pdf <compute> [keyword value ...]`

Bias towards a measured `g(r)` or `D(r) = 4 pi rho r (g(r)-1)` (`output`).

### `fix ID group mad/sq <compute> [...]` and `fix ID group mad/xrd <compute> [...]`

Structure factor, X-ray and neutron diffraction. One class; the style name
picks the default weighting.

| keyword | default | meaning |
|---|---|---|
| `q` | **required** | `<qmin> <qmax> <nq>` |
| `units` | `q` | `q` (1/Angstrom) or `twotheta` (degrees) |
| `weighting` | `sq` / `xray` | `sq`, `xray`, `neutron` |
| `species` | | one element label per atom type; required for `xray`/`neutron` |
| `output` | `raw` / `qFq` | `raw`, `F(q)`, `q*F(q)` |
| `wavelength` | `1.5405981` | Angstrom, Cu K-alpha1 |
| `window` | `lorch` | Lorch window damping the truncation ringing |
| `window_cut` | `rmax` | its cutoff -- the **transform range**, not the neighbour cutoff |
| `lp`, `polarization` | `no`, `1.0` | powder Lorentz-polarization factor |

### `compute ID group mad/property/*`

A per-atom quantity and its adjoint. XPS needs core-electron binding energies;
IR will need local dipoles. Two backends so far:

```lammps
# a fitted TurboGAP GPR -- an existing .gap file, used as it stands
compute q  all mad/property/turbogap gap_files/CO.gap core_electron_be species C O

compute mu all mad/property/dipole charges 0.5 -0.5  # or 'charge' to use atom->q
# any model reachable from Python: MACE, GRACE, a torch module
compute q  all mad/property/python my_model ncomp 1 cutoff 5.0

# an ML-IAP model, with any descriptor ML-IAP supports
compute q  all mad/property/mliap model linear cebe.model descriptor sna cebe.descriptor

compute cn all mad/property/coord cutoff 1.7 2.1     # smooth coordination number
compute q  all mad/property/table values 290.0 536.0
compute q  all mad/property/table file cebe.dat    # "atom-id value" per line
```

`turbogap` evaluates `q_i = V0 + delta^2 sum_s alpha_s (soap_i . Q_s)^zeta` through
soap_turbo, one block per central species. Fitting a core-electron-binding-energy
model is months of work and there is no reason to refit one to move it, so the
potential file is read as it stands -- including the fact that its sparse-set
paths are written relative to the directory TurboGAP was *run* from rather than
to the file. Needs the Fortran backend; without it the style still exists and
says what to build.

`dipole` gives `mu_i = q_i r_i` from per-type charges or from `atom->q`, so any
model that writes charges is a dipole model here. **`sum q_i r_i` is not defined
in a periodic box** -- it depends on the origin and on which images the atoms
happen to be wrapped into -- so a periodic run needs molecule IDs, and gets the
sum of molecular dipoles built from intramolecular vectors; a periodic box
without them is refused rather than given a plausible number.

`coord` has an analytic gradient and no data files, which is what lets the whole
XPS chain be finite-differenced before any ML model is involved. `table` has no
gradient by construction and says so rather than silently exerting no force, so
it is for prediction (`gamma 0.0`) from values computed elsewhere.

**Adding a model is one method.** What the bias needs is never the full Jacobian
`dq_i/dr_j`, only its contraction with a per-atom weight:

```cpp
virtual void adjoint(double **lambda, double *virial);   // f_j -= sum_i lambda_i dq_i/dr_j
```

and that is the force a model already computes from a weighted per-atom energy
`sum_i lambda_i q_i`. A GAP scales its kernel derivative row, MLIAP scales
`betas`, a torch model takes one backward pass. This is the whole of what makes
MADLIB model-agnostic.

### `fix ID group mad/xps <compute> [keyword value ...]`

```
u_l = sum_i exp( -(x_l - q_i)^2 / 2 sigma^2 ),   y = u / ||u||
```

| keyword | default | meaning |
|---|---|---|
| `energy` | **required** | `<emin> <emax> <nsamples>`, in eV for XPS |
| `sigma` | `0.4` | broadening width, same units as the property |
| `norm` | `l2` | `l2` (TurboGAP's), `area`, or `none` |
| `trunc` | `8.0` | Gaussian tails past this many sigma are dropped |

Also registered as `fix mad/spectrum`, since nothing in it is specific to
photoemission -- any observable that is a broadened sum over a per-atom scalar
fits.

### `fix ID group mad/ir <stride> <compute> [keyword value ...]`

An infrared spectrum from the total dipole of a running trajectory, by
Wiener-Khinchin.

```lammps
compute mu all mad/property/dipole charges 0.5 -0.5
fix     ir all mad/ir 2 c_mu nu_max 4000 acf_ratio 0.25 qc classical &
           smooth 5 gaussian file ir_spectrum.dat
```

| keyword | default | meaning |
|---|---|---|
| `nu_max` | `4000.0` | top of the returned range, cm^-1; refused above Nyquist |
| `acf_ratio` | `0.1` | fraction of the buffer kept as lags. **This sets the resolution.** |
| `window` | `hann` | lag window: `hann`, `bartlett`, `welch`, `lorch`, `blackman`, `none` |
| `qc` | `harmonic` | `harmonic`, `classical` (alias `quadratic`), `linear`, `none` |
| `temp` | `300.0` | K, for `harmonic` |
| `smooth` | `10 gaussian` | width in **bins**, not cm^-1; `gaussian` or `box` |
| `frame_dt` | | fs between frames, for a `rerun` where `update->dt` is not the interval |

`compute_array` gives the spectrum as `(nu, intensity, power)`;
`compute_vector` gives `{frames, lags, bin, Nyquist, peak}`.

**Prediction only.** A bias is a much larger job -- the spectrum is a functional
of the whole trajectory while only the newest frame can still be moved, which
is what forces TurboGAP's frozen-past adjoint and its three alternative
estimators -- and none of it is needed to report a spectrum.

**The weighting is a physical choice, not a detail.** `harmonic` is
`nu(1-exp(-hbar c nu/kT))`, `classical` is `nu^2`. They agree only for
`hbar w << kT` -- below about 208 cm^-1 at 300 K -- and at 3400 cm^-1 the harmonic
form is 16 times smaller relative to the librational region. That is a tilt
across the range, so no overall scale absorbs it. Two spectra of one trajectory
disagreeing about band heights is this before it is anything else.

**Resolution is the longest lag, not the run length.** A long run at a small
`acf_ratio` gives a coarse spectrum and no amount of extra sampling changes it.
Both this and the frame interval are written into the output header, because
neither is a property of the sample and both are easy to assume wrongly.

### Shared by every `fix mad/*`

`target <file>` (two columns, no header), `gamma <g>` or `gamma <beg> <end>`,
`every <N>`, `fit <scale> <offset>`, `file <out>`, `write_every <N>`,
`compat turbogap`.

`compute_vector` gives `{E, rel_error, dissimilarity, gamma, scale, offset}`.

## Numerics -- things that would otherwise be silently wrong

Each of these is an error or a guard, not a warning, because each produces a
plausible-looking answer:

- **A zero KDE width** gives zero bias force. Refused.
- **A neighbour cutoff inside the ghost range** silently truncates `g(r)`.
  Refused, naming `comm_modify cutoff`.
- **A prediction grid outside the target's range** would extrapolate a measured
  curve. Refused.
- **A hard cut at `rmax`** lets a pair crossing it step the pattern. Invisible
  in `g(r)` and lethal to the virial -- it made the diagonal stress wrong by a
  factor of order one while the shear stayed perfect. The default `rcut` sits
  `trunc*kde` beyond `rmax` so the largest dropped contribution is `1e-14`.
- **The Lorch window cutoff is the transform range, not the neighbour cutoff.**
  They are the same number in TurboGAP and the coincidence does not survive the
  auto-extended `rcut`.
- **`x = q/2pi`, not `q`.** Form factors are sampled at `x/2 = sin(theta)/lambda`
  and wavevectors are `2 pi x`. A factor of `2 pi` here is the easiest mistake
  in the whole calculation.
- **The volume virial belongs to the cell, not to any pair**, so exactly one
  rank adds it. Every rank adding it gives an `nprocs`-fold stress that the
  1-rank-against-N-rank test is there to catch.
- **LAMMPS's once-per-timestep compute caching is wrong under MC.** An MC fix
  re-evaluates the energy several times within one timestep with different
  configurations, so a cached spectrum would price every trial move against the
  first one -- and the bias would simply never reach the Metropolis test, with
  no error anywhere. The pair pass is repeated instead.
- **A serial run still has ghosts.** The property adjoint reaches neighbours of
  `i`, so it accumulates into a buffer over `nlocal+nghost` and reverse-
  communicates. Guarding that on `nprocs > 1` leaves the *serial* force wrong by
  a constant, because periodic images are ghosts too. The h-scan showed it as a
  flat curve; no single tolerance would have.

## Tests

```sh
tests/run.sh                 # everything
tests/run.sh gradients mpi   # a subset
```

| leg | what it establishes |
|---|---|
| `forward` | partial `g_ab(r)` against an independent numpy reference -- periodic images, triclinic cell, nothing shared with the C++ |
| `gradients` | forces and virial against central differences of MADLIB's own energy. The criterion is the **h-scan**: the error must fall as `h^2`, not merely be small. A wrong constant factor gives a flat curve that any single tolerance would pass. |
| `invariants` | `E = gamma D / 2`; `sum_j f_j = 0` (a rigid translation cannot change any of these observables, so this catches a bad scatter or a bad reduction) |
| `mpi` | 1 rank against 4: energy, forces and virial. The only test that catches every rank adding the volume term. |
| `guards` | the three refusals above actually fire |
| `md` | a biased run closes the gap -- `rel_error` 0.99 to 0.04 |
| `mc` | the bias reaches an unmodified LAMMPS MC fix's acceptance |
| `xps` | the property layer end to end: invariants, the adjoint's reverse communication across ranks, and reverse Monte Carlo |
| `gap` | the soap_turbo backend against TurboGAP's own numbers, its derivatives by finite differences, and rank-invariance. Skips cleanly when built without the backend. |
| `ir` | the spectrum pipeline against TurboGAP's `ir_fft` over 29 configurations, a vibrational band against the force constant that produced it, the molecular dipole across ranks, and the periodic refusal |
| `mliap` | an ML-IAP model's per-atom values against `pair_style mliap`'s own, and its adjoint by finite differences |
| `torch` | a model written in Python against the C++ one it reimplements, and its autograd adjoint by finite differences |
| `quip` | the bias running on a real GAP potential through `pair_style quip`, cross-checked against `quippy`, and a biased run closing the gap on it. Skips cleanly without ML-QUIP or the potentials. |

The gradient legs report an **observed convergence order**, not a tolerance at
each step. A large `h` legitimately has a large truncation error, and demanding
that it be small would reward a gradient that is wrong in a way that happens to
cancel; what a correct gradient must do is converge at order ~2, or already sit
at the round-off floor.

Measured: forward `g(r)` to `5e-15` and the XPS spectrum to `3e-16`; observed
gradient order `2.02` on the soap_turbo chain; all four virial components to
`1e-9`-`7e-6`; 1-vs-4-rank energy to `1e-16` (`1e-14` through the descriptor);
91 binding energies against TurboGAP to `1.2e-9` relative; XPS reverse MC
`rel_error` 0.385 unbiased against 0.045 biased; the IR pipeline to `1.5e-13`
over 29 configurations, and an IR band at 1402 cm^-1 against an analytic 1407.8.

## Building

```sh
ln -s /path/to/madlib/src/MADLIB   /path/to/lammps/src/MADLIB
ln -s /path/to/madlib/lib/madlib   /path/to/lammps/lib/madlib
cp    /path/to/madlib/cmake-MADLIB.cmake \
      /path/to/lammps/cmake/Modules/Packages/MADLIB.cmake
```

Then two edits to `lammps/cmake/CMakeLists.txt`: add `MADLIB` to
`STANDARD_PACKAGES`, which compiles the sources, **and** to the first
`foreach(PKG_WITH_INCL ...)` list, without which `Packages/MADLIB.cmake` is
never read and the backend option is silently ignored.

```sh
cmake ../cmake -D PKG_MADLIB=on -D PKG_MC=on -D BUILD_MPI=on \
               -D MADLIB_ENABLE_TURBOGAP=on
```

### Verifying against a real potential

Every other leg runs on `pair_style zero`, which isolates the bias but says
nothing about whether it composes with real physics. The `quip` leg does that:
a fitted GAP under :doc:`pair_style quip`, with the XRD bias on top.

```sh
git clone --recursive https://github.com/libAtoms/QUIP.git
cd QUIP && meson setup builddir --default-library=static --buildtype=release -Dgap=true
meson compile -C builddir
```

Meson builds one archive per subdirectory, where LAMMPS wants a single
`libquip.a`; combine `libPotentials`, `libGAP`, `liblibAtoms`, `libUtils`,
`libfox` **and `libf90wrap_stub`** -- the last is easy to miss and shows up as
an undefined `f90wrap_abort_` at link time. Then

```sh
cmake ../cmake -D PKG_MADLIB=on -D PKG_ML-QUIP=on \
               -D QUIP_LIBRARY=/path/to/QUIP/builddir/quiplib/libquip.a
```

Measured on the 91-atom cell: energy and forces through LAMMPS agree with
`quippy` on the same frame to `9e-8` eV and `3.7e-8` eV/A, and a biased run
takes the relative error from 2.88 to 0.44.

### The Fortran backend

The Fortran backend is off by default. It vendors seven files from TurboGAP  -- 
`kinds`, the five `soap_turbo_*`, and `local_properties`, about 6k lines -- and
needs LAPACK as well as BLAS, because soap_turbo's radial basis orthonormalises
through `dgesvd` and `dpotrf`. `lib/madlib/Makefile` builds it standalone for a
conventional-make LAMMPS or for testing.

## Porting notes

`TURBOGAP-VS-LAMMPS.md` is the side-by-side: where the atoms live in each code,
what the neighbour list is and what shape it has, how the two decompose work
across ranks, and which of those differences forced a change here rather than a
transcription. `NOTES.md` records what the tests caught that reading did not.

## Provenance

The physics is TurboGAP's. `mad_scattering_data.h` is transcribed from
`src/exp_utils.f90`; the observable definitions, the Ashcroft-Langreth partial
structure factors and the output conventions follow `exp_interface.f90` and
`docs/gle_mad.md`. Where MADLIB and TurboGAP differ deliberately, `compat
turbogap` reproduces TurboGAP -- including its single-precision constants, which
is why the X-ray form factors match it to `4e-16` in that mode and differ at the
8th digit outside it.

## Test data

The frames and fitted potentials are not in this repository: they belong to the
`turbogap_tests` data repository, change on a different cadence, and are 24 MB.
`tests/run.sh` looks for them under `$MADLIB_DATA_ROOT` (default
`~/test/turbogap_tests`) and links them into `tests/data/gap_files`; legs that
need them skip when they are absent.
