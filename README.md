# MADlib

![Madlib](./images/madlib_logo.png)

Not the legendary rap producer, but an efficient Molecular Augmented Dynamics
library for experimental atomistic structure inference for use with LAMMPS.

This library allows you to perform experimental structure inference (optimizing
a structure according to experimental data and its potential energy) primarily
through a Molecular Dynamics driven optimization, Molecular Augmented Dynamics
(MAD), which far outstrips other methods for such inference (such as Hybrid
Reverse Monte-Carlo) in terms of scalability and speed. The underlying
interatomic potential can by any LAMMPS potential. 

MADlib predicts the observable through TurboGAP — a diffraction pattern, a pair
distribution function, an XPS or NMR spectrum — from the given atom positions
at that step, compares it with the input experimental data, and optimizes the
potential energy and experimental agreement. 

---

## Theory

An ordinary MD run follows the forces of a potential. MADlib adds one more
term to the energy: a penalty for disagreeing with the experiment.

$$
V_\text{tot} = V_\text{pot} + \frac{\gamma}{2} \sum_i w_i
\left( y_i(\mathbf{r}) - y^\text{exp}_i \right)^2
$$

- $y_i(\mathbf{r})$ — the observable **predicted** from the current positions, on discretized point $i$
- $y^\text{exp}_i$ — your **measured** value at the same point
- $w_i$ — weights if specified for the particular experimental point (so one can focus on optimizing particular regions of a spectrum) 
- $\gamma$ — how important the experimental agreement is in comparison to the interatomic potential. 

Because $y_i(\mathbf{r})$ is differentiable, the penalty has a force:

$$
\mathbf{F}_j = -\nabla_j V_\text{pot} - \gamma \sum_i w_i \left( y_i - y^\text{exp}_i \right)
\frac{\partial y_i}{\partial \mathbf{r}_j}
$$

The second term allows for gradient-based optimization of the structure with
respect to experimental data. Running MD/MC (at high temperature) will then provide a
multi-objective optimization of both the potential energy and experimental
agreement.  The potential keeps the structure physical; the
experimental data funnels the structure search to areas of experimental agreement. 

Ran as MD and it is *molecular augmented dynamics*; ran using the Energy+Exp disagreement and it is Hybrid Reverse Monte Carlo (HRMC).

When the prediction is in arbitrary units, `fit yes` fits a scale $s$ and
baseline $b$ by weighted least squares and compares $s y_i + b$ instead. Both
sit at their own optimum, so the force picks up no extra term.

### The Dissimilarity

$$ D = \sum_i w_i \left( y_i - y^{\text{exp}}_i \right)^2 \qquad\qquad \text{rel error} = \sqrt{\frac{D}{\sum_i w_i \left(y^{\text{exp}}_i\right)^2}} $$

$D$ is the **dissimilarity** and $E_\text{bias} = \tfrac{1}{2}\gamma D$.
`rel_error` is dimensionless — 0 is a perfect fit — and is the number to
follow during a run.

### Choosing the energy scale for experimental agreement ($\gamma$)

1. Run once with `gamma 0.0`. Nothing is biased; read $D$ from `f_mad[3]`.
2. Pick $\gamma$ so that $\tfrac{1}{2}\gamma D$ is a **small fraction of the
   cohesive energy** of your system.
3. Give `gamma` two values (the starting value and the ending value of the run) to ramp it linearly over the run, so the restraint
   switches on without kicking the structure at step 0.

> A $\gamma$ too large by orders of magnitude pulls the structure apart *while
> the reported mismatch falls* (the potential imparts nothing in terms of the optimization).

---

## What's in it

**Observables** — each is a `fix`, and each both predicts and biases:

| fix | fits |
|---|---|
| `mad/pdf` | pair distribution function $g(r)$, or $D(r)$ |
| `mad/sq` | structure factor $S(q)$ |
| `mad/xrd` | X-ray or neutron diffraction |
| `mad/xps` | XPS, or any spectrum made by broadening a per-atom scalar |
| `mad/nmr` | NMR, per nucleus |
| `mad/ir` | IR, from the dipole autocorrelation of a running trajectory |

**Per-atom property backends** — what `mad/xps`, `mad/nmr` and `mad/ir` read
their input from. Any of these, through one interface:

| compute | source |
|---|---|
| `mad/property/turbogap` | a TurboGAP GPR model in a `.gap` file |
| `mad/property/python` | any Python or torch model (MACE, GRACE, ...) |
| `mad/property/mliap` | a LAMMPS ML-IAP model |
| `mad/property/dipole` | fixed or per-atom charges |
| `mad/property/coord` | coordination number |
| `mad/property/table` | values from a file (prediction only) |

**Pair styles** — MADlib is potential-agnostic, so use whatever LAMMPS has. It
also adds one of its own:

| pair style | what it is |
|---|---|
| `turbogap` | a whole [TurboGAP](https://turbogap.fi) potential, `.gap` file as it stands |
| anything in LAMMPS | `symmetrix/mace`, `quip`, `pace`, `eam`, ... |
| `zero` | no potential at all — classical RMC, the data on its own |

Also: structure search with a genetic algorithm that uses MAD as its local
optimiser — see [`python/`](python/README.md).

---

## Invoking the pair styles

A TurboGAP potential, with all four of its terms (soap_turbo, distance_2b,
angle_3b, core_pot):

```lammps
pair_style  turbogap e0 C -0.16138053 e0 O 0.0
pair_coeff  * * gap_files/CO.gap C O
```

`e0 <species> <value>` is a per-atom energy offset. TS/MBD dispersion is not
included, and a `.gap` file with `dipole_model = .true.` is refused.

A MACE foundation model:

```lammps
atom_modify map yes      # symmetrix/mace needs this
newton      on
pair_style  symmetrix/mace
pair_coeff  * * mace-mpa-0-medium.model-lammps.pt C O
```

No potential at all, for classical reverse Monte Carlo:

```lammps
pair_style  zero 7.5
pair_coeff  * *
```

---

## A full example

A melt–quench of carbon against a measured X-ray pattern, with the bias ramped
in during the quench. This is a complete, runnable input.

```lammps
# ---------------------------------------------------------------- system
units           metal
atom_style      atomic
boundary        p p p
atom_modify     map yes

read_data       start.data
mass            * 12.011

# compute mad/pdf extends its neighbour cutoff to rmax + 8*kde. The ghost
# cutoff must cover that plus the skin -- and stay under half the box, or the
# pair distribution counts the cell's own periodic images.
comm_modify     cutoff 9.3
neighbor        0.5 bin

# ---------------------------------------------------------------- potential
pair_style      turbogap e0 C -0.16138053
pair_coeff      * * gap_files/carbon.gap C

# The potential alone. fix mad/* sets thermo_energy on by default, so `pe`
# carries the bias as well; c_pepair is the unbiased energy.
compute         pepair all pe pair

# ---------------------------------------------------------------- observable
# Differentiable partial pair distributions, which the diffraction fix reads.
compute         gr all mad/pdf r 0.1 8.0 201 kde 0.1

# ---------------------------------------------------------------- melt
# Unbiased: the liquid should be the potential's own, not the target's.
fix             mad all mad/xrd c_gr target measured.fq &
                q 1.05 9.97 201 output qFq species C weighting xray gamma 0.0

thermo_style    custom step temp c_pepair f_mad f_mad[2] f_mad[3] press
thermo_modify   format float %.8g flush yes
thermo          200

timestep        0.001
velocity        all create 5000.0 12345 mom yes rot yes dist gaussian
fix             nve all nve
fix             thm all langevin 5000.0 5000.0 0.1 12345
run             2000
unfix           thm
unfix           mad

# ---------------------------------------------------------------- quench
# gamma ramps 0 -> 30 while the structure is still mobile, so the experiment
# steers it rather than fighting an already-frozen cell.
fix             mad all mad/xrd c_gr target measured.fq &
                q 1.05 9.97 201 output qFq species C weighting xray &
                gamma 0.0 30.0 file pattern.dat write_every 1000

fix             thm all langevin 5000.0 300.0 0.1 12345
run             10000
unfix           thm
unfix           nve

min_style       fire
minimize        1e-8 1e-4 2000 20000

write_data      final.data
```

Output of a `fix mad/*`:

| | |
|---|---|
| `f_mad` | bias energy $\tfrac{1}{2}\gamma D$ (in `pe`; `fix_modify mad energy no` to exclude) |
| `f_mad[2]` | **rel_error** |
| `f_mad[3]` | dissimilarity $D$ |
| `f_mad[4]`, `[5]`, `[6]` | current $\gamma$, fitted scale, fitted offset |

### Reverse Monte Carlo

The same fix next to a LAMMPS MC fix. The bias is in `pe`, so it enters the
Metropolis test with no extra settings:

```lammps
fix mad  all mad/xrd c_gr target measured.fq q 1.05 9.97 201 species C O gamma 5.0
fix swap all atom/swap 5 20 29494 300.0 ke no types 1 2
```

Displacement-only RMC is `fix gcmc` with zero exchanges:

```lammps
fix_modify mad energy yes
fix mc all gcmc 1 0 50 1 29494 1000.0 0.0 0.1 full_energy
```

### Keywords shared by every `fix mad/*`

`target <file>` (two columns, no header) · `gamma <g>` or `gamma <beg> <end>` ·
`every <N>` · `fit <scale> <offset>` · `file <out>` · `write_every <N>` ·
`compat turbogap`

Per-style keywords are in [`doc/src/fix_mad.rst`](doc/src/fix_mad.rst),
[`doc/src/fix_mad_ir.rst`](doc/src/fix_mad_ir.rst),
[`doc/src/compute_mad_pdf.rst`](doc/src/compute_mad_pdf.rst) and
[`doc/src/compute_mad_property.rst`](doc/src/compute_mad_property.rst).

---

## Install

### The easy way

One script builds LAMMPS, MADlib and every potential backend. Stages are
independent and each is skipped when its output already exists, so a re-run
after a failure resumes rather than restarts.

```sh
git clone https://github.com/TiganyZ/madlib
cd madlib
PREFIX=/scratch/$USER/madlib ./tools/install_turbogap_lammps.sh
```

Stages: `deps python lammps quip symmetrix madlib build test`. Name the ones
you want to run only those — `./tools/install_turbogap_lammps.sh madlib build`
rebuilds after a source change, and skipping `quip symmetrix` cuts most of the
build time if you do not need them.

> **Put `PREFIX` on scratch, not in your home directory.** The build is tens of
> gigabytes, but the binding limit is usually the *file count*: a venv with
> torch in it is a few hundred thousand small files, and an inode quota is
> reported as "disk quota exceeded" with plenty of space free.

### Into a LAMMPS tree you already have

```sh
ln -s /path/to/madlib/src/MADLIB            /path/to/lammps/src/MADLIB
ln -s /path/to/madlib/lib/madlib            /path/to/lammps/lib/madlib
cp /path/to/madlib/cmake-MADLIB.cmake \
   /path/to/lammps/cmake/Modules/Packages/MADLIB.cmake
python /path/to/madlib/tools/register_madlib.py /path/to/lammps

cd /path/to/lammps && mkdir -p build && cd build
cmake ../cmake -D PKG_MADLIB=on -D MADLIB_ENABLE_TURBOGAP=on \
               -D PKG_MC=on -D BUILD_MPI=on
make -j8
```

- `MADLIB_ENABLE_TURBOGAP=on` builds the Fortran soap_turbo backend, which
  `pair_style turbogap` and `mad/property/turbogap` need. It requires LAPACK.
- `PKG_MC=on` gives you `fix gcmc` and `fix atom/swap` for the Monte Carlo
  form. MADlib has no MC code of its own.
- `MADLIB_ENABLE_PYTHON=on` enables `mad/property/python`.

For SOAP on a GPU, add `-D MADLIB_ENABLE_GPU=on
-D MADLIB_TURBOGAP_SRC=/path/to/turbogap/src -D MADLIB_GPU_ARCH=<sm>
-D CUDAToolkit_ROOT=/path/to/cuda`.

Check it worked:

```sh
./lmp -h | grep -oE "mad/[a-z/]+|turbogap|gcmc" | sort -u
```

### TL;DR — copy this and change the first three lines

```sh
#!/bin/bash
set -euo pipefail

PREFIX=/scratch/$USER/madlib     # somewhere with no inode quota
JOBS=8
LAMMPS_BRANCH=develop

mkdir -p "$PREFIX" && cd "$PREFIX"
git clone --depth 1 -b "$LAMMPS_BRANCH" https://github.com/lammps/lammps lammps
git clone --depth 1 https://github.com/TiganyZ/madlib madlib

ln -sfn "$PREFIX/madlib/src/MADLIB" "$PREFIX/lammps/src/MADLIB"
ln -sfn "$PREFIX/madlib/lib/madlib" "$PREFIX/lammps/lib/madlib"
cp "$PREFIX/madlib/cmake-MADLIB.cmake" \
   "$PREFIX/lammps/cmake/Modules/Packages/MADLIB.cmake"
python3 "$PREFIX/madlib/tools/register_madlib.py" "$PREFIX/lammps"

mkdir -p "$PREFIX/lammps/build" && cd "$PREFIX/lammps/build"
cmake ../cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_MPI=on \
    -D PKG_MADLIB=on -D MADLIB_ENABLE_TURBOGAP=on \
    -D PKG_MC=on -D PKG_MANYBODY=on -D PKG_MOLECULE=on -D PKG_EXTRA-FIX=on
make -j"$JOBS"

./lmp -h | grep -oE "mad/[a-z/]+|turbogap|gcmc" | sort -u
echo "binary: $PREFIX/lammps/build/lmp"
```

---

## Examples

| | |
|---|---|
| [`examples/mace_foundation`](examples/mace_foundation/README.md) | a MACE foundation model under an experimental bias — the smallest useful MAD run with a machine-learned potential |
| [`examples/method_comparison`](examples/method_comparison/README.md) | MAD, HRMC and a genetic algorithm on one target at a matched budget of potential evaluations |

## Tests

```sh
tests/run.sh                  # everything
tests/run.sh gradients mpi    # a subset
```

Forces and virials are checked by finite differences against independent
references, and a run on 1 rank is checked against a run on 4. Tests needing
the TurboGAP test data look under `$MADLIB_DATA_ROOT` and skip when it is
absent.

## Adding a property model

Implement one method. It returns the forces of a weighted per-atom property,
$\sum_i \lambda_i q_i$:

```cpp
virtual void adjoint(double **lambda, double *virial);
```

The observables are ported from [TurboGAP](https://turbogap.fi).
