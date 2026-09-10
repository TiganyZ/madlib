# M0 — LAMMPS facts, verified against the tree

LAMMPS `stable_22Jul2025_update6` at `/home/tigany/Documents/lammps/lammps`
(`src/version.h`: "22 Jul 2025", "Update 6").

Every item the plan marked *inferred* has now been checked in the source.

## 1. The MC claim holds, and upstream documents it

`MC/fix_gcmc.cpp:2442` calls `modify->post_force(vflag)`, then `:2450`
`c_pe->compute_scalar()`. The comment in between, `:2444-2447`, states the
contract we were relying on:

> all fixes with energy_global_flag set and which operate at pre_force() or
> post_force() and which user has enabled via fix_modify energy yes, will
> contribute to total MC energy via pe->compute_scalar()

`compute_pe.cpp:112` closes it: `if (fixflag && modify->n_energy_global)
scalar += modify->energy_global();`

**So experimentally biased MC needs no MC code at all** — `fix mad/*` with
`energy_global_flag = 1` and `fix_modify <id> energy yes` is priced by
`fix gcmc`, `fix atom/swap`, `fix widom` and `fix tfmc` automatically.

## 2. Large-cutoff neighbour list

`compute_rdf.cpp` is the model:

- `neighbor->add_request(this, NeighConst::REQ_OCCASIONAL)` (`:217`),
  `req->set_cutoff(cutoff_user + skin)` (`:222`), `neighbor->build_one(list)`
  (`:299`).
- It **errors**, not warns, when `cutoff_user + skin > cutghost`, naming
  `comm_modify cutoff` (`:188-190`). MADLIB copies that message.
- A custom cutoff is refused under `neighbor style multi` (`:219-221`).
  MADLIB inherits the restriction.

`REQ_FULL | REQ_OCCASIONAL` on a fix is established
(`EXTRA-FIX/fix_electron_stopping.cpp:135`, `REPLICA/fix_hyper_local.cpp:313`),
and `fix_electron_stopping` is nearly MADLIB's shape: a fix that adds a force in
`post_force` from its own list.

## 3. Correction — ghost forces in `post_force` are dropped

`verlet.cpp:341` runs `comm->reverse_comm()` **before** `:347`
`modify->post_force(vflag)`. A fix that writes into `f[j]` for a ghost `j`
during `post_force` therefore loses that contribution silently.

Two consequences, and they point opposite ways:

- **Pair route: use a full list and write only to local `i`.** With
  `REQ_FULL`, the ordered pair `(i,j)` is visited on the rank owning `i` and
  `(j,i)` on the rank owning `j`, so every atom's force is complete locally and
  **no reverse communication is needed at all**. This is exactly why TurboGAP
  writes forces to one end of the pair only, and it is the right choice here for
  a different reason. The cost is a 2x pair loop against a half list.
- **Property adjoint: reverse comm is unavoidable.** `dq_i/dr_j` lands on
  neighbours of `i`, including ghosts, and cannot be restructured away. The fix
  implements `pack_reverse_comm`/`unpack_reverse_comm` with `comm_reverse = 3`
  (`fix.h:215-217`) and calls `comm->reverse_comm(this)` itself.

## 4. MLIAP is drivable from outside `PairMLIAP`

`MLIAPData`'s constructor takes `PairMLIAP *pairmliap_in` **last and
defaultable**, and `ML-IAP/compute_mliap.cpp:142` already omits it:

```cpp
data = new MLIAPData(lmp, gradgradflag, map, model, descriptor);
```

The plan's top MLIAP risk is therefore closed. `ComputeMLIAP` is the template
for `compute mad/property/mliap`.

## 5. Toolchain

`cmake 3.31.11`, `g++ 15.3.1`, `gfortran`, OpenMPI 5.0.8, numpy 2.3.5,
torch 2.11.0. No `fftw-devel` — IR (M5) uses an in-package radix-2 transform,
as TurboGAP's `ir_fft.f90` does, so there is no new dependency.

---

# M1 — findings from building it

Three things the plan did not anticipate, all found by the tests rather than by
reading.

## 6. The hard cutoff wrecks the virial, not `g(r)`

Cutting the pair sum at `rcut` when `rcut == rmax` means a pair crossing the
cutoff removes a Gaussian that was still contributing to the last bins. The
pattern barely moves, and the forces still finite-difference correctly, but the
**diagonal** virial came out wrong by a factor of order one while the shear
stayed exact — and the finite-difference value jumped around with `h`, which is
the only visible symptom.

Placing the neighbour cutoff `trunc*sigma` beyond `rmax` makes the largest
dropped contribution `exp(-trunc^2/2) = 1e-14` and the discontinuity vanishes.
This is now the default, and it is strictly better than TurboGAP's behaviour,
whose `pdf_virial` test had to drop to `--strain 1e-5` to work around the same
effect.

## 7. The Lorch window cutoff is not the neighbour cutoff

Once `rcut` is auto-extended, using it for the Lorch window silently changes the
observable — the window's job is to damp the ringing from truncating the
*transform*, so its cutoff is `rmax`. The two are the same number in TurboGAP
and the coincidence does not survive. Caught by comparing against TurboGAP's
expected pattern: the discrepancy went from 4% to under 1% on fixing it.

## 8. LAMMPS's once-per-timestep compute cache is wrong under MC

The standard idiom

```cpp
if (cmp->invoked_array != update->ntimestep) cmp->compute_array();
```

is correct for several consumers in one timestep and **wrong** for an MC fix,
which calls `modify->post_force()` repeatedly within a single timestep with
different configurations. With the cache in place every trial move was priced
against the spectrum of the first one, so the bias never reached the Metropolis
test — with no error, no warning, and a plausible trajectory.

The symptom was `fix atom/swap` accepting 1200 of 1200 moves whether the bias
was enabled or not. Recomputing unconditionally fixed it: 335 of 1200, and the
relative error fell to 0.109 against 0.371 unbiased.

The cost is one extra pair pass when several fixes share one `compute mad/pdf`.
An explicit invalidation protocol would avoid it and is the obvious later
optimisation; correctness first.

## Volume-term derivation, corrected

The plan asserted that in `S_ab - delta_ab` the `V` in `g_ab` cancels the `1/V`
in `rho` so that only the subtracted `-1` survives. That is true of the *total*,
but it is not how the code decomposes: `compute mad/pdf` already contributes the
whole `g ∝ V` path, so the fix must contribute the `rho = N/V` prefactor path,
which is

    V d(S_ab)/dV |_g = -(S_ab - delta_ab)   ->   virial += sum (dE/dS_ab)(S_ab - delta_ab)

Getting this wrong left all three diagonals off by the same constant while the
shear was exact — the signature of a pure volume term, and worth remembering as
a diagnostic.

---

# M2 — the property layer and XPS

## 9. A serial run still has ghosts, and guarding the reverse comm on `nprocs` is wrong

`ComputeMADProperty::scatter_adjoint()` accumulates `dq_i/dr_j` into a buffer
covering `nlocal+nghost` and reverse-communicates it, because Verlet
reverse-communicates before `post_force()` and a force written to a ghost there
is lost. The obvious optimisation

```cpp
if (comm->nprocs > 1) comm->reverse_comm(this);   // WRONG
```

is wrong: **a single rank with periodic boundaries still has ghost atoms**, and
their share of the adjoint has to be folded back onto the atoms that own them.
`comm->reverse_comm()` does that on one rank too.

The signature was distinctive and worth remembering: the **virial was exactly
right in all four components** while the **forces were wrong by a constant**,
giving a finite-difference h-scan that was flat at ratio 1.00 instead of falling
as h^2. The virial is formed inside the pair loop from `fpair` and the
separation, so it never touches the scatter; only the forces do. Virial correct
plus forces wrong by a constant therefore points at the scatter and nothing
else.

This is exactly the failure the h-scan exists to catch, and precisely what a
single tolerance would have passed: the error was 0.348 at both step sizes.

## 10. Output precision has to be better than the thing being checked

`fix mad/*`'s spectrum file was written `%20.10f`, which put a 5e-11 floor under
the reference comparison and made a machine-precision agreement look like a
5e-11 disagreement. TurboGAP has the same problem, and its own test notes it:
`xrd_prediction.dat` is `F20.8`, "so 5e-9 per sample is the floor and nothing
here can be checked tighter than that". Written `%24.16e`, the XPS spectrum
agrees with an independent reference to **3e-16**.

## 11. `fix tfmc` cannot test the energy path

It is a force-bias method: it moves atoms from forces and never forms a
Metropolis ratio on the energy, so `fix_modify energy no` changes nothing about
it. `fix gcmc` with `nexchanges = 0` and translation moves is the right stock
fix for a reverse-Monte-Carlo demonstration, and `fix atom/swap` for a
composition one -- but only if the property actually depends on the types being
swapped, which a plain coordination number does not.

The precise test of the mechanism is simpler than either: `compute pe` reads
1.707 with `fix_modify mad energy yes` and 0 with `no`. Every MC fix prices a
trial move with `compute pe`, so that single comparison is the claim.

---

# M3 — the soap_turbo backend

The plan's estimate held up: the vendored set is exactly the seven files its
dependency closure predicted (`kinds`, five `soap_turbo_*`, `local_properties`,
5986 lines), `read_files.f90` was correctly left out in favour of a 250-line C++
parser, and the two Fortran entry points needed nothing wrapped around them.
Four things it got wrong or did not know.

## 12. LAPACK, not just BLAS

soap_turbo's radial basis orthonormalises through `dgesvd`, `dpotrf` and
`dpotri`, so the backend needs LAPACK. The plan said "needing only BLAS".

## 13. Being in `STANDARD_PACKAGES` does not get `Packages/<PKG>.cmake` read

LAMMPS compiles a package's sources from a glob driven by `STANDARD_PACKAGES`,
but only *includes* `cmake/Modules/Packages/<PKG>.cmake` for packages named in
one of two explicit `foreach(PKG_WITH_INCL ...)` lists
(`cmake/CMakeLists.txt:628`, `:765`). A package with an optional external
library has to be in both lists, and the failure mode is quiet: the option is
accepted, cached, and ignored, and the link fails with undefined references to
the library that was never built.

Consequence for the file itself: since the glob already adds the sources,
`MADLIB.cmake` must **not** add them again.

## 14. `.gap` paths are relative to where TurboGAP was run

`gap_files/CO.gap` contains `local_property_qs = 'gap_files/core_electron_be...'`
-- relative to the directory TurboGAP is launched from, not to the potential
file. A LAMMPS run has no reason to share that directory, so the parser tries
the path as given, then beside the `.gap` file, then flattened beside it.

## 15. `make` predefines `FC`

`FC ?= gfortran` in a Makefile does nothing, because `make` has already set
`FC = f77`. Plain `FC = gfortran` still lets a command-line `make FC=ifx` win.

## The agreement, and what limits it

91 core-electron binding energies against TurboGAP's own values for the same
frame: **max 3.4e-7 eV, 1.2e-9 relative**, median 8.9e-8 eV. Not bit-identical,
and the residual is not the sparse-set files or the hyperparameters -- a
standalone driver with a brute-force image loop gets 4.4e-8 against the same
reference. What is left is summation order: LAMMPS's neighbour list presents a
different pair ordering, and that propagates through an SVD-orthonormalised
radial basis. It is eight orders below the accuracy of any CEBE model, but it
is worth stating as "agrees to 1e-9 relative" rather than "reproduces".

## The finite-difference criterion had to change

The soap_turbo chain reaches its round-off floor by `h = 1e-4`, where the error
*grows* as `h` shrinks. Scanning down from `3e-3` shows the h^2 regime cleanly
(ratios 9.04 and 11.51, **observed order 2.02**), but the old pass criterion --
every step below one tolerance -- failed on the largest `h`, whose truncation
error is legitimately large.

The criterion is now the shape of the scan: the smallest `h` must be below
tolerance, and the observed order must exceed 1.5 or the error must already be
at the floor. Demanding a small error at large `h` would reward a gradient that
is wrong in a way that happens to cancel, which is the opposite of what the scan
is for.

---

# M5 — IR prediction

The pipeline port went as planned and inherits an external reference: MADLIB
agrees with TurboGAP's `ir_fft.f90` to **1.5e-13 over 29 configurations**, and
that file agrees with GPUMD's `TNEP/spectroscopy.py` to 3.5e-14 over eleven. An
estimator written from the definition alone would have had neither. Two things
the plan did not anticipate.

## 16. Accumulating a trajectory is a fix's job, not a compute's

The plan said `compute mad/ir`. A compute is only evaluated when something asks
for it, and this has to see every stride-th step whether anything asks or not.
`fix ave/correlate` is a fix for the same reason. It is `fix mad/ir`.

## 17. The molecular dipole was silently wrong under MPI

`mu_i = q_i r_i` is not defined in a periodic box, so the periodic path builds
molecular dipoles from intramolecular vectors, measured from the lowest-tagged
atom of each molecule. Choosing that anchor among the rank's **local** atoms
makes it whichever fragment of the molecule the rank happens to hold, so a
molecule split across two ranks is measured from two different origins and its
dipole comes out as zero.

Measured on 24 dimers in a periodic box, 8 of them straddling: **-13.56 on one
rank, 0.00 on four.** No error, no warning.

Choosing the anchor over local **and ghost** atoms fixes it, because a bonded
molecule is always inside the communication cutoff, so every rank holding part
of a molecule can see all of it and picks the same atom. Verified at 1, 2 and 4
ranks with 8 molecules straddling: identical to the last digit.

Two things about finding this are worth keeping. The first test looked like it
passed: it used `boundary s s s`, which is not periodic, so it exercised the
direct sum and never entered the molecular branch at all — a test can straddle
molecules across ranks and still tell you nothing about the code you meant to
test. The second is that the per-atom dump is what settled it: the values were
exactly `q_i z_i`, which identified the branch immediately, where the totals
alone had been agreeing for the wrong reason.
