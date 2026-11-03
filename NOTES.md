# M0 -- LAMMPS facts, verified against the tree

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

**So experimentally biased MC needs no MC code at all** -- `fix mad/*` with
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

## 3. Correction -- ghost forces in `post_force` are dropped

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
torch 2.11.0. No `fftw-devel` -- IR (M5) uses an in-package radix-2 transform,
as TurboGAP's `ir_fft.f90` does, so there is no new dependency.

---

# M1 -- findings from building it

Three things the plan did not anticipate, all found by the tests rather than by
reading.

## 6. The hard cutoff wrecks the virial, not `g(r)`

Cutting the pair sum at `rcut` when `rcut == rmax` means a pair crossing the
cutoff removes a Gaussian that was still contributing to the last bins. The
pattern barely moves, and the forces still finite-difference correctly, but the
**diagonal** virial came out wrong by a factor of order one while the shear
stayed exact -- and the finite-difference value jumped around with `h`, which is
the only visible symptom.

Placing the neighbour cutoff `trunc*sigma` beyond `rmax` makes the largest
dropped contribution `exp(-trunc^2/2) = 1e-14` and the discontinuity vanishes.
This is now the default, and it is strictly better than TurboGAP's behaviour,
whose `pdf_virial` test had to drop to `--strain 1e-5` to work around the same
effect.

## 7. The Lorch window cutoff is not the neighbour cutoff

Once `rcut` is auto-extended, using it for the Lorch window silently changes the
observable -- the window's job is to damp the ringing from truncating the
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
test -- with no error, no warning, and a plausible trajectory.

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
whole `g  proportional to  V` path, so the fix must contribute the `rho = N/V` prefactor path,
which is

    V d(S_ab)/dV |_g = -(S_ab - delta_ab)   ->   virial += sum (dE/dS_ab)(S_ab - delta_ab)

Getting this wrong left all three diagonals off by the same constant while the
shear was exact -- the signature of a pure volume term, and worth remembering as
a diagnostic.

---

# M2 -- the property layer and XPS

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

# M3 -- the soap_turbo backend

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

# M5 -- IR prediction

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
direct sum and never entered the molecular branch at all -- a test can straddle
molecules across ranks and still tell you nothing about the code you meant to
test. The second is that the per-atom dump is what settled it: the values were
exactly `q_i z_i`, which identified the branch immediately, where the totals
alone had been agreeing for the wrong reason.

## 18. A shared scratch file made a correct gradient look broken

`fd_gradient.py` wrote its perturbed configuration to `data/_fd.data`, beside
the input. Running two suites at once -- which is easy to do by accident with
background jobs -- means one overwrites the other's geometry between the write
and the read, so the finite difference is taken between two *different*
structures.

What that looks like is the worst possible thing: the analytic forces and
virial print their correct values, unchanged, next to finite differences that
are nonsense and inconsistent between step sizes. It reads exactly like a
gradient bug, and the h-scan -- the very thing added to make gradient errors
unambiguous -- reports a negative convergence order, which is a real signature of
a real problem.

Two changes: the perturbed file now lives in the run's own `mkdtemp` directory,
and `run.sh` takes a lock and refuses a second concurrent invocation with a
message. The second matters more than the first. Other legs share scratch names
too, and a suite that silently produces wrong numbers when run twice is worse
than one that refuses.

---

# Verification against a real potential

Every test up to here ran on `pair_style zero`, which isolates the bias but says
nothing about whether it composes with real physics. QUIP supplies that: a
fitted GAP under `pair_style quip`, with the MADLIB bias on top.

## 19. Assembling libquip.a from a meson build

QUIP's meson build produces one archive per subdirectory; LAMMPS's ML-QUIP
wants a single `libquip.a`. Combining `libPotentials`, `libGAP`, `liblibAtoms`,
`libUtils` and `libfox` is the obvious set and links almost everything --
leaving one undefined symbol, `f90wrap_abort_`, which lives in
`libf90wrap_stub.a`. That archive exists for the Python wrapper and looks
irrelevant to a C++ link, which is exactly why it gets left out.

## What it establishes

Energy and forces through `pair_style quip` agree with `quippy` evaluating the
same potential on the same frame to **9e-8 eV and 3.7e-8 eV/A** -- two
independent routes into the same GAP.

A biased run on that potential takes the relative error from **2.88 to 0.44**.
The physical potential energy rises while it does, which is what a restraint
should do: the structure is pulled to a configuration the potential likes less
and the data likes more. The temperature rises too, because the bias does work
on the system and the thermostat has to remove it.

## The test data does not belong here

The fitted potentials are 24 MB and were briefly committed. They belong to the
`turbogap_tests` repository and change on a different cadence, so `run.sh` now
looks for them under `$MADLIB_DATA_ROOT` and skips the legs that need them when
they are absent -- the same arrangement TurboGAP's own suites use.

---

# M4 -- the MLIAP and Python backends

Both confirm the design claim the property layer was built on: what a bias needs
is never the full Jacobian, only its contraction with a per-atom weight, and
that is the force a model already knows how to compute. Neither backend
differentiates anything by hand.

## 20. ML-IAP needs three init calls, and two of them look optional

`ComputeMLIAP` calls `model->init()`, `descriptor->init()` and `data->init()`.
Calling only the last -- which is the one with the object's name on it -- leaves
the SNAP tables unbuilt, and every descriptor comes back zero.

The symptom is worth remembering because it does not look like a missing call:
the per-atom value came out as exactly the model's constant term for every atom.
That reads as a model that simply does not vary with structure, which is a
plausible thing for a badly fitted model to do.

## 21. ML-IAP descriptors write to atom->f directly, not to data->f

`MLIAPData` has an `f` member, and `generate_neighdata` sets it to `atom->f`, so
redirecting it into a private buffer looks like the way to keep this
contribution separable. It is not: `MLIAPDescriptorSNAP::compute_forces` opens
with `double **f = atom->f;` and never consults `data->f` at all.

Separability is not optional here -- the descriptor writes to ghosts, and Verlet
reverse-communicates before `post_force()`, so those contributions would be lost.
The fix is to point `atom->f` itself at the buffer for the duration of the call
and restore it afterwards, which `MGPT/pair_mgpt.cpp:1749` already does.

Without it the analytic virial came out as exactly zero next to a finite
difference of 632, which at least fails loudly.

## 22. The per-atom value is only written when asked for

`MLIAPModelLinear::compute_gradients` fills `betas` unconditionally but guards
`eatoms` with `if (data->eflag)`. A property backend wants the value as well as
its derivative, so `generate_neighdata(list, 1, 0)`: eflag on, vflag off -- the
latter because the virial would otherwise be tallied through a `PairMLIAP` that
this compute deliberately does not have.

## What the two backends are validated against

The Python backend is checked against a torch reimplementation of the C++
coordination number. That choice is the point: a model nobody can check
independently would make the test a test of itself. Values agree to 1e-14 and
the autograd adjoint finite-differences at order 2.00.

The ML-IAP backend is checked against `pair_style mliap`'s own per-atom
energies for the same model, which agree **exactly**, and its adjoint
finite-differences at order 2.00 with all four virial components correct.

## 23. A mixed CUDA toolkit fails as an integer divide-by-zero

On `alt` the GPU build linked cleanly and every style was present, then every
run died at startup:

```
Signal: Floating point exception (8), Integer divide-by-zero
NeighBondKokkos<Kokkos::OpenMP>::NeighBondKokkos
  -> UnorderedMap<int,int,Kokkos::Cuda>::UnorderedMap
    -> deep_copy -> contiguous_fill_or_memset -> parallel_for
```

Nothing in MADLIB is on that stack; `pair_style lj/cut` with `-k on g 1`
reproduces it. The faulting instruction is `div %r12d` in
`Kokkos_Cuda_Parallel_Range.hpp:103`, the `(nwork + block_size - 1) / block_size`
that sizes the grid, with `block_size == 0` from `cuda_deduce_block_size`.

That function returns 0 only when no block size gives one resident block, so the
device properties it reads must be wrong. From the core:

```
maxThreadsPerBlock       1024        (right)
maxThreadsPerMultiProc   1           (should be 1536)
maxBlocksPerMultiProc    1           (should be 16)
sharedMemPerMultiproc    4294968832  = 2^32 + 1536
```

The 1536 sitting four bytes above where it belongs is the whole diagnosis: the
fields are shifted. `cudaGetDeviceProperties` filled the struct in one CUDA
version's layout and Kokkos read it in another's. `ldd` confirmed it --
`libcudart.so.12` under a binary whose device code nvcc 13.4 compiled, because
`FindCUDAToolkit` took its headers from `/usr/local/cuda-13.4` and its library
from the distro's `/usr/lib/x86_64-linux-gnu`.

Two things make this worth writing down. It is silent: the build succeeds, the
styles are all there, and the crash names Kokkos rather than the toolkit. And
`CUDAToolkit_ROOT` alone does not fix an existing build directory, because a
cached `CUDA_cudart_LIBRARY` outranks it -- the installer now drops the cache
when it names a different CUDA, and refuses to finish if the linked runtime's
major version differs from nvcc's.

## 24. `vflag_global` is not set when LAMMPS asks for f.r

`pair_turbogap` takes its virial from `virial_fdotr_compute()` rather than from
TurboGAP's own accumulator, guarded the usual way:

```cpp
if (vflag_fdotr) virial_fdotr_compute();
```

Under `-k on g 1 -sf kk` the pressure came out as exactly zero, every component,
while the energy agreed to all printed digits and the forces to 1.7e-09. The
KOKKOS package selects `newton off`, and `verlet.cpp:68` then sets
`virial_style = VIRIAL_PAIR` instead of `VIRIAL_FDOTR`, so `vflag_fdotr` is 0
and a style with no per-pair tally contributes nothing. Nothing about it is
specific to Kokkos: `newton off` on one CPU rank reproduces it.

Widening the guard to `vflag_global` moved the hole rather than closing it,
because `pair.cpp:998` reads

```cpp
vflag_global = vflag & VIRIAL_PAIR;
if (vflag & VIRIAL_FDOTR && no_virial_fdotr_compute == 1) vflag_global = 1;
```

so with newton **on** `vflag_global` is 0 and `vflag_fdotr` is 1 -- the two
modes arrive through different flags, and testing either one alone is wrong
half the time. Setting `no_virial_fdotr_compute = 1` in the constructor is the
way to say "this style computes its own global virial": `vflag_global` is then
set in both modes and one guard covers them.

The `newton` leg compares energy and all six virial components between the two
modes and fails if they are zero in both, since zero agrees with zero.

## 25. A dipole model reads as a potential, and gives a number

TurboGAP predicts a dipole from a *model*, never from charges. The model is a
`soap_turbo` block flagged `dipole_model = .true.` whose local energy is a
fictitious scalar fitted so that its gradient with respect to the central
atom's own position is the local dipole:

```
mu_i = dE_i/dr_i = zeta delta^2 sum_s alpha_s (q_i.Q_s)^(zeta-1) d(q_i)/d(r_i)
```

which is `get_soap_dipole` in `gap.f90`: the energy kernel's own contraction,
taken at the self pair rather than scattered over the neighbours, with
`+zeta*delta^2` instead of `-zeta*delta^2` because a dipole is a gradient where
a force is its negative. TurboGAP guards it carefully -- `gap_interface.f90`
zeroes the virial and skips `get_soap_energy_and_forces` entirely for such a
block, because "a dipole model's energy is a fitting artefact, not a physical
energy".

MADLIB's `compute mad/property/dipole` is `mu_i = q_i r_i` from partial charges.
That is a different quantity, not an approximation to the above, and the docs
had not said so.

The dangerous half is the parser. A dipole block carries the same
hyperparameters, the same `desc_sparse` and the same `alphas_sparse` as an
energy block; the flag is the only thing that distinguishes them, and
`read_potential` ignored unknown keys. So

```lammps
pair_coeff * * water_dipole.gap O H
```

ran, and reported `PotEng = 2.6949838` on a six-atom water frame -- an entirely
ordinary-looking energy, and meaningless. The flag is now read and both entry
points refuse the file by name: the potential reader because there is no energy
in it, and the local-property reader because a dipole is a different
contraction of the descriptor rather than a missing label.

Nothing is in the way of the real backend. `build_batches` already puts the
self pair first for every site with `rjs = 0`, which is exactly what
`soap_cart_der` needs to give `d(soap_i)/d(r_i)`; what is missing is the
contraction and a shim entry for it.

The oracle for it is already sitting in the data:
`water_dipole/water_dipole_tnep_converted_test.xyz` carries a reference
`mu="..."` in every frame header, and `out_quip_predict` beside it is what QUIP
predicts for the same frames from the same model.

## 26. A torch model segfaults inside LAMMPS, in OpenBLAS

ShiftML through `compute mad/property/python` died on the first forward pass:

```
Caught signal 11 (Segmentation fault: address not mapped to object at address 0x148)
#9  0x0000000000000148 in ?? ()
#10 sgemm_batch_thread ()            from /lib64/libopenblas.so.0
#11 cblas_sgemm_batch ()             from /lib64/libopenblas.so.0
#12 at::native::mkl_gemm_batched(...) from torch/lib/libtorch_cpu.so
```

The same module, the same model and the same 3304-atom cluster run clean in a
standalone interpreter, so nothing about the model or the binding is wrong.

`libtorch_cpu.so` carries its own MKL and **defines** `cblas_sgemm_batch`;
`/lib64/libopenblas.so.0` defines it too, and `lmp` links that one directly for
soap_turbo's LAPACK. The system library is therefore in the global scope first,
so when the embedded interpreter dlopens libtorch_cpu, torch's internal call is
interposed onto OpenBLAS's batched routine -- which is not the one its caller
was compiled against, and jumps through a null pointer.

`OPENBLAS_NUM_THREADS=1` does not help; the path is chosen by torch, not by the
thread count. What works is putting torch's definition in the global scope
first:

```bash
LD_PRELOAD=$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib/libtorch_cpu.so
```

LAMMPS's own BLAS calls then resolve to torch's MKL, which is fine and is
usually faster. Nothing here is specific to ShiftML: any torch model reaching
`bmm`/`matmul` hits it, which is most of them. The reason the existing `torch`
leg never did is that its model is a coordination number -- norms and elementwise
ops, no gemm.

## 27. A ghost cloud is the wrong way to feed a long-ranged model a small cell

`compute mad/property/python` hands a model the owned atoms and their ghosts and
no cell. That is the right general contract -- it needs no boundary conditions,
knows nothing about domain decomposition, and is the only form that survives
MPI. It is also, for a long-ranged model on a small cell, absurd.

ShiftML asks for a 5 A neighbour list and passes two messages over it, so its
interaction range is 10 A. Succinic acid's cell is 6.9 x 7.0 x 5.3 A. Reaching
10 A therefore takes `comm_modify cutoff 12.0`, and LAMMPS then carries 3304
atoms to answer a question about 28: 118 copies of the cell.

Measured, one forward and one backward pass:

```
periodic, 28 atoms            1.48 s
cluster, 8 A ghost shell      7.54 s   (744 atoms)
cluster, 10 A ghost shell    12.17 s   (1202 atoms)
cluster, 12 A ghost shell    17.58 s   (1808 atoms)
```

and in LAMMPS the whole `run 0` went from 21 s to 1 s. `pbc yes` passes the
three cell vectors and the owned atoms instead, and the two agree to 9.2e-05
ppm, which is the model's own float32 noise. The convergence with shell size is
exactly what the 10 A range predicts: 1.1 ppm at 6 A, 6.1e-02 at 8 A, 1.0e-03
at 10 A, 1.5e-04 at 12 A.

The cost of the shortcut is the virial. `sum_a f_a (x) r_a` over owned atoms is
not the virial of a periodic system -- the images carrying the rest of it are
inside the model now, not in the atom arrays. So the module returns three more
rows, `-dE/dh`, and

    W_ab = sum_i r_ia f_ib + sum_c h_ca (-dE/dh_c)_b

For a torch model that is the second output of the same backward pass, and
metatomic propagates it: the neighbour list's shift vectors carry `h`. Without
it the pressure would be wrong rather than absent, which is the worse failure.

## 28. Broadening sets the stiffness of a spectral bias, not just its resolution

The first biased anneal of a scrambled succinic acid cell went to 26742 K on the
second thermo line. Nothing was wrong with the forces: the independent reference
agreed with them to 4e-06.

`sigma` was 0.5 ppm, chosen to match ShiftML's 1H RMSE of 0.4. But the bias
force carries `(x_l - q_i)/sigma^2` from differentiating the Gaussian, so the
loss surface stiffens as `1/sigma^2` while the spectrum only sharpens as
`1/sigma`. On a structure whose shieldings are nowhere near the target, that
came to 131 eV/A per atom against a potential whose own forces are a few.

At sigma 1.5 ppm the same configuration gives 14.8 eV/A, and the run is stable.
So the broadening of a bias is chosen twice over: narrow enough to keep the
features that distinguish structures, wide enough that the gradient is a force
and not a kick. A sharp experimental linewidth is a reason to fit narrow at the
END of a search, not to start there.

## 29. A gap file with no dispersion cannot hold a molecular crystal, and NPT says so loudly

`pair_style turbogap` with the CHO potential reports 31 GPa on the experimental
succinic acid cell. That reads like a compressed crystal. It is not. Released
under `fix npt tri 1.0 1.0`:

```
step      P (bar)     volume (A^3)   density
   0      313945           256        1.53
 500       61086          1060        0.37
2000         329        340000        0.0012
6000           0.6     2.1e+07        1.9e-05
```

The crystal sublimates. A molecular crystal is held by hydrogen bonding and
dispersion; the 31 GPa is Pauli repulsion with nothing pulling back, so at 1 bar
the molecules simply fly apart.

The missing term is the one the reader already documents as absent. TurboGAP
takes its TS/MBD dispersion from its *input file*, not from the `.gap`, so there
is no block here to read and `MADGapFile` cannot supply it -- see the "Genuinely
not handled" note in the README. Nothing is wrong with the four terms that are
read; the potential as MADLIB can see it is simply not the potential TurboGAP
was running.

What this costs: any constant-pressure work with a `.gap` whose TurboGAP run
used `energy_vdw` is meaningless, and it is meaningless in a way that constant
volume hides completely -- at fixed cell the same potential gives a stable
300 K trajectory with the right bond topology for picoseconds. The pressure is
the only diagnostic that shows it, and only if someone looks.

For an organic molecular crystal, MACE-OFF through `pair_style symmetrix/mace`
carries dispersion (it is fitted at wB97M-D3(BJ)) and is what a free-cell run
needs instead.

## 30. A spectrum can be fitted by the wrong structure, and the control is how you find out

Longer anneal, cell free, MACE-MP small-omat-0, gamma 10 on 1H and 13C. The bias
fits far better than at fixed volume -- 1H dissimilarity 1.275 -> 0.0685, a
factor of 19, against a control that ends at 0.742 from the same trajectory.

Then look at where the hydrogens are. Nearest heavy atom of each of the twelve:

```
target    4 O-H, 8 C-H      O1.00 O1.00 O1.01 O1.01 C1.09 x8
biased    7 O-H, 5 C-H      O0.96 x3 O0.97 O1.04 x3 C1.07 C1.09 C1.10 C1.11 C1.16
control   4 O-H, 8 C-H      O0.98 O0.99 O1.01 O1.02 C1.09 C1.10 x5 C1.11 x2
```

The biased structure has moved three protons from carbon to oxygen. Succinic
acid has not been rebuilt; something with the wrong hydrogen chemistry has, and
its 1H spectrum matches the target better than the chemically correct control's
does. The four short O-H give the 16 ppm line and the three long ones sit with
the five C-H under the 28 ppm line, so the two-group pattern comes out right for
the wrong reason.

Nothing is wrong with the bias. A 1H spectrum of this molecule is two lines and
a 13C spectrum is two more: four numbers, against 84 degrees of freedom. The
data does not determine the structure, and at gamma 10 the restraint is strong
enough to buy agreement with those four numbers at the cost of the chemistry --
0.56 eV of it, measured against the control's potential energy. This is the
failure the README warns about under gamma sizing, with a control to prove it
rather than a warning to heed.

What would help, in order: more observables (17O is free here -- the target is
already built -- and the shielding ANISOTROPY is three more numbers per site
that ShiftML already predicts and mad/nmr does not yet use); a smaller gamma; and
a potential that puts the density right, since both runs ended 52% expanded and
the bias's stress did not pull it back.

The one unambiguous reading: **report the per-atom property, not only the
spectrum.** `compute mad/property/*` is dumpable for exactly this reason, and
the twelve numbers above took one dump to find where the dissimilarity alone
said the fit was excellent.

## 31. `thermo_style` throws away `thermo_modify`, and a live run then looks dead

A deck that sets `thermo_modify flush yes` early, installs the bias fixes, and
then switches to a `thermo_style custom` naming them loses the flush: LAMMPS
resets every `thermo_modify` setting to its default when the style is replaced.
The log then sits in a stdio buffer and is written in whole blocks, so on a slow
run -- 0.4 s a step, a thermo line every 500 -- it can be fifteen minutes stale
while the job is perfectly healthy.

Two consequences, both of which cost time here. A watchdog keyed on the log's
mtime declares a stall that is not happening. And any `-var` sweep read back
from logs is missing its tail until the run exits.

Put the `thermo_modify flush yes` **after** the last `thermo_style`, and key a
watchdog on `fix mad/*`'s own spectrum file instead: `write_every` flushes it,
and its `# step N ... rel_error X` header is the progress you actually want.

Unrelated but adjacent: do not grep a LAMMPS log case-insensitively for `error`.
The deck's own header comments are echoed into the log verbatim, so prose like
"its density error is handled by the pressure" matches. Anchor on `^ERROR`.

## 32. A third observable does what a stronger gamma cannot

Same cell, same seed, same pressure, same 2 x 6000 steps, same gamma 10 as #30.
The only change is a third channel, `fix mad/nmr` on the eight oxygens, target
built from the same pristine prediction. Dissimilarity on each channel's own
grid and sigma, from `probe_bias.sh`:

```
                       1H       13C      17O
scrambled start      0.547    1.685    1.171
biased on 1H,13C     0.008    0.145    1.952      <- #30
biased on 1H,13C,17O 0.023    0.105    0.060
control, gamma 0     0.349    0.665    1.155
```

The second row is the whole argument for doing this. Fitting two observables
drove the third **past the scrambled cell it started from** -- 1.95 against
1.17 -- while the control, which fitted nothing, left oxygen where it found it.
That is not a spectrum failing to improve; it is a spectrum being actively
spent to pay for the other two.

Add it as a restraint and the trade reverses. 17O falls 32-fold, 13C improves,
and **1H gets worse** (0.008 -> 0.023). The bias can no longer buy agreement in
hydrogen with the wrong hydrogen chemistry, and the worse number is the good
sign. In bonds and in nearest-heavy-atom distances:

```
                     C-C  C-H  C-O  H-O
target                6    8    8    4      4 O-H at 1.00-1.01, 8 C-H at 1.09
biased on 1H,13C      7    5    6    7      three protons moved C -> O
biased on 1H,13C,17O  5    7    8    5      one did
control, gamma 0      6    8    7    4
```

Seven of the eight oxygens land in the two target environments (-67,-61,-56 and
74,86,91,99 ppm against the target's -64..-59 and 86..90); one sits at +268 and
takes a carbon and a proton with it. One defect site, where two channels left
three.

The cell moved too, for the first time: 362.6 A^3 against the control's 387.9,
so the data pulled 10 points of the density error back. It costs 1.45 eV of
MACE energy relative to the control against #30's 0.57, but this potential puts
+65 kbar on the experimental cell, so on it being closer to the true density
*is* being higher in energy. The two are not separable here and the eV should
not be read as the bias straining the structure by that much.

Two things worth keeping about the 17O channel itself. Its grid is -500..400
ppm where the target only spans -64..+90: a shielding outside the grid lands in
no bin and is invisible to the loss as well as to the force, so the wide grid is
what makes a wrecked oxygen count against the fit instead of silently vanishing
from it. And sigma 20 ppm was chosen against the 150 ppm between the C=O and
O-H environments -- 40 would merge them and lose exactly the contrast that
catches a migrating proton. The reach worry (a stray at -250 is 10 sigma from
any target feature and feels nothing) did not bite, because at 1000 K the
oxygens pass back through the reachable region constantly.

The general form: **the cure for a bias that fits the wrong structure is another
observable, not a smaller gamma.** Same gamma, one more channel, and the failure
in #30 mostly closed.

## 33. The span is real information, and six channels dissociated the crystal

The shielding tensor is already in the forward pass: ShiftML's nine outputs are
a rank-2 tensor in an irreducible basis and the Cartesian form is a fixed 9x9
map of them, so `mad_shiftml_span` reads the graph `mad_shiftml` built and costs
a backward pass, not a forward one. Span -- largest principal value minus
smallest -- was chosen over the Haeberlen anisotropy and the skew because it
uses only the extreme eigenvalues, and those coincide only when all three do.
The others are defined by an ordering that swaps whenever two principal values
cross, which puts a discontinuity in the force for no physical reason.

It is not a restatement of the isotropic shielding. On the pristine crystal the
twelve protons split 8 + 4 by span (C-H at 7-11 ppm, O-H at 26-28) and 4 + 8 by
isotropic shielding (O-H at 16, C-H at 28.5): the two spectra are satisfied by
opposite multiplicities.

And #32's lesson recurred one level down. The three-isotropic-channel structure
has a 13C SPAN dissimilarity of 1.08 against the scrambled cell's own 0.70 --
its carbon spans are 14, 27, 27, 64, 71, 126, 149, 170 where the target is a
tight 39x4 + 171x4. The tensor shapes had been spent to buy the isotropic lines.

Adding all three span channels at gamma 2, keeping the isotropic gamma 10:

```
                  1H     13C     17O    1H sp  13C sp  17O sp   total
scrambled start  0.547  1.685   1.171   0.570  0.697   1.935    6.61
3 iso channels   0.023  0.105   0.060   0.135  1.084   1.227    2.63
6 channels       0.015  0.420   0.063   0.069  0.200   0.696    1.46
control          0.349  0.665   1.155   0.513  0.881   1.455    5.02
```

Five of six improved and the total fell 44%; the carbon span fell 1.08 -> 0.20.
The restraint works.

**The structure did not.** 22 bonds against the target's 26, the C-C skeleton
down to 4 from 6, one carbon left singly coordinated, and a free H2 molecule --
two protons whose nearest heavy atom is 1.6 and 2.3 A away. The cell ended at
514 A^3, +101%, against 362 for three channels and 388 for the control, and the
structure sits 6.5 eV above the control on MACE. 13C isotropic got worse
(0.105 -> 0.420) for the same reason the skeleton broke.

The volume is the proximate cause and it is worth stating plainly: **a shielding
is a local quantity, so at twice the experimental volume a spectrum can be
satisfied by isolated fragments.** The six channels are not wrong to be
satisfied; they simply do not constrain density, and nothing else in the run
did either, because MACE-MP wants +72% at 1 bar on its own (#29).

Two caveats on attribution. The six-channel deck has an extra 1500-step
switch-on leg the three-channel one did not, so protocol and channel count are
confounded; separating them needs a gamma 0 run through the same protocol.
And the switch-on is not optional: six channels arriving at full strength put
33 eV into a 28-atom cell and reached 9247 K in eighty steps. A gamma ramp only
delayed it to 9308 K -- the fix was halving the added restraint, not softening
its arrival. Finite-differencing the span adjoint first (3.7e-4 relative, error
growing as 1/h, i.e. float32 round-off) is what ruled out a noisy gradient as
the cause, since a noisy force heats a run exactly as a too-strong one does.

The next thing to fix is not the observables. It is the density: run at the
6.5 GPa where MACE-MP reproduces the experimental volume, or get a potential
that does not need the crutch.
