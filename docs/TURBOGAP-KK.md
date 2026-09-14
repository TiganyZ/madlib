# `pair_style turbogap/kk`: the plan, and the portability problem under it

Author: Tigany Zarrouk (tiganyzarrouk@gmail.com)

`pair_style turbogap` works on the CPU and reproduces TurboGAP to 1.1e-09 on
forces. Putting it on a GPU inside LAMMPS is not one more step of the same
kind, because it runs into a question the CPU version never had to answer:
**three portability mechanisms now stack, and two of them solve the same
problem.**

This is what they are, where they collide, and what to do about it.

---

## 1. The three layers

| layer | what it makes portable | how it is selected |
| --- | --- | --- |
| `src/hop` | the **vendor API**: one HIP source (`hipMalloc`, `hipMemcpyAsync`, `hipblasDgemm`, `hipStream_t`) compiles for CUDA or for ROCm | `-DHOP_TARGET_CUDA` and which compiler |
| TurboGAP's Kokkos | the **kernel dispatch**: one kernel body becomes a `<<<>>>` launch or a `Kokkos::parallel_for` | `-D_KOKKOS` |
| LAMMPS's Kokkos | the **execution and memory spaces** of the host application, and Kokkos's lifetime | `-D PKG_KOKKOS=on` |

hop and Kokkos overlap. Both exist to let one source target NVIDIA or AMD.
They do it at different layers -- hop at the API call, Kokkos at the dispatch
-- so today they coexist rather than conflict. Inside LAMMPS that stops being
true, because LAMMPS's Kokkos is not optional and not ours.

## 2. Where they actually collide

Measured, not guessed.

**The Kokkos version is pinned by LAMMPS.** `cmake/Modules/Packages/KOKKOS.cmake`
does `find_package(Kokkos 5.2.1 REQUIRED CONFIG)` for `EXTERNAL_KOKKOS`, and
otherwise builds its bundled 5.2.1. So device code compiled into LAMMPS uses
LAMMPS's Kokkos. TurboGAP's own pinned 4.7.04 is irrelevant there, and two
Kokkos runtimes in one process is not a thing to attempt.

**The C++ standard is forced.** `cmake/CMakeLists.txt`:

```cmake
if(PKG_KOKKOS AND (CMAKE_CXX_STANDARD LESS 20))
  set(CMAKE_CXX_STANDARD 20)
endif()
```

Passing `-D CMAKE_CXX_STANDARD=17` does not survive this. Anything compiled
into a KOKKOS build is C++20. `gap_3b.cc` already needs C++20; the `.cu` files
currently build at C++17 under TurboGAP's own makefiles and would have to be
C++20-clean.

**The toolchain floor is higher than TurboGAP's.** Kokkos 5.2.1 at C++20
requires NVCC >= 12.2. `alt` has 12.0.140 on `PATH`, which is what TurboGAP's
`Makefile.Aalto_*` uses -- and a CUDA 13.4 at `/usr/local/cuda-13.4` that is
not on `PATH`. The LAMMPS build needs the latter. **The two builds on this
machine cannot share a compiler**, which is worth knowing before debugging a
link error that is really a toolchain split.

**Kokkos's lifetime is LAMMPS's.** `KokkosLMP` calls `Kokkos::initialize`
before any pair style exists. `tg_backend_initialize` already returns early
when Kokkos is up, which is right. `tg_backend_finalize` did not: reached from
`cuda_device_reset`, it would have finalised LAMMPS's Kokkos out from under the
host application, from a routine whose name promises only to reset a device.
Fixed by tracking ownership -- see `tg_owns_kokkos` in `gpu_backend.h`.

**Nothing checks that hop and Kokkos agree.** hop targets CUDA or ROCm by
`-DHOP_TARGET_CUDA`; Kokkos targets CUDA or HIP by its own configuration.
Nothing currently fails the build when they disagree, and a mismatch is not a
compile error -- it is two runtimes in one binary. This needs a
`static_assert`, whichever option below is taken.

## 3. The vendor-API surface, measured

`grep` over `src/gpu`, distinct `hip*` functions:

| where | distinct calls | what they are |
| --- | --- | --- |
| `gpu_memory.cu` | 31 | malloc/free/memcpy/memset, pinned host memory, pools, device select and query, streams, events, error strings |
| `gpu_blas.cu` | 9 | hipBLAS handle, `Dgemm`, `Dgemv`, `SetStream` |
| the kernel files | ~10 | almost entirely scratch `hipMallocAsync`/`hipFreeAsync`, `hipMemsetAsync`, `hipMemcpyAsync` around their own launches |
| `orthonormalization_kernels.cc` | 13 | hipSOLVER. **Disabled in the Makefile**, so not in the build |

The tail is the important part: the per-kernel files are nearly free of vendor
API already. Almost all of it is concentrated in two files.

## 4. Four resolutions

### Option 1 -- keep hop; Kokkos only dispatches

What exists today, carried into LAMMPS unchanged.

Cheapest, and TurboGAP standalone is untouched. But inside LAMMPS you link
hop's vendor runtime *and* Kokkos's: two allocators, two notions of a stream,
and hipBLAS as a dependency LAMMPS otherwise does not have. The
agreement problem above stays live forever.

### Option 2 -- `-D_KOKKOS` means "no vendor API anywhere"

Under `_KOKKOS`, replace the surface in section 3:

| now | then |
| --- | --- |
| `hipMallocAsync` / `hipFreeAsync` | `Kokkos::kokkos_malloc` / `kokkos_free`, or a scoped `View` |
| `hipMemcpyAsync` / `hipMemsetAsync` | `Kokkos::deep_copy` / a fill through `tg_parallel_for` |
| `hipStream_t` | a `Kokkos::Cuda` / `Kokkos::HIP` execution space instance |
| `hipblasDgemm` / `Dgemv` | KokkosKernels `KokkosBlas::gemm` / `gemv` |
| device select and query | Kokkos |

hop is then not compiled at all in a Kokkos build, and CUDA-versus-ROCm becomes
Kokkos's problem -- which is what Kokkos is for. Costs a KokkosKernels
dependency and the work in section 3.

### Option 3 -- Option 2, staged through the layer that already exists

`gpu_backend.h` grows `tg_malloc`, `tg_free`, `tg_memcpy`, `tg_memset`,
`tg_gemm`, each with a `hip*` body and a Kokkos body, exactly as
`tg_parallel_for` already has. The invariant becomes crisp and checkable:
**under `-D_KOKKOS` there is no `hip*` call left in `src/gpu`** -- a grep, and
therefore a CI check.

**This is the recommendation.** It is the same "one body, two dispatches"
pattern already validated three times over the kernels, it reaches Option 2's
end state, and every step is verifiable against the control in
`tools/verify_kokkos.sh` on the way.

### Option 4 -- do not take TurboGAP's device layer into LAMMPS at all

Reimplement the descriptor in Kokkos C++, the way `symmetrix` did for MACE: a
standalone Kokkos library, no Fortran, no vendor API, taking LAMMPS's Views
directly.

The cleanest result and by far the largest cost -- soap_turbo is ~4100 lines of
Fortran and the device kernels ~2100 of CUDA. Worth naming as the destination
even if nobody walks there soon, because options 1-3 are all steps towards it
and none of them is a step away.

## 5. The staged plan for `turbogap/kk`

The enabler is a fact about Kokkos rather than about TurboGAP: **a Kokkos CUDA
View's memory is CUDA memory.** `view.data()` gives a raw device pointer, and
TurboGAP's `extern "C"` entry points already take raw device pointers. So there
is no upload and no download -- the existing compute can run on LAMMPS's own
device data.

| stage | what | state |
| --- | --- | --- |
| 0 | `pair_style turbogap` on the CPU | **done**: forces to 1.1e-09 against TurboGAP |
| 1 | the neighbour list built on device, host compute unchanged | **done**: energy identical to the host style, forces to 1.7e-09 |
| 2 | 2b and core_pot through TurboGAP's own device kernels | **done**: both terms match the host, 2.9e-12 and 1.3e-12 |
| 3 | Option 3: no `hip*` under `_KOKKOS` | applied at one point of use, via the shim headers |
| 4 | soap_turbo and 3b on the device | not started, and the one that pays |

### What stage 1 settled

The acceptance check is end to end -- `turbogap/kk` against `turbogap` on the
same configuration -- because with the compute held fixed, any difference IS
the neighbour list. Energy matches to every digit the thermo table
prints; forces differ by 1.7e-09 relative and cannot be made to match bitwise,
because the two runs build their lists differently (LAMMPS's CPU binning
against Kokkos's) so a site's neighbours come in a different order and the
per-pair force adjoint reassociates. "Exactly" was claimed here on the strength
of the thermo output, which prints eight digits; at `%.17g` the terms differ at
2e-11 like everything else.

Four things it ran into, all of which stage 2 inherits:

- **KOKKOS refuses `newton on` with a full neighbour list**, and a SOAP
  descriptor must have a full list. So folding ghost forces back onto their
  owners -- which LAMMPS does under newton on -- is the style's own job now,
  through `pack_reverse_comm`. Without it every force a site puts on a
  neighbour it does not own is dropped silently.
- **`execution_space` must name where the forces are actually written.** With
  the compute on the host and the space declared as device, LAMMPS marks
  `atom:f` modified on the device while the style writes it on the host, and
  Kokkos rejects the pair: "Concurrent modification of host and device views".
  Stage 2 moves both together, and that is one commit rather than two.
- **`newton off` also means LAMMPS never clears the ghost forces.**
  `Verlet::force_clear()` extends its `memset` over the ghosts only when newton
  is on. A style that writes ghost forces and folds them back itself therefore
  reads last step's leftovers -- exact at `run 0`, when the array is fresh from
  allocation, and wrong by up to 16 eV/A on every atom from step 1. This sat
  undetected through all of stage 1 and 2 because no leg in the suite took a
  step; it turned up in a benchmark. `tests/in.md_*` and the `pairmd` leg exist
  for it now, and the lesson generalises: a pair style that is only ever
  checked at `run 0` is not checked.
- LAMMPS develop **renamed the view typedefs**:
  `t_kkfloat_1d_3_lr_randomread`, not `t_x_array_randomread`.
- `PairStyle()` takes the template directly. The `#define`d alias form
  symmetrix uses exists because it templates on two parameters and the comma
  would split the macro arguments; with one parameter the alias is actively
  wrong, because the `#undef` closing it leaves the name undefined in the
  generated `style_pair.cpp`.

One number that looks alarming and is not: the device list holds 4929 pairs
where LAMMPS reports 14876 neighbours. LAMMPS builds at `rcut + skin` (6.5 A
here) and both styles then filter to the potential's 4.5 A; the counts sit on
opposite sides of that filter.

### Stage 2: measured scope, and the gate

To vendor:

| | lines |
| --- | --- |
| `src/gpu/*.cu`, `*.cc` | 6281 |
| `soap_turbo_gpu` (5 files) | 4106 |
| `fortran_cuda_interfaces.f90` | 1375 |
| `gpu_context_gpu.f90` | 329 |
| `src/hop` headers | 68 files |

**The first gate is clear**: all 14 `.cu` files compile under CUDA 13.4 at
C++20 -- the standard LAMMPS forces with `PKG_KOKKOS` on, and a newer toolchain
than TurboGAP's own 12.0/C++17 build. That was worth testing before anything
else, because a failure there would have ruled the whole approach out rather
than merely delayed it.

What remains genuinely open is the coexistence described in section 2: hop's
vendor runtime and hipBLAS alongside LAMMPS's Kokkos, two device allocators,
and Kokkos's lifetime. `tg_owns_kokkos` and the `#error` on a hop/Kokkos
vendor mismatch, both in TurboGAP's `gpu_backend.h`, exist for this and are the
first two things stage 2 leans on.

Stage 1 is the right first step: it is small, it is the first genuinely
Kokkos-native kernel in this code, and it is checkable on its own against
something that already works.

One thing the CPU style settled carries straight over: the virial comes from
`virial_fdotr_compute()` rather than TurboGAP's own accumulator, whose sign and
factor would be easy to get wrong by two without looking wrong.

A second thing written here earlier was wrong, and stage 2 measured it. It said
the forces were deterministic because `gpu_scatter.cu` orders its sums by the
neighbour list rather than by the scheduler. The lifted 2b kernel does not go
through `gpu_scatter`; it `atomic_add`s, and four consecutive runs of the
91-atom cell gave four different force dumps. The same input reproduces itself
only at 12 pairs, where nothing collides. So `turbogap/kk` inherits TurboGAP's
run-to-run device non-reproducibility, and a LAMMPS user diffing two GPU
trajectories will see them separate. That is worth stating in the style's own
documentation before anyone runs one.

### What stage 2 settled

`gap_2b.cu` is lifted **verbatim** -- byte-identical to TurboGAP master, checked
by `md5sum` -- behind three shim headers (123 lines) that supply `hipStream_t`,
`TG_LAMBDA`, `TG_INLINE_FUNCTION` and `tg_parallel_for` on LAMMPS's Kokkos. No
hop, no hipBLAS, no second device runtime. That is option 3 of section 4
applied at one point of use.

Both device terms match the host, on two configurations:

| | 2b | core_pot | forces |
| --- | --- | --- | --- |
| glassy carbon, 91 atoms | 2.0e-11 | (zero) | 1.7e-09 |
| core_pot probe, 12 pairs | 2.9e-12 | 1.3e-12 | 7.8e-14 |

The second configuration exists because the first cannot check `core_pot` at
all: the term is defined on r in [0.05, 1.0] A and is identically zero beyond
it, so it contributes exactly nothing to any equilibrated structure. Deleting
the device `core_pot` loop outright still passed every check until
`tests/data/core_probe.data` put three pairs at 0.65-0.85 A. With it, the same
deletion fails at 5.5e-01 on the term and 68 eV on the energy. Two things had
to change before that was visible: the terms are reported separately rather
than as one sum, and at `%.17g` rather than `%.8f`, which was coarser than the
tolerance it was being compared against.

## 6. What it costs today

Ten steps of NVE on `alt` (one RTX A2000, one rank), glassy carbon replicated:

| atoms | `turbogap` | `turbogap/kk` | |
| --- | --- | --- | --- |
| 91 | 0.82 s | 1.16 s | 0.70x |
| 364 | 2.67 s | 2.82 s | 0.95x |
| 728 | 5.11 s | 5.15 s | 0.99x |
| 1638 | 10.37 s | 9.99 s | 1.04x |

`TURBOGAP_PROFILE=1` says why, at 728 atoms:

| phase | ms/step | share |
| --- | --- | --- |
| neighbour build | 2.2 | 0.4% |
| `soap_turbo` | 450 | 87.5% |
| 2b + 3b + core_pot | 66 | 12.9% |

Stages 1 and 2 moved the neighbour build and two of the four terms -- together
about 13% of the work. **Amdahl caps them at 1.15x even if the device ran them
in zero time**, so 1.04x is most of what was available and no amount of tuning
inside them will find more.

Worse, the `kk` pair phase is *slower* than the host's, 66 ms against 61. 2b
and core_pot are cheap enough that the launches and the copy back cost more
than the arithmetic saved. Stage 2 is a correctness step and a plumbing step;
it is not a speed step, and at this size it is a small loss.

This is worth stating plainly rather than quoting the 1.04x: **the reason to
have done stages 1 and 2 is that they are checkable**, and they leave the
device arrays and the dispatch shim that stage 4 needs. SOAP is 87.5% of the
cost and the only thing worth moving.

### MPI, measured

`pair_turbogap` is correct under decomposition -- 4 ranks against 1 gives
max|dF| = 5.5e-08 eV/A, the reassociation floor, with an identical energy. The
`pairmpi` leg checks this now; nothing did before.

It does not scale on `alt`, and the per-phase profile says exactly why. 1638
atoms, bound to physical cores, `OPENBLAS_NUM_THREADS=1`:

| ranks | loop | speedup |
| --- | --- | --- |
| 1 | 5.84 s | -- |
| 2 | 3.24 s | 1.80x |
| 3 | 2.36 s | **2.48x** |
| 6 | 3.02 s | 1.93x |

It peaks at three ranks and then goes backwards. The phases scale very
differently (728 atoms, rank 0):

| phase | 1 rank | 12 ranks | |
| --- | --- | --- | --- |
| neighbour build | 2.76 ms | 0.28 ms | 9.9x |
| 2b + 3b + core_pot | 55.9 ms | 6.3 ms | 8.9x |
| **`soap_turbo`** | **440 ms** | **238 ms** | **1.8x** |

Everything scales except SOAP. Two things are behind it, and neither is a bug
in this code: `alt` is an i5-11600 -- **6 physical cores**, not 12, on
dual-channel memory -- and SOAP is the memory-heavy phase, so it saturates
bandwidth long before it runs out of cores. The ghost fraction makes it worse:
at 12 ranks each rank holds 61 local atoms against 1317 ghosts.

Two things follow. **`alt` is not evidence about production** -- a many-core
node with eight memory channels, or a GPU, is a different machine and this
measurement does not transfer. And **MPI is not a substitute for the GPU work**:
2.5x on this box does not change the case for moving SOAP.

Worth knowing separately: `OPENBLAS_NUM_THREADS=1` is slightly *faster* than
letting OpenBLAS thread (4.55 s against 5.12 s at one rank), so the threaded
BLAS is not helping here and is worth pinning.

## 7. The device SOAP path, measured

It works, and it is a shim rather than a port. TurboGAP's GPU `soap_turbo` is a
drop-in for the CPU one -- same module `soap_turbo_desc`, same entry `get_soap`
-- and takes `cublas_handle` and `gpu_stream` as arguments. `gap.f90`'s
prediction does read module state, but `gpu_context` is `public` and
`soap_backend_begin` needs only five fields of a `soap_turbo`, so both are
reachable without `gpu_context_init` -- whose paired finalize calls
`gpu_device_reset` and would destroy a context this code does not own.

Agreement with the host, on a fixed pseudo-random configuration:

| stage | max abs difference | what it exercises |
| --- | --- | --- |
| descriptor | 1.1e-12 | the forward chain, fully Kokkos-converted |
| derivatives | 1.1e-12 | `gap_soap_descriptor.cu`, 9 raw launches + 1 tree reduction |
| energies | 1.8e-11 | the sparse-GP prediction |
| forces | 2.7e-11 | `gap_soap_forces.cu`, 3 more tree reductions |

### Speed: 1.67x, not the 5x a careless measurement gives

Wall clock on `alt` (one RTX A2000), 40 neighbours per site, host BLAS
unthreaded (it makes no difference here -- 297 ms against 301 ms at 1024):

| sites | host | device | |
| --- | --- | --- | --- |
| 64 | 20.7 ms | 56.8 ms | **0.36x** |
| 256 | 103.7 ms | 71.7 ms | 1.45x |
| 512 | 170.8 ms | 108.9 ms | 1.57x |
| 1024 | 309.3 ms | 187.8 ms | 1.65x |
| 2048 | 604.0 ms | 360.9 ms | 1.67x |

The first measurement of this used `cpu_time` and reported 5.25x. That is
processor time of the calling thread, which does not count a host thread parked
waiting on the device, so it flattered the GPU by about three times. Wall clock
is the honest number and 1.67x is it.

1.67x on the 87.5% phase is about 1.5x overall -- worth having, and modest. The
A2000 is an entry-level card; a datacenter GPU should do considerably better,
and that is the measurement to repeat on Roihu rather than to extrapolate here.
Below roughly 150 sites the device loses outright to launch and transfer.

### Batching is mandatory, and the failure is already diagnosed

At 4096 sites with 40 neighbours the device path dies:

    GPUmem: device allocation FAILED: an illegal memory access was encountered
    GPUmem   this rank      1.153 GB live in 21 buffers, peak    3.340 GB
    GPUmem   The batching keywords are what fix this:
    GPUmem     max_Gbytes_per_process       <- splits the SOAP loop

3.34 GB peak on a 6 GB card, and 3072 sites is the last size that fits. With
a 1 GB budget the same configuration runs in five batches and agrees with the
host to 7.3e-12 on energies and 5.9e-11 on forces.

Splitting costs nothing measurable, and at this size it is slightly *faster*:
2048 sites takes 399 ms in one batch and 350-365 ms in two to eleven. So
batching can be on by default rather than reserved for configurations that
would otherwise fail. This is
the batch loop at `turbogap.f90:1792` with the memory model at
`neighbors.f90:646`, and it is not optional: a LAMMPS rank holding a few
thousand atoms is an ordinary case, not a large one. TurboGAP's own diagnostics
name the fix, which is a good sign for reimplementing it faithfully.

### One soap_turbo per binary, and the linker will not tell you

The CPU and GPU `soap_turbo` share a module name, `soap_turbo_desc`, and
therefore a symbol: `__soap_turbo_desc_MOD_get_soap`. Their `get_soap` takes
**34 arguments and 48**.

Linking both succeeds. An explicit `.o` beats an archive member -- the archive
is only searched for symbols still unresolved -- so whichever is named directly
wins, and a caller compiled against the other `.mod` then makes a 34-argument
call into a 48-argument routine. No diagnostic at any stage: not a link error,
not a warning, just undefined behaviour at runtime.

This is why the tests build two binaries rather than one, and it constrains the
LAMMPS integration: **a MADLIB build has one SOAP backend, chosen at configure
time**, not two selectable at runtime. `pair_style turbogap` and
`turbogap/kk` can both exist in a GPU build, but they share whichever
`soap_turbo` was linked; the CPU fallback for a machine without a GPU is a
separate build, which is how LAMMPS treats KOKKOS anyway.

`tools/check_one_soap_turbo.sh` is the guard, meant to sit in front of any link
that mixes `libmadlib.a` with the device objects.

### Which turns out to be the integration, not an obstacle

If a build has exactly one SOAP backend anyway, the two can export the **same C
symbol with the same arguments** and let the build choose. `madlib_local_property`
now does: a host implementation in `madlib_soap.f90` and a device one in
`madlib_soap_gpu.f90`, identical from the caller's side.

So `pair_turbogap.cpp` needs no change at all to compute SOAP on the device --
it already calls `madlib_local_property`. `tests/soaplocalpropdump.f90` is the
proof and contains no `#ifdef`: one source, compiled twice against the two
backends, agreeing to 5.9e-12 on values and 1.2e-11 on derivatives.

The contraction after the descriptor is on the device too, through
`gpu_local_property_predict`. It was on the host for a long time and for a
reason: the device routine took no `zero_trunc` and did not clamp, while the
host one floors negative values and zeroes their gradients, and core electron
binding energies and Hirshfeld volumes both set `zero_trunc` -- so using it
would have changed the answer for exactly the quantities this exists for.
TurboGAP's device predictor now takes `zero_trunc`, which is what made the
switch safe; `lib/madlib/madlib-gpu.cmake` refuses to configure against a
TurboGAP that predates it. `MADLIB_SOAP_HOST_PREDICT=1` puts the contraction
back on the host, which is how the two are compared inside one process.

## 8. Stage 4, and why it is not a kernel port

The obvious reading of stage 4 is "lift `gap_soap_*.cu` the way `gap_2b.cu` was
lifted". That is the expensive way, and measuring the surface says so:

| | `gap_2b.cu` | the SOAP files |
| --- | --- | --- |
| lines | 201 | ~2700 |
| vendor API calls | **0** | 40+ |
| types needed | none | `hipDoubleComplex`, `double3` |
| allocation | none | `hipMallocAsync` / `hipFreeAsync` |
| BLAS | none | three GEMMs |

`gap_2b.cu` lifted verbatim because it called nothing. The SOAP kernels
allocate, use a complex type, and the prediction step is three GEMMs -- the
sparse-GP kernel evaluation itself, which is what `gpu_blas.cu` exists for.

But TurboGAP already orchestrates all of this **in Fortran**:
`soap_turbo_gpu/src/soap_turbo.f90` (1510 lines) calls the device entry points
in order, and `gap.f90` has a `_GPU` variant of `get_soap_energy_and_forces`
that takes `soap_d`, `soap_der_d` and `n_neigh_d` -- device pointers -- plus a
`cublas_handle` and a `gpu_stream` as `type(c_ptr)`.

MADLIB already vendors `gap.f90` and builds it with `-cpp`, with `_GPU` simply
not defined. So stage 4 is: **turn `_GPU` on for the vendored Fortran and give
it a handle and a stream**, rather than re-deriving 2700 lines of kernels in
C++. The device pointers it wants are what `view.data()` already returns.

### CUDA newer than 12

The device sources are already fine on it. All fourteen `.cu` plus `gap_3b.cc`
compile under **CUDA 13.4** with the flags `makefiles/` already passes, with no
deprecation warnings and no CUDA-version conditional anywhere in them.

An earlier note here said `gap_3b.cc` "uses `__shfl_down`, removed in CUDA 13".
That was wrong, and it was wrong because of a missing flag rather than a real
defect: the file already guards the warp reduction with `#ifdef CUDA` and takes
`__shfl_down_sync` on that branch. `tools/build_soap_gpu.sh` was not passing
`-DCUDA`, so it compiled the HIP branch and failed. It now passes what
TurboGAP's own arch makefile does -- `-DHOP_TARGET_CUDA -DCUDA` and
`--expt-extended-lambda --expt-relaxed-constexpr`, without which every
`tg_parallel_for` is a compile error too.

Where a version genuinely matters is `KOKKOS=1`: Kokkos 5 wants C++20 from a
compiler that supports it, which is **nvcc 12.2 or later**, and below that the
build dies inside Kokkos headers with template errors naming nothing
recognisable. TurboGAP's Makefile now refuses that early and says where to look
for a newer toolkit -- a machine often carries one that is not on `PATH`, which
is exactly the case on `alt` (12.0 on `PATH`, 13.4 in `/usr/local`).

`orthonormalization_kernels.cc` remains the one file that does not compile, and
that is not a CUDA-13 problem either: it is **deliberately disabled** in
TurboGAP's own Makefile, kept for reintegration. hop covers runtime, BLAS, FFT,
RAND and RTC but ships no solver header, so `src/gpu/gpu_solver.h` now maps the
hipSOLVER spellings onto cuSOLVER. That is necessary and not sufficient: the
file declares its helpers as taking a `hipblasHandle_t` and passes it straight
to `hipsolver*`, which compiles on ROCm only because hipSOLVER's handle is a
`void*`. Re-enabling it on CUDA needs a real `cusolverDnHandle_t` threaded
through five helpers. Nothing in the descriptor path touches it.

### The gate, measured -- and two warnings in this document were wrong

`tools/check_gpu_gate.sh` compiles TurboGAP's device layer with LAMMPS's own
include list, compiler and flags. **All nine files compile unmodified**, at
C++20 under CUDA 13.4 with `-D_KOKKOS`:

    gap_soap_radial   gap_soap_angular   gap_soap_descriptor
    gap_soap_forces   gap_soap_radial_operator
    gpu_blas          gpu_scatter        gpu_scan            gap_predict

Two claims made earlier in this file do not survive that measurement.

**"hop means a second vendor runtime beside Kokkos."** It does not. hop is
**header-only** -- 45 headers and some Python tooling, no library at all -- and
is pure `#define` remapping: `hipDeviceSynchronize` to `gpuDeviceSynchronize`
to `cudaDeviceSynchronize`. Putting it on the include path costs nothing at
runtime. `nm` over the nine objects finds **zero** `hip*` or `rocblas*` symbols;
hop resolved every one to CUDA.

**"hipBLAS is a dependency LAMMPS otherwise does not have."** On CUDA it
resolves to cuBLAS, which ships with the toolkit LAMMPS already links under
`PKG_KOKKOS` + CUDA. The whole new link surface is five symbols:
`cublasCreate_v2`, `cublasDestroy_v2`, `cublasDgemm_v2`, `cublasDgemv_v2`,
`cublasSetStream_v2`.

So the vendor-surface problem that shaped section 4 is smaller than it looked.
The 68 hop headers are an **include path**, not a vendor drop, and the ROCm
spelling keeps working for free -- which is the portability answer section 4
was reaching for.

It also means the hand-written shim headers in `lib/madlib/turbogap_gpu/` are
not needed for the SOAP work. They earned their place for `gap_2b.cu`, which
calls no vendor API and so needed no hop, but the SOAP files should use
TurboGAP's real `gpu_common.h` rather than a second copy that can drift.

What is left is the part that was always the real work: there is no C++ compute
entry point and no C++ `.gap` reader, so the orchestration has to be written.
Compiling was never the hard part; it just had to be ruled out first.

Three things to settle when it is done:

- **The BLAS handle.** `cublas_handle` is created by `create_cublas_handle` in
  `gpu_blas.cu`. cuBLAS ships with the CUDA toolkit LAMMPS already links under
  `PKG_KOKKOS` + CUDA, so this is not a second runtime -- it is the same one.
  On ROCm it resolves to hipBLAS through hop. This is the one place where the
  bare-kernel portability question of section 4 has to be answered rather than
  side-stepped.
- **The allocator.** The SOAP kernels `hipMallocAsync` their scratch. Either
  let them (a second allocator beside Kokkos, which is what section 2 warned
  about) or shim those three calls onto Kokkos allocation. The shim is small
  and keeps the files verbatim.
- **`gap_soap_forces.cu` must not be converted to `tg_parallel_for`.** Its
  reductions are `__shared__` trees over doubles, and a Kokkos reduction
  reassociates them. It is fine to *call*; it is not fine to rewrite.

## 9. The CUDA runtime trap

Found by building from clean clones, and worth its own section because the
failure names nothing.

Putting a toolkit's `bin` on `PATH` is not enough. cmake finds `libcudart`
separately, and on a machine that has a system CUDA 12 alongside a newer one
under `/usr/local`, you get **device code compiled by nvcc 13.4 linked against
the CUDA 12 runtime**. It configures, compiles and links without a word.

Then `lmp -k on g 1` dies with **SIGFPE**, before reading any input, in:

    Kokkos::UnorderedMap<int, int, Kokkos::Cuda>::UnorderedMap
      -> deep_copy -> contiguous_fill_or_memset -> parallel_for

An integer divide by zero computing launch geometry, because the occupancy
query against a fatbin the runtime does not understand returns zero. Nothing in
that trace mentions CUDA versions, MADLIB, or the pair style. `-k on` without a
GPU works fine, which makes it look like a device problem rather than a build
one.

The check is one line, and both `tools/install_turbogap_lammps.sh` and the
clean-room script now do it after building:

    ldd lmp | grep libcudart     # .so.13
    nvcc --version               # release 13.4

The fix is to name the toolkit root, not just its bin:

    -D CUDAToolkit_ROOT=/usr/local/cuda-13.4

`install_turbogap_lammps.sh` was missing exactly this: it exported `CUDA_HOME`
and prepended `$CUDA_HOME/bin` to `PATH`, and never told cmake.

## 10. Toolchain notes for `alt`

```sh
export PATH=/usr/local/cuda-13.4/bin:$PATH      # NOT the 12.0 on the default PATH
cmake -S cmake -B build \
  -D CMAKE_CXX_COMPILER=<lammps>/lib/kokkos/bin/nvcc_wrapper \
  -D CUDAToolkit_ROOT=/usr/local/cuda-13.4 \
  -D PKG_KOKKOS=on -D Kokkos_ENABLE_CUDA=on -D Kokkos_ARCH_AMPERE86=on \
  -D PKG_MADLIB=on -D MADLIB_ENABLE_TURBOGAP=on
```

Build under `/tmp`, not in `$HOME`. The home filesystem on `alt` is at
948k of its 1000k **inode** quota with 294 GB of space free, and a LAMMPS
checkout is ~15k files before anything is compiled. A previous attempt there
died with "Disk quota exceeded" while cloning, which reads as a space problem
and is not one. `/tmp` has its own inode pool, over a million free.

## 11. Building `lmp` against the device SOAP backend

Sections 7 and 8 verified the device path through standalone drivers. This is
what makes `lmp` itself use it.

```sh
cmake -S cmake -B build \
  -D PKG_MADLIB=on -D MADLIB_ENABLE_TURBOGAP=on \
  -D MADLIB_ENABLE_GPU=on \
  -D MADLIB_TURBOGAP_SRC=<turbogap>/src \
  -D MADLIB_GPU_ARCH=86 \
  -D CUDAToolkit_ROOT=/usr/local/cuda-13.4
```

`lib/madlib/madlib-gpu.cmake` replaces the vendored source list with TurboGAP's
GPU one, compiled **in place** from that checkout: five Fortran files from
`soap_turbo_gpu/src`, nine more from `src`, and fifteen CUDA translation units
from `src/gpu`. Nothing is copied into MADLIB.

**`src/MADLIB` does not change.** Both `madlib_soap.f90` and
`madlib_soap_gpu.f90` define `madlib_local_property`, which is the whole SOAP
term for `pair_style turbogap` — the sparse-GP prediction that `pair_turbogap.cpp`
calls once per descriptor block. Swapping the library swaps the backend, and
the pair style cannot tell.

That is also the constraint: the two soap_turbo forks declare the **same module
and the same entry name**, with 34 and 48 arguments, so a binary holds one of
them. This is a build-time choice, not a runtime one, which is how LAMMPS
treats KOKKOS anyway.

### What had to be split

`madlib_soap_dim` and `madlib_compress_indices` were in `madlib_soap.f90`,
which a GPU build cannot compile — it uses the 34-argument `get_soap`. They are
now in `madlib_soap_dims.f90`, which both builds compile, and which keeps using
the **CPU** compression module in both.

That is not a workaround. The two forks diverge here for real:

| | module | `get_compress_indices` |
|---|---|---|
| `soap_turbo/src` | `soap_turbo_compress_module` | 9 args, returns `P_i, P_j, P_el` |
| `soap_turbo_gpu/src` | `soap_turbo_compress` | 6 args, returns an index list |

Only the CPU one knows `P_transformation`. Because the **module names differ**,
both fit in one binary, so sizing and index expansion always come from the CPU
one and the device descriptor path uses the `P_j` column as its index list. A
`P_transformation` potential still has no device path — TurboGAP aborts on one
too (`read_files.f90:6370`).

### Three traps in the cmake, all silent-ish

**cmake picks the wrong nvcc.** On a machine with a distribution CUDA in
`/usr/bin` and a newer one under `/usr/local`, cmake takes `/usr/bin/nvcc` even
with the newer first on `PATH`. nvcc 12 then calls the **ptxas it finds on
PATH**, which is 13, and the error is

    ptxas fatal : Value 'sm_52' is not defined for option 'gpu-name'

— sm_52 being nvcc 12's default arch, dropped in CUDA 13. It reads as a missing
architecture; it is two toolkits in one compile. `madlib-gpu.cmake` sets
`CMAKE_CUDA_COMPILER` from `CUDAToolkit_ROOT` and then asserts
`CMAKE_CUDA_COMPILER_VERSION` equals `CUDAToolkit_VERSION`. Same family as §9.

**`CMAKE_CUDA_ARCHITECTURES` must be set before `enable_language(CUDA)`**, not
after. cmake compiles a probe to identify the compiler and defaults it to
`compute_52`, so the identification step fails on CUDA 13 before any of our
sources are seen.

**The Fortran module scanner does not run the preprocessor by default.**
`gap.f90`'s `use gpu_context` sits inside `#ifdef _GPU`, so a scanner that skips
the conditional misses the edge and orders `gap.f90` before `gpu_context_gpu.f90`
— on a source list that is already in dependency order. The symptom is
`Cannot open module file 'gpu_context.mod'`. The fix is `Fortran_PREPROCESS ON`
plus `_GPU` as a compile **definition** rather than a `-D` in
`target_compile_options`, so the scanner sees it too.

### C++20 everywhere

`tools/build_soap_gpu.sh` compiles `.cu` at C++17 and `.cc` at C++20, following
TurboGAP's Makefile. Checked: all fourteen `.cu` compile at C++20 as well, so
the cmake path uses one standard and copies the `.cc` in as `.cu` rather than
carrying per-source language and `-x cu` properties whose flag order against the
standard flag is not guaranteed.

### The entry the pair style calls had to be batched

Building it was not enough. `pair_style turbogap` with the device backend ran
91 and 728 atoms and then **aborted out of memory at 2457** on a 6 GB card:

    GPUmem: device allocation of 0.219 GB FAILED: out of memory
    GPUassert: out of memory src/gpu/gpu_memory.cu 227

`madlib_soap_energy_forces_gpu` had the batch loop; `madlib_local_property` —
the entry `pair_turbogap.cpp` actually calls, and the whole SOAP term for the
pair style — did not, and ran the rank in one shot. It has it now.

Two differences from the other entry. It takes **no budget argument**: its C
signature is TurboGAP's local-property one and the pair style passes what that
needs, so the budget comes from `gpu_mem_stats` (half of free device memory;
`MADLIB_SOAP_MAX_GB` overrides). And the split is **exact rather than
approximate** — `local_property_predict` is per site and per pair with no
coupling between them, so a batch's V and dV/dr go straight into their slice of
the caller's arrays.

The decomposition is measured per call, not cached. It therefore settles rather
than being fixed: the first call sees an empty device and takes three batches,
every later one takes six, because cuBLAS and the sparse set are resident by
then. That moves the last digits between step 0 and step 1. Deliberately not a
remembered minimum — a budget that can only tighten is the batch-count ratchet,
where one transient dip runs the count to 900 and never recovers. Pin
`MADLIB_SOAP_MAX_GB` for a fixed decomposition.

### Measured, RTX A2000, glassy carbon with CO.gap

Per-step SOAP, `TURBOGAP_PROFILE=1`, host build against device build:

| atoms | host | device | |
|---|---|---|---|
| 728 | 539 ms | 441 ms | 1.22x |
| 2457 | 1537 ms | 1175 ms | 1.31x |

Those were measured while the shim still copied the **whole derivative
tensor** back — 3 × dim × n_pairs doubles, hundreds of MB per call — to run the
sparse-GP contraction on the host, which was most of what the device path was
saving. Keeping the contraction on the card removes that, and it is the larger
effect of the two:

| atoms | host backend | device SOAP, host contraction | device SOAP, device contraction |
|---|---|---|---|
| 728 | 10.31 s | 8.34 s (1.24x) | 4.48 s (**2.30x**) |
| 2457 | 30.46 s | 26.23 s (1.16x) | 15.09 s (**2.02x**) |

20 steps of `in.bench`, LAMMPS's Loop time, one A2000, two runs each, all three
columns measured the same way so they are comparable with each other. The
middle column is the same binary as the right-hand one with
`MADLIB_SOAP_HOST_PREDICT=1`, so those two differ only in where the contraction
runs. Every figure repeated to within 1% except the 728-atom middle column,
which gave 8.92 s and 7.75 s -- report it as approximate.

These are not the 1.22x/1.31x recorded above for the host-contraction path:
those came from per-step profiling rather than Loop time, and the two are not
comparable. The ratio to trust for a decision is the one between the last two
columns, which is one binary against itself.

### Proof the device path is the one running

A passing comparison alone would not distinguish "the device reproduces the
host" from "it quietly fell back". Three checks, in the same session:

- the device build links cuBLAS, the host build does not;
- 23 `madlib_soap_gpu` symbols against zero;
- with `CUDA_VISIBLE_DEVICES=` the device build **dies** (rc 100) where the
  host build completes.

The `pairgpu` leg also forces a twelve-batch split and fails if the run reports
no batches at all, so it cannot pass by testing the one-batch case twice.

### The trap the clean room caught: modules beside the sources

The device build worked from a synced working tree and **failed from a clean
clone**, with 32 errors that all look like a broken shim:

    madlib_soap_gpu.f90:293: Error: Type mismatch in argument 'alpha_max';
                                    passed TYPE(c_ptr) to INTEGER(4)
    madlib_soap_gpu.f90:284: Error: Rank mismatch in argument 'l_max'
    ...                     Error: More actual than formal arguments

Those are the arguments landing one slot early: the shim was compiled against
the **34-argument CPU** `get_soap` rather than the 48-argument GPU one. But the
target's source list holds only `soap_turbo_gpu/src/soap_turbo.f90`, and
`build-madlib/mod/soap_turbo_desc.mod` was the GPU one — byte-identical to the
standalone build's, which compiles the same file without complaint.

**gfortran searches the directory of the file it is compiling for `.mod`
files.** `make -C lib/madlib` wrote the CPU modules there; a cmake build of
`lib/madlib/madlib_soap_gpu.f90` then preferred `lib/madlib/soap_turbo_desc.mod`
over its own. Hiding the module in the `-I` directory did not change the error
at all — which is what finally located it: the module being read was never the
one on the command line.

Two things follow, and both are now in the tree. The standalone `Makefile`
compiles with `-Jmod` so it never writes beside a source again. And
`lib/madlib/CMakeLists.txt` refuses to configure when it finds a `.mod` there,
naming `make clean` — because a tree that predates the Makefile change still has
them, and the silent version of this failure is worse than the loud one.

This is the same "one soap_turbo per binary" constraint as the link-level trap
in §4, one level lower down: at the **module** level it does not even need two
object files, only two files with the same name on a search path nobody passed.

### Ranks sharing a card

Batching by itself was still not enough. Every rank asks the device how much
memory is free and takes a fraction of it, and they all get the same answer, so
four ranks on one GPU budget four halves of the same memory. Measured: 5824
atoms on 4 ranks aborted with twelve out-of-memory hits.

The budget is now divided by the ranks sharing the card — node-local rank count
over visible devices, because `cuda_set_device` assigns them round-robin. The
same run then takes seven batches per rank instead of two and completes. The
`pairgpu` leg asserts the divisor rather than running something large enough to
OOM, which no leg can afford: with `MADLIB_SOAP_BATCH_REPORT=1` the shim prints
the budget and the rank share, and the leg checks the share equals the rank
count.
