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
| 2 | vendor the device layer and call its entry points with `view.data()` | gate cleared, not started; scope below |
| 3 | Option 3: no `hip*` under `_KOKKOS` | grep is empty; the regression control stays at zero attributable |
| 4 | optionally Option 4 | -- |

### What stage 1 settled

The acceptance check is end to end -- `turbogap/kk` against `turbogap` on the
same configuration -- because with the compute held fixed, any difference IS
the neighbour list. Energy matches exactly; forces differ by 1.7e-09 relative
and cannot be made to match bitwise, because the two runs build their lists
differently (LAMMPS's CPU binning against Kokkos's) so a site's neighbours come
in a different order and the per-pair force adjoint reassociates. The energy is
a sum over sites whose own order is unchanged, which is why it agrees exactly
and the forces do not.

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

## 6. Toolchain notes for `alt`

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
