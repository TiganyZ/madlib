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

| stage | what | check |
| --- | --- | --- |
| 0 | `pair_style turbogap` on the CPU | **done**: forces to 1.1e-09 against TurboGAP |
| 1 | build the TurboGAP-layout neighbour list on device with `tg_parallel_for`, from LAMMPS's Views | the device list equals the host list built by the CPU style, element for element |
| 2 | vendor the device layer into `lib/madlib` and call the existing device entry points with `view.data()` | `turbogap/kk` energy and forces against `turbogap` on the same configuration |
| 3 | Option 3: no `hip*` under `_KOKKOS` | grep is empty; the regression control stays at zero attributable |
| 4 | optionally Option 4 | -- |

Stage 1 is the right first step: it is small, it is the first genuinely
Kokkos-native kernel in this code, and it is checkable on its own against
something that already works.

Two things the CPU style settled that carry straight over. The virial comes
from `virial_fdotr_compute()` rather than TurboGAP's own accumulator, whose
sign and factor would be easy to get wrong by two without looking wrong. And
the forces are deterministic because `gpu_scatter.cu` sums them in an order
fixed by the neighbour list rather than by the scheduler -- which a pair style
inherits for free, and which matters more in LAMMPS than in TurboGAP because
LAMMPS users diff trajectories.

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
