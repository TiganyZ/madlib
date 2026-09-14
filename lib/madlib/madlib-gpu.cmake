# The device SOAP backend: libmadlib built against TurboGAP's GPU soap_turbo
# and its CUDA kernels, in place, from an external TurboGAP checkout.
#
# Nothing is vendored. TurboGAP's GPU soap_turbo is a drop-in for the CPU one
# at the module level -- same module name, same entry point, different argument
# count -- so a binary holds one of the two and this file chooses the GPU one.
# That is also why madlib_soap.f90 is absent here: madlib_soap_gpu.f90 defines
# the same madlib_local_property symbol, and the pair style calls it either
# way, which is what makes the swap invisible to src/MADLIB.
#
# See docs/TURBOGAP-KK.md and tools/build_soap_gpu.sh, which builds the same
# object set standalone for the test suite.

if(NOT MADLIB_TURBOGAP_SRC)
  message(FATAL_ERROR
    "MADLIB_ENABLE_GPU needs MADLIB_TURBOGAP_SRC set to a TurboGAP src/ directory")
endif()
set(TG ${MADLIB_TURBOGAP_SRC})
foreach(d gpu hop soap_turbo_gpu/src)
  if(NOT IS_DIRECTORY ${TG}/${d})
    message(FATAL_ERROR "MADLIB_TURBOGAP_SRC=${TG} has no ${d}/ -- is that TurboGAP's src/?")
  endif()
endforeach()
if(NOT EXISTS ${TG}/hop/hop)
  message(FATAL_ERROR "${TG}/hop is empty; run git submodule update --init in TurboGAP")
endif()

# The device predictor has to be able to clamp. Without zero_trunc it returns
# unfloored values and keeps the gradients of sites the host would have
# floored, which is not a build error and not a crash -- it is core electron
# binding energies and Hirshfeld volumes coming out different from the host
# backend, in a build whose whole point is that they do not.
file(READ ${TG}/gpu/gap_predict.cu MADLIB_TG_PREDICT)
string(FIND "${MADLIB_TG_PREDICT}" "gpu_zero_trunc" MADLIB_TG_HAS_TRUNC)
if(MADLIB_TG_HAS_TRUNC EQUAL -1)
  message(FATAL_ERROR
    "${TG} predates zero_trunc on the device local-property predictor.\n"
    "       Update TurboGAP, or build without -D MADLIB_ENABLE_GPU.")
endif()

# Before enable_language, not after: cmake compiles a probe to identify the
# CUDA compiler and defaults it to compute_52, which CUDA 13 has dropped. The
# failure is ptxas rejecting sm_52 while identifying nvcc, which reads as a
# broken toolkit rather than a missing architecture.
if(MADLIB_GPU_ARCH)
  set(CMAKE_CUDA_ARCHITECTURES ${MADLIB_GPU_ARCH})
elseif(NOT CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "set MADLIB_GPU_ARCH to the device's compute capability, e.g. 86")
endif()

# Name nvcc, do not let cmake search. A machine with a distribution CUDA in
# /usr/bin and a newer one under /usr/local has two, and cmake picks the
# distribution one even with the newer first on PATH. nvcc 12 then calls the
# ptxas it finds on PATH, which is 13, and the failure is ptxas rejecting the
# sm_52 that nvcc 12 defaults to -- a missing-architecture message for what is
# really two toolkits in one compile. Same trap as the libcudart mismatch in
# tools/install_turbogap_lammps.sh.
if(NOT CMAKE_CUDA_COMPILER AND EXISTS ${CUDAToolkit_ROOT}/bin/nvcc)
  set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc)
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
if(NOT CMAKE_CUDA_COMPILER_VERSION VERSION_EQUAL CUDAToolkit_VERSION)
  message(FATAL_ERROR
    "nvcc is ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER}) but the toolkit "
    "found is ${CUDAToolkit_VERSION} -- pass -D CUDAToolkit_ROOT for one of them")
endif()
# gap.f90's _GPU half opens with an unconditional `use mpi`, so the device
# backend cannot be built without an MPI Fortran module. A serial LAMMPS can
# still use the CPU backend.
find_package(MPI REQUIRED COMPONENTS Fortran)

# orthonormalization_kernels.cc is left out here as it is in TurboGAP's own
# Makefile: it calls hipSOLVER with its #include commented out, and hop ships
# no solver header.
file(GLOB MADLIB_GPU_CU ${TG}/gpu/*.cu)
set(MADLIB_GPU_CC ${TG}/gpu/gap_3b.cc)

# The .cc sources are CUDA compiled as C++20; copying them in under a .cu name
# says so once, rather than per-source language and -x cu properties whose
# flag order against the standard flag is not guaranteed.
foreach(f ${MADLIB_GPU_CC})
  get_filename_component(b ${f} NAME_WE)
  configure_file(${f} ${CMAKE_CURRENT_BINARY_DIR}/gpu/${b}.cu COPYONLY)
  list(APPEND MADLIB_GPU_CU ${CMAKE_CURRENT_BINARY_DIR}/gpu/${b}.cu)
endforeach()

# The CPU soap_turbo's compression module, under a name that cannot collide
# with the GPU fork's file of the same basename. Both modules live in this
# binary: they are named differently (soap_turbo_compress_module against
# soap_turbo_compress) and only the CPU one knows P_transformation, so
# madlib_soap_dims sizes a descriptor the same way in either build.
configure_file(turbogap/soap_turbo_compress.f90
               ${CMAKE_CURRENT_BINARY_DIR}/soap_turbo_compress_cpu.f90 COPYONLY)

add_library(madlib STATIC
  ${TG}/kinds.f90
  ${TG}/fortran_cuda_interfaces.f90
  ${TG}/soap_turbo_gpu/src/soap_turbo_functions.f90
  ${TG}/soap_turbo_gpu/src/soap_turbo_radial.f90
  ${TG}/soap_turbo_gpu/src/soap_turbo_angular.f90
  ${TG}/soap_turbo_gpu/src/soap_turbo_compress.f90
  ${TG}/soap_turbo_gpu/src/soap_turbo.f90
  ${TG}/elements.f90
  ${TG}/topology.f90
  ${TG}/splines.f90
  ${TG}/types.f90
  ${TG}/gpu_context_gpu.f90
  ${TG}/local_properties.f90
  ${TG}/gap.f90
  ${CMAKE_CURRENT_BINARY_DIR}/soap_turbo_compress_cpu.f90
  madlib_soap_gpu.f90
  ${MADLIB_COMMON_SOURCES}
  ${MADLIB_GPU_CU})

# Fortran_PREPROCESS rather than a bare -cpp, and _GPU as a compile DEFINITION
# rather than an option, because cmake's Fortran module scanner has to see them
# too: gap.f90's `use gpu_context` sits inside #ifdef _GPU, and a scanner that
# does not evaluate the conditional misses the edge and orders gap.f90 before
# the module it needs. The symptom is "Cannot open module file gpu_context.mod"
# on a source list that is already in dependency order.
set_target_properties(madlib PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  Fortran_PREPROCESS ON)
target_compile_definitions(madlib PRIVATE _GPU _MPIF90)

# What makefiles/Makefile.Aalto_*_hip_cuda passes, and every one matters.
# Without --expt-extended-lambda every tg_parallel_for is a compile error;
# without -DCUDA gap_3b takes the HIP branch of its warp reduction and fails on
# __shfl_down. hop is header-only, so -DHOP_TARGET_CUDA is the whole port.
target_compile_options(madlib PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-DHOP_TARGET_CUDA>
  $<$<COMPILE_LANGUAGE:CUDA>:-DCUDA>
  $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>
  $<$<COMPILE_LANGUAGE:Fortran>:-fallow-argument-mismatch>)

target_include_directories(madlib PRIVATE ${TG}/gpu ${TG}/hop ${TG}/hop/source/hip)
target_link_libraries(madlib PUBLIC CUDA::cudart CUDA::cublas MPI::MPI_Fortran)

message(STATUS "MADLIB: device SOAP from ${TG} (sm_${CMAKE_CUDA_ARCHITECTURES})")
