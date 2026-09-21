# The MADLIB soap_turbo backend, as a static library the package links against.

project(madlib Fortran)

set(MADLIB_ENABLE_GPU OFF CACHE BOOL "Compute SOAP on the device, from a TurboGAP source tree")
set(MADLIB_TURBOGAP_SRC "" CACHE PATH "TurboGAP's src/ directory; required by MADLIB_ENABLE_GPU")
set(MADLIB_GPU_ARCH "" CACHE STRING "CUDA architecture for the device SOAP sources, e.g. 86")

# Backend-independent. madlib_soap_dims sizes a descriptor and expands a
# compression mode, and does it from the CPU soap_turbo's compression module in
# both builds -- only that one knows P_transformation, and its module name
# differs from the GPU fork's, so it coexists with either.
set(MADLIB_COMMON_SOURCES
  madlib_soap_dims.f90
  madlib_gap.f90)

# gfortran searches the directory of the file it is compiling for .mod files,
# so a stale module beside these sources is preferred over the one this build
# produced. That is not a cosmetic clash: the CPU and GPU soap_turbo both
# define soap_turbo_desc, and picking up the wrong one resolves get_soap to the
# wrong argument count -- 34 against 48 -- which reads as a broken shim rather
# than a stale artefact. `make` here used to leave them; it writes to mod/ now.
file(GLOB MADLIB_STALE_MODS ${CMAKE_CURRENT_SOURCE_DIR}/*.mod
                            ${CMAKE_CURRENT_SOURCE_DIR}/turbogap/*.mod)
if(MADLIB_STALE_MODS)
  message(FATAL_ERROR
    "Fortran modules are sitting beside the sources in ${CMAKE_CURRENT_SOURCE_DIR} "
    "and gfortran would prefer them to this build's own. Run: make -C "
    "${CMAKE_CURRENT_SOURCE_DIR} clean")
endif()

if(MADLIB_ENABLE_GPU)
  include(madlib-gpu.cmake)
else()
  # Module dependency order. tools/vendor_turbogap.sh lists the same order and
  # the two must agree: kinds first, gap.f90 last, types before anything that
  # uses the soap_turbo derived type.
  add_library(madlib STATIC
    turbogap/kinds.f90
    turbogap/elements.f90
    turbogap/topology.f90
    turbogap/splines.f90
    turbogap/soap_turbo_functions.f90
    turbogap/soap_turbo_radial.f90
    turbogap/soap_turbo_angular.f90
    turbogap/soap_turbo_compress.f90
    turbogap/soap_turbo.f90
    turbogap/types.f90
    turbogap/local_properties.f90
    turbogap/gap.f90
    madlib_soap.f90
    ${MADLIB_COMMON_SOURCES})

  # -cpp is load-bearing. turbogap/local_properties.f90 is a verbatim copy of
  # TurboGAP's and carries #ifdef _GPU around its device path; without the
  # preprocessor those lines reach the compiler and it fails. _GPU is never
  # defined here, so the blocks compile out. See tools/vendor_turbogap.sh.
  target_compile_options(madlib PRIVATE -cpp)
endif()

set_target_properties(madlib PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod
  POSITION_INDEPENDENT_CODE ON)

target_link_libraries(madlib PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
