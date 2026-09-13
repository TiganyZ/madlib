# MADLIB - Molecular Augmented Dynamics
#
# The package's own sources are picked up by the STANDARD_PACKAGES glob, so all
# this file does is the optional Fortran backend, which evaluates TurboGAP
# soap_turbo local-property models and lets an existing .gap file be used
# unchanged. Everything else builds and runs without it; only
# compute mad/property/turbogap needs it, and that style reports what to build
# rather than disappearing.
#
# For this file to be read at all, MADLIB has to appear in the PKG_WITH_INCL
# list in cmake/CMakeLists.txt -- being in STANDARD_PACKAGES is not enough.

set(MADLIB_ENABLE_TURBOGAP OFF CACHE BOOL "Build the soap_turbo local-property backend")
set(MADLIB_ENABLE_PYTHON OFF CACHE BOOL "Build the embedded-Python local-property backend")

# The soap_turbo backend has two implementations of one interface, and a binary
# gets one of them: TurboGAP's CPU and GPU soap_turbo declare the same module
# and entry names, so they cannot be linked together and cannot be selected at
# runtime. MADLIB_ENABLE_GPU picks the device one, which needs an external
# TurboGAP checkout (MADLIB_TURBOGAP_SRC) and MADLIB_GPU_ARCH; the options are
# declared in lib/madlib/CMakeLists.txt. Nothing in src/MADLIB changes -- both
# export the same C entry points.

# The ML-IAP backend needs no option: it compiles when ML-IAP is enabled, since
# that is the only thing it depends on.
if(PKG_ML-IAP)
  target_compile_definitions(lammps PRIVATE -DMADLIB_MLIAP)
  target_include_directories(lammps PRIVATE ${LAMMPS_SOURCE_DIR}/ML-IAP)
  message(STATUS "MADLIB: ML-IAP property backend enabled")
endif()

if(MADLIB_ENABLE_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Embed)
  target_link_libraries(lammps PRIVATE Python::Python)
  target_compile_definitions(lammps PRIVATE -DMADLIB_PYTHON)
  message(STATUS "MADLIB: embedded-Python backend enabled (${Python_VERSION})")
endif()

# The lifted TurboGAP device kernels, when there is a Kokkos to run them on.
# Only an include path: the sources are picked up with the rest of src/KOKKOS,
# and the shim headers beside them are what let the kernel file be a verbatim
# copy of TurboGAP's. See docs/TURBOGAP-KK.md.
if(PKG_KOKKOS)
  target_include_directories(lammps PRIVATE ${LAMMPS_LIB_SOURCE_DIR}/madlib/turbogap_gpu)
  # Named explicitly, unlike the pair style beside it. LAMMPS gathers KOKKOS
  # package sources from a fixed list plus the style globs (pair_*, fix_* and
  # so on); this file is neither a style nor on that list, so a symlink into
  # src/KOKKOS is not enough to get it compiled -- it is silently ignored, and
  # the link then fails on the kernels it would have defined.
  target_sources(lammps PRIVATE ${LAMMPS_SOURCE_DIR}/KOKKOS/mad_gap_2b_kokkos.cpp)
  message(STATUS "MADLIB: TurboGAP device kernels on LAMMPS's Kokkos")
endif()

if(MADLIB_ENABLE_TURBOGAP)
  enable_language(Fortran)
  find_package(LAPACK REQUIRED)    # soap_turbo's radial basis needs dgesvd, dpotrf, dpotri
  find_package(BLAS REQUIRED)
  if(NOT MADLIB_LIB_DIR)
    set(MADLIB_LIB_DIR ${LAMMPS_LIB_SOURCE_DIR}/madlib)
  endif()
  add_subdirectory(${MADLIB_LIB_DIR} ${CMAKE_BINARY_DIR}/build-madlib)
  target_link_libraries(lammps PRIVATE madlib)
  target_compile_definitions(lammps PRIVATE -DMADLIB_TURBOGAP)
  if(MADLIB_ENABLE_GPU)
    message(STATUS "MADLIB: soap_turbo backend enabled, SOAP on the device")
  else()
    message(STATUS "MADLIB: soap_turbo backend enabled, SOAP on the host")
  endif()
else()
  message(STATUS "MADLIB: soap_turbo backend off (-D MADLIB_ENABLE_TURBOGAP=on to build it)")
  if(MADLIB_ENABLE_GPU)
    message(WARNING "MADLIB_ENABLE_GPU has no effect without MADLIB_ENABLE_TURBOGAP")
  endif()
endif()
