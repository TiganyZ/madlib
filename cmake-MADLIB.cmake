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
  message(STATUS "MADLIB: soap_turbo backend enabled")
else()
  message(STATUS "MADLIB: soap_turbo backend off (-D MADLIB_ENABLE_TURBOGAP=on to build it)")
endif()
