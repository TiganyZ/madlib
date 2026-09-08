# MADLIB - Molecular Augmented Dynamics
#
# Optional Fortran backend for soap_turbo local-property models. The package
# builds and runs without it; only compute mad/property/turbogap needs it.

set(MADLIB_SOURCE_DIR ${LAMMPS_SOURCE_DIR}/MADLIB)
file(GLOB MADLIB_SOURCES ${MADLIB_SOURCE_DIR}/[^.]*.cpp)
target_sources(lammps PRIVATE ${MADLIB_SOURCES})
target_include_directories(lammps PRIVATE ${MADLIB_SOURCE_DIR})

option(MADLIB_ENABLE_TURBOGAP "Build the soap_turbo local-property backend" OFF)
if(MADLIB_ENABLE_TURBOGAP)
  enable_language(Fortran)
  find_package(BLAS REQUIRED)
  add_subdirectory(${LAMMPS_LIB_SOURCE_DIR}/madlib build-madlib)
  target_link_libraries(lammps PRIVATE madlib)
  target_compile_definitions(lammps PRIVATE -DMADLIB_TURBOGAP)
endif()
