# alt (sazerac): user-space toolchain; C++ from the system g++, Fortran/MPI/BLAS from conda.
source /l/zarrout1/toolchain/env.sh
export OMPI_CXX=/usr/bin/g++
export CXX=/usr/bin/g++
TOOLCHAIN_PREFIX=/l/zarrout1/toolchain/mamba/envs/tg
PREFIX=${PREFIX:-/l/zarrout1/madlib-grace}
NPROC=${NPROC:-10}
UV=${UV:-$HOME/.local/bin/uv}
AUTOPLEX_ENV=${AUTOPLEX_ENV:-/l/autoplex-soap-turbo_tutorial/env.sh}
JF_PROJECT=${JF_PROJECT:-lico_tigany}
