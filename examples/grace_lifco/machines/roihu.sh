# Roihu CPU (x86_64): LAMMPS + grace/fs for production MD; CSC's gcc/openmpi modules.
export CSC_ENV_INIT_NON_INTERACTIVE=yes
export LMOD_TERM_WIDTH=80
# CSC's profile and Lmod reference unset variables and return non-zero.
set +eu
source /etc/profile
module load gcc/15.2.0 openmpi/5.0.10 >/dev/null 2>&1
set -eu
PREFIX=${PREFIX:-/scratch/project_2003566/zarrout1/madlib-grace}
NPROC=${NPROC:-16}
UV=${UV:-$HOME/.local/bin/uv}
# No conda toolchain here; the system prefix makes the link flags harmless.
TOOLCHAIN_PREFIX=/usr
