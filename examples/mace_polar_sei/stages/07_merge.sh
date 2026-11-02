#!/bin/bash
# Every system's selection as one, names prefixed by system (grace_sei_na's merge).
set -eu
source "$(dirname "$0")/lib.sh"
PARAMS=$MP_EX/../grace_sei_na/params/combined.sh bash "$MP_EX/../grace_sei_na/stages/06_merge.sh"
