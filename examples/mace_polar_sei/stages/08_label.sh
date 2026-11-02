#!/bin/bash
# The merged selection labelled by VASP on Roihu at OMat24's level (grace_lifco's stage 07).
set -eu
source "$(dirname "$0")/lib.sh"
PARAMS=$MP_EX/../grace_sei_na/params/combined.sh LABEL=vasp bash "$GRACE_STAGES/07_label.sh"
