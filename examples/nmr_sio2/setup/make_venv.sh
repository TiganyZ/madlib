#!/bin/bash
# The python environment these models need: torch, e3nn and the authors' fork
# of NequIP, which is what supplies the MagResTensorModel builder their
# config.yaml names. Without the fork the weights cannot be rebuilt into a
# model at all.
#
# ONE VENV PER MODEL FAMILY. This fork pins e3nn below 0.6 and mace-torch
# pins 0.4.4, so a single environment holding both is a coincidence rather
# than a plan. compute mad/property/python reads PYTHONPATH, so pointing it at
# a different venv per deck is the whole of the switch.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

make_venv() {
    mkdir -p "$PREFIX"
    export UV_CACHE_DIR=$PREFIX/uvcache
    uv venv --python "$PYTHON_VERSION" "$VENV"
}

install_torch() {
    uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu torch
}

install_deps() {
    uv pip install --python "$PY" numpy "e3nn<0.6" ase pyyaml tqdm torch-runstats torch-ema
}

# A clone on PYTHONPATH rather than an install. The fork is pure Python, so
# `pip install` buys nothing over this, and nothing in the package is executed
# until a model is actually loaded.
clone_nequip_fork() {
    [ -d "$FORK_DIR/.git" ] || git clone -q --depth 1 "$NEQUIP_FORK" "$FORK_DIR"
    git -C "$FORK_DIR" log --oneline -1
}

report() {
    PYTHONPATH=$FORK_DIR "$PY" - <<'PYEOF'
import ase, e3nn, nequip, torch
print("torch ", torch.__version__)
print("e3nn  ", e3nn.__version__)
print("ase   ", ase.__version__)
print("nequip", nequip.__version__)
from nequip.model import MagResTensorModel  # noqa: F401
print("MagResTensorModel: present")
PYEOF
    echo
    echo "put both of these on PYTHONPATH for a deck that uses backend=nequip_nmr:"
    echo "  $FORK_DIR"
    echo "  $("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
}

export UV_CACHE_DIR=$PREFIX/uvcache
make_venv
install_torch
install_deps
clone_nequip_fork
report
