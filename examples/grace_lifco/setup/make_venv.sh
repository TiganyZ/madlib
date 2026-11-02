#!/bin/bash
# The python environment, from the locked requirements.txt.
set -eu
source "$(dirname "$0")/../env.sh"

lock() {
    "$UV" pip compile --python-version "$PYTHON_VERSION" --generate-hashes \
        "$EX/setup/requirements.in" -o "$EX/setup/requirements.txt"
}

create() {
    "$UV" venv --clear --python "$PYTHON_VERSION" "$VENV_DIR"
    "$UV" pip install --python "$PY" --require-hashes -r "$EX/setup/requirements.txt"
}

check() {
    "$PY" -c "import tensorpotential, pyace, ase; print('tensorpotential', tensorpotential.__version__)"
    "$PY" -c "import tensorflow as tf; print('tf', tf.__version__, tf.config.list_physical_devices())"
}

[ -f "$EX/setup/requirements.txt" ] || lock
create
check
