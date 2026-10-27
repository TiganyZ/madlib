#!/bin/bash
# alt: the LSC-MACE environment (mace-scf's LocalSplitCharges). mace-scf is tied to mace 0.3.14,
# so it cannot share the PolarMACE venv.
set -eu
PREFIX=${PREFIX:-/l/zarrout1/mace-scf}
UV=${UV:-$HOME/.local/bin/uv}
TORCH=${TORCH:-2.8.0}
CUDA_INDEX=${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}
MACE=0.3.14
MACE_SCF_REPO=https://github.com/ACEsuit/mace-scf.git
MACE_SCF_COMMIT=784a510339aacc4412b61b6926ef1a92bca387c0
GRAPH_LONGRANGE=git+https://github.com/WillBaldwin0/graph_electrostatics.git

checkout() {
    [ -d "$PREFIX/mace-scf/.git" ] || git clone -q "$MACE_SCF_REPO" "$PREFIX/mace-scf"
    git -C "$PREFIX/mace-scf" checkout -q "$MACE_SCF_COMMIT"
}

make_venv() {
    [ -x "$PREFIX/venv/bin/python" ] || "$UV" venv -q -p 3.12 --python-preference only-managed "$PREFIX/venv"
    VIRTUAL_ENV=$PREFIX/venv "$UV" pip install -q --index-strategy unsafe-best-match \
        --extra-index-url "$CUDA_INDEX" "torch==$TORCH" "mace-torch[schedulefree]==$MACE" "$GRAPH_LONGRANGE" \
        -e "$PREFIX/mace-scf"
}

mkdir -p "$PREFIX"
checkout
make_venv
"$PREFIX/venv/bin/python" -c "import mace, mace_scf, graph_longrange, torch; print('mace', mace.__version__, 'torch', torch.__version__, torch.cuda.is_available())"
