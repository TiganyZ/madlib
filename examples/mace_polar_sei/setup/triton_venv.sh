#!/bin/bash
# The MACE venv on Triton (run on a login node): torch with CUDA wheels, mace-torch[polar], torch-dftd.
set -eu
PREFIX=${PREFIX:-/scratch/work/zarrout1/mace-polar}
UV=${UV:-$HOME/.local/bin/uv}
TORCH=${TORCH:-2.8.0}
CUDA_INDEX=${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}
MACE=${MACE:-0.3.17}
FOUNDATIONS=${FOUNDATIONS:-"polar-1-s polar-1-m"}

# Home is a 15 GB quota.
export UV_CACHE_DIR=$PREFIX/uv_cache
export XDG_CACHE_HOME=$PREFIX/cache

make_venv() {
    [ -x "$PREFIX/venv/bin/python" ] || "$UV" venv -q -p 3.12 "$PREFIX/venv"
    VIRTUAL_ENV=$PREFIX/venv "$UV" pip install -q --index-strategy unsafe-best-match \
        --extra-index-url "$CUDA_INDEX" "torch==$TORCH" "mace-torch[polar]==$MACE" torch-dftd
}

fetch_foundations() {
    for name in $FOUNDATIONS; do
        "$PREFIX/venv/bin/python" -c "from mace.calculators import mace_polar; mace_polar(model='$name', device='cpu', return_raw_model=True)"
    done
}

mkdir -p "$PREFIX"
make_venv
fetch_foundations
"$PREFIX/venv/bin/python" -c "import mace, torch, torch_dftd, graph_longrange; print('mace', mace.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
