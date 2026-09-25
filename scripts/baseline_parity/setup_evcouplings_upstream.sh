#!/usr/bin/env bash
# Fetch the pinned upstream EVCouplings and ProteinGym sources and build a Python
# environment matching the rewirebench NumPy/SciPy versions, for the same-runtime
# parity test and diagnostic. Usage: setup_evcouplings_upstream.sh DESTINATION
# Afterwards: REWIRE_EVCOUPLINGS_PYTHON=DESTINATION/venv/bin/python and
# REWIRE_EVCOUPLINGS_UPSTREAM=DESTINATION. The receipt script re-checks every
# commit and source hash before running anything.
set -euo pipefail
dest=$1
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$dest"

git init -q "$dest/evcouplings"
git -C "$dest/evcouplings" fetch -q --depth 1 https://github.com/debbiemarkslab/EVcouplings.git \
  e1362407a0b65d63ca07df55f44cb17b0a3722b7
git -C "$dest/evcouplings" checkout -q FETCH_HEAD

git init -q "$dest/proteingym"
git -C "$dest/proteingym" sparse-checkout set --no-cone /config.json /proteingym/baselines/EVmutation/
git -C "$dest/proteingym" fetch -q --depth 1 --filter=blob:none https://github.com/OATML-Markslab/ProteinGym.git \
  144fe22b07dfaeec2b366f2346203a9838a55b4c
git -C "$dest/proteingym" checkout -q FETCH_HEAD

uv venv -q --python 3.11 "$dest/venv"
VIRTUAL_ENV="$dest/venv" uv pip install -q -r "$here/evcouplings-upstream-requirements.txt"
VIRTUAL_ENV="$dest/venv" uv pip install -q --no-deps "$dest/evcouplings"
"$dest/venv/bin/python" -c "import numpy, scipy; print('upstream environment numpy', numpy.__version__, 'scipy', scipy.__version__)"
