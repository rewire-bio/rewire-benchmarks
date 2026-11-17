#!/usr/bin/env bash
# CPU smoke for the scIB run guide (guides/discovery-benchmark-scib.json), commands as written,
# on the supplied test-data configuration. The guide's last step allows ten cores; the runner's
# core count is used instead.
set -euo pipefail
REV=e97631a478e063883cee5db0dcebbf42d8268ab0
WORK=${WORK:-$RUNNER_TEMP/scib}
HERE=$(cd "$(dirname "$0")" && pwd)
STARTED=$(date +%s)

# Step 1: check out the reviewed repository.
git clone --quiet https://github.com/theislab/scib-pipeline.git "$WORK"
cd "$WORK"
git checkout --quiet --detach "$REV"

# Step 2: create and activate the documented environments.
source "$CONDA/etc/profile.d/conda.sh"
# Prerequisite "Conda or Mambaforge/Mamba": the workflow provides a user-writable Miniforge,
# which includes the mamba that the setup script calls.
command -v mamba
bash envs/create_conda_environments.sh -r 4.0
conda activate scib-pipeline-R4.0
pip freeze > "$RUNNER_TEMP/scib-freeze.txt"

# Step 3: generate the included test data.
(cd data && python generate_data.py)
# Step 4: inspect scheduled work.
snakemake --configfile configs/test_data-R4.0.yaml -n
# Step 5: run the supplied test-data benchmark.
snakemake --configfile configs/test_data-R4.0.yaml --cores "$(nproc)"

find . -newer "$RUNNER_TEMP/scib-freeze.txt" -type f \( -name '*.csv' -o -name '*.h5ad' \) | sort \
  | tee "$RUNNER_TEMP/scib-outputs.txt"
mapfile -t FILES < "$RUNNER_TEMP/scib-outputs.txt"
python3 "$HERE/receipt.py" --recipe discovery-benchmark-scib --revision "$REV" \
  --started "$STARTED" --out "$RUNNER_TEMP/receipt-scib.json" \
  --deviation "snakemake --cores \$(nproc) instead of 10." \
  data/adata_norm.h5ad "${FILES[@]}"
