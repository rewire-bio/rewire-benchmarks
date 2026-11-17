#!/usr/bin/env bash
# CPU smoke for the MFASS v2 run guide (guides/rewire-mfass-v2.json), commands as written.
# The optional DNABERT-2 resource pilot is skipped: the guide marks it optional and it does
# not produce an evaluation.
set -euo pipefail
REV=68d1ee53a0fcd1104d9f427d68dedffdccf2c47f
WORK=${WORK:-$RUNNER_TEMP/mfass}
HERE=$(cd "$(dirname "$0")" && pwd)
STARTED=$(date +%s)

# Step 1: check out the reviewed repository.
git clone --quiet https://github.com/rewire-bio/rewire-benchmarks.git "$WORK"
cd "$WORK"
git checkout --quiet --detach "$REV"

# Step 2: fetch the two published input tables.
mkdir -p benchmarks/mfass/data
curl -fsSL -o benchmarks/mfass/data/snv_data_clean.txt \
  https://raw.githubusercontent.com/KosuriLab/MFASS/master/processed_data/snv/snv_data_clean.txt
curl -fsSL -o benchmarks/mfass/data/snv_func_annot.txt \
  https://raw.githubusercontent.com/KosuriLab/MFASS/master/processed_data/snv/snv_func_annot.txt
# The upstream URLs are mutable; the README records the expected hashes.
sha256sum -c - <<'SUMS'
a637ca0e307e66ff48811ec7efa22b9ce453bc7883b04f0cacb867f7283132d8  benchmarks/mfass/data/snv_data_clean.txt
71a857fe647c4e68acbb41ca61e959c47e1176de89b1442bd6ca1772aa60d5a1  benchmarks/mfass/data/snv_func_annot.txt
999ebcb7e63a5c5eaa8780fa468e59ac1f934260ad50102814174c396317f052  benchmarks/mfass/splits/split-v2.tsv
SUMS

# Step 3: install, validate the cohort and run the corrected baseline.
# The guide requires Python >=3.11. Under 3.12, tokenizers 0.13.3 (from the dnabert2-pilot extra)
# has no wheel and its source build fails, so the lowest documented version is used.
export UV_PYTHON=3.11
uv sync --package mfass --extra dnabert2-pilot
uv run --package mfass --extra dnabert2-pilot mfass-build --check
uv run --package mfass --extra dnabert2-pilot mfass-build
uv run --package mfass --extra dnabert2-pilot mfass-baseline

# The baseline result must match the corrected values published in the README.
OUT=benchmarks/mfass/results/baseline-kmer-position-v2.json
python3 - "$OUT" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["metrics"]
print(m)
assert m["n"] == 8324 and m["positives"] == 315, m
assert round(m["precision_at_capacity"], 3) == 0.610, m
assert round(m["average_precision_sklearn"], 3) == 0.286, m
assert round(m["auroc"], 3) == 0.778, m
PY
uv pip freeze --python .venv/bin/python > "$RUNNER_TEMP/mfass-freeze.txt" 2>/dev/null || true
python3 "$HERE/receipt.py" --recipe rewire-mfass-v2 --revision "$REV" --started "$STARTED" \
  --out "$RUNNER_TEMP/receipt-mfass-v2.json" \
  --deviation "Optional DNABERT-2 resource pilot not run (optional in the guide; measures resources only)." \
  --note "Input tables verified against the README SHA-256 values." \
  --note "Python 3.11 (guide: >=3.11). Under Python 3.12, uv sync fails building tokenizers 0.13.3 from source." \
  "$OUT" "$RUNNER_TEMP/mfass-freeze.txt"
