#!/usr/bin/env bash
# CPU smoke for the ProteinGym run guide (guides/discovery-benchmark-proteingym.json).
# Inputs are shrunk to three small assays: the DMS archive (43 MB) is downloaded whole, but only
# those three assays are unpacked, and their score files are taken from the 1.9 GB score archive
# with HTTP range requests. Both scoring scripts skip assays whose files are absent.
# The guide gives no environment install; the pinned environments/proteingym_env.txt is used.
set -euo pipefail
REV=144fe22b07dfaeec2b366f2346203a9838a55b4c
ASSAYS=(NPC1_HUMAN_Erwood_2022_RPE1 B2L11_HUMAN_Dutta_2010_binding-Mcl-1 OPSD_HUMAN_Wan_2019)
WORK=${WORK:-$RUNNER_TEMP/proteingym}
HERE=$(cd "$(dirname "$0")" && pwd)
STARTED=$(date +%s)
CACHE="$HOME/.cache/ProteinGym"   # default PROTEINGYM_CACHE in scripts/zero_shot_config.sh

# Step 1: check out the reviewed repository.
git clone --quiet https://github.com/OATML-Markslab/ProteinGym.git "$WORK"
cd "$WORK"
git checkout --quiet --detach "$REV"

# Prerequisite: a conda environment named proteingym_env (merge_all_scores.sh activates it).
source "$CONDA/etc/profile.d/conda.sh"
conda create -q -y -n proteingym_env python=3.10 >/dev/null
conda activate proteingym_env
pip install -q -r environments/proteingym_env.txt
pip freeze > "$RUNNER_TEMP/proteingym-freeze.txt"
conda deactivate
export PATH="$CONDA/bin:$PATH"   # so `source activate` in the upstream script resolves

# Step 2: download the two documented resource archives (shrunk to three assays).
mkdir -p "$CACHE"
BASE=https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3
curl -fsSL -o "$CACHE/DMS_ProteinGym_substitutions.zip" "$BASE/DMS_ProteinGym_substitutions.zip"
GLOBS=()
MEMBERS=()
for a in "${ASSAYS[@]}"; do
  GLOBS+=(--member-glob "$a.csv")
  MEMBERS+=("DMS_ProteinGym_substitutions/$a.csv")
done
(cd "$CACHE" && unzip -q -o DMS_ProteinGym_substitutions.zip "${MEMBERS[@]}")
mkdir -p "$CACHE/zero_shot_substitutions_scores"
python3 "$HERE/rangezip.py" "$BASE/zero_shot_substitutions_scores.zip" \
  --dest "$CACHE/zero_shot_substitutions_scores" "${GLOBS[@]}"
find "$CACHE" -name '*.csv' | sed "s#$CACHE/##" | sort

# Step 3: merge scores and calculate substitution metrics.
cd scripts/scoring_DMS_zero_shot
bash merge_all_scores.sh
bash performance_substitutions.sh
cd "$WORK"
OUT=benchmarks/DMS_zero_shot/substitutions
find "$OUT" -newer "$RUNNER_TEMP/proteingym-freeze.txt" -type f | sort | tee "$RUNNER_TEMP/proteingym-outputs.txt"
mapfile -t FILES < "$RUNNER_TEMP/proteingym-outputs.txt"
python3 "$HERE/receipt.py" --recipe discovery-benchmark-proteingym --revision "$REV" \
  --started "$STARTED" --out "$RUNNER_TEMP/receipt-proteingym.json" \
  --deviation "Three assays (${ASSAYS[*]}) instead of all 217; score files taken by HTTP range." \
  --deviation "Environment created from environments/proteingym_env.txt with Python 3.10; the guide gives no install step." \
  "${FILES[@]}" "$RUNNER_TEMP/proteingym-freeze.txt"
