#!/usr/bin/env bash
# Run the frozen annotation-matched MFASS specialist study, one step at a time:
# S0, S1, P0, P1, verification, the three contrasts, then the report.
# Rerunning resumes: finished steps are re-verified and skipped, and an interrupted
# condition continues from its checkpoint. The first failure stops the sequence.
#
# Usage: run_matched_study.sh MANIFEST [ENV_FILE]
# May be run from any directory; it changes to the repository root first. ENV_FILE
# is sourced before that; it must set MFASS_STUDY_PYTHON (an environment with mfass
# installed) and should place TMPDIR and caches on the study disk.
set -euo pipefail
manifest_arg=${1:?usage: run_matched_study.sh MANIFEST [ENV_FILE]}
env_arg=${2:-}
absolute() { (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd -P)" "$(basename "$1")"); }
manifest=$(absolute "$manifest_arg")
[[ -f $manifest ]] || { echo "manifest not found: $manifest_arg" >&2; exit 2; }
if [[ -n $env_arg ]]; then
  env_file=$(absolute "$env_arg")
  # shellcheck disable=SC1090
  source "$env_file"
fi
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
cd "$repo_root"
python=${MFASS_STUDY_PYTHON:?set MFASS_STUDY_PYTHON}
# One value per line: paths may contain spaces.
{ IFS= read -r out_dir; IFS= read -r cpu_threads; } < <("$python" -c '
import json, sys
m = json.load(open(sys.argv[1]))
print(m["out_dir"]); print(m["threads"]["cpu_threads"])' "$manifest")
[[ -n $out_dir && $cpu_threads =~ ^[1-9][0-9]*$ ]] || { echo "unreadable manifest" >&2; exit 2; }
# Bound the orchestrating process as well; each step's own settings come from the manifest.
for key in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS VECLIB_MAXIMUM_THREADS \
           NUMEXPR_NUM_THREADS; do
  export "$key=$cpu_threads"
done
mkdir -p "$out_dir/logs"
echo "$(date -u +%FT%TZ) launching $manifest from $repo_root (TMPDIR=${TMPDIR:-unset})" \
  | tee -a "$out_dir/logs/all.log"
# caffeinate keeps macOS awake while the study runs; elsewhere run without it.
keep_awake=()
if command -v caffeinate >/dev/null; then keep_awake=(caffeinate -i -s); fi
${keep_awake[@]+"${keep_awake[@]}"} "$python" -m mfass.matched_study all --manifest "$manifest" 2>&1 \
  | tee -a "$out_dir/logs/all.log"
