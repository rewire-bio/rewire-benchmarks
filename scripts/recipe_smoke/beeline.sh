#!/usr/bin/env bash
# CPU smoke for the BEELINE run guide (guides/discovery-benchmark-beeline.json).
# Shrunk to two of the configured algorithms (PPCOR and GRNBOOST2) on the supplied GSD example:
# a copy of config-files/config.yaml keeps should_run only for those two, and only their images
# are pulled instead of running utils/initialize.sh, which pulls all twelve.
set -euo pipefail
REV=37464085eb8a95d6cc6a3d3a3c649d36db6052ed
KEEP=(PPCOR GRNBOOST2)
WORK=${WORK:-$RUNNER_TEMP/beeline}
HERE=$(cd "$(dirname "$0")" && pwd)
STARTED=$(date +%s)

# Step 1: check out the reviewed repository.
git clone --quiet https://github.com/Murali-group/Beeline.git "$WORK"
cd "$WORK"
git checkout --quiet --detach "$REV"

# Step 2: environment and algorithm images (images for the kept algorithms only).
source "$CONDA/etc/profile.d/conda.sh"
bash utils/setupAnacondaVENV.sh
conda activate BEELINE
pip freeze > "$RUNNER_TEMP/beeline-freeze.txt"
python3 - "${KEEP[@]}" <<'PY'
import sys, yaml
keep = set(sys.argv[1:])
cfg = yaml.safe_load(open("config-files/config.yaml"))
for alg in cfg["input_settings"]["algorithms"]:
    alg["should_run"] = [alg["algorithm_id"] in keep]
images = sorted({a["image"] for a in cfg["input_settings"]["algorithms"] if a["should_run"][0]})
yaml.safe_dump(cfg, open("config-files/config-smoke.yaml", "w"), sort_keys=False)
open("smoke-images.txt", "w").write("\n".join(images) + "\n")
print("images:", images)
PY
while read -r image; do
  docker pull -q "$image"
  docker inspect --format '{{index .RepoDigests 0}}' "$image" >> "$RUNNER_TEMP/beeline-image-digests.txt"
done < smoke-images.txt
cat "$RUNNER_TEMP/beeline-image-digests.txt"

# Step 3: run the configured algorithms.
python BLRunner.py -c config-files/config-smoke.yaml
# Step 4: evaluate and plot ROC/precision-recall performance.
python BLEvaluator.py -c config-files/config-smoke.yaml -a
python BLPlotter.py -c config-files/config-smoke.yaml -o ./plots -a -r

find outputs plots -type f | sort | tee "$RUNNER_TEMP/beeline-outputs.txt"
mapfile -t RANKED < <(find outputs -name rankedEdges.csv | sort)
mapfile -t EVALS < <(find outputs -maxdepth 2 -name '*AUPRC*.csv' -o -maxdepth 2 -name '*AUROC*.csv' | sort)
mapfile -t PLOTS < <(find plots -type f | sort)
[ "${#RANKED[@]}" -eq "${#KEEP[@]}" ] || { echo "expected ${#KEEP[@]} rankedEdges.csv"; exit 1; }
python3 "$HERE/receipt.py" --recipe discovery-benchmark-beeline --revision "$REV" \
  --started "$STARTED" --out "$RUNNER_TEMP/receipt-beeline.json" \
  --deviation "Only ${KEEP[*]} enabled (config-smoke.yaml); only their images pulled, not utils/initialize.sh." \
  --note "Image digests in beeline-image-digests.txt; the guide notes tags are not digest-pinned." \
  "${RANKED[@]}" "${EVALS[@]}" "${PLOTS[@]}" "$RUNNER_TEMP/beeline-image-digests.txt"
