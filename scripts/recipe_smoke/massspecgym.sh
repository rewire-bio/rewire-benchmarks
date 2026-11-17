#!/usr/bin/env bash
# CPU smoke for the MassSpecGym run guide (guides/discovery-benchmark-massspecgym.json).
# The tutorial script is taken verbatim from the guide; only its Trainer line is changed to
# one epoch over a few batches, so the run reaches trainer.test in CI time. Data loading is
# unchanged (the full dataset and candidate file are downloaded).
set -euo pipefail
REV=f259fe3780d5bd227fc6ece36ce6f397c2eef716
WORK=${WORK:-$RUNNER_TEMP/massspecgym}
HERE=$(cd "$(dirname "$0")" && pwd)
GUIDE="$HERE/guides/discovery-benchmark-massspecgym.json"
STARTED=$(date +%s)

# Step 1: check out the reviewed repository.
git clone --quiet https://github.com/pluskal-lab/MassSpecGym.git "$WORK"
cd "$WORK"
git checkout --quiet --detach "$REV"

# Step 2: install into the recommended environment.
source "$CONDA/etc/profile.d/conda.sh"
conda create -q -y -n massspecgym python==3.11 >/dev/null
conda activate massspecgym
pip install -q massspecgym
pip freeze > "$RUNNER_TEMP/massspecgym-freeze.txt"
grep -i '^massspecgym==' "$RUNNER_TEMP/massspecgym-freeze.txt"

# Step 3: the documented script, written from the guide, with a smaller Trainer.
python3 - "$GUIDE" <<'PY'
import json, re, sys
shell = json.load(open(sys.argv[1]))["steps"][2]["shell"]
script = re.search(r"<<'PY'\n(.*)\nPY", shell, re.S).group(1)
documented = 'trainer = Trainer(accelerator="cpu", devices=1, max_epochs=5)'
assert documented in script, "guide script changed"
smoke = ('trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1, limit_train_batches=20, '
         'limit_val_batches=5, limit_test_batches=5, log_every_n_steps=5)')
open("massspecgym_retrieval_example.py", "w").write(script.replace(documented, smoke) + "\n")
PY
/usr/bin/time -v python massspecgym_retrieval_example.py 2>&1 | tee "$RUNNER_TEMP/massspecgym-run.log"
grep -E "Maximum resident|Elapsed" "$RUNNER_TEMP/massspecgym-run.log"
# trainer.test prints the retrieval metrics table; require it.
grep -E "test_hit_rate@|test_mces_dist@|test_.*hit" "$RUNNER_TEMP/massspecgym-run.log" | head -20
grep -q -E "test_hit_rate@" "$RUNNER_TEMP/massspecgym-run.log"
grep -E "test_" "$RUNNER_TEMP/massspecgym-run.log" > "$RUNNER_TEMP/massspecgym-test-metrics.txt"
python3 "$HERE/receipt.py" --recipe discovery-benchmark-massspecgym --revision "$REV" \
  --started "$STARTED" --out "$RUNNER_TEMP/receipt-massspecgym.json" \
  --deviation "Trainer limited to 1 epoch, 20 training, 5 validation and 5 test batches (guide: 5 full epochs)." \
  --note "massspecgym installed unpinned from PyPI as documented; resolved versions in massspecgym-freeze.txt." \
  "$RUNNER_TEMP/massspecgym-test-metrics.txt" "$RUNNER_TEMP/massspecgym-freeze.txt"
