"""Offline, archived-score/native-container parity; no model inference or training."""

import csv
import json
from pathlib import Path
from rewirebench.protocols import mfass

root = Path(__file__).resolve().parents[1]
with (root / "benchmarks/mfass/results/baseline-kmer-position-v2.predictions.tsv").open() as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
dataset = {
    "protocol_id": "mfass-v2",
    "scope": "full",
    "metadata": {"canonical_test_count": 8324},
    "rows": [
        {"id": r["id"], "split": "test", "target": int(r["label"]), "group": r["group"]}
        for r in rows
    ],
}
result = mfass.score(dataset, {r["id"]: float(r["score"]) for r in rows})
print(json.dumps(result["metrics"], sort_keys=True, allow_nan=False))
