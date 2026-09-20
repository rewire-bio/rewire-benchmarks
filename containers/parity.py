"""Offline native/container scoring checks; synthetic probes are not scientific results.

Legacy environments retain the archived MFASS check. The sequence environment also
exercises all three new protocol readers, private adapters and embedding imports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path

from rewirebench.protocols import mfass


def archived_mfass():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/mfass/results/baseline-kmer-position-v2.predictions.tsv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    dataset = {"protocol_id": "mfass-v2", "scope": "full",
               "metadata": {"canonical_test_count": 8324},
               "rows": [{"id": r["id"], "split": "test", "target": int(r["label"]),
                         "group": r["group"]} for r in rows]}
    return mfass.score(dataset, {r["id"]: float(r["score"]) for r in rows})["metrics"]


def compare(left, right, path="root"):
    """Reject missing metrics and scope drift; permit bounded floating-point drift."""
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"Different fields at {path}")
        for key in left:
            compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise AssertionError(f"Different list shape at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{path}[{index}]")
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        if not isinstance(right, (int, float)) or isinstance(right, bool) or not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(f"Numeric mismatch at {path}: {left} vs {right}")
    elif left != right:
        raise AssertionError(f"Mismatch at {path}: {left!r} vs {right!r}")


def sequence_checks():
    import h5py
    import numpy as np
    import pandas as pd
    from rewirebench import sdk
    from rewirebench.adapters.sequence import SequenceComposition
    from rewirebench.protocols import flip2

    def summary(report):
        return {key: report[key] for key in ("metrics", "coverage", "scope", "completion")}

    with tempfile.TemporaryDirectory(prefix="rewire-sequence-parity-") as folder:
        directory = Path(folder)
        # FLIP2's native four-column file; never mistaken for an official split.
        flip = directory / "flip.csv"
        with flip.open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sequence", "set", "validation", "target"])
            for i in range(40):
                split = "train" if i < 24 else "validation" if i < 32 else "test"
                writer.writerow(["ACDE"*(i % 5 + 1)+"A"*(i+1)+"G"*(i % 3 + 1),
                                 split, split == "validation", i/20 + math.sin(i)])
        selection = next(key for key, entry in flip2.DATASETS.items() if entry["dataset"] != "pdz3")
        flip_data = sdk.prepare("flip2-fitness-v1", source=flip, output=directory/"flip-prepared",
                                dataset=selection, allow_unverified=True)
        flip_result = sdk.run(flip_data, SequenceComposition("ACDEFGHIKLMNPQRSTVWY"),
                              output=directory/"flip-run", prediction_type="embedding")

        # Parquet exercises optional Arrow, not only a CSV fallback.
        mrna = directory / "mrna.parquet"
        pd.DataFrame({"sequence": ["ACGT"*(i % 4+1)+"A"*(i+1) for i in range(60)],
                      "target_mrl_designed": [i/10+math.sin(i) for i in range(60)]}).to_parquet(mrna)
        mrna_data = sdk.prepare("mrnabench-sample-mrl-v1", source=mrna, output=directory/"mrna-prepared",
                                dataset="designed")
        adapter = SequenceComposition()
        mrna_result = sdk.run(mrna_data, adapter, output=directory/"mrna-run", prediction_type="embedding")
        vectors = adapter.embed([dict(row["inputs"], id=row["id"]) for row in mrna_data["rows"]])
        embedding_file = directory/"embeddings.json"
        embedding_file.write_text(json.dumps(vectors))
        imported = sdk.evaluate(mrna_data, embeddings=embedding_file, output=directory/"mrna-imported")
        compare(summary(mrna_result), summary(imported))

        # HDF5 exercises upstream-shaped sequence pairs without restricted data.
        dart = directory/"dart.h5"
        elements, controls = [], []
        for i in range(12):
            elements.append(np.eye(4, dtype=np.uint8)[[2]*(20+i*2)+[0]*(330-i*2)])
            controls.append(np.eye(4, dtype=np.uint8)[[2]*(23+i)+[0]*(327-i)])
        with h5py.File(dart, "w") as handle:
            group = handle.create_group("test")
            group.create_dataset("seqs", data=elements)
            group.create_dataset("ctrls", data=controls)
            group.create_dataset("idxs", data=np.arange(12))
        dart_data = sdk.prepare("dart-eval-task1-zero-shot-v1", source=dart,
                                output=directory/"dart-prepared",
                                source_sha256=hashlib.sha256(dart.read_bytes()).hexdigest())

        class PrivateGCAdapter:
            def predict(self, inputs):
                assert all(set(row) == {"id", "sequence"} for row in inputs)
                return {row["id"]: row["sequence"].count("G")/len(row["sequence"]) for row in inputs}

        dart_result = sdk.run(dart_data, PrivateGCAdapter(), output=directory/"dart-run")
        results = {"flip2_synthetic": summary(flip_result), "mrnabench_synthetic": summary(mrna_result),
                   "dart_synthetic": summary(dart_result), "imported_embeddings_match": True}
        assert all(results[key]["scope"] != "full" for key in ("flip2_synthetic", "mrnabench_synthetic", "dart_synthetic"))
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["core", "esm", "dnabert2", "sequence"], default="core")
    parser.add_argument("--compare", nargs=2, metavar=("NATIVE_JSON", "CONTAINER_JSON"))
    args = parser.parse_args()
    if args.compare:
        compare(*(json.loads(Path(path).read_text()) for path in args.compare))
        print("Parity passed (absolute/relative tolerance 1e-9)")
        return
    result = {"mfass_archived": archived_mfass()}
    if args.environment == "sequence":
        result.update(sequence_checks())
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
