"""Verify local MFASS v2 inputs and execute a training-prior null control.

No downloads, model training, raw-data export or upload. Use a frozen package
snapshot and new output/evidence directories. Source files remain private.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from rewirebench import sdk
from rewirebench.baselines import run_baselines
from rewirebench.protocols import mfass
from sklearn.metrics import average_precision_score, roc_auc_score


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.evidence.mkdir(parents=True, exist_ok=False)
    code_start = sdk._code_digest()
    cohort = args.source / "cohort.tsv"
    expected = {"cohort.tsv": mfass.COHORT_SHA256, "snv_data_clean.txt": mfass.RAW_SHA256,
                "snv_func_annot.txt": mfass.ANNOTATION_SHA256}
    for name, digest in expected.items():
        assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == digest
    split_bytes = args.split.read_bytes()
    assert hashlib.sha256(split_bytes).hexdigest() == mfass.SPLIT_SHA256
    assert mfass._build_cohort(args.source) == cohort.read_bytes()
    raw = list(csv.DictReader(io.StringIO(cohort.read_text()), delimiter="\t"))
    splits = {r["id"]: r for r in csv.DictReader(io.StringIO(split_bytes.decode()), delimiter="\t")}
    assert len(raw) == len(splits) == 27733
    # Independent orientation checks do not invoke the package's pair validator.
    reverse_count = 0
    complement = str.maketrans("ACGT", "TGCA")
    for row in raw:
        ref, mutant, legacy = row["reference_sequence"], row["mutant_sequence"], row["sequence"]
        changed = [i for i, (a, b) in enumerate(zip(ref, mutant, strict=True)) if a != b]
        assert changed == [int(row["rel_position"]) - 1]
        reverse = legacy == mutant.translate(complement)[::-1]
        assert legacy == mutant or reverse
        reverse_count += int(reverse)
    assert reverse_count == 7770
    groups = {part: {r["group"] for r in splits.values() if r["split"] == part} for part in ("train", "test")}
    assert groups["train"].isdisjoint(groups["test"])
    prepared = sdk.prepare("mfass-v2", source=cohort, split=args.split, output=args.output / "prepared")
    batch = run_baselines(prepared, output=args.output / "runs", baseline_ids=["training-prior-v1"])
    assert batch["status"] == "evaluated"
    run = args.output / "runs/training-prior-v1"
    report = json.loads((run / "report.json").read_text())
    predictions = sdk.read_predictions(run / "predictions.json")
    if sdk._digest(predictions) != report.get("predictions_sha256"):
        raise ValueError("Predictions digest differs from the executed report")
    if report["prepared_sha256"] != prepared["prepared_sha256"]:
        raise ValueError("Report prepared digest differs from evaluated inputs")
    if report["environment"]["sdk_code_sha256"] != code_start:
        raise ValueError("Report code digest differs from frozen implementation")
    train = [r for r in raw if splits[r["id"]]["split"] == "train"]
    test = [r for r in raw if splits[r["id"]]["split"] == "test"]
    assert len(train) == 19409 and len(test) == 8324
    prior = sum(int(r["sdv"]) for r in train) / len(train)
    assert set(predictions) == {r["id"] for r in test}
    assert set(predictions.values()) == {prior}
    labels = np.asarray([int(r["sdv"]) for r in test])
    scores = np.asarray([predictions[r["id"]] for r in test])
    assert labels.sum() == 315
    # Recreate protocol's label-independent fixed tie break, not source-order ranking.
    tie_order = np.random.default_rng(0).permutation(len(test))
    chosen = np.argsort(tie_order)[:100]
    independent = {"n": 8324, "positives": 315, "prevalence": float(labels.mean()),
        "capacity": 100, "precision_at_capacity": float(labels[chosen].sum() / 100),
        "recall_at_capacity": float(labels[chosen].sum() / labels.sum()),
        "average_precision_sklearn": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores))}
    for key, value in independent.items():
        assert abs(value - report["metrics"][key]) < 1e-12
    assert report["coverage"] == {"denominator": 8324, "scored": 8324, "unscored": 0}
    sdk.export(report, output=args.evidence / "training-prior.bundle.json")
    write(args.evidence / "training-prior.report.json", report)
    assert code_start == sdk._code_digest()
    write(args.evidence / "verification.json", {
        "schema_version": "1.0", "verified_at": "2026-09-21", "review_method": "automated independent source parsing and metric recomputation",
        "source_files_sha256": {**expected, "split-v2.tsv": mfass.SPLIT_SHA256},
        "cohort_rebuild": "byte-identical from pinned local raw and annotation tables",
        "cohort_count": len(raw), "train_count": len(train), "test_count": len(test),
        "training_positives": sum(int(r["sdv"]) for r in train), "test_positives": int(labels.sum()),
        "reverse_complement_legacy_rows": reverse_count, "split_groups_disjoint": True,
        "training_prior": prior, "metrics": independent, "absolute_tolerance": 1e-12,
        "predictions_sha256": report["predictions_sha256"], "prepared_sha256": report["prepared_sha256"],
        "prediction_digest_verification": "passed", "prepared_and_code_binding": "passed",
        "code_hash_start": code_start, "code_hash_end": sdk._code_digest(),
        "source_retrieval": "existing local source files; no network retrieval",
        "scientific_reproduction": False, "submission_status": "not_submitted",
        "limitation": "All scores tied: top100 is one fixed label-independent tie break, not a ranking ability estimate or expected chance yield",
    })
    print(json.dumps({"training_prior": prior, "metrics": independent, "coverage": report["coverage"]}))


if __name__ == "__main__":
    main()
