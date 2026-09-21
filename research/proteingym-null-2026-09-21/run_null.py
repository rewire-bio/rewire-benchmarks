"""One fixed random-ranking control on the complete AMFR assay, offline.

Prepared inputs and local assay bytes are verified against their prior recorded
hashes, which are observed local hashes, not upstream-published data checksums.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import numpy as np
from rewirebench import sdk
from rewirebench.baselines import run_baselines
from rewirebench.protocols import proteingym as pg
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, roc_auc_score

ASSAY = "AMFR_HUMAN_Tsuboyama_2023_4G3O"


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--assay-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.evidence.exists():
        raise FileExistsError("Use new output and evidence directories")
    prepared = sdk._load(args.prepared, "prepared.json")
    sdk._validate_prepared(prepared)
    assert prepared["protocol_id"] == pg.PROTOCOL_ID and prepared["scope"] == "subset"
    assert len(prepared["rows"]) == 2972
    assert set(prepared["metadata"]["assays"]) == {ASSAY}
    assert prepared["metadata"]["assays"][ASSAY]["expected_count"] == 2972
    assert prepared["provenance"]["data_verification"] == "local_bytes_hashed_not_independently_source_verified"
    source_hash = hashlib.sha256(args.assay_source.read_bytes()).hexdigest()
    assert source_hash == prepared["provenance"]["assay_sha256"][ASSAY]
    with args.assay_source.open() as source:
        raw = list(csv.DictReader(source))
    source_by_id = {ASSAY + "::" + row["mutant"]: row for row in raw}
    assert len(raw) == len(source_by_id) == 2972
    for row in prepared["rows"]:
        original = source_by_id[row["id"]]
        assert row["target"] == float(original["DMS_score"])
        assert row["target_binary"] == int(original["DMS_score_bin"])
        assert row["inputs"]["mutated_sequence"] == original["mutated_sequence"]
    code_start = sdk._code_digest()
    args.evidence.mkdir(parents=True)
    write(args.evidence / "selection.json", {"selected_before_execution": True,
        "baseline_id": "seeded-random-v1", "seed": 0, "assay": ASSAY,
        "selection": "One pre-existing complete assay, matched to the prior ESM-2 evaluation",
        "scope": "one assay; partial ProteinGym track", "seed_search": False})
    batch = run_baselines(prepared, output=args.output, baseline_ids=["seeded-random-v1"])
    assert batch["status"] == "evaluated"
    run = args.output / "seeded-random-v1"
    report = json.loads((run / "report.json").read_text())
    predictions = sdk.read_predictions(run / "predictions.json")
    assert sdk._digest(predictions) == report["predictions_sha256"]
    assert prepared["prepared_sha256"] == report["prepared_sha256"]
    assert code_start == report["environment"]["sdk_code_sha256"] == sdk._code_digest()
    assert set(predictions) == set(source_by_id)
    assert report["scope"] == "subset" and report["completion"] == "partial" and not report["metrics"]
    assert report["coverage"] == {"denominator": 2972, "scored": 2972, "unscored": 0}
    assert report["protocol_results"]["status"] == "partial_track"
    assay_report = report["protocol_results"]["per_assay"][ASSAY]
    assert assay_report["status"] == "complete"
    for ident, value in predictions.items():
        expected = int.from_bytes(hashlib.sha256(f"0:{ident}".encode()).digest()[:8], "big") / 2**64
        assert value == expected
    # Execute pinned upstream NDCG/top-recall functions instead of SDK implementations.
    upstream = pg.resource_path("upstream_performance.py")
    source_ledger = json.loads(pg.resource_path("sources.json").read_text())
    evaluator_receipt = next(r for r in source_ledger if r.get("local_file") == upstream.name)
    assert hashlib.sha256(upstream.read_bytes()).hexdigest() == evaluator_receipt["sha256"]
    spec = importlib.util.spec_from_file_location("pinned_proteingym_performance", upstream)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    labels = np.asarray([row["target"] for row in prepared["rows"]])
    binary = np.asarray([row["target_binary"] for row in prepared["rows"]])
    values = np.asarray([predictions[row["id"]] for row in prepared["rows"]])
    raw_metrics = {"Spearman": float(spearmanr(labels, values).statistic),
                   "AUC": float(roc_auc_score(binary, values)),
                   "MCC": float(matthews_corrcoef(binary, values >= np.median(values))),
                   "NDCG": float(original.calc_ndcg(labels, values)),
                   "Top_recall": float(original.calc_toprecall(labels, values))}
    rounded = {name: float(np.round(value, 3)) for name, value in raw_metrics.items()}
    assert rounded == assay_report["metrics"]
    bundle = sdk.export(report, output=args.evidence / "seeded-random.bundle.json")
    assert bundle["data_verification"] == "local_bytes_hashed_not_independently_source_verified"
    write(args.evidence / "seeded-random.report.json", report)
    write(args.evidence / "verification.json", {
        "review_date": "2026-09-21", "review_method": "automated independent source alignment and pinned-upstream metric execution",
        "assay": ASSAY, "denominator": 2972, "scored": 2972,
        "assay_status": "complete", "track_status": "partial_track", "official_assays": 217,
        "baseline_id": "seeded-random-v1", "seed": 0, "seed_search": False,
        "source_assay_sha256": source_hash, "source_hash_status": "observed_local_bytes_not_upstream_published_checksum",
        "data_verification": bundle["data_verification"], "source_alignment": "all 2972 mutation identities, sequences and labels match prior prepared inputs",
        "prepared_sha256": prepared["prepared_sha256"], "predictions_sha256": report["predictions_sha256"],
        "prediction_digest_verification": "passed", "prepared_and_code_binding": "passed",
        "upstream_evaluator_sha256": evaluator_receipt["sha256"], "upstream_revision": pg.UPSTREAM_REVISION,
        "raw_upstream_metrics": raw_metrics, "rounded_metrics": rounded, "rounded_maximum_absolute_difference": 0.0,
        "code_hash_start": code_start, "code_hash_end": sdk._code_digest(),
        "environment": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
        "scientific_reproduction": False, "submission_status": "not_submitted",
        "limitation": "One seeded random ranking; not a chance-performance confidence interval or a full-suite score",
    })
    print(json.dumps({"rounded_metrics": rounded, "coverage": report["coverage"], "scope": report["scope"]}))


if __name__ == "__main__":
    main()
