"""FLIP2 archived fitness splits, with source-pinned rank metrics.

The Zenodo v3 files and manuscript contain several count conflicts. We keep the
archived assignments exactly and expose those conflicts, never repair a split
by guessing. Metric definitions follow baselines/aggregate.py at the pinned
revision. This is an independent implementation, not copied AFL-3.0 code.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import secrets
from collections import Counter
from importlib.resources import files
from pathlib import Path

import numpy as np

PROTOCOL_ID = "flip2-fitness-v1"
PROTOCOL_VERSION = "1"
CAPABILITIES = {
    "prediction_types": ["scalar", "embedding"], "allow_fit": True,
    "allow_validation": True, "embedding_kind": "sequence",
}
INPUT_CONTRACT = "opaque-sequence-inputs-v1"
SOURCES = json.loads(files("rewirebench").joinpath("resources/flip2/sources.json").read_text())
UPSTREAM_REVISION = SOURCES["upstream_revision"]
DATASETS = {entry["dataset_id"]: entry for entry in SOURCES["datasets"]}
METRICS = ["spearman", "ndcg"]
PROBE = {
    "name": "rewire-frozen-embedding-ridge-v1", "alpha": 10.0,
    "solver": "auto", "tol": 1e-5, "max_iter": 1000000,
    "fit_intercept": True, "feature_scaling": "none",
    "target_scaling": "StandardScaler fitted only on train labels",
    "validation_use": "none; fixed hyperparameters", "refit": "train only",
    "relationship_to_published_baseline": (
        "Frozen-embedding extension of upstream alpha-10 ridge. The published "
        "baseline uses one-hot features and scales targets using train plus "
        "validation rows. This extension is not that published baseline."
    ),
}


def describe():
    return {
        "protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION,
        "title": "FLIP2 archived v3 fitness splits", "capabilities": CAPABILITIES,
        "datasets": list(DATASETS.values()), "metrics": METRICS,
        "upstream_revision": UPSTREAM_REVISION, "embedding_probe": PROBE,
        "suite_completion": "Each run covers one of 16 splits; never a suite aggregate",
        "access": "Open download; CC-BY-4.0 redistribution, per-dataset attribution packaged",
    }


def submission_contract(dataset_id):
    if dataset_id not in DATASETS:
        raise ValueError("Unknown FLIP2 dataset/split identity")
    return {"metrics": METRICS, "protocol_version": PROTOCOL_VERSION,
            "upstream_revision": UPSTREAM_REVISION,
            "required_hashes": {"source_csv_sha256": DATASETS[dataset_id]["csv_sha256"]},
            "ranges": {"spearman": [-1, 1], "ndcg": [0, 1]}}


def _selection(options):
    name = str(options.get("dataset") or "")
    if name in DATASETS:
        entry = DATASETS[name]
        if options.get("split") not in (None, entry["split"]):
            raise ValueError("dataset_id and split disagree")
        return entry
    matches = [v for v in DATASETS.values() if v["dataset"] == name
               and v["split"] == options.get("split")]
    if len(matches) != 1:
        raise ValueError("Choose an exact FLIP2 dataset_id or dataset and split from describe()")
    return matches[0]


def _rows_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def prepare(source: Path, **options):
    entry = _selection(options)
    source = Path(source)
    candidates = [source] if source.is_file() else [source / entry["path"],
        source / (entry["split"] + ".csv.gz"), source / (entry["split"] + ".csv")]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Download the pinned file first: {entry['url']}")
    raw = path.read_bytes()
    csv_bytes = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    csv_sha = hashlib.sha256(csv_bytes).hexdigest()
    verified = csv_sha == entry["csv_sha256"]
    if not verified and options.get("allow_unverified") is not True:
        raise ValueError("FLIP2 source checksum differs from pinned v3; no split substitution allowed")
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
    if set(reader.fieldnames or []) != {"sequence", "set", "validation", "target"}:
        raise ValueError("Expected FLIP2 sequence,set,validation,target columns")
    grouped = {key: [] for key in ("train", "validation", "test")}
    for record in reader:
        flag, split = record["validation"], record["set"]
        if flag not in ("True", "False") or split not in grouped:
            raise ValueError("Invalid source split or validation flag")
        if split == "test" and flag == "True":
            raise ValueError("Test row cannot also be validation")
        if split == "validation" and flag != "True":
            raise ValueError("Contradictory validation assignment")
        split = "validation" if flag == "True" else split
        sequence = record["sequence"]
        allowed = "ACDEFGHIKLMNPQRSTVWY" + (":" if entry["dataset"] == "pdz3" else "")
        if not sequence or any(c not in allowed for c in sequence):
            raise ValueError("Malformed biological input; source representation must be retained")
        if entry["dataset"] == "pdz3" and sequence.count(":") != 1:
            raise ValueError("PDZ3 input requires its original protein:peptide delimiter")
        target = float(record["target"])
        if not math.isfinite(target):
            raise ValueError("Nonfinite target")
        grouped[split].append({"id": secrets.token_hex(32), "split": split,
                               "inputs": {"sequence": sequence}, "target": target})
    counts = {key: len(value) for key, value in grouped.items()}
    if not all(counts.values()):
        raise ValueError("FLIP2 requires nonempty train, validation and test splits")
    if verified and counts != entry["counts"]:
        raise ValueError("Source split counts do not match the pinned manifest")
    limit = options.get("limit")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    rows = []
    for group in grouped.values():
        group.sort(key=lambda r: r["id"])
        rows.extend(group[:limit] if limit else group)
    result = {
        "protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION,
        "dataset_id": entry["dataset_id"],
        "scope": "smoke" if limit else "full" if verified else "subset",
        "rows": rows,
        "provenance": {
            "upstream_revision": UPSTREAM_REVISION, "source_url": entry["url"],
            "source_sha256": hashlib.sha256(raw).hexdigest(), "source_csv_sha256": csv_sha,
            "dataset_version": SOURCES["dataset_version"],
            "data_verification": "pinned_source_bytes" if verified else "local_bytes_hashed_not_independently_source_verified",
            "split_origin": "Archived assignments: validation=True held out of training",
            "evaluator_url": f"https://github.com/J-SNACKKB/FLIP/blob/{UPSTREAM_REVISION}/baselines/aggregate.py#L131-L135",
        },
        "metadata": {
            "dataset": entry["dataset"], "split": entry["split"],
            "data_verification": "pinned_source_bytes" if verified else "local_bytes_hashed_not_independently_source_verified",
            "adapter_input_contract": INPUT_CONTRACT, "canonical_test_count": counts["test"],
            "canonical_split_counts": counts, "smoke_limit_per_split": limit,
            "prepared_rows_sha256": _rows_digest(rows), "metric": "spearman", "metrics": METRICS,
            "metric_direction": "higher", "task": "regression", "target_units": entry["target_units"],
            "input_representation": entry["input_representation"], "embedding_probe": PROBE,
            "source_reuse_terms": "CC-BY-4.0 archive; source attribution under resources/flip2",
            "source_note": entry["source_note"], "suite_complete": False,
        },
    }
    validate_prepared(result)
    return result


def validate_prepared(data):
    entry = DATASETS.get(data.get("dataset_id"))
    meta = data.get("metadata", {})
    if not entry or data.get("protocol_id") != PROTOCOL_ID or data.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported FLIP2 prepared protocol or dataset")
    if meta.get("adapter_input_contract") != INPUT_CONTRACT:
        raise ValueError("Re-prepare FLIP2 with the opaque input contract")
    rows = data["rows"]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or any(len(i) != 64 or set(i) - set("0123456789abcdef") for i in ids):
        raise ValueError("Duplicate or nonopaque FLIP2 IDs")
    if _rows_digest(rows) != meta.get("prepared_rows_sha256"):
        raise ValueError("Prepared FLIP2 rows or split assignments changed")
    for split in ("train", "validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        if not subset or [r["id"] for r in subset] != sorted(r["id"] for r in subset):
            raise ValueError("FLIP2 needs ordered opaque IDs in each split")
    if any(r["split"] not in ("train", "validation", "test") or set(r["inputs"]) != {"sequence"}
           or not math.isfinite(r["target"]) for r in rows):
        raise ValueError("Invalid FLIP2 prepared rows")
    if data.get("scope") == "full":
        if data["provenance"].get("source_csv_sha256") != entry["csv_sha256"]:
            raise ValueError("Full FLIP2 split requires verified source bytes")
        if dict(Counter(r["split"] for r in rows)) != entry["counts"]:
            raise ValueError("Full FLIP2 split requires every archived row")
    elif data.get("scope") not in ("smoke", "subset"):
        raise ValueError("Invalid FLIP2 scope")
    if meta.get("canonical_test_count", 0) < sum(r["split"] == "test" for r in rows):
        raise ValueError("Invalid FLIP2 denominator")


def score(data, predictions):
    from scipy.stats import spearmanr
    from sklearn.metrics import ndcg_score

    validate_prepared(data)
    known = {r["id"] for r in data["rows"]}
    if set(predictions) - known:
        raise ValueError("Unknown prediction IDs")
    for value in predictions.values():
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(float(value)):
            raise ValueError("Predictions must be finite scalar scores")
    test = [r for r in data["rows"] if r["split"] == "test"]
    selected = [r for r in test if r["id"] in predictions]
    y = np.asarray([r["target"] for r in selected], dtype=float)
    p = np.asarray([predictions[r["id"]] for r in selected], dtype=float)
    metrics = {"spearman": None, "ndcg": None, "n": len(selected)}
    reasons = {}
    if len(y) < 2:
        reasons = {name: "At least two scored test rows required" for name in METRICS}
    else:
        if np.ptp(y) == 0 or np.ptp(p) == 0:
            reasons["spearman"] = "Undefined correlation for constant targets or predictions"
        else:
            metrics["spearman"] = float(spearmanr(y, p).statistic)
        # Upstream shifts the scored test targets by their minimum, then uses
        # sklearn's linear gain, full rank and tie-averaged NDCG defaults.
        metrics["ndcg"] = float(ndcg_score((y - y.min())[None, :], p[None, :]))
    denominator = data["metadata"]["canonical_test_count"]
    return {
        "protocol_id": PROTOCOL_ID, "scope": data["scope"], "metrics": metrics,
        "coverage": {"denominator": denominator, "selected": len(test),
                     "scored": len(selected), "unscored": denominator - len(selected)},
        "complete": data["scope"] == "full" and len(selected) == denominator,
        "suite_complete": False, "score_direction": "higher",
        "metrics_scope": "Scored test rows of this archived split only; no suite aggregate",
        "unavailable_metrics": reasons,
        "unscored_reasons": {"not_selected": denominator - len(test),
                             "missing_prediction": len(test) - len(selected)},
        "protocol_notes": [data["metadata"].get("source_note")] if data["metadata"].get("source_note") else [],
    }


def fit_embeddings(data, embeddings):
    """Fixed ridge on train only; validation/test targets are never read."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    validate_prepared(data)
    rows = data["rows"]
    if set(embeddings) != {r["id"] for r in rows}:
        raise ValueError("Embedding probe requires every selected row exactly once")
    vectors = [np.asarray(embeddings[r["id"]], dtype=float) for r in rows]
    if any(v.ndim != 1 or not v.size or not np.isfinite(v).all() for v in vectors):
        raise ValueError("Embeddings must be nonempty finite one-dimensional vectors")
    if len({v.shape for v in vectors}) != 1:
        raise ValueError("Embedding dimensions differ")
    x = np.asarray(vectors)
    train = np.asarray([r["split"] == "train" for r in rows])
    test = np.asarray([r["split"] == "test" for r in rows])
    y = np.asarray([r["target"] for r in rows if r["split"] == "train"])[:, None]
    scaler = StandardScaler().fit(y)
    head = Ridge(alpha=10, solver="auto", tol=1e-5, max_iter=1000000, fit_intercept=True)
    head.fit(x[train], scaler.transform(y).ravel())
    values = scaler.inverse_transform(head.predict(x[test])[:, None]).ravel()
    return {r["id"]: float(value) for r, value in zip((r for r in rows if r["split"] == "test"), values)}
