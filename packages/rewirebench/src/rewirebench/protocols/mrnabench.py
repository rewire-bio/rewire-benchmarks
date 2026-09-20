"""Sequence-only Sample MRL evaluation, independently implemented from methods.

Upstream software is AGPL-3.0; no upstream implementation is distributed here.
Data remain under their original (unreported) terms. This is one target and one
held-out split, never a score for the complete mRNABench catalogue.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from importlib.resources import files
from pathlib import Path

import numpy as np

PROTOCOL_ID = "mrnabench-sample-mrl-v1"
PROTOCOL_VERSION = "1"
UPSTREAM_REVISION = "74f96b8e6ae9f41cc3cccff089d826a62d5604b8"
DATA_REVISION = "ef67f7cf8a999bb1c412ad6551aa7d9f901cbb95"
CAPABILITIES = {"prediction_types": ["scalar", "embedding"], "allow_fit": True,
                "allow_validation": True, "embedding_kind": "sequence"}
TARGETS = {
    "egfp": ("target_mrl_egfp_m1pseudo", "target_mrl_egfp_pseudo", "target_mrl_egfp_unmod"),
    "mcherry": ("target_mrl_mcherry",), "designed": ("target_mrl_designed",),
    "varying": ("target_mrl_varying_length",),
}
DATASETS = tuple(TARGETS)
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]
SEED = 2541


def _pins():
    return json.loads(files("rewirebench").joinpath("resources/mrnabench/sources.json").read_text())


def _identity(dataset, target):
    return f"mrnabench-sample-mrl-{dataset}-{target}"


def describe():
    return {"protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION,
            "capabilities": CAPABILITIES, "datasets": TARGETS,
            "metrics": {"mse": "lower", "pearson": "higher", "spearman": "higher"},
            "split": {"train": 0.7, "validation": 0.15, "test": 0.15, "seed": SEED},
            "input": "Full processed sequence; excludes CDS/splice and derived biological tracks",
            "sources": _pins(), "scope": "One target, held-out test split; not entire mRNABench"}


def submission_contract(dataset_id):
    valid = {_identity(dataset, target) for dataset, targets in TARGETS.items() for target in targets}
    if dataset_id not in valid:
        raise ValueError("Unsupported mRNABench dataset/target identity")
    return {"metrics": ["mse", "pearson", "spearman"], "protocol_version": PROTOCOL_VERSION}


def prepare(source: Path, **options):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    dataset = str(options.get("dataset") or "").removeprefix("mrl-sample-")
    if dataset not in TARGETS:
        raise ValueError(f"Choose an mRNABench Sample dataset: {', '.join(TARGETS)}")
    target = options.get("target")
    if target is None and len(TARGETS[dataset]) == 1:
        target = TARGETS[dataset][0]
    if target not in TARGETS[dataset]:
        raise ValueError(f"Select target from {TARGETS[dataset]}; eGFP targets are scored separately")
    if options.get("seed", SEED) != SEED:
        raise ValueError("This protocol pins split seed 2541; other seeds require a separate protocol")
    limit = options.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    source = Path(source)
    if source.is_dir():
        source = source / f"mrl-sample-{dataset}.parquet"
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    pin = _pins()["datasets"][dataset]
    official = digest == pin["sha256"]
    if options.get("require_official", False) and not official:
        raise ValueError("Input checksum does not match the pinned official dataset")
    if source.suffix == ".parquet":
        frame = pd.read_parquet(source, columns=["sequence", target])
    elif source.suffix == ".csv":
        frame = pd.read_csv(source, usecols=["sequence", target])
    else:
        raise ValueError("Use a parquet or CSV file; executable/pickle formats are not accepted")
    # Source ordinals, not arbitrary dataframe labels, identify split membership.
    frame = frame.reset_index(drop=True)
    missing = frame[target].isna()
    excluded = [{"source_index": int(i), "reason": "missing_target"} for i in frame.index[missing]]
    eligible = frame.loc[~missing]
    values = pd.to_numeric(eligible[target], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Infinite or invalid MRL targets are not supported")
    if len(eligible) < 7:
        raise ValueError("At least seven eligible rows are required for train/validation/test splitting")
    for sequence in eligible["sequence"]:
        if not isinstance(sequence, str) or not sequence or any(b not in "ACGTUNacgtun" for b in sequence):
            raise ValueError("Invalid or missing biological sequence")
    train_indices, heldout = train_test_split(eligible.index.to_numpy(), test_size=0.3, random_state=SEED)
    validation_indices, test_indices = train_test_split(heldout, test_size=0.5, random_state=SEED)
    assignments = {"train": train_indices, "validation": validation_indices, "test": test_indices}
    rows = []
    split_membership = {}
    for split, indices in assignments.items():
        split_membership[split] = [int(i) for i in indices]
        # Preserve the split's upstream order privately. Adapter ordering is opaque.
        selected = indices if limit is None else indices[:limit]
        split_rows = [{"id": secrets.token_hex(32), "split": split, "source_index": int(i),
                       "inputs": {"sequence": frame.at[i, "sequence"]},
                       "target": float(frame.at[i, target])} for i in selected]
        rows.extend(sorted(split_rows, key=lambda row: row["id"]))
    split_hash = hashlib.sha256(json.dumps(split_membership, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION,
            "dataset_id": _identity(dataset, target), "scope": "smoke" if limit is not None else ("full" if official else "subset"),
            "rows": rows,
            "provenance": {"upstream_revision": UPSTREAM_REVISION, "dataset_revision": DATA_REVISION,
                           "source_sha256": digest, "split_sha256": split_hash,
                           "source_url": pin["url"],
                           "data_verification": "pinned_source_bytes" if official else "local_bytes_hashed_not_independently_source_verified",
                           "source_license": "unreported", "upstream_code_license": "AGPL-3.0"},
            "metadata": {"dataset": dataset, "target": target, "task": "regression",
                         "metric": "mse", "metric_direction": "lower", "seed": SEED,
                         "canonical_test_count": len(test_indices), "source_count": len(frame),
                         "eligible_count": len(eligible), "excluded": excluded,
                         "split_membership": split_membership, "split_ratios": [0.7, 0.15, 0.15],
                         "adapter_input_contract": "opaque-sequence-only-v1",
                         "input_context": "Full processed sequence without CDS/splice/derived tracks",
                         "head": {"type": "RidgeCV", "alphas": ALPHAS, "fit_split": "train", "normalization": "none"},
                         "smoke_limit_per_split": limit,
                         "scope_note": "One Sample dataset/target only, not a complete mRNABench score"}}


def validate_prepared(data):
    if data.get("protocol_id") != PROTOCOL_ID or data.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported mRNABench protocol version")
    submission_contract(data["dataset_id"])
    meta = data["metadata"]
    if meta.get("seed") != SEED or meta.get("adapter_input_contract") != "opaque-sequence-only-v1":
        raise ValueError("Invalid mRNABench split/input contract")
    if data["dataset_id"] != _identity(meta.get("dataset"), meta.get("target")):
        raise ValueError("Dataset target does not match identity")
    membership = meta["split_membership"]
    split_hash = hashlib.sha256(json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if split_hash != data["provenance"]["split_sha256"]:
        raise ValueError("Split membership checksum mismatch")
    flat = [i for split in ("train", "validation", "test") for i in membership[split]]
    if len(flat) != len(set(flat)) or len(flat) != meta["eligible_count"]:
        raise ValueError("Duplicate or missing source positions in split membership")
    if len(membership["test"]) != meta["canonical_test_count"]:
        raise ValueError("Test denominator differs from split")
    membership_sets = {split: set(indices) for split, indices in membership.items()}
    seen, positions = set(), set()
    for row in data["rows"]:
        ident = row["id"]
        if not isinstance(ident, str) or len(ident) != 64 or any(c not in "0123456789abcdef" for c in ident) or ident in seen:
            raise ValueError("Duplicate or nonopaque row ID")
        seen.add(ident)
        index = row["source_index"]
        if row["split"] not in membership or index not in membership_sets[row["split"]] or index in positions:
            raise ValueError("Row split mismatch or duplicate source position")
        positions.add(index)
        if set(row["inputs"]) != {"sequence"} or not np.isfinite(float(row["target"])):
            raise ValueError("Invalid model inputs or target")
    if data["scope"] == "full":
        if positions != set(flat):
            raise ValueError("Full preparation cannot omit source rows")
        receipt = json.loads(files("rewirebench").joinpath(
            "resources/mrnabench/validation-receipt.json").read_text())
        canonical = next(row for row in receipt["checks"]
                         if row["dataset"] == meta["dataset"] and row["target"] == meta["target"])
        if (data["provenance"].get("source_sha256") != canonical["source_sha256"]
                or data["provenance"].get("dataset_revision") != DATA_REVISION
                or split_hash != canonical["canonical_split_sha256"]
                or meta["canonical_test_count"] != canonical["canonical_test_count"]):
            raise ValueError("Full mRNABench preparation requires pinned source and canonical split")


def score(data, predictions):
    from scipy.stats import pearsonr, spearmanr

    validate_prepared(data)
    known = {row["id"] for row in data["rows"]}
    if set(predictions) - known:
        raise ValueError("Unknown prediction IDs")
    if any(np.ndim(value) != 0 or not np.isfinite(float(value)) for value in predictions.values()):
        raise ValueError("Predictions must be finite scalars")
    rows = [row for row in data["rows"] if row["split"] == "test"]
    selected = [row for row in rows if row["id"] in predictions]
    y = np.asarray([row["target"] for row in selected], dtype=float)
    p = np.asarray([predictions[row["id"]] for row in selected], dtype=float)
    metrics = {"mse": float(np.mean((p-y)**2)) if len(y) else None, "pearson": None, "spearman": None}
    reasons = {}
    if len(y) >= 2 and np.ptp(y) > 0 and np.ptp(p) > 0:
        metrics.update(pearson=float(pearsonr(p, y).statistic), spearman=float(spearmanr(p, y).statistic))
    else:
        reasons.update(pearson="Fewer than two scored rows or constant values", spearman="Fewer than two scored rows or constant values")
    if not len(y):
        reasons["mse"] = "No scored rows"
    denominator = data["metadata"]["canonical_test_count"]
    return {"protocol_id": PROTOCOL_ID, "scope": data["scope"], "metrics": metrics,
            "metric_unavailable_reasons": reasons,
            "metric_directions": {"mse": "lower", "pearson": "higher", "spearman": "higher"},
            "coverage": {"scored": len(selected), "unscored": denominator-len(selected),
                         "denominator": denominator, "selected": len(rows)},
            "complete": data["scope"] == "full" and len(selected) == denominator,
            "suite_complete": False, "score_direction": "metric-specific",
            "metrics_scope": "Scored test sequences for this dataset and target only",
            "unscored_reasons": {"not_selected": denominator-len(rows), "missing_prediction": len(rows)-len(selected)}}


def fit_embeddings(data, embeddings):
    from sklearn.linear_model import RidgeCV

    validate_prepared(data)
    rows = data["rows"]
    if set(embeddings) != {row["id"] for row in rows}:
        raise ValueError("One embedding per selected sequence is required")
    vectors = []
    for row in rows:
        vector = np.asarray(embeddings[row["id"]], dtype=float)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("Embeddings must be finite nonempty one-dimensional vectors")
        vectors.append(vector)
    if len({vector.shape for vector in vectors}) != 1:
        raise ValueError("Embedding dimensions differ")
    x = np.asarray(vectors)
    train = np.asarray([row["split"] == "train" for row in rows])
    test = np.asarray([row["split"] == "test" for row in rows])
    if train.sum() < 2 or not test.any():
        raise ValueError("Probe needs at least two training rows and a held-out row")
    y_train = [row["target"] for row in rows if row["split"] == "train"]
    # RidgeCV selects alpha internally on train only. No held-out labels or
    # validation feature statistics enter fitting. Do not silently switch solvers.
    model = RidgeCV(alphas=ALPHAS).fit(x[train], y_train)
    return {row["id"]: float(value) for row, value in
            zip([row for row in rows if row["split"] == "test"], model.predict(x[test]))}
