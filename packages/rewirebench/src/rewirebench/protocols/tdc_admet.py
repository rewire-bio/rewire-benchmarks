"""Therapeutics Data Commons ADMET benchmark group, scored as TDC scores it.

The group's own evaluator decides both the metric per dataset and how that
metric is computed. Both are transcribed from upstream at the pinned revision:
the dataset-to-metric map is `admet_metrics` in tdc/metadata.py lines 517-540,
and the metric functions are the ones tdc/evaluator.py lines 385-417 assigns,
with the spearman special case at line 487 that takes the correlation and drops
the p-value. Copyright (c) the TDC authors; upstream is MIT licensed.

Nothing here downloads data. TDC's BenchmarkGroup writes the scaffold split to
disk as train_val.csv and test.csv, and this protocol reads what it wrote and
records the digest of those exact bytes. It does not claim the split matches
any particular TDC release, because that is a property of the copy on disk, and
recording its hash is what lets a reader tell two copies apart.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

PROTOCOL_ID = "tdc-admet-group-v1"
UPSTREAM_REVISION = "c310c35f27e3f506411018ac43d97b8ba23ca652"
GROUP_DIRECTORY = "admet_group"

#: tdc/metadata.py, admet_metrics, lines 517-540 at UPSTREAM_REVISION.
ADMET_METRICS = {
    "caco2_wang": "mae",
    "hia_hou": "roc-auc",
    "pgp_broccatelli": "roc-auc",
    "bioavailability_ma": "roc-auc",
    "lipophilicity_astrazeneca": "mae",
    "solubility_aqsoldb": "mae",
    "bbb_martins": "roc-auc",
    "ppbr_az": "mae",
    "vdss_lombardo": "spearman",
    "cyp2c9_veith": "pr-auc",
    "cyp2d6_veith": "pr-auc",
    "cyp3a4_veith": "pr-auc",
    "cyp2c9_substrate_carbonmangels": "pr-auc",
    "cyp3a4_substrate_carbonmangels": "roc-auc",
    "cyp2d6_substrate_carbonmangels": "pr-auc",
    "half_life_obach": "spearman",
    "clearance_hepatocyte_az": "spearman",
    "clearance_microsome_az": "spearman",
    "ld50_zhu": "mae",
    "herg": "roc-auc",
    "ames": "roc-auc",
    "dili": "roc-auc",
}

#: Lower is better for an error; every other ADMET metric is better higher.
LOWER_IS_BETTER = {"mae"}
CLASSIFICATION = {"roc-auc", "pr-auc"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    if not rows:
        raise ValueError(f"{path} has no rows")
    for column in ("Drug_ID", "Drug", "Y"):
        if column not in rows[0]:
            raise ValueError(f"{path} has no {column} column; expected TDC's ADMET layout")
    return rows, _sha(raw)


def _directory(source: Path, dataset: str) -> Path:
    """Accept the path TDC was given, the group directory, or the dataset itself."""
    for candidate in (
        source / GROUP_DIRECTORY / dataset,
        source / dataset,
        source,
    ):
        if (candidate / "test.csv").exists():
            return candidate
    raise FileNotFoundError(
        f"No test.csv for {dataset} under {source}. Run TDC's BenchmarkGroup first: "
        f"BenchmarkGroup(name='ADMET_Group', path=...).get('{dataset}')"
    )


def prepare(source: Path, **options):
    dataset = str(options.get("dataset") or "").strip().lower()
    if dataset not in ADMET_METRICS:
        raise ValueError(
            f"Unknown ADMET dataset {dataset!r}; choose one of {', '.join(sorted(ADMET_METRICS))}"
        )
    directory = _directory(Path(source), dataset)
    test_rows, test_sha = _read(directory / "test.csv")
    train_path = directory / "train_val.csv"
    train_rows, train_sha = ([], None)
    if train_path.exists():
        train_rows, train_sha = _read(train_path)

    metric = ADMET_METRICS[dataset]
    rows = []
    for split, source_rows in (("train", train_rows), ("test", test_rows)):
        for row in source_rows:
            target = float(row["Y"])
            if metric in CLASSIFICATION and target not in (0.0, 1.0):
                raise ValueError(
                    f"{dataset} is scored with {metric} but Y={row['Y']} is not a label"
                )
            rows.append(
                {
                    "id": row["Drug_ID"],
                    "split": split,
                    "inputs": {"smiles": row["Drug"]},
                    "target": target,
                }
            )
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Drug_ID across the prepared splits")

    limit = options.get("limit")
    if limit is not None and (isinstance(limit, bool) or int(limit) != limit or limit < 1):
        raise ValueError("limit must be a positive integer")
    scope = "full"
    if limit:
        # Deterministic source order, and never selected on the outcome.
        rows = (
            [r for r in rows if r["split"] == "train"][:limit]
            + [r for r in rows if r["split"] == "test"][:limit]
        )
        scope = "smoke"

    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": UPSTREAM_REVISION,
        "dataset_id": f"tdc-admet-{dataset}",
        "scope": scope,
        "rows": rows,
        "provenance": {
            "upstream_revision": UPSTREAM_REVISION,
            "test_sha256": test_sha,
            "train_val_sha256": train_sha,
            "split_origin": "TDC BenchmarkGroup scaffold split, as written to disk locally",
            "source_url": f"https://github.com/mims-harvard/TDC/tree/{UPSTREAM_REVISION}",
        },
        "metadata": {
            "dataset": dataset,
            "metric": metric,
            "metric_direction": "lower" if metric in LOWER_IS_BETTER else "higher",
            "canonical_test_count": len(test_rows),
            "task": "classification" if metric in CLASSIFICATION else "regression",
            "source_reuse_terms": "TDC is MIT licensed; per-dataset terms are upstream and unreported here",
            "smoke_limit_per_split": limit,
        },
    }


def _metric_value(metric: str, labels: np.ndarray, values: np.ndarray) -> float:
    if metric == "mae":
        from sklearn.metrics import mean_absolute_error

        return float(mean_absolute_error(labels, values))
    if metric == "roc-auc":
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, values))
    if metric == "pr-auc":
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(labels, values))
    if metric == "spearman":
        from scipy import stats

        # tdc/evaluator.py line 487 keeps the correlation and drops the p-value.
        return float(stats.spearmanr(labels, values)[0])
    raise ValueError(f"Unsupported ADMET metric {metric!r}")


def score(dataset, predictions):
    metric = dataset["metadata"]["metric"]
    rows = [r for r in dataset["rows"] if r["split"] == "test"]
    known = {r["id"] for r in dataset["rows"]}
    unknown = set(predictions) - known
    if unknown:
        raise ValueError(f"Unknown prediction IDs, first: {sorted(unknown)[0]}")
    selected = [r for r in rows if r["id"] in predictions]
    values = np.asarray([predictions[r["id"]] for r in selected], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite prediction scores")
    labels = np.asarray([r["target"] for r in selected], dtype=float)

    usable = len(selected) > 1 and (
        metric not in CLASSIFICATION or len(set(labels.tolist())) == 2
    )
    if usable:
        metrics = {metric: _metric_value(metric, labels, values), "n": len(selected)}
    else:
        metrics = {
            metric: None,
            "n": len(selected),
            "unavailable_reason": "Too few scored rows, or only one outcome class",
        }

    denominator = dataset["metadata"].get("canonical_test_count", len(rows))
    return {
        "metrics": metrics,
        "coverage": {
            "scored": len(selected),
            "unscored": denominator - len(selected),
            "denominator": denominator,
            "selected": len(rows),
        },
        "protocol_id": dataset["protocol_id"],
        "scope": dataset["scope"],
        "complete": dataset["scope"] == "full" and len(selected) == denominator,
        "score_direction": dataset["metadata"]["metric_direction"],
        "metrics_scope": "scored test rows only",
        "unscored_reasons": {
            "not_selected": denominator - len(rows),
            "missing_prediction": len(rows) - len(selected),
        },
    }


def fit_embeddings(dataset, embeddings):
    raise NotImplementedError(
        "The ADMET group scores predicted property values, not embeddings. "
        "Use a score adapter that returns one value per Drug_ID."
    )
