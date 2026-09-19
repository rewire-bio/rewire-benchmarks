"""Genomic Benchmarks sequence classification, scored on the packaged split.

The package writes each dataset to disk as <dataset>/<split>/<class>/<id>.txt,
one sequence per file, which is what this protocol reads. Upstream assigns the
integer label from the order the filesystem happens to return the class
directories in (dataset_getters/pytorch_datasets.py at the pinned revision,
GenomicClfDataset.__init__). That order is not stable between machines, so the
label here is assigned from the sorted class name instead and the mapping is
recorded in the prepared dataset. A run is reproducible; matching upstream's
integer for a given class is not something either of us can promise.

The paper reports accuracy and F1. Binary F1 uses class index 1 as positive;
comparisons require the same positive class. For a dataset with more than two
classes the paper does not say which averaging it
used, so macro and weighted are both reported and neither is presented as the
paper's number.
"""
from __future__ import annotations

import hashlib
import secrets
from pathlib import Path

import numpy as np

PROTOCOL_ID = "genomic-benchmarks-v2"
PROTOCOL_VERSION = "2"
INPUT_CONTRACT = "random-opaque-ids-sorted-v2"
UPSTREAM_REVISION = "605d8539830e16c85abe7826990958303ffc5e1c"

#: The nine datasets the paper's Table 2 scores.
DATASETS = (
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
    "dummy_mouse_enhancers_ensembl",
    "drosophila_enhancers_stark",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
)


def _directory(source: Path, dataset: str) -> Path:
    for candidate in (source / dataset, source):
        if (candidate / "test").is_dir() and (candidate / "train").is_dir():
            return candidate
    raise FileNotFoundError(
        f"No train and test directories for {dataset} under {source}. "
        f"Download it first: download_dataset('{dataset}')"
    )


def _read_split(directory: Path, split: str, classes: list[str]):
    rows = []
    digest = hashlib.sha256()
    for label, name in enumerate(classes):
        for path in sorted((directory / split / name).glob("*.txt")):
            raw = path.read_bytes()
            sequence = raw.decode("utf-8").strip()
            if not sequence:
                raise ValueError(f"Empty sequence file {path}")
            # Frame the full relative path and original bytes, including class
            # membership and whitespace, so this identifies the local copy.
            for value in (str(path.relative_to(directory)).encode(), raw):
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
            rows.append(
                {
                    # IDs must not reveal class directories, source filenames,
                    # or positions in the class-sorted source traversal.
                    "id": secrets.token_hex(32),
                    "split": split,
                    "inputs": {"sequence": sequence},
                    "target": label,
                }
            )
    if not rows:
        raise ValueError(f"No sequences found in {directory / split}")
    # Sorting independently random IDs gives a label-independent input order.
    # IDs/order are persisted in prepared.json and reused for every later run.
    rows.sort(key=lambda row: row["id"])
    return rows, digest.hexdigest()


def prepare(source: Path, **options):
    dataset = str(options.get("dataset") or "").strip()
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset!r}; choose one of {', '.join(DATASETS)}"
        )
    directory = _directory(Path(source), dataset)
    classes = sorted(
        path.name for path in (directory / "train").iterdir() if path.is_dir()
    )
    expected_classes = 3 if dataset == "human_ensembl_regulatory" else 2
    if len(classes) != expected_classes:
        raise ValueError(f"{dataset} requires {expected_classes} classes on disk")
    test_classes = sorted(
        path.name for path in (directory / "test").iterdir() if path.is_dir()
    )
    if test_classes != classes:
        raise ValueError("Train and test classes differ on disk")

    train_rows, train_sha = _read_split(directory, "train", classes)
    test_rows, test_sha = _read_split(directory, "test", classes)
    rows = train_rows + test_rows

    limit = options.get("limit")
    if limit is not None and (isinstance(limit, bool) or int(limit) != limit or limit < 1):
        raise ValueError("limit must be a positive integer")
    scope = "full"
    if limit:
        rows = train_rows[:limit] + test_rows[:limit]
        scope = "smoke"

    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_id": f"genomic-benchmarks-{dataset}",
        "scope": scope,
        "rows": rows,
        "provenance": {
            "upstream_revision": UPSTREAM_REVISION,
            "train_sha256": train_sha,
            "test_sha256": test_sha,
            "split_origin": "Packaged train and test directories, as downloaded locally",
            "data_verification": "local_bytes_hashed_not_independently_source_verified",
            "source_url": f"https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/tree/{UPSTREAM_REVISION}",
        },
        "metadata": {
            "dataset": dataset,
            "classes": classes,
            "label_mapping": {name: index for index, name in enumerate(classes)},
            "adapter_input_contract": INPUT_CONTRACT,
            "metric": "accuracy",
            "metric_direction": "higher",
            "canonical_test_count": len(test_rows),
            "task": "classification",
            "source_reuse_terms": "Package is Apache-2.0; per-dataset terms are upstream and unreported here",
            "smoke_limit_per_split": limit,
        },
    }


def validate_prepared(dataset):
    """Reject old or relabelled v1 artifacts before adapter code can see them."""
    if (
        dataset.get("protocol_version") != PROTOCOL_VERSION
        or dataset.get("metadata", {}).get("adapter_input_contract") != INPUT_CONTRACT
    ):
        raise ValueError("Genomic Benchmarks inputs require v2 preparation; run prepare again")
    for split in ("train", "test"):
        ids = [row["id"] for row in dataset["rows"] if row["split"] == split]
        if any(
            len(ident) != 64 or any(char not in "0123456789abcdef" for char in ident)
            for ident in ids
        ) or ids != sorted(ids):
            raise ValueError("Genomic Benchmarks requires opaque, ordered v2 IDs; run prepare again")


def _predicted_label(value, classes: int) -> int:
    """A class index, or for a binary dataset a score thresholded at 0.5."""
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Nonfinite prediction")
    if classes == 2 and 0.0 <= number <= 1.0 and number not in (0.0, 1.0):
        return int(number > 0.5)
    if number != int(number) or not 0 <= int(number) < classes:
        raise ValueError(
            f"Prediction {value!r} is neither a class index below {classes} "
            "nor a probability for a binary dataset"
        )
    return int(number)


def score(dataset, predictions):
    from sklearn.metrics import accuracy_score, f1_score

    classes = dataset["metadata"]["classes"]
    rows = [r for r in dataset["rows"] if r["split"] == "test"]
    known = {r["id"] for r in dataset["rows"]}
    unknown = set(predictions) - known
    if unknown:
        raise ValueError(f"Unknown prediction IDs, first: {min(unknown)}")
    selected = [r for r in rows if r["id"] in predictions]
    labels = [r["target"] for r in selected]
    guesses = [_predicted_label(predictions[r["id"]], len(classes)) for r in selected]

    if selected:
        metrics = {
            "accuracy": float(accuracy_score(labels, guesses)),
            "n": len(selected),
        }
        if len(classes) == 2:
            metrics["f1"] = float(f1_score(labels, guesses, zero_division=0))
        else:
            metrics["f1_macro"] = float(
                f1_score(labels, guesses, average="macro", zero_division=0)
            )
            metrics["f1_weighted"] = float(
                f1_score(labels, guesses, average="weighted", zero_division=0)
            )
            metrics["f1_note"] = (
                "The paper does not state which averaging its F1 column uses for "
                "a dataset with more than two classes."
            )
    else:
        metrics = {
            "accuracy": None,
            "n": 0,
            "unavailable_reason": "No scored rows",
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
        "score_direction": "higher",
        "metrics_scope": "scored test sequences only",
        "unscored_reasons": {
            "not_selected": denominator - len(rows),
            "missing_prediction": len(rows) - len(selected),
        },
    }


def fit_embeddings(dataset, embeddings):
    raise NotImplementedError(
        "This protocol scores predicted classes, not embeddings. Use a score "
        "adapter that returns a class index, or a probability for a binary dataset."
    )
