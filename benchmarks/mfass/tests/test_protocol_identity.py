"""Reject split drift before loading a checkpoint or reporting v2 scores."""
import csv
from pathlib import Path

import pytest
from mfass.validation import CANONICAL_SPLIT_SHA256, validate_canonical_split

SPLIT = Path("benchmarks/mfass/splits/split-v2.tsv")


def test_checked_in_split_is_the_predeclared_protocol():
    assert validate_canonical_split(SPLIT) == CANONICAL_SPLIT_SHA256
    with SPLIT.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == len({r["id"] for r in rows}) == 27733
    train_groups = {r["group"] for r in rows if r["split"] == "train"}
    test_groups = {r["group"] for r in rows if r["split"] == "test"}
    assert train_groups.isdisjoint(test_groups)
    assert len(test_groups) == 463


def test_rejects_reassigned_rows_even_when_split_sizes_are_preserved(tmp_path):
    with SPLIT.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    train = next(r for r in rows if r["split"] == "train")
    test = next(r for r in rows if r["split"] == "test")
    train["split"], test["split"] = "test", "train"
    altered = tmp_path / "split.tsv"
    with altered.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "group", "split"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="canonical split"):
        validate_canonical_split(altered)
