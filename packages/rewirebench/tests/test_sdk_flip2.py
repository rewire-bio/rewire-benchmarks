"""Offline contracts plus independently calculated FLIP2 ranking fixtures."""
import copy
import csv

import numpy as np
import pytest
from rewirebench.protocols import flip2
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


@pytest.fixture
def prepared(tmp_path):
    path = tmp_path / "fixture.csv"
    rows = [
        ["AAA", -2, "train", False], ["AAC", 1, "train", False],
        ["ACC", 2, "train", False], ["CCC", 100, "train", True],
        ["AAD", -1, "test", False], ["ADD", 0, "test", False],
        ["DDD", 2, "test", False], ["DDA", 2, "test", False],
    ]
    with path.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "target", "set", "validation"])
        writer.writerows(rows)
    return flip2.prepare(path, dataset="flip2-amylase-one-to-many", allow_unverified=True)


def test_all_declared_splits_are_unique_and_contracted():
    datasets = flip2.describe()["datasets"]
    assert len(datasets) == 16
    assert len({d["dataset"] for d in datasets}) == 7
    assert len({d["dataset_id"] for d in datasets}) == 16
    for entry in datasets:
        contract = flip2.submission_contract(entry["dataset_id"])
        assert contract["metrics"] == ["spearman", "ndcg"]
        assert contract["required_hashes"]["source_csv_sha256"] == entry["csv_sha256"]
        assert sum(entry["counts"].values()) > 0
    with pytest.raises(ValueError):
        flip2.submission_contract("flip2-gb1-one-vs-rest")


def test_unverified_source_rejected_unless_explicit(prepared, tmp_path):
    with pytest.raises(ValueError, match="checksum"):
        flip2.prepare(tmp_path / "fixture.csv", dataset="amylase", split="one_to_many")
    assert prepared["scope"] == "subset"
    assert prepared["provenance"]["data_verification"] == "local_bytes_hashed_not_independently_source_verified"


def test_splits_and_opaque_order(prepared):
    assert len([r for r in prepared["rows"] if r["split"] == "train"]) == 3
    assert len([r for r in prepared["rows"] if r["split"] == "validation"]) == 1
    for row in prepared["rows"]:
        assert len(row["id"]) == 64
        assert set(row["inputs"]) == {"sequence"}
    for split in ("train", "validation", "test"):
        ids = [r["id"] for r in prepared["rows"] if r["split"] == split]
        assert ids == sorted(ids)


def test_rank_metrics_match_upstream_definitions_with_negative_labels_and_ties(prepared):
    test = [r for r in prepared["rows"] if r["split"] == "test"]
    scores = {r["id"]: float(r["inputs"]["sequence"].count("D")) for r in test}
    result = flip2.score(prepared, scores)
    y = np.asarray([r["target"] for r in test])
    p = np.asarray([scores[r["id"]] for r in test])
    assert result["metrics"]["spearman"] == pytest.approx(spearmanr(y, p).statistic)
    assert result["metrics"]["ndcg"] == pytest.approx(ndcg_score((y-y.min())[None, :], p[None, :]))
    assert result["complete"] is False
    assert result["suite_complete"] is False


def test_constant_predictions_and_targets(prepared):
    test = [r for r in prepared["rows"] if r["split"] == "test"]
    result = flip2.score(prepared, {r["id"]: 1 for r in test})
    assert result["metrics"]["spearman"] is None
    assert 0 <= result["metrics"]["ndcg"] <= 1
    for row in test:
        row["target"] = 5
    prepared["metadata"]["prepared_rows_sha256"] = flip2._rows_digest(prepared["rows"])
    result = flip2.score(prepared, {r["id"]: 1 for r in test})
    assert result["metrics"]["ndcg"] == 0
    assert result["metrics"]["spearman"] is None


def test_partial_coverage_and_unknown_ids(prepared):
    test = [r for r in prepared["rows"] if r["split"] == "test"]
    result = flip2.score(prepared, {test[0]["id"]: 1})
    assert result["coverage"] == {"denominator": 4, "selected": 4, "scored": 1, "unscored": 3}
    assert result["metrics"]["ndcg"] is None
    assert result["unscored_reasons"]["missing_prediction"] == 3
    with pytest.raises(ValueError, match="Unknown"):
        flip2.score(prepared, {"unknown": 0})
    for bad in [float("nan"), float("inf"), [1], True]:
        with pytest.raises(ValueError, match="finite scalar"):
            flip2.score(prepared, {test[0]["id"]: bad})


def test_changed_split_duplicate_and_false_full_rejected(prepared):
    for mutate in [
        lambda d: d["rows"][0].update(split="test"),
        lambda d: d["rows"][1].update(id=d["rows"][0]["id"]),
        lambda d: d.update(scope="full"),
    ]:
        changed = copy.deepcopy(prepared)
        mutate(changed)
        with pytest.raises(ValueError):
            flip2.validate_prepared(changed)


def test_embedding_head_train_only_and_dimension_validation(prepared):
    embeddings = {r["id"]: [r["inputs"]["sequence"].count("A"), r["inputs"]["sequence"].count("C")]
                  for r in prepared["rows"]}
    expected = flip2.fit_embeddings(prepared, embeddings)
    changed = copy.deepcopy(prepared)
    for row in changed["rows"]:
        if row["split"] != "train":
            row["target"] = 123456
    changed["metadata"]["prepared_rows_sha256"] = flip2._rows_digest(changed["rows"])
    assert flip2.fit_embeddings(changed, embeddings) == expected
    assert set(expected) == {r["id"] for r in prepared["rows"] if r["split"] == "test"}
    key = next(iter(embeddings))
    for bad in [[], [float("nan"), 0], [1, 2, 3], [[1, 2]]]:
        broken = dict(embeddings)
        broken[key] = bad
        with pytest.raises(ValueError):
            flip2.fit_embeddings(prepared, broken)
    broken = dict(embeddings)
    del broken[key]
    with pytest.raises(ValueError):
        flip2.fit_embeddings(prepared, broken)


def test_pdz3_preserves_delimiter(tmp_path):
    path = tmp_path / "pdz.csv"
    path.write_text("sequence,target,set,validation\nAAA:,0,train,False\nAAA:CCC,1,train,True\nAAD:CCD,2,test,False\n")
    prepared = flip2.prepare(path, dataset="flip2-pdz3-single-to-double", allow_unverified=True)
    assert {r["inputs"]["sequence"] for r in prepared["rows"]} == {"AAA:", "AAA:CCC", "AAD:CCD"}


def test_no_silent_split_conflict(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("sequence,target,set,validation\nAAA,0,test,True\n")
    with pytest.raises(ValueError, match="Test row"):
        flip2.prepare(path, dataset="flip2-amylase-one-to-many", allow_unverified=True)


def test_limit_keeps_canonical_denominator(prepared, tmp_path):
    small = flip2.prepare(tmp_path / "fixture.csv", dataset="flip2-amylase-one-to-many", allow_unverified=True, limit=1)
    assert small["scope"] == "smoke"
    result = flip2.score(small, {})
    assert result["coverage"]["denominator"] == 4
    assert result["coverage"]["selected"] == 1
    assert result["unscored_reasons"] == {"not_selected": 3, "missing_prediction": 1}
