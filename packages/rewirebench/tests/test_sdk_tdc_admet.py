"""The ADMET group protocol must score the way TDC scores, or not at all."""
import csv
import math
from pathlib import Path

import pytest

from rewirebench.protocols import tdc_admet


def write(directory: Path, name: str, rows):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Drug_ID", "Drug", "Y"])
        writer.writerows(rows)


def regression(tmp_path: Path, dataset="caco2_wang"):
    root = tmp_path / "data" / tdc_admet.GROUP_DIRECTORY / dataset
    write(root, "train_val.csv", [(f"t{i}", "CCO", i / 10) for i in range(6)])
    write(root, "test.csv", [(f"s{i}", "CCN", i / 5) for i in range(5)])
    return tmp_path / "data"


def classification(tmp_path: Path, dataset="hia_hou"):
    root = tmp_path / "data" / tdc_admet.GROUP_DIRECTORY / dataset
    write(root, "train_val.csv", [(f"t{i}", "CCO", i % 2) for i in range(6)])
    write(root, "test.csv", [(f"s{i}", "CCN", i % 2) for i in range(6)])
    return tmp_path / "data"


def test_every_admet_dataset_has_a_metric():
    # The paper's ADMET benchmark group is 22 datasets.
    assert len(tdc_admet.ADMET_METRICS) == 22
    assert set(tdc_admet.ADMET_METRICS.values()) == {
        "mae", "roc-auc", "pr-auc", "spearman",
    }


def test_prepare_reads_the_layout_tdc_writes(tmp_path):
    prepared = tdc_admet.prepare(regression(tmp_path), dataset="caco2_wang")
    assert prepared["protocol_id"] == "tdc-admet-group-v1"
    assert prepared["metadata"]["metric"] == "mae"
    assert prepared["metadata"]["metric_direction"] == "lower"
    assert prepared["metadata"]["canonical_test_count"] == 5
    assert len(prepared["provenance"]["test_sha256"]) == 64
    assert {r["split"] for r in prepared["rows"]} == {"train", "test"}
    assert prepared["rows"][0]["inputs"] == {"smiles": "CCO"}


def test_prepare_refuses_an_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="Unknown ADMET dataset"):
        tdc_admet.prepare(regression(tmp_path), dataset="not_a_dataset")


def test_prepare_refuses_a_missing_split(tmp_path):
    with pytest.raises(FileNotFoundError, match="BenchmarkGroup"):
        tdc_admet.prepare(tmp_path, dataset="caco2_wang")


def test_prepare_refuses_continuous_labels_on_a_classification_dataset(tmp_path):
    root = tmp_path / "data" / tdc_admet.GROUP_DIRECTORY / "hia_hou"
    write(root, "test.csv", [("s0", "CCN", 0.42)])
    with pytest.raises(ValueError, match="not a label"):
        tdc_admet.prepare(tmp_path / "data", dataset="hia_hou")


def test_mae_matches_sklearn(tmp_path):
    from sklearn.metrics import mean_absolute_error

    prepared = tdc_admet.prepare(regression(tmp_path), dataset="caco2_wang")
    predictions = {f"s{i}": i / 5 + 0.1 for i in range(5)}
    report = tdc_admet.score(prepared, predictions)
    expected = mean_absolute_error([i / 5 for i in range(5)], list(predictions.values()))
    assert math.isclose(report["metrics"]["mae"], expected, rel_tol=1e-12)
    assert report["score_direction"] == "lower"
    assert report["complete"] is True


def test_roc_auc_matches_sklearn(tmp_path):
    from sklearn.metrics import roc_auc_score

    prepared = tdc_admet.prepare(classification(tmp_path), dataset="hia_hou")
    predictions = {f"s{i}": 0.9 if i % 2 else 0.1 for i in range(6)}
    report = tdc_admet.score(prepared, predictions)
    expected = roc_auc_score([i % 2 for i in range(6)], list(predictions.values()))
    assert math.isclose(report["metrics"]["roc-auc"], expected, rel_tol=1e-12)


def test_spearman_drops_the_p_value(tmp_path):
    from scipy import stats

    prepared = tdc_admet.prepare(
        regression(tmp_path, "vdss_lombardo"), dataset="vdss_lombardo"
    )
    predictions = {f"s{i}": (5 - i) / 5 for i in range(5)}
    report = tdc_admet.score(prepared, predictions)
    expected = stats.spearmanr([i / 5 for i in range(5)], list(predictions.values()))[0]
    assert math.isclose(report["metrics"]["spearman"], expected, rel_tol=1e-12)


def test_partial_predictions_are_scored_but_not_called_complete(tmp_path):
    prepared = tdc_admet.prepare(regression(tmp_path), dataset="caco2_wang")
    report = tdc_admet.score(prepared, {"s0": 0.0, "s1": 0.2})
    assert report["coverage"] == {
        "scored": 2, "unscored": 3, "denominator": 5, "selected": 5,
    }
    assert report["complete"] is False
    assert report["unscored_reasons"]["missing_prediction"] == 3


def test_unknown_and_nonfinite_predictions_are_refused(tmp_path):
    prepared = tdc_admet.prepare(regression(tmp_path), dataset="caco2_wang")
    with pytest.raises(ValueError, match="Unknown prediction IDs"):
        tdc_admet.score(prepared, {"nope": 1.0})
    with pytest.raises(ValueError, match="Nonfinite"):
        tdc_admet.score(prepared, {"s0": float("nan")})


def test_one_class_leaves_the_metric_unavailable_rather_than_guessing(tmp_path):
    prepared = tdc_admet.prepare(classification(tmp_path), dataset="hia_hou")
    report = tdc_admet.score(prepared, {"s0": 0.1, "s2": 0.2})
    assert report["metrics"]["roc-auc"] is None
    assert "one outcome class" in report["metrics"]["unavailable_reason"]


def test_a_smoke_limit_keeps_the_real_denominator(tmp_path):
    prepared = tdc_admet.prepare(regression(tmp_path), dataset="caco2_wang", limit=2)
    assert prepared["scope"] == "smoke"
    report = tdc_admet.score(prepared, {"s0": 0.0, "s1": 0.2})
    assert report["coverage"]["denominator"] == 5
    assert report["complete"] is False


def test_embeddings_are_refused_with_a_reason(tmp_path):
    prepared = tdc_admet.prepare(regression(tmp_path), dataset="caco2_wang")
    with pytest.raises(NotImplementedError, match="not embeddings"):
        tdc_admet.fit_embeddings(prepared, {})
