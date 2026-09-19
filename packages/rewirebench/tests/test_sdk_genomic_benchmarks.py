"""Genomic Benchmarks scoring, including the parts the paper leaves unstated."""
import math
from pathlib import Path

import pytest

from rewirebench.protocols import genomic_benchmarks as gb


def build(tmp_path: Path, dataset="human_nontata_promoters", classes=("negative", "positive"), n=4):
    root = tmp_path / dataset
    for split in ("train", "test"):
        for name in classes:
            directory = root / split / name
            directory.mkdir(parents=True)
            for i in range(n):
                (directory / f"{i}.txt").write_text("ACGT" * (i + 1))
    return tmp_path


def test_the_nine_paper_datasets_are_named():
    assert len(gb.DATASETS) == 9
    assert "human_ensembl_regulatory" in gb.DATASETS


def test_labels_come_from_the_sorted_class_name(tmp_path):
    prepared = gb.prepare(build(tmp_path), dataset="human_nontata_promoters")
    assert prepared["metadata"]["label_mapping"] == {"negative": 0, "positive": 1}
    assert prepared["metadata"]["canonical_test_count"] == 8
    assert len(prepared["provenance"]["train_sha256"]) == 64


def test_prepare_refuses_an_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        gb.prepare(build(tmp_path), dataset="nope")


def test_prepare_refuses_a_missing_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_dataset"):
        gb.prepare(tmp_path, dataset="human_nontata_promoters")


def test_prepare_refuses_mismatched_classes(tmp_path):
    root = build(tmp_path)
    (root / "human_nontata_promoters" / "test" / "extra").mkdir()
    with pytest.raises(ValueError, match="classes differ"):
        gb.prepare(root, dataset="human_nontata_promoters")


def test_binary_scoring_matches_sklearn(tmp_path):
    from sklearn.metrics import accuracy_score, f1_score

    prepared = gb.prepare(build(tmp_path), dataset="human_nontata_promoters")
    rows = [r for r in prepared["rows"] if r["split"] == "test"]
    predictions = {r["id"]: r["target"] for r in rows}
    predictions[rows[0]["id"]] = 1 - rows[0]["target"]
    report = gb.score(prepared, predictions)
    truth = [r["target"] for r in rows]
    guesses = [predictions[r["id"]] for r in rows]
    assert math.isclose(report["metrics"]["accuracy"], accuracy_score(truth, guesses))
    assert math.isclose(report["metrics"]["f1"], f1_score(truth, guesses))
    assert report["complete"] is True


def test_a_binary_probability_is_thresholded_like_upstream(tmp_path):
    prepared = gb.prepare(build(tmp_path), dataset="human_nontata_promoters")
    rows = [r for r in prepared["rows"] if r["split"] == "test"]
    report = gb.score(prepared, {r["id"]: 0.9 if r["target"] else 0.1 for r in rows})
    assert report["metrics"]["accuracy"] == 1.0


def test_multiclass_reports_both_averages_and_says_why(tmp_path):
    root = build(tmp_path, "human_ensembl_regulatory", ("enhancer", "ocr", "promoter"))
    prepared = gb.prepare(root, dataset="human_ensembl_regulatory")
    rows = [r for r in prepared["rows"] if r["split"] == "test"]
    report = gb.score(prepared, {r["id"]: r["target"] for r in rows})
    assert report["metrics"]["f1_macro"] == 1.0
    assert report["metrics"]["f1_weighted"] == 1.0
    assert "does not state" in report["metrics"]["f1_note"]
    assert "f1" not in report["metrics"]


def test_an_out_of_range_class_is_refused(tmp_path):
    prepared = gb.prepare(build(tmp_path), dataset="human_nontata_promoters")
    rows = [r for r in prepared["rows"] if r["split"] == "test"]
    with pytest.raises(ValueError, match="class index"):
        gb.score(prepared, {rows[0]["id"]: 7})
    with pytest.raises(ValueError, match="Nonfinite"):
        gb.score(prepared, {rows[0]["id"]: float("inf")})


def test_partial_coverage_is_reported_not_hidden(tmp_path):
    prepared = gb.prepare(build(tmp_path), dataset="human_nontata_promoters")
    rows = [r for r in prepared["rows"] if r["split"] == "test"]
    report = gb.score(prepared, {rows[0]["id"]: rows[0]["target"]})
    assert report["coverage"]["scored"] == 1
    assert report["coverage"]["denominator"] == 8
    assert report["complete"] is False


def test_embeddings_are_refused_with_a_reason(tmp_path):
    prepared = gb.prepare(build(tmp_path), dataset="human_nontata_promoters")
    with pytest.raises(NotImplementedError, match="not embeddings"):
        gb.fit_embeddings(prepared, {})
