import sys
from pathlib import Path

import pytest
from mfass.compare import load_predictions, main
from mfass.run_pangolin import _max_abs_score


def test_duplicate_missing_prediction_refused(tmp_path):
    p = tmp_path / "p.tsv"
    p.write_text("id\tgroup\tlabel\tscore\nx\tg\t1\t\nx\tg\t1\t0.5\n")
    with pytest.raises(ValueError, match="unique"):
        load_predictions(p)


@pytest.mark.parametrize("score", ["nan", "inf", "-inf"])
def test_nonfinite_refused(tmp_path, score):
    p = tmp_path / "p.tsv"
    p.write_text(f"id\tgroup\tlabel\tscore\nx\tg\t1\t{score}\n")
    with pytest.raises(ValueError, match="Nonfinite"):
        load_predictions(p)


def test_paired_metadata_disagreement(tmp_path, monkeypatch):
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    a.write_text("id\tgroup\tlabel\tscore\nx\tg\t1\t0.5\ny\th\t0\t0.2\n")
    b.write_text("id\tgroup\tlabel\tscore\nx\tg\t0\t0.5\ny\th\t0\t0.2\n")
    monkeypatch.setattr(sys, "argv", ["compare", "--baseline", str(a), "--candidate", str(b)])
    with pytest.raises(ValueError, match="disagree"):
        main()


def test_existing_outputs_refused_before_model_import(tmp_path, monkeypatch):
    from mfass.run_spliceai import main as spliceai
    root = Path(__file__).parents[3]
    cohort = root / "benchmarks/mfass/data/cohort.tsv"
    if not cohort.exists():
        pytest.skip("Source data not distributed")
    p = tmp_path / "result.json"
    p.write_text("preserved")
    monkeypatch.setattr(sys, "argv", ["spliceai", "--cohort", str(cohort),
                                     "--split", str(root / "benchmarks/mfass/splits/split-v2.tsv"),
                                     "--out", str(p)])
    with pytest.raises(FileExistsError):
        spliceai()
    assert p.read_text() == "preserved"


def test_pangolin_invalid_values_are_not_zero_scores():
    assert _max_abs_score("gene|10:nan|11:0.2") is None
    assert _max_abs_score("gene|Warnings:missing") is None
    assert _max_abs_score("gene|10:-0.3|11:0.2") == 0.3


def test_comparison_conflicting_split_manifest(tmp_path):
    import json

    from mfass.compare import check_protocol_metadata
    for name, digest in (("a", "one"), ("b", "two")):
        (tmp_path / f"{name}.json").write_text(json.dumps({"config": {"split_sha256": digest}}))
    with pytest.raises(ValueError, match="split_sha256"):
        check_protocol_metadata(tmp_path / "a.predictions.tsv", tmp_path / "b.predictions.tsv")
