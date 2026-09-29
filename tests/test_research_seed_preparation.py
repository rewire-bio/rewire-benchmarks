"""Regression checks for connecting historical evaluations to research artifacts."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/prepare-research-seeds.py"
spec = importlib.util.spec_from_file_location("research_seeds", path)
seeds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seeds)


def test_changed_source_never_inherits_old_identity(tmp_path):
    source = tmp_path / "cohort.tsv"
    source.write_text("id\nold\n")
    digest = seeds.sha(source)
    source.write_text("id\nnew\n")
    with pytest.raises(ValueError, match="Artifact drift"):
        seeds.require_hash(source, digest)


def test_duplicate_source_identifiers_are_rejected(tmp_path):
    source = tmp_path / "predictions.tsv"
    source.write_text("id\tscore\na\t1\na\t2\n")
    with pytest.raises(ValueError, match="Duplicate"):
        seeds.tsv(source)


def test_constant_predictor_is_not_a_defined_correlation():
    rows = [{"y": y, "scores": {"mean": 550.0}} for y in [500.0, 550.0, 600.0]]
    metrics = seeds.regression_metrics(rows, "mean")
    assert metrics["spearman"] is None
    assert metrics["pearson"] is None
    assert 0 < metrics["ndcg"] < 1


def test_flip_ndcg_uses_shifted_targets_not_absolute_wavelength():
    rows = [{"y": y, "scores": {"model": p}} for y, p in [(500, 3), (550, 1), (600, 2)]]
    shifted = [{"y": r["y"] + 1000, "scores": r["scores"]} for r in rows]
    assert seeds.regression_metrics(rows, "model")["ndcg"] == pytest.approx(
        seeds.regression_metrics(shifted, "model")["ndcg"], abs=1e-12
    )
