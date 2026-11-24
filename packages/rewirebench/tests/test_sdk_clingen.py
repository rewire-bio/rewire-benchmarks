"""SDK protocol and adapter tests for ClinGen PP3/BP4 local posterior calibration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
from rewirebench import sdk
from rewirebench.adapters.clingen import ScoreTableLookup
from rewirebench.protocols.clingen_pp3_bp4 import PROTOCOL_ID

ROOT = Path(__file__).resolve().parents[3]
CLINGEN_DIR = ROOT / "benchmarks/clingen-pp3-bp4"
COHORTS_DIR = CLINGEN_DIR / "cohorts"
SCORES_DIR = CLINGEN_DIR / "scores"
SUMMARY_PATH = CLINGEN_DIR / "results/calibration_summary.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_describe_protocol_lists_clingen_strata():
    info = sdk.describe(PROTOCOL_ID)
    assert info["protocol_id"] == PROTOCOL_ID
    assert info["capabilities"] == {
        "prediction_types": ["scalar"],
        "allow_fit": False,
        "allow_validation": False,
        "embedding_kind": None,
    }
    assert info["default_stratum"] == "post_2023_12_af001"
    assert "post_2023_12_af001" in info["strata"]


def test_prepare_and_smoke_hide_labels_from_adapter(tmp_path: Path):
    prepared = sdk.prepare(
        PROTOCOL_ID,
        source=COHORTS_DIR,
        output=tmp_path / "prepared",
        stratum="post_2023_12_af001",
        limit=10,
    )
    assert prepared["scope"] == "smoke"
    assert len(prepared["rows"]) == 30
    seen_inputs: list[dict[str, Any]] = []

    class CaptureAdapter:
        def predict(self, inputs: list[dict[str, Any]]) -> dict[str, float]:
            seen_inputs.extend(inputs)
            return {row["id"]: float(index) for index, row in enumerate(inputs)}

    report = sdk.run(prepared, CaptureAdapter(), output=tmp_path / "smoke_run")
    assert report["scope"] == "smoke"
    assert len(seen_inputs) == 30
    for item in seen_inputs:
        assert set(item) == {"id", "chrom", "pos", "ref", "alt"}
        assert "target" not in item
        assert "role" not in item
        assert "label" not in item
    with pytest.raises(ValueError, match="Smoke tests cannot be exported"):
        sdk.export(report, output=tmp_path / "bundle.json")


def test_score_table_lookup_rejects_bad_sha256():
    revel_path = SCORES_DIR / "scores_revel.tsv"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ScoreTableLookup(str(revel_path), "0" * 64, "revel")


@pytest.mark.parametrize(
    ("score_file", "column", "summary_key", "increment", "decimals"),
    [
        ("scores_revel.tsv", "revel", "revel", 0.001, 3),
        ("scores_avi.tsv", "avi", "alphagenome_avi", 0.005, 3),
        ("scores_gpnstar_v.tsv", "gpnstar_v", "gpnstar_v", 0.01, 2),
    ],
)
def test_full_protocol_matches_published_calibration_summary(
    tmp_path: Path,
    score_file: str,
    column: str,
    summary_key: str,
    increment: float,
    decimals: int,
):
    summary = json.loads(SUMMARY_PATH.read_text())["strata"]["post_2023_12_af001"]
    expected = summary["avi" if summary_key == "alphagenome_avi" else summary_key]
    prepared = sdk.prepare(
        PROTOCOL_ID,
        source=COHORTS_DIR,
        output=tmp_path / f"prepared_{column}",
        stratum="post_2023_12_af001",
        increment=increment,
        decimals=decimals,
    )
    assert prepared["scope"] == "full"
    tsv = SCORES_DIR / score_file
    adapter = ScoreTableLookup(str(tsv), _sha256(tsv), column)
    report = sdk.run(
        prepared,
        adapter,
        output=tmp_path / f"run_{column}",
        allow_partial=True,
        batch_size=4096,
    )
    metrics = report["metrics"]
    assert metrics["n_pos"] == expected["n_pos"]
    assert metrics["n_neg"] == expected["n_neg"]
    assert metrics["n_ref"] == expected["n_g_ref"]
    assert math.isclose(metrics["auroc"], expected["auroc"], abs_tol=1e-12)
    assert math.isclose(metrics["auprc"], expected["auprc"], abs_tol=1e-12)
    disc_p = expected["discounted_thresholds"]["PP3"]
    disc_b = expected["discounted_thresholds"]["BP4"]
    for key, metric_key in (
        ("Supporting_1", "pp3_supporting"),
        ("Moderate_2", "pp3_moderate"),
        ("Intermediate_3", "pp3_intermediate"),
        ("Strong_4", "pp3_strong"),
    ):
        assert metrics[metric_key] == pytest.approx(disc_p[key], abs=1e-12)
    for key, metric_key in (
        ("Supporting_1", "bp4_supporting"),
        ("Moderate_2", "bp4_moderate"),
        ("Intermediate_3", "bp4_intermediate"),
        ("Strong_4", "bp4_strong"),
    ):
        assert metrics[metric_key] == pytest.approx(disc_b[key], abs=1e-12)
