"""Audit bindings must reject altered data even when aggregate ranks agree."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from rewirebench import sdk
from scipy.stats import spearmanr

SCRIPT = Path(__file__).resolve().parents[1] / "research/baseline-programme-2026-09-21/run_rhomax.py"
spec = importlib.util.spec_from_file_location("rhomax_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def saved(tmp_path):
    prepared = {"protocol_id": "flip2-fitness-v1", "protocol_version": "1",
                "dataset_id": "fixture", "scope": "full", "prepared_sha256": "a" * 64,
                "rows": [{"id": str(i), "split": "test"} for i in range(3)]}
    predictions = {str(i): float(i) for i in range(3)}
    report = {**{k: v for k, v in prepared.items() if k != "rows"},
              "predictions_sha256": sdk._digest(predictions),
              "environment": {"sdk_code_sha256": "b" * 64}}
    (tmp_path / "predictions.json").write_text(json.dumps(predictions))
    return prepared, report, predictions, tmp_path


def test_original_predictions_verify(saved):
    prepared, report, predictions, directory = saved
    assert audit.validate_saved_run(prepared, directory, report, "b" * 64) == predictions


@pytest.mark.parametrize("shift", [10.0, 0.000001])
def test_rank_equivalent_changed_predictions_are_rejected(saved, shift):
    prepared, report, predictions, directory = saved
    altered = {k: v + shift for k, v in predictions.items()}
    assert spearmanr(list(predictions.values()), list(altered.values())).statistic == 1
    (directory / "predictions.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="Predictions digest"):
        audit.validate_saved_run(prepared, directory, report, "b" * 64)


@pytest.mark.parametrize("field", ["prepared_sha256", "protocol_id", "protocol_version", "scope", "dataset_id"])
def test_all_model_and_control_context_fields_bound(saved, field):
    prepared, report, _, directory = saved
    changed = copy.deepcopy(report)
    changed[field] = "different"
    with pytest.raises(ValueError, match="Report"):
        audit.validate_saved_run(prepared, directory, changed, "b" * 64)


def test_implementation_binding_is_checked(saved):
    prepared, report, _, directory = saved
    with pytest.raises(ValueError, match="code digest"):
        audit.validate_saved_run(prepared, directory, report, "c" * 64)


def test_wrong_checkpoint_and_identity_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"recorded bytes")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {"execution": {"adapter_provenance": {"checkpoint_sha256": digest, "model": "model-a"}},
              "model_configuration": {"checkpoint": "model-a"}}
    assert audit.validate_checkpoint(checkpoint, report, "model-a") == digest
    with pytest.raises(ValueError, match="identity"):
        audit.validate_checkpoint(checkpoint, report, "model-b")
    checkpoint.write_bytes(b"different checkpoint bytes")
    with pytest.raises(ValueError, match="digest"):
        audit.validate_checkpoint(checkpoint, report, "model-a")
