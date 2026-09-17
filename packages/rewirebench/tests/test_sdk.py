import json
from types import SimpleNamespace
import pytest
from rewirebench import sdk


def data():
    value = {
        "protocol_id": "mfass-v2",
        "protocol_version": "2",
        "dataset_id": "test-fixture",
        "scope": "full",
        "rows": [
            {
                "id": "train-a",
                "split": "train",
                "inputs": {"sequence": "ACGT"},
                "target": 1,
                "group": "a",
            },
            {
                "id": "test-a",
                "split": "test",
                "inputs": {"sequence": "AAAA"},
                "target": 0,
                "group": "b",
            },
            {
                "id": "test-b",
                "split": "test",
                "inputs": {"sequence": "TTTT"},
                "target": 1,
                "group": "c",
            },
        ],
        "provenance": {"source_sha256": "a" * 64, "local_path": "/private/weights"},
    }
    value["prepared_sha256"] = sdk._digest(value)
    return value


@pytest.fixture
def plugin(monkeypatch):
    def score(dataset, values):
        return {"metrics": {"mean": sum(values.values()) / len(values)}}

    plugin = SimpleNamespace(score=score)
    monkeypatch.setattr(sdk, "_plugin", lambda protocol: plugin)
    return plugin


def test_private_adapter_only_sees_allowed_inputs_and_train_labels(tmp_path, plugin):
    seen = []

    class PrivateAdapter:
        def fit(self, inputs, targets):
            assert [r["id"] for r in inputs] == ["train-a"]
            assert targets == [1]

        def predict(self, inputs):
            for row in inputs:
                assert set(row) == {"id", "sequence"}
                seen.append(row["id"])
            return {r["id"]: 0.5 for r in inputs}

    report = sdk.run(data(), PrivateAdapter(), output=tmp_path / "run", batch_size=1)
    assert seen == ["test-a", "test-b"]
    assert report["coverage"] == {"scored": 2, "unscored": 0, "denominator": 2}
    assert report["review_status"] == "unreviewed_local_run"
    bundle = sdk.export(report, output=tmp_path / "bundle.json")
    assert "/private" not in json.dumps(bundle)
    assert "sequence" not in json.dumps(bundle)
    assert "environment" not in bundle
    assert bundle["provenance"]["source_sha256"] == "a" * 64
    assert len(bundle["provenance"]["sdk_code_sha256"]) == 64


@pytest.mark.parametrize(
    "predictions",
    [
        {"test-a": float("nan"), "test-b": 1},
        {"test-a": True, "test-b": 1},
        {"test-a": 0},
        {"test-a": 0, "test-b": 1, "unknown": 1},
        {"test-a": {"score": None}, "test-b": 1},
    ],
)
def test_invalid_predictions_refused(tmp_path, plugin, predictions):
    with pytest.raises(ValueError):
        sdk.evaluate(data(), predictions, output=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_missing_prediction_remains_in_denominator(tmp_path, plugin):
    report = sdk.evaluate(data(), {"test-a": 0}, output=tmp_path / "partial", allow_partial=True)
    assert report["coverage"] == {"denominator": 2, "scored": 1, "unscored": 1}
    assert report["completion"] == "partial"
    assert json.loads((tmp_path / "partial/unscored.json").read_text()) == {
        "test-b": "missing_prediction"
    }


def test_smoke_retains_original_denominator_and_cannot_export(tmp_path, plugin):
    fixture = data()
    fixture.pop("prepared_sha256")
    fixture["scope"] = "smoke"
    fixture["prepared_sha256"] = sdk._digest(fixture)
    plugin.score = lambda *_: {"metrics": {"score": 1}, "coverage": {"denominator": 100}}
    report = sdk.evaluate(fixture, {"test-a": 0, "test-b": 1}, output=tmp_path / "smoke")
    assert report["coverage"]["unscored"] == 98
    with pytest.raises(ValueError, match="Smoke"):
        sdk.export(report, output=tmp_path / "export.json")


def test_no_overwrite_or_prepared_tampering(tmp_path, plugin):
    dest = tmp_path / "existing"
    dest.mkdir()
    with pytest.raises(FileExistsError):
        sdk.evaluate(data(), {}, output=dest)
    changed = data()
    changed["rows"][1]["target"] = 1
    with pytest.raises(ValueError, match="checksum"):
        sdk.evaluate(changed, {}, output=tmp_path / "tampered")


def test_duplicate_prediction_file_ids(tmp_path):
    for filename, contents in [("x.csv", "id,score\na,1\na,2\n"), ("x.json", '{"a":1,"a":2}')]:
        path = tmp_path / filename
        path.write_text(contents)
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            sdk.read_predictions(path)


def test_zero_shot_rejects_training_hook(tmp_path, plugin):
    fixture = data()
    fixture.pop("prepared_sha256")
    fixture["protocol_id"] = "proteingym-v1.3-dms-substitutions"
    fixture["prepared_sha256"] = sdk._digest(fixture)
    adapter = SimpleNamespace(fit=lambda *_: None, predict=lambda *_: {})
    with pytest.raises(ValueError, match="Fitting is forbidden"):
        sdk.run(fixture, adapter, output=tmp_path / "leak")


def test_numpy_adapter_outputs_are_normalized_before_serializing(tmp_path, plugin):
    import numpy as np

    report = sdk.evaluate(
        data(), {"test-a": np.float32(0.25), "test-b": np.int64(1)}, output=tmp_path / "numpy"
    )
    assert report["coverage"]["scored"] == 2
    assert json.loads((tmp_path / "numpy/predictions.json").read_text()) == {
        "test-a": 0.25,
        "test-b": 1.0,
    }


def test_export_preserves_public_checkpoint_evidence_without_paths(tmp_path, plugin):
    from rewirebench.adapters.mfass import DNABERT2

    report = sdk.evaluate(data(), {"test-a": 0, "test-b": 1}, output=tmp_path / "run")
    report["execution"]["adapter_provenance"] = {
        "artifact_sha256": {
            "model.safetensors": DNABERT2.ARTIFACTS["model.safetensors"],
            "/private/model.bin": "0" * 64,
        }
    }
    report["environment"].update({"sif_sha256": "e" * 64, "container_digest": "sha256:" + "f" * 64})
    bundle = sdk.export(report, output=tmp_path / "bundle.json")
    evidence = json.loads((tmp_path / "bundle.evidence.json").read_text())
    assert evidence["public_model_artifacts"] == {
        "model.safetensors": DNABERT2.ARTIFACTS["model.safetensors"]
    }
    assert bundle["provenance"]["evidence_manifest_sha256"] == sdk._digest(evidence)
    assert bundle["provenance"]["sif_sha256"] == "e" * 64
    assert "/private" not in json.dumps(evidence)
    assert bundle["data_verification"] == "unreported"
