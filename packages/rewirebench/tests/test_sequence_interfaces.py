import json
from types import SimpleNamespace

import numpy as np
import pytest
from rewirebench import sdk


@pytest.fixture
def supervised(monkeypatch):
    plugin = SimpleNamespace(
        CAPABILITIES={"prediction_types": ["scalar", "embedding"], "allow_fit": True,
                      "allow_validation": True, "embedding_kind": "sequence"},
        score=lambda data, scores: {"metrics": {"mse": 0.0, "n": len(scores)}},
        fit_embeddings=lambda data, vectors: {"test": float(vectors["test"][0])},
    )
    monkeypatch.setattr(sdk, "_plugin", lambda protocol: plugin)
    data = {"protocol_id": "test-sequence", "protocol_version": "1",
            "dataset_id": "fixture", "scope": "full", "rows": [
                {"id": name, "split": split, "inputs": {"sequence": "ACGT"}, "target": target}
                for name, split, target in [("train", "train", 1),
                                             ("validation", "validation", 5),
                                             ("test", "test", 900)]
            ]}
    data["prepared_sha256"] = sdk._digest(data)
    return plugin, data


def test_supervised_adapter_receives_only_allowed_labels(supervised, tmp_path):
    _, data = supervised

    class Model:
        def fit_with_validation(self, inputs, targets, val_inputs, val_targets):
            assert targets == [1] and val_targets == [5]
            assert inputs == [{"id": "train", "sequence": "ACGT"}]
            assert val_inputs == [{"id": "validation", "sequence": "ACGT"}]
            inputs[0]["sequence"] = "changed locally"

        def predict(self, inputs):
            assert inputs == [{"id": "test", "sequence": "ACGT"}]
            return {"test": 3}

    report = sdk.run(data, Model(), output=tmp_path / "run")
    assert data["rows"][0]["inputs"]["sequence"] == "ACGT"
    assert report["execution"]["fitting"] == "train_fit_validation_selection"


def test_zero_shot_refuses_fitting(supervised, tmp_path):
    plugin, data = supervised
    plugin.CAPABILITIES["allow_fit"] = False
    model = SimpleNamespace(fit=lambda *args: pytest.fail("fit called"), predict=lambda x: {})
    with pytest.raises(ValueError, match="Fitting is forbidden"):
        sdk.run(data, model, output=tmp_path / "run")


def test_embedding_import_and_adapter_have_same_scores(supervised, tmp_path):
    _, data = supervised
    vectors = {r["id"]: [float(i), 2.0] for i, r in enumerate(data["rows"])}
    adapter = SimpleNamespace(embed=lambda rows: {r["id"]: np.array(vectors[r["id"]]) for r in rows})
    run = sdk.run(data, adapter, output=tmp_path / "run")
    path = tmp_path / "vectors.npz"
    np.savez(path, ids=np.array(list(vectors)), embeddings=np.array(list(vectors.values())))
    imported = sdk.evaluate(data, embeddings=path, output=tmp_path / "import")
    assert imported["metrics"] == run["metrics"]
    assert imported["execution"]["mode"] == "imported_embeddings"
    assert imported["execution"]["inference_seconds"] is None
    assert not imported["independently_reproduced"]


@pytest.mark.parametrize("vectors", [
    {"train": [1], "test": [2]},
    {"train": [1], "validation": [2, 3], "test": [4]},
    {"train": [1], "validation": [float("nan")], "test": [4]},
    {"train": [True], "validation": [2], "test": [4]},
])
def test_bad_embedding_matrices_are_rejected(supervised, tmp_path, vectors):
    _, data = supervised
    with pytest.raises(ValueError):
        sdk.evaluate(data, embeddings=vectors, output=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_safe_embedding_file_parser(tmp_path):
    path = tmp_path / "unsafe.npz"
    np.savez(path, ids=np.array(["a"], dtype=object), embeddings=np.array([[1.0]]))
    with pytest.raises(ValueError):
        sdk.read_embeddings(path)
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": [1], "a": [2]}')
    with pytest.raises(ValueError, match="Duplicate"):
        sdk.read_embeddings(path)


def test_inspection_exposes_permissions(supervised):
    info = sdk.describe("test-sequence")
    assert json.loads(json.dumps(info))["capabilities"]["allow_validation"] is True


def test_composition_control_rejects_case_duplicate_alphabet():
    from rewirebench.adapters.sequence import SequenceComposition

    with pytest.raises(ValueError, match="unique"):
        SequenceComposition("Aa")
    assert SequenceComposition("acgt").embed([{ "id": "x", "sequence": "ACGTN" }])["x"][-1] == pytest.approx(0.2)
