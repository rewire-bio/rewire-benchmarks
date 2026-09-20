"""Shared SDK/API contribution contract; fixtures are not experimental results."""
import copy
import hashlib
import json
from importlib.resources import files

import pytest
from rewirebench import sdk
from rewirebench.sequence_submission import contract
from rewirebench.submission import _validate_bundle

RESOURCE = files("rewirebench").joinpath("resources")
FIXTURES = json.loads(RESOURCE.joinpath("sequence-submission-fixtures.json").read_text())["cases"]


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_shared_sequence_intake_contract(fixture):
    if fixture["valid"]:
        _validate_bundle(fixture["bundle"])
    else:
        with pytest.raises(ValueError):
            _validate_bundle(fixture["bundle"])


def test_contract_matches_all_checked_source_pins():
    from rewirebench.protocols import flip2, mrnabench
    rules = contract()["protocols"]
    assert len(rules[flip2.PROTOCOL_ID]["datasets"]) == 16
    for identity, entry in flip2.DATASETS.items():
        published = rules[flip2.PROTOCOL_ID]["datasets"][identity]
        assert published["verified_provenance"]["source_csv_sha256"] == entry["csv_sha256"]
        assert published["denominator"] == entry["counts"]["test"]
    receipt = json.loads(RESOURCE.joinpath("mrnabench/validation-receipt.json").read_text())
    assert len(rules[mrnabench.PROTOCOL_ID]["datasets"]) == 6
    for item in receipt["checks"]:
        identity = mrnabench._identity(item["dataset"], item["target"])
        published = rules[mrnabench.PROTOCOL_ID]["datasets"][identity]
        assert published["verified_provenance"] == {
            "source_sha256": item["source_sha256"],
            "split_sha256": item["canonical_split_sha256"],
            "dataset_revision": mrnabench.DATA_REVISION,
        }
        assert published["denominator"] == item["canonical_test_count"]


def test_local_mrna_export_preserves_source_and_method(tmp_path):
    import pandas as pd
    from rewirebench.protocols import mrnabench
    path = tmp_path / "local.csv"
    pd.DataFrame({"sequence": ["ACGU" + "A" * i for i in range(20)],
                  "target_mrl_mcherry": list(range(20))}).to_csv(path, index=False)
    data = sdk.prepare(mrnabench.PROTOCOL_ID, source=path, output=tmp_path / "prepared", dataset="mcherry")
    predictions = {r["id"]: r["target"] + 0.1 for r in data["rows"] if r["split"] == "test"}
    report = sdk.evaluate(data, predictions, output=tmp_path / "scores")
    bundle = sdk.export(report, output=tmp_path / "bundle.json")
    _validate_bundle(bundle)
    assert bundle["evaluation_method"] == "imported_predictions"
    assert bundle["data_verification"] == "local_bytes_hashed_not_independently_source_verified"
    assert bundle["provenance"]["dataset_revision"] == mrnabench.DATA_REVISION
    assert bundle["scope"] == "subset" and bundle["completion"] == "partial"
    embeddings = {r["id"]: [float(i), 1.0] for i, r in enumerate(data["rows"])}
    report = sdk.evaluate(data, embeddings=embeddings, output=tmp_path / "embeddings")
    bundle = sdk.export(report, output=tmp_path / "embedding-bundle.json")
    _validate_bundle(bundle)
    assert bundle["evaluation_method"] == "frozen_embedding_probe"
    assert bundle["execution_status"] == "imported_embeddings"


def test_dart_local_subset_export_retains_pair_coverage(tmp_path):
    import h5py
    import numpy as np
    from rewirebench.protocols import dart_eval
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("test")
        group["seqs"] = np.eye(4, dtype=np.uint8)[np.zeros((6, 350), dtype=int)]
        group["ctrls"] = np.eye(4, dtype=np.uint8)[np.ones((6, 350), dtype=int)]
        group["idxs"] = list(range(6))
    data = sdk.prepare(dart_eval.PROTOCOL_ID, source=path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), output=tmp_path / "prepared")
    values = {r["id"]: float(r["target"]) for r in data["rows"]}
    report = sdk.evaluate(data, values, output=tmp_path / "evaluated")
    bundle = sdk.export(report, output=tmp_path / "export.json")
    _validate_bundle(bundle)
    assert bundle["metrics"]["n"] == 12
    assert bundle["metrics"]["n_pairs"] == 6
    assert bundle["metrics"]["pairs_denominator"] == 6
    assert bundle["scope"] == "subset"
    assert bundle["completion"] == "partial"
    bad = copy.deepcopy(bundle)
    bad["provenance"]["private_path"] = "a" * 64
    with pytest.raises(ValueError, match="allowlisted"):
        _validate_bundle(bad)
