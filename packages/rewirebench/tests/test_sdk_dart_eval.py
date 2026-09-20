"""DART contract tests use synthetic data, not a biological execution claim."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from rewirebench.protocols import dart_eval as dart


def demo():
    return dart.prepare(Path("demo"))


def predictions(data):
    return {r["id"]: float(r["source_index"] - 3) if r["pair_role"] == "element" else 0.0 for r in data["rows"]}


def test_upstream_resources_are_pinned():
    index = json.loads(dart._resource("sources.json").read_text())
    assert index["upstream_revision"] == dart.UPSTREAM_REVISION
    for item in index["files"]:
        assert hashlib.sha256(dart._resource(item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_pinned_evaluate_method_execution_receipt_parity():
    """Reference values were executed using upstream code, not reimplemented."""
    reference = json.loads(dart._resource("upstream-evaluator-reference.json").read_text())
    assert reference["scipy"] == "1.12.0"
    data = demo()
    values = {r["id"]: reference["scores" if r["pair_role"] == "element" else "controls"][r["source_index"]] for r in data["rows"]}
    observed = dart.score(data, values)
    for key, value in reference["expected_metrics"].items():
        assert observed["metrics"][key] == pytest.approx(value, rel=1e-14, abs=1e-14)
    assert observed["metrics"]["acc"] == 8 / 12
    assert observed["metrics"]["n"] == 24
    assert observed["metrics"]["n_pairs"] == 12
    assert not observed["complete"]


def test_missing_pair_member_and_smoke_denominators():
    data = dart.prepare(Path("demo"), limit=3)
    values = predictions(data)
    missing = next(r for r in data["rows"] if r["pair_role"] == "control")
    values.pop(missing["id"])
    report = dart.score(data, values)
    assert report["coverage"] == {"denominator": 24, "selected": 6, "scored": 5, "unscored": 19}
    assert report["pair_coverage"] == {"denominator": 12, "selected": 3, "scored": 2, "unscored": 10}
    assert report["unscored_pair_reasons"] == {"not_selected": 9, "missing_control": 1}
    assert report["metrics"]["acc"] == 0
    assert not report["complete"]


def test_all_ties_and_no_predictions_have_explicit_undefined_statistics():
    data = demo()
    tied = dart.score(data, dict.fromkeys((r["id"] for r in data["rows"]), 0.0))
    assert tied["metrics"]["acc"] == 0
    assert tied["metrics"]["pval"] is None
    assert "undefined" in tied["metrics"]["unavailable_reason"]
    missing = dart.score(data, {})
    assert missing["metrics"]["acc"] is None
    assert missing["unscored_pair_reasons"]["missing_both"] == 12
    json.dumps(tied, allow_nan=False)
    json.dumps(missing, allow_nan=False)


@pytest.mark.parametrize("mutation", [
    lambda d: d["rows"].append(copy.deepcopy(d["rows"][0])),
    lambda d: d["rows"][0]["inputs"].update(pair_role="element"),
    lambda d: d["rows"][0].update(split="train"),
    lambda d: d["rows"][0].update(target=7),
    lambda d: d["metadata"].update(shuffle_seed=1),
    lambda d: d.update(scope="full"),
    lambda d: d["metadata"].update(canonical_test_count=25),
    lambda d: d["rows"].pop(),
])
def test_rejects_invalid_preparation(mutation):
    data = demo()
    mutation(data)
    with pytest.raises(ValueError):
        dart.validate_prepared(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, [1, 2], "2"])
def test_invalid_scores(value):
    data = demo()
    with pytest.raises(ValueError):
        dart.score(data, {data["rows"][0]["id"]: value})


def test_unknown_prediction_and_dataset():
    with pytest.raises(ValueError):
        dart.score(demo(), {"unknown": 1.0})
    with pytest.raises(ValueError):
        dart.submission_contract("other")


def test_hdf5_layout_hash_and_duplicate_indices(tmp_path):
    import h5py
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as out:
        grp = out.create_group("test")
        grp["seqs"] = np.eye(4, dtype=np.uint8)[np.zeros((3, 350), dtype=int)]
        grp["ctrls"] = np.eye(4, dtype=np.uint8)[np.ones((3, 350), dtype=int)]
        grp["idxs"] = [11, 27, 90]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source_sha256"):
        dart.prepare(path)
    with pytest.raises(ValueError, match="source_sha256"):
        dart.prepare(path, source_sha256="0" * 64)
    data = dart.prepare(path, source_sha256=sha)
    assert data["scope"] == "subset"
    assert data["metadata"]["canonical_test_count"] == 6
    assert data["provenance"]["data_verification"] == "local_bytes_hashed_not_independently_source_verified"
    with h5py.File(path, "a") as out:
        out["test/idxs"][1] = 11
    with pytest.raises(ValueError, match="unique"):
        dart.prepare(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_private_adapter_offline_and_only_allowlisted_inputs(tmp_path, monkeypatch):
    import socket

    from rewirebench import sdk
    monkeypatch.setitem(sdk.PROTOCOLS, dart.PROTOCOL_ID, "dart_eval")
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Network access"))
    prepared = sdk.prepare(dart.PROTOCOL_ID, source="demo", output=tmp_path / "prepared")

    class Private:
        def predict(self, inputs):
            assert all(set(row) == {"id", "sequence"} for row in inputs)
            assert all(len(row["id"]) == 64 for row in inputs)
            # This local toy score is not a published baseline or a biological result.
            return {row["id"]: float(row["sequence"].count("AAA")) for row in inputs}

    report = sdk.run(prepared, Private(), output=tmp_path / "run")
    assert report["coverage"]["scored"] == 24
    assert report["metrics"]["n_pairs"] == 12
    assert report["completion"] == "partial"


def test_frozen_statistics_match_actual_scipy_112_receipt():
    receipt = json.loads(dart._resource("statistical-reference.json").read_text())
    assert receipt["scipy"] == "1.12.0"
    for case in receipt["cases"]:
        statistic, pvalue = dart._wilcoxon_112(np.asarray(case["differences"], dtype=float))
        assert statistic == case["signed_rank_sum"]
        assert pvalue == pytest.approx(case["pval"], rel=1e-14, abs=1e-14)


def test_control_generation_is_dinucleotide_preserving_demo():
    from collections import Counter
    fixture = json.loads(dart._resource("demo.json").read_text())
    for pair in fixture["pairs"]:
        assert Counter(zip(pair["sequence"], pair["sequence"][1:])) == Counter(zip(pair["control"], pair["control"][1:]))
