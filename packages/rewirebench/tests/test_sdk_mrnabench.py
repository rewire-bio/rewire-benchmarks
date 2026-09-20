import copy
import csv

import pytest
from rewirebench.protocols import mrnabench as m
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split


def fixture(tmp_path, dataset="egfp", target="target_mrl_egfp_unmod", limit=None):
    path = tmp_path / "sample.csv"
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", target])
        writer.writeheader()
        for i in range(40):
            writer.writerow({"sequence": "A"*(i+1)+"CGTU", target: "" if i == 3 else i/3})
    options = {"dataset": dataset, "target": target}
    if limit is not None:
        options["limit"] = limit
    return m.prepare(path, **options)


@pytest.mark.parametrize("dataset,target", [(d,t) for d,ts in m.TARGETS.items() for t in ts])
def test_every_target(tmp_path, dataset, target):
    data = fixture(tmp_path, dataset, target)
    m.validate_prepared(data)
    assert data["metadata"]["excluded"] == [{"source_index": 3, "reason": "missing_target"}]
    assert data["provenance"]["data_verification"] == "local_bytes_hashed_not_independently_source_verified"
    assert all(set(r["inputs"]) == {"sequence"} for r in data["rows"])
    score = m.score(data, {r["id"]: r["target"] for r in data["rows"] if r["split"] == "test"})
    assert score["metrics"]["mse"] == 0
    assert score["metrics"]["pearson"] == pytest.approx(1)
    assert not score["complete"] and not score["suite_complete"]
    assert score["scope"] == "subset"
    assert m.submission_contract(data["dataset_id"])["metrics"] == ["mse", "pearson", "spearman"]


def test_split_and_probe_match_reference(tmp_path):
    data = fixture(tmp_path)
    train, tv = train_test_split([i for i in range(40) if i != 3], test_size=.3, random_state=2541)
    val, test = train_test_split(tv, test_size=.5, random_state=2541)
    assert data["metadata"]["split_membership"] == {"train": train, "validation": val, "test": test}
    embeddings = {r["id"]: [len(r["inputs"]["sequence"]), r["inputs"]["sequence"].count("A")**2] for r in data["rows"]}
    result = m.fit_embeddings(data, embeddings)
    training = [r for r in data["rows"] if r["split"] == "train"]
    testing = [r for r in data["rows"] if r["split"] == "test"]
    reference = RidgeCV(alphas=[.001,.01,.1,1.,10.]).fit([embeddings[r["id"]] for r in training], [r["target"] for r in training])
    assert list(result.values()) == pytest.approx(reference.predict([embeddings[r["id"]] for r in testing]))
    changed = copy.deepcopy(data)
    for row in changed["rows"]:
        if row["split"] != "train":
            row["target"] += 10000
    assert m.fit_embeddings(changed, embeddings) == result


def test_partial_undefined_and_invalid(tmp_path):
    data = fixture(tmp_path, limit=3)
    result = m.score(data, {r["id"]: 1 for r in data["rows"] if r["split"] == "test"})
    assert result["scope"] == "smoke" and not result["complete"]
    assert result["coverage"]["denominator"] == 6
    assert result["metrics"]["pearson"] is None
    assert result["metrics"]["spearman"] is None
    assert result["unscored_reasons"]["not_selected"] == 3
    assert m.score(data, {})["metrics"]["mse"] is None
    with pytest.raises(ValueError, match="Unknown"):
        m.score(data, {"unrecognized": 1})
    with pytest.raises(ValueError, match="finite"):
        m.score(data, {data["rows"][0]["id"]: float("nan")})
    with pytest.raises(ValueError, match="Select target"):
        m.prepare(tmp_path / "sample.csv", dataset="egfp")
    with pytest.raises(ValueError, match="checksum"):
        m.prepare(tmp_path / "sample.csv", dataset="egfp", target="target_mrl_egfp_unmod", require_official=True)
    with pytest.raises(ValueError, match="Unsupported"):
        m.submission_contract("mrnabench-sample-mrl-egfp-unrecognized")


@pytest.mark.parametrize("kind", ["duplicate", "split", "hash", "dimension", "nonfinite"])
def test_validation_guards(tmp_path, kind):
    data = fixture(tmp_path)
    embeddings = {r["id"]: [1., 2.] for r in data["rows"]}
    if kind == "duplicate":
        data["rows"][1]["id"] = data["rows"][0]["id"]
    elif kind == "split":
        data["rows"][0]["split"] = "test"
    elif kind == "hash":
        data["metadata"]["split_membership"]["train"].append(1234)
    elif kind == "dimension":
        embeddings[data["rows"][0]["id"]] = [1.]
    else:
        embeddings[data["rows"][0]["id"]] = [float("inf"), 2.]
    with pytest.raises(ValueError):
        m.fit_embeddings(data, embeddings)


def test_local_source_cannot_claim_full_official_scope(tmp_path):
    data = fixture(tmp_path)
    data["scope"] = "full"
    with pytest.raises(ValueError, match="pinned source"):
        m.validate_prepared(data)
