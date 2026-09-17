"""MFASS protocol identity, privacy, frozen fitting and archived-score regression."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from rewirebench.adapters.mfass import DNABERT2, KmerBaseline, featurise
from rewirebench.protocols import mfass

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/mfass/data/cohort.tsv"
RESULTS = ROOT / "benchmarks/mfass/results"


def test_mfass_rejects_changed_cohort(tmp_path):
    (tmp_path / "cohort.tsv").write_text("id\tsdv\nnot-canonical\t1\n")
    with pytest.raises(ValueError, match="cohort SHA"):
        mfass.prepare(tmp_path)


def test_wrong_split_is_refused_before_cohort(tmp_path):
    (tmp_path / "split-v2.tsv").write_text("different split")
    with pytest.raises(ValueError, match="canonical split"):
        mfass.prepare(tmp_path)


@pytest.fixture(scope="module")
def canonical():
    if not DATA.exists():
        pytest.skip("Local upstream MFASS data is intentionally not distributed")
    return mfass.prepare(DATA)


def test_inputs_are_allowlisted_and_canonical(canonical):
    assert len(canonical["rows"]) == 27733
    assert sum(r["target"] for r in canonical["rows"]) == 1050
    forbidden = {"sdv", "target", "label", "strong_lof", "delta_dpsi", "category", "cadd_score", "sequence"}
    assert all(not forbidden.intersection(r["inputs"]) for r in canonical["rows"])
    assert all(set(r["inputs"]) == set(mfass.INPUT_FIELDS) | {"region"} for r in canonical["rows"])
    train_groups = {r["group"] for r in canonical["rows"] if r["split"] == "train"}
    assert train_groups.isdisjoint(r["group"] for r in canonical["rows"] if r["split"] == "test")


def test_assay_orientation_and_position():
    row = {"id": "x", "reference_sequence": "A" * 170,
           "mutant_sequence": "A" * 40 + "C" + "A" * 129,
           "sequence": "T" * 129 + "G" + "T" * 40,
           "rel_position": "41", "strand": "-", "ref_allele": "T", "alt_allele": "G",
           "legacy_sequence_orientation": "reverse_complement"}
    mfass.validate_pair(row)
    with pytest.raises(ValueError, match="rel_position"):
        mfass.validate_pair({**row, "rel_position": "130"})
    with pytest.raises(ValueError, match="source variant"):
        mfass.validate_pair({**row, "strand": "+"})


@pytest.mark.parametrize("name", ["baseline-kmer-position-v2", "dnabert2-117m-frozen-pair-logreg"])
def test_archived_predictions_reproduce_metrics(canonical, name):
    with (RESULTS / f"{name}.predictions.tsv").open() as fh:
        predictions = {r["id"]: float(r["score"]) for r in csv.DictReader(fh, delimiter="\t")}
    result = mfass.score(canonical, predictions)
    archived = json.loads((RESULTS / f"{name}.json").read_text())
    # TSV prints ten decimals, introducing ties absent from full-precision arrays.
    for key, value in archived["metrics"].items():
        assert result["metrics"][key] == pytest.approx(value, abs=1e-6)
    full_precision = np.load(RESULTS / f"{name}.scores.npy")
    exact = mfass.score(canonical, dict(zip(predictions, full_precision)))
    for key, value in archived["metrics"].items():
        assert exact["metrics"][key] == pytest.approx(value, abs=1e-12)
    assert result["coverage"]["denominator"] == 8324
    assert result["complete"]


def test_smoke_retains_canonical_denominator(canonical):
    dataset = mfass.prepare(DATA, limit=20)
    assert len(dataset["rows"]) == 40
    scores = {r["id"]: 0.5 for r in dataset["rows"] if r["split"] == "test"}
    report = mfass.score(dataset, scores)
    assert report["scope"] == "smoke"
    assert not report["complete"]
    assert report["coverage"] == {"scored": 20, "unscored": 8304, "denominator": 8324, "selected": 20}


def test_bad_scores(canonical):
    with pytest.raises(ValueError, match="Unknown"):
        mfass.score(canonical, {"unknown": 1.0})
    rid = next(r["id"] for r in canonical["rows"] if r["split"] == "test")
    with pytest.raises(ValueError, match="Nonfinite"):
        mfass.score(canonical, {rid: float("nan")})


def _tiny_dataset():
    rows = [{"id": str(i), "split": "train" if i < 8 else "test",
             "target": i % 2, "inputs": {}, "group": str(i)} for i in range(10)]
    vectors = {str(i): {"reference": [float(i), 1.], "mutant": [float(i), float(i % 2)]}
               for i in range(10)}
    return {"rows": rows, "scope": "smoke"}, vectors


def test_fixed_head_never_reads_test_targets():
    dataset, vectors = _tiny_dataset()
    first = mfass.fit_embeddings(dataset, vectors)
    changed = copy.deepcopy(dataset)
    for row in changed["rows"]:
        if row["split"] == "test":
            row["target"] = 99999
    assert first == mfass.fit_embeddings(changed, vectors)
    assert set(first) == {"8", "9"}


def test_embeddings_must_reconcile():
    dataset, vectors = _tiny_dataset()
    del vectors["0"]
    with pytest.raises(ValueError, match="every selected"):
        mfass.fit_embeddings(dataset, vectors)
    dataset, vectors = _tiny_dataset()
    vectors["0"]["mutant"] = [float("inf"), 0.]
    with pytest.raises(ValueError, match="Nonfinite"):
        mfass.fit_embeddings(dataset, vectors)


def test_baseline_is_label_free_and_centered(canonical):
    row = canonical["rows"][0]
    inputs = {**row["inputs"], "id": row["id"]}
    got = featurise([inputs])
    assert got.shape == (1, 85)
    from mfass.run_baseline import featurise as previous
    with DATA.open() as fh:
        original = next(csv.DictReader(fh, delimiter="\t"))
    np.testing.assert_array_equal(got, previous([original]))
    with pytest.raises(ValueError, match="Fit"):
        KmerBaseline().predict([inputs])


def test_checkpoint_checked_before_framework_loading(tmp_path):
    with pytest.raises(ValueError, match="artifact"):
        DNABERT2(tmp_path)


def test_raw_builder_is_byte_identical_and_does_not_modify_source(tmp_path):
    if not (DATA.parent / "snv_data_clean.txt").exists():
        pytest.skip("Local source tables not distributed")
    for name in ("snv_data_clean.txt", "snv_func_annot.txt"):
        (tmp_path / name).symlink_to(DATA.parent / name)
    prepared = mfass.prepare(tmp_path, limit=80)
    assert prepared["provenance"]["cohort_sha256"] == mfass.COHORT_SHA256
    assert prepared["scope"] == "smoke"
    assert not (tmp_path / "cohort.tsv").exists()
