"""Genomic Benchmarks scoring, including the parts the paper leaves unstated."""
import json
import math
from pathlib import Path

import pytest
from rewirebench import sdk
from rewirebench.protocols import genomic_benchmarks as gb


def build(tmp_path: Path, dataset="human_nontata_promoters", classes=("negative", "positive"), n=4):
    root = tmp_path / dataset
    for split in ("train", "test"):
        for name in classes:
            directory = root / split / name
            directory.mkdir(parents=True)
            for i in range(n):
                (directory / f"{i}.txt").write_text("ACGT" * (i + 1))
    return tmp_path


def test_the_nine_paper_datasets_are_named():
    assert len(gb.DATASETS) == 9
    assert "human_ensembl_regulatory" in gb.DATASETS


def test_labels_come_from_the_sorted_class_name(tmp_path):
    prepared = gb.prepare(build(tmp_path), dataset="human_nontata_promoters")
    assert prepared["metadata"]["label_mapping"] == {"negative": 0, "positive": 1}
    assert prepared["metadata"]["canonical_test_count"] == 8
    assert len(prepared["provenance"]["train_sha256"]) == 64


def test_prepare_refuses_an_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        gb.prepare(build(tmp_path), dataset="nope")


def test_prepare_refuses_a_missing_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_dataset"):
        gb.prepare(tmp_path, dataset="human_nontata_promoters")


def test_prepare_refuses_mismatched_classes(tmp_path):
    root = build(tmp_path)
    (root / "human_nontata_promoters" / "test" / "extra").mkdir()
    with pytest.raises(ValueError, match="classes differ"):
        gb.prepare(root, dataset="human_nontata_promoters")


def test_binary_scoring_matches_sklearn(tmp_path):
    from sklearn.metrics import accuracy_score, f1_score

    prepared = gb.prepare(build(tmp_path), dataset="human_nontata_promoters")
    rows = [r for r in prepared["rows"] if r["split"] == "test"]
    predictions = {r["id"]: r["target"] for r in rows}
    predictions[rows[0]["id"]] = 1 - rows[0]["target"]
    report = gb.score(prepared, predictions)
    truth = [r["target"] for r in rows]
    guesses = [predictions[r["id"]] for r in rows]
    assert math.isclose(report["metrics"]["accuracy"], accuracy_score(truth, guesses))
    assert math.isclose(report["metrics"]["f1"], f1_score(truth, guesses))
    assert report["complete"] is True


def test_a_binary_probability_is_thresholded_like_upstream(tmp_path):
    prepared = gb.prepare(build(tmp_path), dataset="human_nontata_promoters")
    rows = [r for r in prepared["rows"] if r["split"] == "test"]
    report = gb.score(prepared, {r["id"]: 0.9 if r["target"] else 0.1 for r in rows})
    assert report["metrics"]["accuracy"] == 1.0


def test_multiclass_reports_both_averages_and_says_why(tmp_path):
    root = build(tmp_path, "human_ensembl_regulatory", ("enhancer", "ocr", "promoter"))
    prepared = gb.prepare(root, dataset="human_ensembl_regulatory")
    rows = [r for r in prepared["rows"] if r["split"] == "test"]
    report = gb.score(prepared, {r["id"]: r["target"] for r in rows})
    assert report["metrics"]["f1_macro"] == 1.0
    assert report["metrics"]["f1_weighted"] == 1.0
    assert "does not state" in report["metrics"]["f1_note"]
    assert "f1" not in report["metrics"]


def test_an_out_of_range_class_is_refused(tmp_path):
    prepared = gb.prepare(build(tmp_path), dataset="human_nontata_promoters")
    rows = [r for r in prepared["rows"] if r["split"] == "test"]
    with pytest.raises(ValueError, match="class index"):
        gb.score(prepared, {rows[0]["id"]: 7})
    with pytest.raises(ValueError, match="Nonfinite"):
        gb.score(prepared, {rows[0]["id"]: float("inf")})


def test_partial_coverage_is_reported_not_hidden(tmp_path):
    prepared = gb.prepare(build(tmp_path), dataset="human_nontata_promoters")
    rows = [r for r in prepared["rows"] if r["split"] == "test"]
    report = gb.score(prepared, {rows[0]["id"]: rows[0]["target"]})
    assert report["coverage"]["scored"] == 1
    assert report["coverage"]["denominator"] == 8
    assert report["complete"] is False


def test_embeddings_are_refused_with_a_reason(tmp_path):
    prepared = gb.prepare(build(tmp_path), dataset="human_nontata_promoters")
    with pytest.raises(NotImplementedError, match="not embeddings"):
        gb.fit_embeddings(prepared, {})


def test_adapter_ids_and_batch_order_do_not_reveal_class(tmp_path, monkeypatch):
    # Deliberately interleave the random IDs across source classes. This makes
    # the ordering check deterministic, including at one-record batch size.
    # The train/test IDs must also be globally unique.
    ids = iter(f"{i:064x}" for i in [18, 12, 16, 14, 11, 17, 13, 15, 8, 2, 6, 4, 1, 7, 3, 5])
    monkeypatch.setattr(gb.secrets, "token_hex", lambda _: next(ids))
    prepared = sdk.prepare(
        gb.PROTOCOL_ID, source=build(tmp_path / "source"), dataset="human_nontata_promoters",
        output=tmp_path / "prepared",
    )
    test_rows = [row for row in prepared["rows"] if row["split"] == "test"]
    assert [row["target"] for row in test_rows] == [1, 0, 1, 0, 1, 0, 1, 0]
    seen = []

    class IdOnlyAdapter:
        def predict(self, inputs):
            for row in inputs:
                assert set(row) == {"id", "sequence"}
                assert len(row["id"]) == 64
                assert "/" not in row["id"]
                assert "positive" not in row["id"] and "negative" not in row["id"]
                seen.append(row.copy())
            return {row["id"]: int("/positive/" in row["id"]) for row in inputs}

    report = sdk.run(
        tmp_path / "prepared", IdOnlyAdapter(), output=tmp_path / "run", batch_size=1,
    )
    assert report["metrics"]["accuracy"] == 0.5
    first = list(seen)
    seen.clear()
    sdk.run(tmp_path / "prepared", IdOnlyAdapter(), output=tmp_path / "rerun", batch_size=3)
    assert seen == first


def test_preparation_reuses_source_hashes_but_not_ids(tmp_path):
    source = build(tmp_path)
    first = gb.prepare(source, dataset="human_nontata_promoters")
    second = gb.prepare(source, dataset="human_nontata_promoters")
    assert first["provenance"] == second["provenance"]
    assert {r["id"] for r in first["rows"]}.isdisjoint(r["id"] for r in second["rows"])
    assert first["protocol_version"] == "2"
    assert first["metadata"]["adapter_input_contract"] == gb.INPUT_CONTRACT


def test_source_digest_includes_class_paths_and_original_bytes(tmp_path):
    source = build(tmp_path)
    first = gb.prepare(source, dataset="human_nontata_promoters")
    file = source / "human_nontata_promoters/test/positive/0.txt"
    file.write_text(file.read_text() + "\n")
    second = gb.prepare(source, dataset="human_nontata_promoters")
    assert first["provenance"]["test_sha256"] != second["provenance"]["test_sha256"]
    assert first["provenance"]["train_sha256"] == second["provenance"]["train_sha256"]
    for split in ("train", "test"):
        path = source / "human_nontata_promoters" / split / "positive"
        path.rename(path.with_name("zzpositive"))
    third = gb.prepare(source, dataset="human_nontata_promoters")
    for key in ("train_sha256", "test_sha256"):
        assert third["provenance"][key] != second["provenance"][key]


@pytest.mark.parametrize("operation", ["run", "evaluate", "export"])
def test_v1_artifacts_cannot_be_reused(tmp_path, operation):
    old = {"protocol_id": "genomic-benchmarks-v1", "kind": "rewire_local_evaluation"}
    with pytest.raises(ValueError, match="v1 exposed class labels"):
        if operation == "export":
            sdk.export(old, output=tmp_path / "export.json")
        elif operation == "run":
            sdk.run(old, object(), output=tmp_path / "run")
        else:
            sdk.evaluate(old, {}, output=tmp_path / "evaluate")


@pytest.mark.parametrize("change", ["contract", "ids", "order", "version"])
def test_v2_reloads_validate_the_input_contract(tmp_path, change):
    prepared = sdk.prepare(
        gb.PROTOCOL_ID, source=build(tmp_path / "source"), dataset="human_nontata_promoters",
        output=tmp_path / "prepared",
    )
    if change == "contract":
        prepared["metadata"].pop("adapter_input_contract")
    elif change == "version":
        prepared["protocol_version"] = gb.UPSTREAM_REVISION
    elif change == "ids":
        prepared["rows"][0]["id"] = "train/negative/0"
    else:
        prepared["rows"].reverse()
    prepared.pop("prepared_sha256")
    prepared["prepared_sha256"] = sdk._digest(prepared)
    with pytest.raises(ValueError, match="run prepare again"):
        sdk.run(prepared, object(), output=tmp_path / "run")


def test_sdk_export_and_submission_describe_only_a_local_copy(tmp_path):
    from rewirebench.submission import submit

    prepared = sdk.prepare(
        gb.PROTOCOL_ID, source=build(tmp_path / "source"), dataset="human_nontata_promoters",
        output=tmp_path / "prepared",
    )
    predictions = {r["id"]: r["target"] for r in prepared["rows"] if r["split"] == "test"}
    report = sdk.evaluate(prepared, predictions, output=tmp_path / "run")
    bundle = sdk.export(report, output=tmp_path / "bundle.json")
    assert bundle["evaluation_claim"] == "local_evaluation_not_paper_reproduction"
    assert bundle["data_verification"] == "local_bytes_hashed_not_independently_source_verified"
    assert bundle["independently_reproduced"] is False
    payload = submit(
        bundle, title="Local sequence evaluation", summary="Evaluation of the downloaded copy.",
        source_url="https://example.org/evaluation", metric="accuracy", value="1.0",
        source_locator="Local report", dry_run=True,
    )
    assert payload["contribution"]["details"]["rewire_bundle"] == bundle
    assert json.loads((tmp_path / "bundle.json").read_text()) == bundle
