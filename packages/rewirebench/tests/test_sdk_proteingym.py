"""Parity with pinned upstream code, malformed input and local ESM boundaries."""
import csv
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest
from rewirebench.protocols import proteingym as pg


def _dataset():
    """Synthetic scoring inputs, deliberately unequal assays per protein/category."""
    categories = ["Activity", "Activity", "Activity", "Binding", "Expression",
                  "OrganismalFitness", "Stability", "Stability"]
    proteins = ["P1", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]
    rows, assays, predictions = [], {}, {}
    for i, (category, protein) in enumerate(zip(categories, proteins)):
        assay_id = f"synthetic-{i}"
        assays[assay_id] = {"expected_count": 30, "prepared_count": 30,
                           "UniProt_ID": protein, "selection_type": category,
                           "taxon": ["Human", "Eukaryote", "Prokaryote", "Virus"][i % 4],
                           "MSA_Neff_L_category": ["Low", "Medium", "High"][i % 3]}
        rng = np.random.default_rng(51 + i)
        for j in range(30):
            identifier = f"{assay_id}::{j}"
            rows.append({"id": identifier, "assay_id": assay_id, "group": protein,
                         "split": "test", "selection_type": category,
                         "target": float(j), "target_binary": int(j >= 15),
                         "inputs": {"mutant": str(j)}})
            predictions[identifier] = float(np.round(rng.normal() + (j / (i + 1)), 1))
    dataset = {"protocol_id": pg.PROTOCOL_ID, "scope": "full", "rows": rows,
               "metadata": {"assays": assays, "official_assay_count": len(assays),
                            "aggregation": "pinned_upstream"}}
    return dataset, predictions


def test_all_metrics_and_aggregation_match_real_upstream_main(tmp_path, monkeypatch):
    """Run the original scorer end-to-end; only bootstrap repetitions reduced."""
    dataset, predictions = _dataset()
    data_dir, scores_dir = tmp_path / "data", tmp_path / "scores"
    data_dir.mkdir()
    scores_dir.mkdir()
    references = []
    for assay_id, meta in dataset["metadata"]["assays"].items():
        rows = [r for r in dataset["rows"] if r["assay_id"] == assay_id]
        frame = pd.DataFrame({"mutant": [r["id"].split("::")[1] for r in rows],
                              "DMS_score": [r["target"] for r in rows],
                              "DMS_score_bin": [r["target_binary"] for r in rows],
                              "fixture_model": [predictions[r["id"]] for r in rows]})
        frame.to_csv(data_dir / f"{assay_id}.csv", index=False)
        frame.to_csv(scores_dir / f"{assay_id}.csv", index=False)
        references.append({"DMS_id": assay_id, "DMS_filename": f"{assay_id}.csv",
                           "UniProt_ID": meta["UniProt_ID"],
                           "coarse_selection_type": meta["selection_type"],
                           "MSA_Neff_L_category": meta["MSA_Neff_L_category"],
                           "taxon": meta["taxon"]})
    pd.DataFrame(references).to_csv(tmp_path / "reference.csv", index=False)
    (tmp_path / "config.json").write_text(json.dumps({"model_list_zero_shot_substitutions_DMS": {
        "fixture_model": {"model_type": "Synthetic fixture"}}}))
    (tmp_path / "constants.json").write_text(json.dumps({"model_details": {},
                                                        "model_references": {}, "clean_names": {}}))
    spec = importlib.util.spec_from_file_location("upstream_proteingym", pg.resource_path("upstream_performance.py"))
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    monkeypatch.setattr(upstream, "__file__", str(tmp_path / "upstream.py"))
    original_bootstrap = upstream.compute_bootstrap_standard_error_functional_categories
    monkeypatch.setattr(upstream, "compute_bootstrap_standard_error_functional_categories",
                        lambda df: original_bootstrap(df, number_assay_reshuffle=3))
    monkeypatch.setattr(sys, "argv", ["upstream", "--input_scoring_files_folder", str(scores_dir),
        "--output_performance_file_folder", str(tmp_path / "out"),
        "--DMS_reference_file_path", str(tmp_path / "reference.csv"),
        "--DMS_data_folder", str(data_dir), "--config_file", str(tmp_path / "config.json")])
    upstream.main()
    actual = pg.score(dataset, predictions)
    for metric in pg.METRICS:
        frame = pd.read_csv(tmp_path / "out" / metric / f"DMS_substitutions_{metric}_DMS_level.csv")
        for _, row in frame.iterrows():
            assert actual["per_assay"][row["DMS ID"]]["metrics"][metric] == pytest.approx(row["fixture_model"], abs=1e-12)
        summary = pd.read_csv(tmp_path / "out" / metric / f"Summary_performance_DMS_substitutions_{metric}.csv")
        assert actual["metrics"][metric] == pytest.approx(summary[f"Average_{metric}"].iloc[0], abs=1e-12)


@pytest.mark.parametrize("scope", ["subset", "smoke"])
def test_partial_scope_never_has_suite_metric(scope):
    dataset, predictions = _dataset()
    dataset["scope"] = scope
    scored = pg.score(dataset, predictions)
    assert scored["metrics"] == {}
    assert scored["status"] == "partial_track"


def test_missing_predictions_keep_denominator_and_remove_suite_aggregate():
    dataset, predictions = _dataset()
    predictions.pop(next(iter(predictions)))
    scored = pg.score(dataset, predictions)
    assert scored["metrics"] == {}
    assert scored["per_assay"]["synthetic-0"]["eligible"] == 30
    assert scored["per_assay"]["synthetic-0"]["scored"] == 29
    assert scored["per_assay"]["synthetic-0"]["status"] == "partial"


def test_single_class_auc_and_constant_scores_are_explicit_nulls():
    dataset, predictions = _dataset()
    for row in dataset["rows"]:
        row["target_binary"] = 1
    predictions = dict.fromkeys(predictions, 0.0)
    result = pg.score(dataset, predictions)
    assert result["per_assay"]["synthetic-0"]["metrics"]["Spearman"] is None
    assert result["per_assay"]["synthetic-0"]["metrics"]["AUC"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("prediction", [float("nan"), float("inf"), "1.2", True])
def test_invalid_predictions_rejected(prediction):
    dataset, predictions = _dataset()
    predictions[next(iter(predictions))] = prediction
    with pytest.raises(ValueError, match="finite scalar"):
        pg.score(dataset, predictions)


def test_unknown_prediction_id_rejected():
    dataset, predictions = _dataset()
    predictions["unexpected"] = 1.
    with pytest.raises(ValueError, match="unknown"):
        pg.score(dataset, predictions)


def _local_assay(tmp_path, *, duplicate=False, invalid_mutation=False):
    ref = list(csv.DictReader(pg.resource_path("DMS_substitutions.csv").open()))[1]
    sequence = ref["target_seq"]
    wt = sequence[0]
    alt = "A" if wt != "A" else "C"
    mutant = f"{wt}1{alt}"
    rows = [{"mutant": "Z1A" if invalid_mutation else mutant,
             "mutated_sequence": alt + sequence[1:], "DMS_score": 0.3, "DMS_score_bin": 1}]
    if duplicate:
        rows += rows
    with (tmp_path / ref["DMS_filename"]).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return ref


def test_prepare_label_free_inputs_with_smoke_denominator(tmp_path):
    ref = _local_assay(tmp_path)
    prepared = pg.prepare(tmp_path, assay_ids=[ref["DMS_id"]], limit=1)
    assert prepared["scope"] == "smoke"
    assert "DMS_score" not in prepared["rows"][0]["inputs"]
    assert "target" not in prepared["rows"][0]["inputs"]
    assert prepared["metadata"]["assays"][ref["DMS_id"]]["expected_count"] == int(ref["DMS_total_number_mutants"])
    assert prepared["provenance"]["reference_sha256"] == pg.REFERENCE_SHA256


def test_incomplete_assay_requires_explicit_smoke(tmp_path):
    ref = _local_assay(tmp_path)
    with pytest.raises(ValueError, match="expected"):
        pg.prepare(tmp_path, assay_ids=[ref["DMS_id"]])


@pytest.mark.parametrize("kw", [{"duplicate": True}, {"invalid_mutation": True}])
def test_malformed_assay_rejected_even_for_smoke(tmp_path, kw):
    ref = _local_assay(tmp_path, **kw)
    with pytest.raises(ValueError):
        pg.prepare(tmp_path, assay_ids=[ref["DMS_id"]], limit=1)


def test_altered_reference_and_hash_rejected(tmp_path):
    ref = _local_assay(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        pg.prepare(tmp_path, assay_ids=[ref["DMS_id"]], limit=1,
                   expected_hashes={ref["DMS_id"]: "0" * 64})
    bad_reference = tmp_path / "bad.csv"
    bad_reference.write_text("not the reference")
    with pytest.raises(ValueError, match="Reference metadata"):
        pg.prepare(tmp_path, reference=bad_reference, assay_ids=[ref["DMS_id"]], limit=1)


def test_esm_requires_pinned_checkpoint_before_loading(tmp_path):
    from rewirebench.adapters.esm import ESM2Adapter
    path = tmp_path / "untrusted.pt"
    path.write_text("not a checkpoint")
    with pytest.raises(ValueError, match="Checkpoint"):
        ESM2Adapter(path)


def test_private_model_runs_offline_without_receiving_labels(tmp_path, monkeypatch):
    import socket

    from rewirebench import prepare, run

    source = tmp_path / "input"
    source.mkdir()
    ref = _local_assay(source)
    prepared = prepare(pg.PROTOCOL_ID, source=source, output=tmp_path / "prepared",
                       assay_ids=[ref["DMS_id"]], limit=1)

    def deny_network(*args, **kwargs):
        raise AssertionError("No network during local evaluation")

    monkeypatch.setattr(socket.socket, "connect", deny_network)

    class PrivateModel:
        def predict(self, inputs):
            assert set(inputs[0]) == {"id", "assay_id", "wild_type_sequence",
                                      "mutant", "mutated_sequence"}
            return {row["id"]: 0.1 for row in inputs}

    report = run(prepared, PrivateModel(), output=tmp_path / "out")
    assert report["completion"] == "partial"
    assert report["metrics"] == {}
    assert report["coverage"]["denominator"] == int(ref["DMS_total_number_mutants"])


def test_zero_shot_rejects_fitting(tmp_path):
    from rewirebench import prepare, run
    source = tmp_path / "input"
    source.mkdir()
    ref = _local_assay(source)
    prepared = prepare(pg.PROTOCOL_ID, source=source, output=tmp_path / "prepared",
                       assay_ids=[ref["DMS_id"]], limit=1)

    class WrongTrack:
        def predict(self, inputs):
            raise AssertionError("Must reject before prediction")

        def fit(self, inputs, labels):
            raise AssertionError("Must reject before fitting")

    with pytest.raises(ValueError, match="Fitting is forbidden"):
        run(prepared, WrongTrack(), output=tmp_path / "out")


def test_vendored_evidence_receipts_match_bytes():
    import hashlib
    receipts = json.loads(pg.resource_path("sources.json").read_text())
    for receipt in receipts:
        if receipt.get("local_file"):
            path = pg.resource_path(receipt["local_file"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]


def _cached_esm_without_weights():
    """Test input routing without downloading model weights."""
    from rewirebench.adapters.esm import ESM2Adapter

    adapter = ESM2Adapter.__new__(ESM2Adapter)

    class Alphabet:
        @staticmethod
        def get_idx(amino_acid):
            return {"A": 0, "C": 1}[amino_acid]

    adapter.alphabet = Alphabet()
    adapter._cache = {("AC", 0): np.array([-0.4, -0.2])}
    return adapter


def test_esm_long_sequence_unscored_and_batch_continues():
    adapter = _cached_esm_without_weights()
    output = adapter.predict([
        {"id": "long", "wild_type_sequence": "A" * 1023, "mutant": "A1C"},
        {"id": "short", "wild_type_sequence": "AC", "mutant": "A1C"},
    ])
    assert output["long"] == {
        "score": None,
        "reason": "ESM example supports at most 1022 residues; windowed inference not implemented",
    }
    assert output["short"] == pytest.approx(0.2)


@pytest.mark.parametrize("sequence", ["", "AC*", "AC DEF", "A" * 1023 + "*", None])
def test_esm_invalid_sequence_still_rejected(sequence):
    adapter = _cached_esm_without_weights()
    with pytest.raises(ValueError, match="valid amino-acid sequence"):
        adapter.predict([{"id": "invalid", "wild_type_sequence": sequence, "mutant": "A1C"}])


def test_esm_long_sequence_invalid_mutation_is_not_silently_unscored():
    adapter = _cached_esm_without_weights()
    with pytest.raises(ValueError, match="Mutation does not match"):
        adapter.predict([{"id": "invalid", "wild_type_sequence": "A" * 1023, "mutant": "C1A"}])
