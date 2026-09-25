"""Reference-runner tests: real protocol scoring, leakage and failure boundaries."""
import copy
import csv
import json

import pytest
from rewirebench import sdk
from rewirebench.adapters.baselines import PairedComposition, StringNgramReference, TrainingPrior
from rewirebench.baselines import REGISTRY, describe_baselines, run_baselines


def genomic(tmp_path, *, classes=("negative", "positive")):
    dataset = "human_nontata_promoters" if len(classes) == 2 else "human_ensembl_regulatory"
    for split in ("train", "test"):
        for j, name in enumerate(classes):
            path = tmp_path / "source" / dataset / split / name
            path.mkdir(parents=True)
            for i in range(4):
                (path / f"{i}.txt").write_text("ACG"[j] * (i + 8))
    return sdk.prepare("genomic-benchmarks-v2", source=tmp_path / "source", dataset=dataset,
                       output=tmp_path / "prepared")


def tdc(tmp_path, task):
    from rewirebench.protocols.tdc_admet import GROUP_DIRECTORY

    dataset = "hia_hou" if task == "classification" else "caco2_wang"
    directory = tmp_path / "source" / GROUP_DIRECTORY / dataset
    directory.mkdir(parents=True)
    for filename, count in (("train_val.csv", 8), ("test.csv", 6)):
        with (directory / filename).open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Drug_ID", "Drug", "Y"])
            for i in range(count):
                writer.writerow([f"{filename}-{i}", "C" * (i + 1) + "O",
                                 i % 2 if task == "classification" else i / 2])
    return sdk.prepare("tdc-admet-group-v1", source=tmp_path / "source", dataset=dataset,
                       output=tmp_path / "prepared")


def rehash(data):
    data["prepared_sha256"] = sdk._digest({k: v for k, v in data.items() if k != "prepared_sha256"})
    return data


def test_registry_covers_every_sdk_protocol_with_explicit_roles_and_no_execution_claim():
    assert set(REGISTRY) == set(sdk.PROTOCOLS)
    for protocol in REGISTRY:
        registry = describe_baselines(protocol)
        assert {b["role"] for b in registry["baselines"]} == {"null", "conventional"}
        assert all(b["execution_status"] == "not_established_by_registry" for b in registry["baselines"])
    describe_baselines("mfass-v2")["baselines"].clear()
    assert len(describe_baselines("mfass-v2")["baselines"]) == 2
    with pytest.raises(ValueError, match="Unsupported"):
        describe_baselines("invented")


def test_reviewed_conventional_methods_keep_states_separate_and_old_proposals_blocked():
    expected = {"proteingym-v1.3-dms-substitutions": ("proteingym-conventional-pending-v1",
                                                      "proteingym-evcouplings-independent-v1"),
                "dart-eval-task1-zero-shot-v1": ("dart-conventional-pending-v1",
                                                 "dart-h12core-fimo-hit-count-v1")}
    for protocol, (pending, reviewed) in expected.items():
        entries = {b["baseline_id"]: b for b in describe_baselines(protocol)["baselines"]}
        assert entries[pending]["status"] == "blocked"
        entry = entries[reviewed]
        assert entry["status"] == "blocked" and entry["implementation_status"] == "implemented"
        assert entry["biological_evaluation_status"] == "not_executed"
        assert entry["measurement_release_status"] == "none"
        assert "no pretraining" not in entry["training_overlap"]
    assert expected["proteingym-v1.3-dms-substitutions"][1] == describe_baselines(
        "proteingym-v1.3-dms-substitutions")["baselines"][1]["replaced_by"]
    dart = describe_baselines("dart-eval-task1-zero-shot-v1")["baselines"]
    assert "not a published Task 1 comparator" in dart[1]["related"]["dart-h12core-fimo-hit-count-v1"]
    assert "not a reproduced DART Task 1 baseline" in dart[2]["configuration"]["claim"]


@pytest.mark.parametrize("classes", [("negative", "positive"), ("enhancer", "ocr", "promoter")])
def test_genomic_end_to_end_null_and_train_only_conventional(tmp_path, classes):
    prepared = genomic(tmp_path, classes=classes)
    manifest = run_baselines(prepared, output=tmp_path / "run", batch_size=1)
    assert manifest["status"] == "evaluated"
    assert len(manifest["baselines"]) == 2
    reference = json.loads((tmp_path / "run/dna-ngram-logistic-v1/report.json").read_text())
    assert reference["metrics"]["accuracy"] == 1.0
    assert reference["execution"]["fitting"] == "train_only"
    assert reference["coverage"]["scored"] == len(classes) * 4
    assert reference["model_configuration"]["baseline_id"] == "dna-ngram-logistic-v1"
    assert not reference["independently_reproduced"]
    # Test labels affect scoring, never features, head selection or predictions.
    changed = copy.deepcopy(prepared)
    for row in changed["rows"]:
        if row["split"] == "test":
            row["target"] = (row["target"] + 1) % len(classes)
    run_baselines(rehash(changed), output=tmp_path / "changed")
    for ident in ("training-prior-v1", "dna-ngram-logistic-v1"):
        assert (tmp_path / "run" / ident / "predictions.json").read_text() == (tmp_path / "changed" / ident / "predictions.json").read_text()
    with pytest.raises(FileExistsError):
        run_baselines(prepared, output=tmp_path / "run")


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_tdc_task_specific_controls_and_reference(tmp_path, task):
    data = tdc(tmp_path, task)
    result = run_baselines(data, output=tmp_path / "run")
    assert result["status"] == "evaluated"
    assert len(result["baselines"]) == 2
    predictions = json.loads((tmp_path / "run/training-control-v1/predictions.json").read_text())
    assert set(predictions.values()) == {0.5 if task == "classification" else 1.75}
    report = json.loads((tmp_path / "run/smiles-ngram-linear-v1/report.json").read_text())
    assert "not SMILES-invariant" in report["model_configuration"]["limitation"]


def test_priors_do_not_use_test_inputs_and_multiclass_ties_are_explicit():
    prior = TrainingPrior()
    prior.fit([{}, {}, {}], [0, 0, 1])
    assert prior.predict([{"id": "test", "target": 1}]) == {"test": 1 / 3}
    majority = TrainingPrior(multiclass=True)
    majority.fit([{}, {}, {}, {}], [2, 1, 2, 1])
    assert majority.predict([{"id": "t"}]) == {"t": 1.0}


def test_tdc_vocabulary_fits_training_only_and_preserves_smiles_case():
    reference = StringNgramReference(field="smiles", classification=False)
    reference.fit([{"smiles": "CCO"}, {"smiles": "CCN"}], [0.0, 1.0])
    before = dict(reference.vectorizer.vocabulary_)
    reference.predict([{"id": "test", "smiles": "[Xe]cccc"}])
    assert before == reference.vectorizer.vocabulary_
    assert "c" not in before and "C" in before and "X" not in before


def test_dart_zero_shot_control_runs_without_fitting_and_keeps_gap(tmp_path):
    data = sdk.prepare("dart-eval-task1-zero-shot-v1", source="demo", output=tmp_path / "prepared")
    manifest = run_baselines(data, output=tmp_path / "run")
    assert manifest["status"] == "incomplete"
    assert [r["status"] for r in manifest["baselines"]] == ["evaluated", "blocked", "blocked"]
    report = json.loads((tmp_path / "run/seeded-random-v1/report.json").read_text())
    assert report["execution"]["fitting"] == "none"
    assert report["scope"] == "smoke"
    assert report["completion"] == "partial"
    with pytest.raises(ValueError, match="Smoke"):
        sdk.export(report, output=tmp_path / "submission.json")


def test_paired_constant_and_composition_have_distinct_semantics():
    inputs = [{"id": "x", "reference_sequence": "AAA", "mutant_sequence": "AAT"}]
    assert PairedComposition(constant=True).embed(inputs)["x"] == {"reference": [0.0], "mutant": [0.0]}
    pair = PairedComposition().embed(inputs)["x"]
    assert pair["reference"] != pair["mutant"]
    assert len(pair["reference"]) == 6


@pytest.mark.parametrize("ids", [[], ["missing"], ["training-prior-v1", "training-prior-v1"], "training-prior-v1"])
def test_invalid_selection_creates_no_outputs(tmp_path, ids):
    data = genomic(tmp_path)
    with pytest.raises(ValueError):
        run_baselines(data, output=tmp_path / "bad", baseline_ids=ids)
    assert not (tmp_path / "bad").exists()


def test_prepared_corruption_is_rejected_before_execution(tmp_path):
    data = genomic(tmp_path)
    data["rows"][0]["target"] = 23
    with pytest.raises(ValueError, match="checksum"):
        run_baselines(data, output=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_baseline_failure_is_recorded_redacted_and_others_continue(tmp_path, monkeypatch):
    from rewirebench import baselines

    data = genomic(tmp_path)
    original = baselines._adapter
    def broken(entry, data, options=None):
        if entry["role"] == "null":
            raise RuntimeError("secret/private/sample must not enter the manifest")
        return original(entry, data, options)
    monkeypatch.setattr(baselines, "_adapter", broken)
    manifest = run_baselines(data, output=tmp_path / "run")
    assert manifest["status"] == "incomplete"
    assert [r["status"] for r in manifest["baselines"]] == ["failed", "evaluated"]
    text = (tmp_path / "run/baseline-manifest.json").read_text()
    assert "RuntimeError" in text and "secret/private" not in text
    with pytest.raises(RuntimeError):
        run_baselines(data, output=tmp_path / "fail-fast", fail_fast=True)
    assert json.loads((tmp_path / "fail-fast/baseline-manifest.json").read_text())["status"] == "interrupted_by_failure"


def mfass_fixture(protocol):
    rows = []
    for i in range(48):
        ref = "A" * 50
        rows.append({"id": f"fixture-{i}", "split": "train" if i < 40 else "test",
                     "target": i % 2, "inputs": {
                         "reference_sequence": ref, "mutant_sequence": ref[:24] + "C" + ref[25:],
                         "rel_position": 25, "rel_position_scaled": 0.5,
                         "intron1_len": 10, "intron2_len": 10, "exon_len": 30,
                         "ref_allele": "A", "alt_allele": "C", "region": "exon",
                         "phylop_score": 0.0, "mean_phastCons_score": 0.0}})
    return rehash({"protocol_id": protocol, "protocol_version": "fixture",
                   "dataset_id": "synthetic-mfass-smoke", "scope": "smoke", "rows": rows,
                   "metadata": {"canonical_test_count": 8324}})


@pytest.mark.parametrize("protocol", ["mfass-v2", "mfass-v2-frozen-encoder"])
def test_mfass_both_interfaces_execute_with_original_protocol_scoring(tmp_path, protocol):
    result = run_baselines(mfass_fixture(protocol), output=tmp_path / "run")
    assert result["status"] == "evaluated"
    for baseline in result["baselines"]:
        assert baseline["completion"] == "partial"
        assert baseline["coverage"] == {"scored": 8, "denominator": 8324, "unscored": 8316}
        report = json.loads((tmp_path / "run" / baseline["report"]).read_text())
        assert report["scope"] == "smoke"
        assert report["execution"]["fitting"] == ("train_only" if protocol == "mfass-v2" else "protocol_owned_probe")


def test_flip_and_mrna_probes_run_train_only(tmp_path):
    flip_path = tmp_path / "flip.csv"
    with flip_path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "target", "set", "validation"])
        writer.writerows([["A" * (i+1) + "C", float(i), "train" if i < 10 else "test", i == 9]
                          for i in range(16)])
    flip = sdk.prepare("flip2-fitness-v1", source=flip_path, dataset="flip2-amylase-one-to-many",
                       allow_unverified=True, output=tmp_path / "flip-prepared")
    mrna_path = tmp_path / "sample.csv"
    with mrna_path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "target_mrl_egfp_unmod"])
        writer.writerows([["A" * (i+1) + "CGTU", i / 3] for i in range(40)])
    mrna = sdk.prepare("mrnabench-sample-mrl-v1", source=mrna_path, dataset="egfp",
                       target="target_mrl_egfp_unmod", output=tmp_path / "mrna-prepared")
    for i, data in enumerate((flip, mrna)):
        manifest = run_baselines(data, output=tmp_path / f"run-{i}")
        assert manifest["status"] == "evaluated"
        changed = copy.deepcopy(data)
        for row in changed["rows"]:
            if row["split"] != "train":
                row["target"] += 999
        if i == 0:
            from rewirebench.protocols.flip2 import _rows_digest
            changed["metadata"]["prepared_rows_sha256"] = _rows_digest(changed["rows"])
        run_baselines(rehash(changed), output=tmp_path / f"changed-{i}")
        for baseline in manifest["baselines"]:
            ident = baseline["baseline_id"]
            assert (tmp_path / f"run-{i}" / ident / "predictions.json").read_text() == (tmp_path / f"changed-{i}" / ident / "predictions.json").read_text()
        control = json.loads((tmp_path / f"run-{i}/training-mean-v1/report.json").read_text())
        assert control["metrics"]["spearman"] is None


def test_proteingym_null_is_scored_per_assay_without_claiming_full_track(tmp_path):
    from rewirebench.protocols import proteingym

    source = tmp_path / "source"
    source.mkdir()
    # Use the real packaged reference identity with an explicit truncated source.
    with proteingym.resource_path("DMS_substitutions.csv").open() as stream:
        reference = next(csv.DictReader(stream))
    source_file = source / reference["DMS_filename"]
    with source_file.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"])
        for i in range(12):
            wild_type = reference["target_seq"]
            alternate = "C" if wild_type[i] != "C" else "A"
            mutant = wild_type[:i] + alternate + wild_type[i + 1:]
            writer.writerow([f"{wild_type[i]}{i+1}{alternate}", mutant, i / 10, int(i > 5)])
    data = sdk.prepare("proteingym-v1.3-dms-substitutions", source=source,
                       assay_ids=[reference["DMS_id"]], limit=12, output=tmp_path / "prepared")
    result = run_baselines(data, output=tmp_path / "run")
    assert result["status"] == "incomplete"
    assert result["baselines"][1]["status"] == "blocked"
    report = json.loads((tmp_path / "run/seeded-random-v1/report.json").read_text())
    assert report["completion"] == "partial"
    assert report["execution"]["fitting"] == "none"
    assert reference["DMS_id"] in report["protocol_results"]["per_assay"]
