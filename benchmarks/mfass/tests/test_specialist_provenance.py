"""Provenance and configuration tests with synthetic inputs, without inference."""

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from mfass.specialist_provenance import (
    file_sha256,
    resolve_spliceai_annotation,
    specialist_artifacts,
)


@pytest.mark.parametrize("release", [None, "", "  ", "unreported"])
def test_annotation_filename_does_not_establish_release(tmp_path, release):
    annotation = tmp_path / "gencode.v44.annotation.db"
    reference = tmp_path / "ref.fa"
    annotation.write_bytes(b"synthetic annotation")
    reference.write_bytes(b">chr1\nACGT\n")
    metadata = specialist_artifacts(annotation, reference, release)
    assert metadata["annotation_release"] == "unreported"
    assert metadata["annotation_release_status"] == "unreported"
    assert metadata["annotation_sha256"] == hashlib.sha256(annotation.read_bytes()).hexdigest()
    assert metadata["reference_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert str(tmp_path) not in json.dumps(metadata)


def test_declaration_is_not_verified_and_hashes_track_the_bytes(tmp_path):
    annotation, reference = tmp_path / "custom.db", tmp_path / "ref.fa"
    annotation.write_bytes(b"original")
    reference.write_bytes(b">chr1\nACGT\n")
    first = specialist_artifacts(annotation, reference, "GENCODE v44")
    assert first["annotation_release"] == "GENCODE v44"
    assert first["annotation_release_status"] == "declared"
    annotation.write_bytes(b"changed")
    second = specialist_artifacts(annotation, reference, "GENCODE v44")
    assert first["annotation_sha256"] != second["annotation_sha256"]
    assert first["reference_sha256"] == second["reference_sha256"]
    assert "upstream identity not verified" in second["artifact_identity_status"]


def test_hash_streams_multiple_blocks_and_rejects_missing_files(tmp_path):
    artifact = tmp_path / "reference.fa"
    data = b"ACGT" * 600_000
    artifact.write_bytes(data)
    assert file_sha256(artifact) == hashlib.sha256(data).hexdigest()
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "missing.fa")


@pytest.mark.parametrize("alias", ["grch37", "grch38"])
def test_bundled_alias_has_precedence_but_explicit_local_path_is_custom(
    tmp_path, monkeypatch, alias,
):
    monkeypatch.chdir(tmp_path)
    Path(alias).write_text("local file")
    bundled = tmp_path / "installed" / "annotations" / f"{alias}.txt"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("packaged annotation")

    def resource_filename(package, path):
        assert package == "spliceai"
        assert path == f"annotations/{alias}.txt"
        return str(bundled)

    monkeypatch.setitem(sys.modules, "pkg_resources", SimpleNamespace(
        resource_filename=resource_filename,
    ))
    assert resolve_spliceai_annotation(alias) == bundled
    custom = resolve_spliceai_annotation(f"./{alias}")
    assert custom == tmp_path / alias
    assert str(custom) != alias  # Annotator must not reinterpret this as a bundled alias.


@pytest.mark.parametrize("mask", ["-1", "2", "True"])
def test_spliceai_invalid_mask_fails_before_preparation_or_model_import(monkeypatch, mask):
    from mfass.run_spliceai import main

    monkeypatch.setattr(sys, "argv", ["mfass-spliceai", "--mask", mask])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


PATCHED_IDENTITY = {"pangolin_source_identity": "pangolin-5cf94b8-mask-per-gene-1",
                    "pangolin_source_sha256": "fixture", "pangolin_upstream_files_verified": True,
                    "pangolin_upstream_file_mismatches": []}


def fake_torch(threads=4):
    state = {"threads": threads, "interop": 8}
    return SimpleNamespace(set_num_threads=lambda n: state.update(threads=n),
                           get_num_threads=lambda: state["threads"],
                           set_num_interop_threads=lambda n: state.update(interop=n),
                           get_num_interop_threads=lambda: state["interop"],
                           __version__="fixture")


def fake_tensorflow():
    state = {"intra": 0, "inter": 0}
    threading = SimpleNamespace(
        set_intra_op_parallelism_threads=lambda n: state.update(intra=n),
        set_inter_op_parallelism_threads=lambda n: state.update(inter=n),
        get_intra_op_parallelism_threads=lambda: state["intra"],
        get_inter_op_parallelism_threads=lambda: state["inter"])
    return SimpleNamespace(config=SimpleNamespace(threading=threading), __version__="fixture")


@pytest.mark.parametrize("mask", [None, "0", "1"])
def test_spliceai_records_and_uses_the_resolved_file(synthetic_run, monkeypatch, mask):
    from mfass import run_spliceai as runner

    fixture = synthetic_run
    # The upstream reader treats this bare filename as an alias, making the
    # absolute path passed by the runner important for custom ./grch38 files.
    monkeypatch.chdir(fixture.annotation.parent)
    Path("grch38").write_text("custom grch38 file")
    annotation = Path("grch38").resolve()
    expected_mask = int(mask) if mask is not None else 0
    module, utils = ModuleType("spliceai"), ModuleType("spliceai.utils")

    def annotator(reference, actual_annotation):
        assert reference == str(fixture.reference)
        assert actual_annotation == str(annotation)
        return object()

    def scores(record, ann, distance, selected_mask):
        assert selected_mask == expected_mask
        return ["G|gene|0.1|0.2|0.3|0.4"]

    utils.Annotator, utils.get_delta_scores = annotator, scores
    monkeypatch.setitem(sys.modules, "spliceai", module)
    monkeypatch.setitem(sys.modules, "spliceai.utils", utils)
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow())
    monkeypatch.setattr(runner, "_patch_numpy_fromstring", lambda: None)
    monkeypatch.setattr(runner, "installed_identity", lambda: {"spliceai_version": "fixture"})
    args = ["spliceai", *fixture.args, "--annotation", "./grch38"]
    if mask is not None:
        args += ["--mask", mask]
    monkeypatch.setattr(sys, "argv", args)
    runner.main()
    result = json.loads(fixture.output.read_text())
    assert result["config"]["annotation_sha256"] == file_sha256(annotation)
    assert result["config"]["reference_sha256"] == file_sha256(fixture.reference)
    assert result["config"]["annotation_release_status"] == "unreported"
    assert result["config"]["mask_M"] == expected_mask
    assert result["benchmark"] == "mfass-v2-smoke"


@pytest.mark.parametrize("mask", [None, "True", "False"])
@pytest.mark.parametrize("release", [None, "A declared custom release"])
def test_pangolin_records_only_the_selected_configuration(
    synthetic_run, monkeypatch, mask, release,
):
    from mfass import run_pangolin as runner

    fixture = synthetic_run
    expected_mask = mask or "True"
    module, pp = ModuleType("pangolin"), ModuleType("pangolin.pangolin")

    def process_variant(i, chromosome, position, ref, alt, db, models, args):
        assert args.mask == expected_mask
        return "gene|10:0.3|11:-0.2"

    pp.process_variant = process_variant
    pp.pyfastx = SimpleNamespace(Fasta=lambda *a, **kw: object())
    module.pangolin = pp
    monkeypatch.setitem(sys.modules, "pangolin", module)
    monkeypatch.setitem(sys.modules, "pangolin.pangolin", pp)
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "gffutils", SimpleNamespace(FeatureDB=lambda path: object()))
    monkeypatch.setattr(runner, "_load_models", list)
    monkeypatch.setattr(runner, "installed_identity", lambda: PATCHED_IDENTITY)
    args = ["pangolin", *fixture.args, "--db", str(fixture.annotation)]
    if mask is not None:
        args += ["--mask", mask]
    if release is not None:
        args += ["--annotation-release", release]
    monkeypatch.setattr(sys, "argv", args)
    runner.main()
    result = json.loads(fixture.output.read_text())
    config = result["config"]
    assert config["annotation_release"] == (release or "unreported")
    assert config["annotation_release_status"] == ("declared" if release else "unreported")
    assert config["annotation_sha256"] == file_sha256(fixture.annotation)
    assert config["reference_sha256"] == file_sha256(fixture.reference)
    assert config["mask_m"] == expected_mask
    assert result["notes"].startswith(f"This run used mask={expected_mask}.")
    assert "Both are run" not in result["notes"]
    assert "hash_reference_and_annotation" in result["timing_seconds"]
    assert "artifact hashing" in config["timing_scope"]
    assert result["benchmark"] == "mfass-v2-smoke"
