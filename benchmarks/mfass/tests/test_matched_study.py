"""Checkpoint, runner failure policy, output binding and study orchestration.

Synthetic inputs and mocked model modules only; no inference and no MFASS data.
"""
import csv
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from mfass import matched_study
from mfass.checkpoint import Checkpoint
from mfass.specialist_provenance import file_sha256
from test_specialist_provenance import PATCHED_IDENTITY, fake_tensorflow, fake_torch

SETTINGS = {"mask": "True", "annotation_sha256": "a", "reference_sha256": "r"}


# Checkpoint


def test_checkpoint_resumes_only_with_identical_settings(tmp_path):
    path = tmp_path / "checkpoint.tsv"
    log = Checkpoint(path, SETTINGS)
    log.append("v1", 0.25, 1.5, raw="gene|1:0.25|2:-0.1")
    log.append("v2", None, 0.1, reason="no hg38 coordinate")
    log.close()
    resumed = Checkpoint(path, dict(SETTINGS))
    assert resumed.resumed and set(resumed.rows) == {"v1", "v2"}
    assert float(resumed.get("v1")["score"]) == 0.25
    assert resumed.get("v2")["status"] == "unscored"
    with pytest.raises(ValueError, match="duplicate"):
        resumed.append("v1", 0.3, 1.0)
    resumed.close()
    with pytest.raises(ValueError, match="different settings"):
        Checkpoint(path, {**SETTINGS, "mask": "False"})


def test_checkpoint_discards_a_line_cut_off_by_a_crash(tmp_path):
    path = tmp_path / "checkpoint.tsv"
    log = Checkpoint(path, SETTINGS)
    log.append("v1", 0.25, 1.0)
    log.close()
    with path.open("a") as handle:
        handle.write("v2\tscored\t0.9")  # interrupted mid-write
    log = Checkpoint(path, SETTINGS)
    assert log.discarded_partial_line and set(log.rows) == {"v1"}
    log.append("v2", 0.5, 1.0)
    log.close()
    assert set(Checkpoint(path, SETTINGS).rows) == {"v1", "v2"}


def test_checkpoint_is_exclusive_while_open(tmp_path):
    path = tmp_path / "checkpoint.tsv"
    first = Checkpoint(path, SETTINGS)
    with pytest.raises(RuntimeError, match="in use"):
        Checkpoint(path, SETTINGS)
    first.close()
    Checkpoint(path, SETTINGS).close()


@pytest.mark.parametrize("row, message", [
    ("v1\tscored\tnan\t1.0\t\t", "finite score"),
    ("v1\tscored\t\t1.0\t\t", "finite score"),
    ("v1\tscored\t0.2\t1.0\tsome reason\t", "no unscored reason"),
    ("v1\tunscored\t0.2\t1.0\tno hg38 coordinate\t", "carry no score"),
    ("v1\tunscored\t\t1.0\t\t", "need a reason"),
    ("v1\tscored\t0.2\t-1\t\t", "seconds"),
    ("v1\tscored\t0.2\tinf\t\t", "seconds"),
    ("v1\tscored\t0.2\tfast\t\t", "seconds"),
    ("v1\tdone\t0.2\t1.0\t\t", "invalid status"),
    ("\tscored\t0.2\t1.0\t\t", "without an ID"),
    ("v1\tscored\t0.2\t1.0", "malformed"),
])
def test_checkpoint_rejects_malformed_rows(tmp_path, row, message):
    path = tmp_path / "checkpoint.tsv"
    Checkpoint(path, SETTINGS).close()
    with path.open("a") as handle:
        handle.write(row + "\n")
    with pytest.raises(ValueError, match=message):
        Checkpoint(path, SETTINGS)


def test_checkpoint_refuses_duplicate_rows_on_disk(tmp_path):
    path = tmp_path / "checkpoint.tsv"
    log = Checkpoint(path, SETTINGS)
    log.append("v1", 0.25, 1.0)
    log.close()
    line = path.read_text().splitlines()[-1]
    with path.open("a") as handle:
        handle.write(line + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        Checkpoint(path, SETTINGS)


def test_incomplete_checkpoint_is_not_a_result(tmp_path):
    log = Checkpoint(tmp_path / "c.tsv", SETTINGS)
    log.append("v1", 0.1, 1.0)
    with pytest.raises(ValueError, match="1 missing"):
        log.check_complete(["v1", "v2"])
    with pytest.raises(ValueError, match="1 unexpected"):
        log.check_complete([])
    with pytest.raises(ValueError, match="finite score"):
        log.append("v3", float("nan"), 1.0)


# Runners


@pytest.fixture
def pangolin_env(synthetic_run, monkeypatch):
    from mfass import run_pangolin as runner

    calls, behaviour = [], {}
    module, pp = ModuleType("pangolin"), ModuleType("pangolin.pangolin")

    def process_variant(i, chromosome, position, ref, alt, db, models, args):
        calls.append(position)
        if position in behaviour:
            return behaviour[position]()
        if position == 2:
            print("[Line 1] WARNING, skipping variant: Mismatch between FASTA and variant file.")
            return -1
        return f"gene|10:0.{position}|11:-0.2|Warnings:"

    pp.process_variant = process_variant
    pp.pyfastx = SimpleNamespace(Fasta=lambda *a, **kw: object())
    module.pangolin = pp
    torch = fake_torch()
    monkeypatch.setitem(sys.modules, "pangolin", module)
    monkeypatch.setitem(sys.modules, "pangolin.pangolin", pp)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "gffutils", SimpleNamespace(FeatureDB=lambda path: object()))
    monkeypatch.setattr(runner, "_load_models", list)
    monkeypatch.setattr(runner, "installed_identity", lambda: PATCHED_IDENTITY)
    args = ["pangolin", *synthetic_run.args, "--db", str(synthetic_run.annotation),
            "--threads", "6", "--interop-threads", "1"]
    return SimpleNamespace(runner=runner, calls=calls, args=args, fixture=synthetic_run,
                           behaviour=behaviour)


@pytest.mark.parametrize("identity, argv", [
    ({"pangolin_source_identity": "upstream-5cf94b8-unpatched"}, ["--mask", "True"]),
    ({"pangolin_upstream_files_verified": False,
      "pangolin_upstream_file_mismatches": ["model.py"]}, ["--mask", "True"]),
    ({"pangolin_upstream_files_verified": False,
      "pangolin_upstream_file_mismatches": ["models/final.1.0.3.v2"]},
     ["--mask", "False", "--require-verified-code"]),
])
def test_pangolin_refuses_unreviewed_code_before_scoring(pangolin_env, monkeypatch, identity,
                                                        argv):
    monkeypatch.setattr(pangolin_env.runner, "installed_identity",
                        lambda: {**PATCHED_IDENTITY, **identity})
    monkeypatch.setattr(sys, "argv", [*pangolin_env.args, *argv])
    with pytest.raises(SystemExit, match="Refusing to run"):
        pangolin_env.runner.main()
    assert not pangolin_env.fixture.output.exists() and pangolin_env.calls == []


def test_unmasked_pangolin_without_the_patch_is_allowed_outside_the_study(pangolin_env,
                                                                          monkeypatch):
    unpatched = {**PATCHED_IDENTITY, "pangolin_source_identity": "upstream-5cf94b8-unpatched"}
    monkeypatch.setattr(pangolin_env.runner, "installed_identity", lambda: unpatched)
    monkeypatch.setattr(sys, "argv", [*pangolin_env.args, "--mask", "False"])
    pangolin_env.runner.main()
    config = json.loads(pangolin_env.fixture.output.read_text())["config"]
    assert config["mask_m"] == "False"
    assert config["pangolin_source_identity"] == "upstream-5cf94b8-unpatched"


def _predictions(path):
    with path.with_suffix(".predictions.tsv").open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_resumed_pangolin_run_matches_an_uninterrupted_run(pangolin_env, monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.tsv"
    argv = [*pangolin_env.args, "--mask", "True", "--checkpoint", str(checkpoint)]
    monkeypatch.setattr(sys, "argv", argv)
    original = pangolin_env.runner._score_variant
    scored = []

    def interrupt_after_one(*a, **kw):
        if scored:
            raise KeyboardInterrupt
        scored.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(pangolin_env.runner, "_score_variant", interrupt_after_one)
    with pytest.raises(KeyboardInterrupt):
        pangolin_env.runner.main()
    assert not pangolin_env.fixture.output.exists()
    assert len(checkpoint.read_text().splitlines()) == 3  # fingerprint, header, one row

    monkeypatch.setattr(pangolin_env.runner, "_score_variant", original)
    pangolin_env.calls.clear()
    pangolin_env.runner.main()
    assert pangolin_env.calls == [2, 3]  # the first variant was not rescored
    resumed = json.loads(pangolin_env.fixture.output.read_text())
    config = resumed["config"]
    assert config["checkpoint"]["resumed"] is True
    assert (config["torch_threads"]["intra_op"], config["torch_threads"]["inter_op"]) == (6, 1)
    assert "final segment" in config["timing_scope"]
    unscored = pangolin_env.fixture.output.with_suffix(".unscored.tsv").read_text()
    assert "Mismatch between FASTA and variant file" in unscored

    fresh = tmp_path / "fresh.json"
    fresh_args = [a if a != str(pangolin_env.fixture.output) else str(fresh)
                  for a in pangolin_env.args] + ["--mask", "True"]
    monkeypatch.setattr(sys, "argv", fresh_args)
    pangolin_env.runner.main()
    assert _predictions(fresh) == _predictions(pangolin_env.fixture.output)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError):
        pangolin_env.runner.main()


def test_unexpected_error_stops_the_run_and_is_retried_on_resume(pangolin_env, monkeypatch,
                                                                  tmp_path):
    checkpoint = tmp_path / "checkpoint.tsv"
    monkeypatch.setattr(sys, "argv", [*pangolin_env.args, "--mask", "True",
                                      "--checkpoint", str(checkpoint)])

    def transient():
        raise OSError("transient read failure")

    pangolin_env.behaviour[3] = transient
    with pytest.raises(OSError, match="transient"):
        pangolin_env.runner.main()
    rows = checkpoint.read_text().splitlines()[2:]
    assert [r.split("\t")[0] for r in rows] == ["0", "1"]  # nothing logged for the failure
    assert not pangolin_env.fixture.output.exists()
    del pangolin_env.behaviour[3]
    pangolin_env.calls.clear()
    pangolin_env.runner.main()
    assert pangolin_env.calls == [3]
    assert "OSError" not in pangolin_env.fixture.output.with_suffix(".unscored.tsv").read_text()


def test_nonfinite_and_empty_pangolin_outputs(pangolin_env, monkeypatch):
    pangolin_env.behaviour[1] = lambda: "gene|10:nan|11:-0.2|Warnings:"
    monkeypatch.setattr(sys, "argv", [*pangolin_env.args, "--mask", "True"])
    pangolin_env.runner.main()
    unscored = pangolin_env.fixture.output.with_suffix(".unscored.tsv").read_text()
    assert "0\tnonfinite model score" in unscored
    pangolin_env.fixture.output.unlink()
    for path in pangolin_env.fixture.output.parent.glob("result.*.tsv"):
        path.unlink()
    pangolin_env.behaviour[1] = lambda: "gene|Warnings:"
    with pytest.raises(RuntimeError, match="no scores"):
        pangolin_env.runner.main()


def test_outputs_are_bound_and_orphans_moved_aside(pangolin_env, monkeypatch):
    output = pangolin_env.fixture.output
    stale = output.with_suffix(".predictions.tsv")
    stale.write_text("left by an interrupted final write\n")
    monkeypatch.setattr(sys, "argv", [*pangolin_env.args, "--mask", "True"])
    pangolin_env.runner.main()
    config = json.loads(output.read_text())["config"]
    tables = config["output_tables"]
    assert tables["predictions"]["sha256"] == file_sha256(stale)
    assert tables["unscored"]["sha256"] == file_sha256(output.with_suffix(".unscored.tsv"))
    assert len(config["orphaned_tables_moved_aside"]) == 1
    aside = output.parent / config["orphaned_tables_moved_aside"][0]
    assert aside.read_text() == "left by an interrupted final write\n"
    assert not list(output.parent.glob(".*.partial"))


def test_spliceai_thread_pools_are_set_separately(synthetic_run, monkeypatch):
    from mfass import run_spliceai as runner

    module, utils = ModuleType("spliceai"), ModuleType("spliceai.utils")
    utils.Annotator = lambda reference, annotation: object()
    utils.get_delta_scores = lambda *a: ["G|gene|0.1|0.2|0.3|0.4|1|2|3|4"]
    tf = fake_tensorflow()
    monkeypatch.setitem(sys.modules, "spliceai", module)
    monkeypatch.setitem(sys.modules, "spliceai.utils", utils)
    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    monkeypatch.setattr(runner, "_patch_numpy_fromstring", lambda: None)
    monkeypatch.setattr(runner, "installed_identity", lambda: {
        "spliceai_upstream_files_verified": False, "spliceai_upstream_file_mismatches": ["x"]})
    base = ["spliceai", *synthetic_run.args, "--annotation", str(synthetic_run.annotation),
            "--threads", "5", "--inter-threads", "1"]
    monkeypatch.setattr(sys, "argv", [*base, "--require-verified-code"])
    with pytest.raises(SystemExit, match="differs from the pinned upstream"):
        runner.main()
    monkeypatch.setattr(sys, "argv", base)
    runner.main()
    threads = json.loads(synthetic_run.output.read_text())["config"]["tensorflow_threads"]
    assert (threads["intra_op"], threads["inter_op"]) == (5, 1)


def test_spliceai_skip_reason_is_taken_from_its_warning():
    import logging

    from mfass.run_spliceai import _score_variant

    row = {"chr": "chr1", "snp_position_hg38_1based": "5", "ref_allele": "A", "alt_allele": "G"}

    def mismatch(record, ann, distance, mask):
        logging.warning("Skipping record (ref issue): %s", record.pos)  # noqa: LOG015 (as upstream)
        return []

    def broken(record, ann, distance, mask):
        raise KeyError("chr1")

    assert _score_variant(mismatch, row, None, 50, 0) == (
        None, "skipped by SpliceAI: Skipping record (ref issue): 5", "")
    assert _score_variant(lambda *a: [], row, None, 50, 0)[1] == (
        "no annotated gene overlapping the variant")
    assert _score_variant(lambda *a: ["G|g1|0.10|0.20|0.00|0.00|1|2|3|4",
                                      "G|g2|0.05|0.00|0.30|0.00|1|2|3|4"],
                          row, None, 50, 1)[:2] == (0.3, "")
    assert _score_variant(None, {**row, "snp_position_hg38_1based": "NA"}, None, 50, 0)[1] == (
        "no hg38 coordinate")
    with pytest.raises(KeyError):
        _score_variant(broken, row, None, 50, 0)


def test_unexpected_unscored_reason_stops_the_loop():
    from mfass.specialist_run import score_all

    with pytest.raises(RuntimeError, match="unexpected unscored reason"):
        score_all([{"id": "v1"}], lambda i, r: (None, "ValueError: boom", ""))


# Comparison


def _write_predictions(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "group", "label", "score"])
        writer.writerows(rows)


def test_compare_reports_the_common_population_and_enforces_a_minimum(tmp_path, monkeypatch):
    from mfass import compare

    rows = [(f"v{i}", f"g{i % 40}", int(i % 5 == 0), f"{(i * 37 % 101) / 101:.4f}")
            for i in range(150)]
    baseline, candidate = tmp_path / "b.predictions.tsv", tmp_path / "c.predictions.tsv"
    _write_predictions(baseline, rows)
    _write_predictions(candidate, [r if i % 10 else (*r[:3], "") for i, r in enumerate(rows)])
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["compare", "--baseline", str(baseline), "--candidate",
                                      str(candidate), "--draws", "50", "--min-common", "100",
                                      "--out", str(out)])
    compare.main()
    report = json.loads(out.read_text())
    assert report["denominators"]["common"] == 135
    assert report["denominators"]["baseline_only"] == 15
    assert report["denominators"]["common_positives"] == sum(
        r[2] for i, r in enumerate(rows) if i % 10)
    assert len(report["denominators"]["common_id_sha256"]) == 64
    assert report["baseline_sha256"] == file_sha256(baseline)
    assert report["candidate_sha256"] == file_sha256(candidate)
    monkeypatch.setattr(sys, "argv", ["compare", "--baseline", str(baseline), "--candidate",
                                      str(candidate), "--min-common", "136"])
    with pytest.raises(SystemExit, match="at least 136"):
        compare.main()


# Freeze gates


def _verified_code():
    from mfass.pangolin_patch import PATCH_ID, PATCHED_SOURCE_SHA256
    return {"spliceai": {"spliceai_source_identity": "upstream-b3c7f17",
                         "spliceai_upstream_files_verified": True,
                         "spliceai_upstream_file_mismatches": []},
            "pangolin": {"pangolin_source_identity": PATCH_ID,
                         "pangolin_source_sha256": PATCHED_SOURCE_SHA256,
                         "pangolin_upstream_files_verified": True,
                         "pangolin_upstream_file_mismatches": []}}


@pytest.mark.parametrize("tool, change, message", [
    ("pangolin", {"pangolin_source_identity": "upstream-5cf94b8-unpatched"}, "reviewed patch"),
    ("pangolin", {"pangolin_upstream_files_verified": False,
                  "pangolin_upstream_file_mismatches": ["model.py"]}, "model.py"),
    ("spliceai", {"spliceai_upstream_files_verified": False,
                  "spliceai_upstream_file_mismatches": ["models/spliceai3.h5"]}, "spliceai3"),
    ("spliceai", {"spliceai_source_identity": "unrecognised"}, "SpliceAI source"),
])
def test_freeze_identity_gate(tool, change, message):
    code = _verified_code()
    matched_study.check_identities(code)
    code[tool].update(change)
    with pytest.raises(SystemExit, match=message):
        matched_study.check_identities(code)


def test_real_upstream_pangolin_source_is_refused(tmp_path):
    """Uses the pinned upstream bytes when the checkout is available."""
    import os
    import subprocess

    from mfass.pangolin_patch import UPSTREAM_REVISION, source_identity
    checkout = os.environ.get("MFASS_PANGOLIN_CHECKOUT")
    if not checkout:
        pytest.skip("MFASS_PANGOLIN_CHECKOUT not set")
    source = tmp_path / "pangolin.py"
    source.write_bytes(subprocess.run(
        ["git", "-C", checkout, "show", f"{UPSTREAM_REVISION}:pangolin/pangolin.py"],
        check=True, capture_output=True).stdout)
    code = _verified_code()
    code["pangolin"].update(source_identity(source))
    with pytest.raises(SystemExit, match="reviewed patch"):
        matched_study.check_identities(code)


def test_thread_plan_budget():
    plan = matched_study.thread_plan(5, 1, 6, 1, 6)
    assert plan["spliceai"]["env"]["OMP_NUM_THREADS"] == "5"
    assert (plan["pangolin"]["intra_op"], plan["pangolin"]["inter_op"]) == (6, 1)
    for bad in ((6, 1, 6, 1, 6), (5, 1, 7, 1, 6), (0, 1, 6, 1, 6)):
        with pytest.raises(SystemExit):
            matched_study.thread_plan(*bad)


def test_eligibility_must_come_from_this_annotation_and_source():
    from mfass.pangolin_patch import PATCHED_SOURCE_SHA256
    resources = {"cohort": "c", "split": "s", "fasta": "f"}
    good = {"test_variants": 8324, "labels_read": False, "annotation_dir_summary_sha256": "a",
            "pangolin_source_sha256": PATCHED_SOURCE_SHA256, "cohort_sha256": "c",
            "split_sha256": "s", "fasta_sha256": "f"}
    matched_study.check_eligibility(good, "a", resources)
    for key, value in (("annotation_dir_summary_sha256", "b"), ("pangolin_source_sha256", "x"),
                       ("fasta_sha256", "g"), ("labels_read", None), ("test_variants", 10)):
        with pytest.raises(SystemExit, match="eligibility"):
            matched_study.check_eligibility({**good, key: value}, "a", resources)


# Synthetic study end to end


N_TEST = 150


@pytest.fixture
def study(tmp_path, monkeypatch):
    """A synthetic manifest over 150 canonical test rows and four conditions."""
    cohort, split = tmp_path / "cohort.tsv", tmp_path / "split.tsv"
    ids = [f"v{i:03d}" for i in range(N_TEST + 20)]
    with cohort.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "sdv"])
        writer.writerows([[v, int(i % 5 == 0)] for i, v in enumerate(ids)])
    with split.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "group", "split"])
        writer.writerows([[v, f"g{i % 40}", "test" if i < N_TEST else "train"]
                          for i, v in enumerate(ids)])
    code = _verified_code()
    code["spliceai"]["spliceai_version"] = "1.3.1"
    out_dir = tmp_path / "runs"
    manifest = {
        "frozen_at_utc": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(
            timespec="seconds"),
        "draft": False,
        "paths": {"cohort": str(cohort), "split": str(split)},
        "resources_sha256": {"spliceai_annotation": "s", "pangolin_db": "p", "fasta": "f"},
        "distance": 50, "code": code, "out_dir": str(out_dir),
        "threads": matched_study.thread_plan(5, 1, 6, 1, 6),
        "contrasts": [{"name": n, "candidate": c, "baseline": b}
                      for n, c, b in matched_study.CONTRASTS],
        "steps": {},
    }
    for name, candidate, baseline in matched_study.CONTRASTS:
        def predictions(c):
            return str(out_dir / c / f"{matched_study.CONDITIONS[c]['stem']}.predictions.tsv")
        manifest["steps"][name] = [
            sys.executable, "-m", "mfass.compare", "--baseline", predictions(baseline),
            "--candidate", predictions(candidate), "--draws", "50", "--seed", "20260914",
            "--min-common", "100", "--out", str(out_dir / "contrasts" / f"{name}.json")]
    for condition in matched_study.CONDITIONS:
        manifest["steps"][condition] = ["--out", str(matched_study._result_path(manifest,
                                                                                  condition))]
    monkeypatch.setattr(matched_study, "CANONICAL_TEST_ROWS", N_TEST)
    monkeypatch.setattr(matched_study, "_check_frozen", lambda manifest, step: None)
    return manifest


def fake_condition(manifest, condition):
    """Write a condition's outputs through the real checkpoint and output writer."""
    from mfass.specialist_run import score_all, write_outputs
    from rewirebench.metrics import point_metrics
    from rewirebench.results import BenchmarkResult

    spec = matched_study.CONDITIONS[condition]
    spliceai = spec["tool"] == "spliceai"
    canonical = matched_study.canonical_test_rows(manifest)
    test = [{"id": r["id"], "sdv": r["label"]} for r in canonical]
    groups = {r["id"]: r["group"] for r in canonical}
    offset = {"S0": 0, "S1": 3, "P0": 7, "P1": 11}[condition]

    def score_one(i, row):
        if i % 37 == 5:
            return None, "no annotated gene overlapping the variant", ""
        return round(((i * 13 + offset) % 97) / 97, 2), "", "raw"

    out = matched_study._result_path(manifest, condition)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = Checkpoint(out.parent / "checkpoint.tsv", {"condition": condition})
    scores, unscored, seconds = score_all(test, score_one, log)
    log.close()
    ok = np.isfinite(scores)
    labels = [int(r["sdv"]) for r, s in zip(test, scores) if np.isfinite(s)]
    threads = manifest["threads"][spec["tool"]]
    config = {
        "scope": "full", "selected_test_rows": N_TEST,
        "cohort_sha256": matched_study.COHORT_SHA256, "split_sha256": matched_study.SPLIT_SHA256,
        "annotation_sha256": manifest["resources_sha256"][
            "spliceai_annotation" if spliceai else "pangolin_db"],
        "reference_sha256": manifest["resources_sha256"]["fasta"],
        ("mask_M" if spliceai else "mask_m"): int(spec["mask"]) if spliceai else spec["mask"],
        ("distance_D" if spliceai else "distance_d"): 50,
        ("tensorflow_threads" if spliceai else "torch_threads"): {
            "intra_op": threads["intra_op"], "inter_op": threads["inter_op"]},
        "checkpoint": {"sha256": file_sha256(out.parent / "checkpoint.tsv")},
        **manifest["code"][spec["tool"]],
    }
    result = BenchmarkResult(
        benchmark="mfass-v2", method=condition, family="specialist", description="synthetic",
        split="synthetic", metrics=point_metrics(labels, scores[ok], 100),
        coverage={"scored": int(ok.sum()), "unscored": int((~ok).sum()), "denominator": N_TEST},
        timing_seconds={"score_test": seconds},
        independent_groups=len({groups[r["id"]] for r, s in zip(test, scores) if np.isfinite(s)}),
        config=config)
    write_outputs(result, out, test, groups, scores, unscored)


def _results(manifest):
    return {c: matched_study._result(manifest, c) for c in matched_study.CONDITIONS}


def test_verify_binds_rows_tables_checkpoint_and_metrics(study):
    for condition in matched_study.CONDITIONS:
        fake_condition(study, condition)
    canonical = matched_study.canonical_test_rows(study)
    report = matched_study.verify_results(study, _results(study), canonical)
    assert report["passed"], report["problems"]
    assert report["metrics_reported"] is False
    assert report["tables"]["S0"]["unscored_reasons"] == {
        "no annotated gene overlapping the variant": 4}


def _tamper(study, condition, key, edit, rebind=False):
    from mfass.specialist_run import output_paths
    result_path = matched_study._result_path(study, condition)
    path = output_paths(result_path)[key] if key != "checkpoint" else (
        result_path.parent / "checkpoint.tsv")
    path.write_text(edit(path.read_text()))
    if rebind:
        result = json.loads(result_path.read_text())
        result["config"]["output_tables"][key]["sha256"] = file_sha256(path)
        result_path.write_text(json.dumps(result))


def _swap_first_score(text):
    lines = text.splitlines(keepends=True)
    fields = lines[1].rstrip("\r\n").split("\t")
    fields[3] = "0.9999"
    lines[1] = "\t".join(fields) + "\r\n"
    return "".join(lines)


@pytest.mark.parametrize("condition, key, edit, rebind, problem", [
    ("S1", "predictions", _swap_first_score, False, "predictions table bytes differ"),
    ("S1", "predictions", _swap_first_score, True, "checkpoint and predictions disagree"),
    ("P0", "predictions", lambda t: "".join(t.splitlines(keepends=True)[:-1]), True,
     "not the canonical test IDs"),
    ("P0", "predictions", lambda t: t.replace("\tg1\t", "\tg2\t", 1), True, "labels or groups"),
    ("P1", "unscored", lambda t: t.replace("no annotated gene", "ValueError: boom"), True,
     "unexpected unscored reasons"),
    ("S0", "checkpoint", lambda t: t + "", False, None),
    ("S0", "checkpoint", lambda t: t.replace("\traw", "\tRAW", 1), False, "checkpoint bytes"),
])
def test_verify_rejects_tampered_outputs(study, condition, key, edit, rebind, problem):
    for c in matched_study.CONDITIONS:
        fake_condition(study, c)
    _tamper(study, condition, key, edit, rebind)
    report = matched_study.verify_results(study, _results(study),
                                          matched_study.canonical_test_rows(study))
    if problem is None:
        assert report["passed"], report["problems"]
    else:
        assert any(problem in p for p in report["problems"]), report["problems"]


def test_verify_rejects_metric_and_coverage_drift(study):
    for c in matched_study.CONDITIONS:
        fake_condition(study, c)
    path = matched_study._result_path(study, "P1")
    result = json.loads(path.read_text())
    result["metrics"]["auroc"] += 0.01
    result["coverage"]["scored"] -= 1
    path.write_text(json.dumps(result))
    problems = matched_study.verify_results(study, _results(study),
                                            matched_study.canonical_test_rows(study))["problems"]
    assert any("metric auroc" in p for p in problems)
    assert any("coverage differs" in p for p in problems)


@pytest.mark.parametrize("condition, change, problem", [
    ("S1", {"extra_setting": 4}, "S0/S1 differ beyond mask"),
    ("P1", {"reference_sha256": "other"}, "P1: reference_sha256"),
    ("P0", {"pangolin_source_identity": "upstream-5cf94b8-unpatched"},
     "P0: pangolin_source_identity"),
    ("S0", {"mask_M": 1}, "S0: mask_M"),
    ("P0", {"torch_threads": {"intra_op": 10, "inter_op": 1}}, "P0: thread configuration"),
    ("S1", {"selected_test_rows": 10}, "S1: not a full canonical run"),
])
def test_verify_rejects_configuration_drift(study, condition, change, problem):
    for c in matched_study.CONDITIONS:
        fake_condition(study, c)
    results = _results(study)
    results[condition]["config"].update(change)
    report = matched_study.verify_results(study, results)
    assert any(p.startswith(problem) for p in report["problems"]), report["problems"]


def test_result_created_before_freeze_is_rejected(study):
    for c in matched_study.CONDITIONS:
        fake_condition(study, c)
    study["frozen_at_utc"] = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    problems = matched_study.verify_results(study, _results(study))["problems"]
    assert any("predates the frozen manifest" in p for p in problems)


def test_all_runs_in_order_resumes_and_fails_closed(study, monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(study))
    real_execute = matched_study._execute
    ran, fail = [], {"P0"}

    def execute(manifest, digest, step):
        ran.append(step)
        if step in fail:
            raise SystemExit(f"{step} failed with exit code 1")
        if step in matched_study.CONDITIONS:
            return fake_condition(manifest, step)
        return real_execute(manifest, digest, step)

    monkeypatch.setattr(matched_study, "_execute", execute)
    args = SimpleNamespace(manifest=str(manifest_path))
    with pytest.raises(SystemExit, match="P0 failed"):
        matched_study.run_all(args)
    assert ran == ["S0", "S1", "P0"]
    out = tmp_path / "runs"
    assert not (out / "contrasts").exists() and not (out / "report.json").exists()

    fail.clear()
    ran.clear()
    matched_study.run_all(args)
    assert ran == ["P0", "P1", "S1-S0", "P1-P0", "P0-S0"]  # S0 and S1 were not rerun
    report = json.loads((out / "report.json").read_text())
    assert set(report["contrasts"]) == {"S1-S0", "P1-P0", "P0-S0"}
    assert report["contrasts"]["P0-S0"]["denominators"]["common"] == N_TEST - 4
    assert "tied_at_cutoff" in report["conditions"]["S0"]["ties"]
    assert (out / "report.md").read_text().startswith("# MFASS annotation-matched")
    assert (out / "logs" / "S1-S0.log").exists()
    events = [json.loads(line) for line in (out / "logs" / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "complete"
    assert {e["manifest_sha256"] for e in events} == {file_sha256(manifest_path)}

    ran.clear()
    matched_study.run_all(args)
    assert ran == []  # everything already complete and still verifies

    contrast = out / "contrasts" / "P1-P0.json"
    data = json.loads(contrast.read_text())
    data["baseline_sha256"] = "0" * 64
    contrast.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="other prediction bytes"):
        matched_study.run_all(args)


def test_tie_summary():
    values = [0.9] * 3 + [0.5] * 200 + [0.1] * 10
    summary = matched_study.tie_summary(values, capacity=100)
    assert summary == {"scored": 213, "score_at_capacity": 0.5, "above_cutoff": 3,
                       "tied_at_cutoff": 200, "slots_filled_from_ties": 97,
                       "tie_break": summary["tie_break"]}
    assert matched_study.tie_summary([0.2] * 5)["capacity_exceeds_list"] is True


def test_draft_manifests_are_never_executed(study):
    for draft in (True, None):
        manifest = {**study, "draft": draft} if draft else {
            k: v for k, v in study.items() if k != "draft"}
        with pytest.raises(SystemExit, match="draft"):
            matched_study.run_step(manifest, "digest", "S0")
        assert matched_study.run_step(manifest, "digest", "S0", check_only=True) == "checked"


@pytest.mark.parametrize("step, threads", [("S0", "5"), ("P1", "6"), ("S1-S0", "6")])
def test_every_step_runs_with_budgeted_thread_environment(study, monkeypatch, step, threads):
    seen = {}

    def fake_run(argv, stdout, stderr, env, check):
        seen.update(env)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(matched_study.subprocess, "run", fake_run)
    monkeypatch.setenv("OMP_NUM_THREADS", "10")
    matched_study._execute(study, "digest", step)
    assert {key: seen[key] for key in matched_study.THREAD_ENV_KEYS} == {
        key: threads for key in matched_study.THREAD_ENV_KEYS}


def test_check_frozen_refuses_drift_before_a_step(tmp_path, monkeypatch):
    """The pre-step gate re-hashes runner files, registration, resources and code."""
    repo = Path(matched_study._git("rev-parse", "--show-toplevel"))
    resource = tmp_path / "cohort.tsv"
    resource.write_text("cohort")
    names = ("cohort", "split", "fasta", "fai", "spliceai_annotation")
    identity = {"spliceai_source_identity": "upstream-b3c7f17"}
    manifest = {
        "runner_files_sha256": {"uv.lock": file_sha256(repo / "uv.lock")},
        "registration": "docs/mfass-matched-study-registration.md",
        "registration_sha256": file_sha256(repo / "docs/mfass-matched-study-registration.md"),
        "paths": dict.fromkeys(names, str(resource)),
        "resources_sha256": dict.fromkeys(names, file_sha256(resource)),
        "environments": {"spliceai": {"python": "py", "distributions": ["a==1"]}},
        "code": {"spliceai": identity},
    }
    answers = {"identity": identity, "distributions": ["a==1"]}
    monkeypatch.setattr(matched_study, "_python_json", lambda python, snippet: (
        answers["distributions"] if snippet == matched_study.DISTRIBUTIONS_SNIPPET
        else answers["identity"]))
    matched_study._check_frozen(manifest, "S0")
    matched_study._check_frozen(manifest, "S1-S0")
    for change, expected in (
        (lambda m: m["runner_files_sha256"].update({"uv.lock": "0" * 64}), "uv.lock"),
        (lambda m: m.update(registration_sha256="0" * 64), "registration"),
        (lambda m: resource.write_text("changed"), "cohort"),
        (lambda m: answers.update(identity={"spliceai_source_identity": "unrecognised"}),
         "spliceai code or weights"),
        (lambda m: answers.update(distributions=["a==2"]), "spliceai environment"),
    ):
        changed = json.loads(json.dumps(manifest))
        change(changed)
        with pytest.raises(SystemExit, match=expected):
            matched_study._check_frozen(changed, "S0")
        resource.write_text("cohort")
        answers.update(identity=identity, distributions=["a==1"])
