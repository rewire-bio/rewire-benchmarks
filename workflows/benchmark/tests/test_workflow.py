"""Offline orchestration contracts; optional real Nextflow test requires Java and Nextflow."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("workflow_stage", ROOT / "bin/workflow_stage.py")
STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE)


def example():
    job = json.loads((ROOT / "examples/smoke.jobs.json").read_text())[0]
    job["environment_id"] = "pytest-local-environment"
    return job


def test_credentials_ids_and_assets_are_rejected(tmp_path):
    job = example()
    for invalid in [dict(job, id="../../escape"), dict(job, adapter_options={"api_key": "secret"}),
                    dict(job, adapter="module:Class;bad"), dict(job, environment_id="")]:
        with pytest.raises(ValueError):
            STAGE.validate_job(invalid)
    with pytest.raises(ValueError, match="inside"):
        STAGE.resolve_assets({"checkpoint": "asset:../private"}, tmp_path)
    with pytest.raises(ValueError, match="Missing"):
        STAGE.resolve_assets({"checkpoint": "asset:missing"}, tmp_path)


def test_native_stages_preserve_results_and_block_smoke_exports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    shutil.copy(ROOT / "examples/smoke.csv", tmp_path / "input/source.csv")
    job = example()
    for stage in ("prepare", "run", "evaluate", "export"):
        STAGE.execute(stage, job)
    original = json.loads((tmp_path / "runs/model/report.json").read_text())
    checked = json.loads((tmp_path / "evaluated/report.json").read_text())
    assert checked == original
    assert checked["scope"] == "smoke"
    assert (tmp_path / "bundles/NOT_SUBMITTABLE.txt").exists()
    assert not (tmp_path / "bundles/contribution.json").exists()
    receipt = json.loads((tmp_path / "evaluated/workflow-receipt.json").read_text())
    assert receipt["rescoring_matches"] is True
    assert receipt["submission_attempted"] is False


def test_changed_predictions_fail_rescoring(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    shutil.copy(ROOT / "examples/smoke.csv", tmp_path / "input/source.csv")
    job = example()
    STAGE.execute("prepare", job)
    STAGE.execute("run", job)
    path = tmp_path / "runs/model/predictions.json"
    predictions = json.loads(path.read_text())
    predictions[next(iter(predictions))] = 999
    path.write_text(json.dumps(predictions))
    with pytest.raises(ValueError, match="rescoring changed"):
        STAGE.execute("evaluate", job)


@pytest.mark.parametrize("status", ["failed", "blocked"])
def test_zero_exit_baseline_failure_is_rejected_in_run(tmp_path, monkeypatch, status):
    monkeypatch.chdir(tmp_path)
    job = dict(example(), kind="baseline", baseline="training-mean-v1")

    def zero_exit_cli(*args, **kwargs):
        Path("runs").mkdir()
        Path("runs/baseline-manifest.json").write_text(json.dumps({
            "status": "incomplete",
            "baselines": [{"baseline_id": job["baseline"], "status": status}],
        }))
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(STAGE.subprocess, "run", zero_exit_cli)
    with pytest.raises(ValueError, match="Selected baseline did not evaluate successfully"):
        STAGE.execute("run", job)
    assert not Path("evaluated").exists()


def test_workflow_has_no_upload_or_shard_averaging():
    text = (ROOT / "main.nf").read_text()
    assert "process PREPARE" in text and "process EXPORT" in text
    assert "cache 'deep'" in text
    assert "submit(" not in text and "--token" not in text
    assert "@sha256:" in text and "google_service_account" in text


def test_nextflow_real_smoke_and_resume(tmp_path):
    if not shutil.which("nextflow") or not shutil.which("java"):
        pytest.skip("Nextflow and Java required")
    if subprocess.run(["java", "-version"], check=False, capture_output=True).returncode:
        pytest.skip("No usable Java runtime; set JAVA_HOME")
    job = example()
    job["source"] = str(ROOT / "examples/smoke.csv")
    manifest = tmp_path / "jobs.json"
    baseline = dict(json.loads(json.dumps(job)), id="synthetic-baseline", kind="baseline", baseline="training-mean-v1")
    manifest.write_text(json.dumps([job, baseline]))
    checkout = tmp_path / "checkout"
    pipeline = checkout / "workflows/benchmark"
    shutil.copytree(ROOT, pipeline, ignore=shutil.ignore_patterns("__pycache__"))
    for source in ("packages/rewirebench/src", "benchmarks/mfass/src"):
        shutil.copytree(ROOT.parents[1] / source, checkout / source,
                        ignore=shutil.ignore_patterns("__pycache__"))
    command = ["nextflow", "run", str(pipeline), "-profile", "native", "--jobs", str(manifest),
               "--environment_id", "pytest-smoke", "--outdir", str(tmp_path / "output"),
               "-work-dir", str(tmp_path / "work"), "-ansi-log", "false"]
    first = subprocess.run(command, cwd=tmp_path, check=False, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stdout + first.stderr
    second = subprocess.run(command + ["-resume"], cwd=tmp_path, check=False, capture_output=True, text=True, timeout=120)
    assert second.returncode == 0, second.stdout + second.stderr
    assert second.stdout.count("Cached process") == 8
    receipts = list((tmp_path / "output").rglob("workflow-receipt.json"))
    assert len(receipts) == 2
    assert not list((tmp_path / "output").rglob("contribution.json"))

    job["model"]["name"] = "Changed model configuration invalidates cached job"
    manifest.write_text(json.dumps([job, baseline]))
    changed = subprocess.run(command + ["-resume"], cwd=tmp_path, check=False, capture_output=True,
                             text=True, timeout=120)
    assert changed.returncode == 0, changed.stdout + changed.stderr
    assert changed.stdout.count("Cached process") == 4
    assert changed.stdout.count("Submitted process") == 4
    assert len(list((tmp_path / "output").rglob("workflow-receipt.json"))) == 3

    source_code = checkout / "packages/rewirebench/src/rewirebench/sdk.py"
    source_code.write_text(source_code.read_text() + "\n# Cache invalidation fixture\n")
    code_changed = subprocess.run(command + ["-resume"], cwd=tmp_path, check=False,
                                  capture_output=True, text=True, timeout=120)
    assert code_changed.returncode == 0, code_changed.stdout + code_changed.stderr
    assert code_changed.stdout.count("Submitted process") == 8
    assert "Cached process" not in code_changed.stdout
    assert len(list((tmp_path / "output").rglob("workflow-receipt.json"))) == 5
