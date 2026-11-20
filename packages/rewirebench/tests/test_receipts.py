"""Every run writes a receipt, and verify recomputes it instead of trusting it."""
from __future__ import annotations

import datetime
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from rewirebench import cli, receipts, recipes

ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "recipes"
FIXTURE = ROOT / "workflows/benchmark/examples"
SMOKE = "flip2-synthetic-smoke"


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[Any, int]:
    code = 0
    try:
        cli.main(list(argv))
    except SystemExit as stop:
        code = int(stop.code or 0)
    out = capsys.readouterr().out
    return (json.loads(out) if out.strip() else None), code


def statuses(result: dict[str, Any]) -> dict[str, str]:
    return {entry["check"]: entry["status"] for entry in result["results"]}


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One offline run of the synthetic recipe, shared by the read-only tests."""
    folder = tmp_path_factory.mktemp("runs") / "smoke"
    loaded = recipes.get_recipe(SMOKE, RECIPES)
    receipt = receipts.run_recipe(loaded, output=folder, recipes_dir=RECIPES, started_by="agent")
    assert receipt.checks_passed, receipt.checks
    return folder


@pytest.fixture
def run_copy(smoke_run: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(smoke_run, tmp_path / "run"))


def test_receipt_records_everything_the_run_depends_on(smoke_run: Path):
    data = json.loads((smoke_run / "receipt.json").read_text())
    loaded = recipes.get_recipe(SMOKE, RECIPES)
    assert data["schema_version"] == "1.0" and data["kind"] == "rewire_run_receipt"
    assert data["recipe"]["id"] == SMOKE
    assert data["recipe"]["sha256"] == loaded.sha256
    assert data["recipe"]["path"] == "recipes/flip2/flip2-synthetic-smoke.yaml"
    # The pointer has a commit only when the file at HEAD has the bytes that ran.
    assert data["recipe"]["commit"] in (None, data["code"]["commit"])
    assert data["code"]["commit"] is None or len(data["code"]["commit"]) == 40
    assert len(data["code"]["sdk_code_sha256"]) == 64
    assert data["environment"]["images"] == {"linux/amd64": None, "linux/arm64": None}
    assert data["inputs"] == [{
        "name": "smoke.csv", "source": "repository:workflows/benchmark/examples/smoke.csv",
        "obtained": "repository",
        "sha256": recipes.sha256_file(FIXTURE / "smoke.csv"),
        "bytes": (FIXTURE / "smoke.csv").stat().st_size,
    }]
    hardware = data["hardware"]
    assert hardware["architecture"] and hardware["cpu"]["logical_cores"] >= 1
    assert hardware["memory_bytes"] > 0 and isinstance(hardware["gpus"], list)
    assert data["wall_seconds"] > 0
    assert data["peak_memory_bytes"] > 10 * 2**20
    assert [command["name"] for command in data["commands"]] == ["prepare", "run-baselines"]
    assert all(command["exit_code"] == 0 for command in data["commands"])
    # The recipe's own argv is recorded, so the receipt holds no local path.
    assert data["commands"][0]["argv"][:3] == ["rewirebench", "prepare", "flip2-fitness-v1"]
    assert "{work}/prepared" in data["commands"][0]["argv"]
    assert str(smoke_run) not in json.dumps(data)
    listed = {record["path"]: record for record in data["outputs"]}
    report = "baselines/training-mean-v1/report.json"
    assert listed[report]["sha256"] == recipes.sha256_file(smoke_run / report)
    assert "receipt.json" not in listed and "SHA256SUMS" not in listed
    assert data["metrics"] == {"n": 3.0, "coverage_scored": 3.0}
    assert data["checks_passed"] is True and len(data["checks"]) == 5
    assert data["cost"] == {"paid_compute": False, "amount": None, "currency": None,
                            "reason": "owned hardware", "hourly_rate": None,
                            "rate_source": None, "rate_date": None}
    assert data["backend"] == "local" and data["started_by"] == "agent"
    assert data["scope"] == "synthetic_smoke"
    assert data["verified_on"] == data["started_at"][:10]
    sums = (smoke_run / "SHA256SUMS").read_text().splitlines()
    assert f"{recipes.sha256_file(smoke_run / 'receipt.json')}  ./receipt.json" in sums
    # The steps ran as nested runs: the recipe's receipt is the only one in the folder.
    assert [path.name for path in smoke_run.rglob("receipt.json")] == ["receipt.json"]


def test_verify_passes_on_an_untouched_run(smoke_run: Path, capsys: pytest.CaptureFixture[str]):
    result, code = run_cli(capsys, "receipt", "verify", str(smoke_run), "--strict",
                           "--recipes-dir", str(RECIPES), "--inputs", str(FIXTURE))
    found = statuses(result)
    assert list(found) == ["sha256sums", "outputs", "run", "recipe", "inputs", "checks",
                           "metrics", "commits"]
    # `commits` depends on whether this checkout is clean, so it decides --strict here.
    commits = found.pop("commits")
    assert set(found.values()) == {"match"} and result["ok"]
    assert result["complete"] == (commits == "match") and code == (0 if result["complete"] else 1)
    # Without --inputs the receipt's inputs are still compared with the recipe's pins.
    result, code = run_cli(capsys, "receipt", "verify", str(smoke_run),
                           "--recipes-dir", str(RECIPES))
    assert code == 0 and statuses(result)["inputs"] == "match"
    # The path of receipt.json works as well as its folder.
    by_file, code = run_cli(capsys, "receipt", "verify", str(smoke_run / "receipt.json"),
                            "--recipes-dir", str(RECIPES))
    assert code == 0 and by_file == result


def test_verify_without_the_recipe_cannot_be_complete(smoke_run: Path):
    result = receipts.verify(smoke_run, recipes_dir=None)
    found = statuses(result)
    assert found["recipe"] == found["inputs"] == found["checks"] == "not_available"
    assert found["metrics"] == "not_available" and found["run"] == "match"
    assert result["ok"] and not result["complete"]


def test_strict_needs_a_clean_code_commit_and_a_committed_recipe(smoke_run: Path):
    receipt = receipts.load_receipt(smoke_run)
    clean = receipt.model_copy(update={
        "code": receipt.code.model_copy(update={"commit": "a" * 40, "dirty": False}),
        "recipe": receipt.recipe.model_copy(update={"commit": "a" * 40}),
    })
    assert receipts._verify_commits(clean)["status"] == "match"
    for update in ({"code": clean.code.model_copy(update={"commit": None})},
                   {"code": clean.code.model_copy(update={"dirty": True})},
                   {"code": clean.code.model_copy(update={"dirty": None})},
                   {"recipe": clean.recipe.model_copy(update={"commit": None})}):
        assert receipts._verify_commits(clean.model_copy(update=update))["status"] \
            == "not_available"


def test_verify_catches_a_changed_output(run_copy: Path, capsys: pytest.CaptureFixture[str]):
    target = run_copy / "baselines/training-mean-v1/predictions.json"
    target.write_text(target.read_text().replace("0.3", "0.4"))
    result, code = run_cli(capsys, "receipt", "verify", str(run_copy),
                           "--recipes-dir", str(RECIPES))
    assert code == 1 and not result["ok"]
    assert statuses(result)["outputs"] == statuses(result)["sha256sums"] == "mismatch"


def test_verify_recomputes_the_metric_checks(run_copy: Path):
    """Rewriting the report and both digest lists still fails: the value is re-read."""
    report = run_copy / "baselines/training-mean-v1/report.json"
    data = json.loads(report.read_text())
    data["metrics"]["n"] = 99
    report.write_text(json.dumps(data))
    receipt = json.loads((run_copy / "receipt.json").read_text())
    for record in receipt["outputs"]:
        if record["path"] == "baselines/training-mean-v1/report.json":
            record.update(sha256=recipes.sha256_file(report), bytes=report.stat().st_size)
    (run_copy / "receipt.json").write_text(json.dumps(receipt))
    receipts.write_sums(run_copy)
    result = receipts.verify(run_copy, recipes_dir=RECIPES)
    assert statuses(result)["sha256sums"] == statuses(result)["outputs"] == "match"
    assert statuses(result)["checks"] == "mismatch" and not result["ok"]


def test_verify_catches_edited_sums_and_unlisted_files(run_copy: Path):
    sums = run_copy / "SHA256SUMS"
    sums.write_text(sums.read_text().replace("./receipt.json", "./other.json"))
    assert statuses(receipts.verify(run_copy, recipes_dir=RECIPES))["sha256sums"] == "mismatch"
    receipts.write_sums(run_copy)
    assert receipts.verify(run_copy, recipes_dir=RECIPES)["ok"]
    (run_copy / "extra.txt").write_text("added after the run")
    result = receipts.verify(run_copy, recipes_dir=RECIPES)
    assert statuses(result)["sha256sums"] == statuses(result)["outputs"] == "mismatch"
    (run_copy / "extra.txt").unlink()
    intact = sums.read_text()
    # A second line for the same file must not win over the first.
    sums.write_text(f"{'0' * 64}  ./receipt.json\n" + intact)
    result = receipts.verify(run_copy, recipes_dir=RECIPES)
    assert statuses(result)["sha256sums"] == "mismatch" and "twice" in result["results"][0]["detail"]
    sums.write_text(intact + "not a digest line\n")
    assert statuses(receipts.verify(run_copy, recipes_dir=RECIPES))["sha256sums"] == "mismatch"
    # Every run writes SHA256SUMS, so a folder without one has been changed.
    sums.unlink()
    result = receipts.verify(run_copy, recipes_dir=RECIPES)
    assert statuses(result)["sha256sums"] == "mismatch" and not result["ok"]


def rewrite_receipt(folder: Path, change: Any) -> None:
    """Edit receipt.json and bring SHA256SUMS in line, as someone hiding an edit would."""
    data = json.loads((folder / "receipt.json").read_text())
    change(data)
    (folder / "receipt.json").write_text(json.dumps(data))
    receipts.write_sums(folder)


def test_verify_catches_edited_metrics_and_inputs(run_copy: Path, tmp_path: Path):
    other = Path(shutil.copytree(run_copy, tmp_path / "other"))
    rewrite_receipt(run_copy, lambda data: data["metrics"].update(n=99.0))
    result = receipts.verify(run_copy, recipes_dir=RECIPES)
    found = statuses(result)
    assert found["sha256sums"] == found["outputs"] == found["checks"] == "match"
    assert found["metrics"] == "mismatch" and not result["ok"]
    # Without the recipe the metrics are still compared with the receipt's own checks.
    result = receipts.verify(run_copy, recipes_dir=None)
    assert statuses(result)["metrics"] == "mismatch" and not result["ok"]

    rewrite_receipt(other, lambda data: data["inputs"][0].update(sha256="b" * 64))
    result = receipts.verify(other, recipes_dir=RECIPES)
    assert statuses(result)["inputs"] == "mismatch" and not result["ok"]
    rewrite_receipt(other, lambda data: data["inputs"].clear())
    assert statuses(receipts.verify(other, recipes_dir=RECIPES))["inputs"] == "mismatch"


def test_verify_catches_a_changed_recipe_and_changed_inputs(run_copy: Path, tmp_path: Path):
    copied = Path(shutil.copytree(RECIPES, tmp_path / "recipes"))
    recipe = copied / "flip2/flip2-synthetic-smoke.yaml"
    recipe.write_text(recipe.read_text().replace("value: 3\n", "value: 4\n"))
    result = receipts.verify(run_copy, recipes_dir=copied)
    assert statuses(result)["recipe"] == "mismatch" and not result["ok"]
    assert statuses(result)["checks"] == "not_available"
    result = receipts.verify(run_copy, recipes_dir=None)
    assert statuses(result)["recipe"] == "not_available" and result["ok"]
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/smoke.csv").write_text("sequence,target\nAAA,1\n")
    result = receipts.verify(run_copy, recipes_dir=RECIPES, inputs=tmp_path / "inputs")
    assert statuses(result)["inputs"] == "mismatch" and not result["ok"]


def test_verify_rejects_a_receipt_that_breaks_the_schema(run_copy: Path):
    receipt = json.loads((run_copy / "receipt.json").read_text())
    receipt["cost"] = {"paid_compute": False, "amount": 1.5, "currency": "USD"}
    (run_copy / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValidationError, match="Unpaid compute carries no money field"):
        receipts.verify(run_copy, recipes_dir=RECIPES)


WRITE = ("import json, pathlib, sys; target = pathlib.Path(sys.argv[1]) / 'result.json'; "
         "target.write_text(json.dumps({'nested': [%s]}))")
SCRIPTS = {
    "write": WRITE % "2.004",
    "write-nan": WRITE % "float('nan')",
    "fail": "raise SystemExit(3)",
    # The step starts a child of its own; the timeout must stop that one too.
    "sleep": ("import subprocess, sys, time; "
              "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
              "time.sleep(60)"),
    "environment": WRITE % "2.004" + "; import os; print(sorted(os.environ.items()))",
    "write-receipt": WRITE % "2.004" + "; (target.parent / 'receipt.json').write_text('{}')",
}


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A recipe step can only be a rewirebench command, so the tests swap the process
    behind `rewirebench evaluate <script>` for a short script."""

    def program(argv: list[str]) -> list[str]:
        assert argv[:2] == ["rewirebench", "evaluate"]
        return [sys.executable, "-c", SCRIPTS[argv[2]], *argv[3:]]

    monkeypatch.setattr(receipts, "_program", program)


def custom_recipes(tmp_path: Path, script: str = "write", **fields: Any) -> Path:
    """A recipes directory with one recipe that runs a test script on a small input."""
    argv = ["rewirebench", "evaluate", script, "{output}"]
    root = tmp_path / "checkout"
    (root / "recipes/test").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "data/input.txt").write_text("pinned input\n")
    data = yaml.safe_load((RECIPES / "flip2/flip2-synthetic-smoke.yaml").read_text())
    data.update(
        id="custom", purpose="generate_and_evaluate", rewirebench_protocol=None,
        inputs=[{"name": "input.txt", "repo_path": "data/input.txt",
                 "sha256": recipes.sha256_file(root / "data/input.txt")}],
        commands=[{"name": "step", "argv": argv}, {"name": "after", "argv": argv}],
        outputs=["result.json"],
        expected_output={"description": "one value", "values": [{
            "label": "Value", "metric_id": "value", "file": "result.json",
            "key": ["nested", 0], "value": 2.0, "tolerance": 0.01}]},
    )
    data.update(fields)
    (root / "recipes/test/custom.yaml").write_text(yaml.safe_dump(data))
    return root / "recipes"


def run_custom(tmp_path: Path, script: str = "write", **fields: Any) -> receipts.Receipt:
    directory = custom_recipes(tmp_path, script, **fields)
    loaded = recipes.get_recipe("custom", directory)
    return receipts.run_recipe(loaded, output=tmp_path / "out", recipes_dir=directory)


def expected(**fields: Any) -> dict[str, Any]:
    value = {"label": "Value", "metric_id": "value", "file": "result.json",
             "key": ["nested", 0], "value": 2.0, "tolerance": 0.01}
    return {"description": "one value", "values": [{**value, **fields}]}


def test_a_step_runs_only_this_installation_of_rewirebench():
    assert receipts._program(["rewirebench", "prepare", "mfass-v2"]) == [
        sys.executable, "-m", "rewirebench.cli", "prepare", "mfass-v2"]
    for argv in (["sh", "-c", "true"], ["runpodctl", "create", "pod"], ["rewirebench"],
                 ["rewirebench", "run", "--adapter", "subprocess:Popen"],
                 ["rewirebench", "submit"], ["rewirebench", "recipe", "run", "x"]):
        with pytest.raises(ValueError, match="runs only rewirebench data commands"):
            receipts._program(argv)


def test_a_step_gets_a_reduced_environment_with_no_cuda_device(
    tmp_path: Path, scripted: None, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_SECRET123")
    monkeypatch.setenv("REWIRE_SUBMISSION_TOKEN", "token-SECRET456")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    environment = receipts.step_environment()
    assert environment[receipts.NESTED_VARIABLE] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "" and "PATH" in environment
    assert not {"RUNPOD_API_KEY", "REWIRE_SUBMISSION_TOKEN"} & set(environment)
    assert run_custom(tmp_path, "environment").checks_passed
    log = (tmp_path / "out/logs/01-step.stdout.txt").read_text()
    assert "SECRET" not in log and "('CUDA_VISIBLE_DEVICES', '')" in log
    assert f"('{receipts.NESTED_VARIABLE}', '1')" in log


def test_a_step_that_outlives_its_timeout_is_stopped(tmp_path: Path, scripted: None):
    directory = custom_recipes(tmp_path, "sleep")
    recipe = directory / "test/custom.yaml"
    data = yaml.safe_load(recipe.read_text())
    data["commands"][0]["timeout_seconds"] = 1
    recipe.write_text(yaml.safe_dump(data))
    loaded = recipes.get_recipe("custom", directory)
    receipt = receipts.run_recipe(loaded, output=tmp_path / "out", recipes_dir=directory)
    assert receipt.wall_seconds < 30 and not receipt.checks_passed
    assert receipt.commands[0].exit_code not in (0, None) and receipt.commands[1].exit_code is None
    assert "timeout" in (tmp_path / "out/logs/01-step.stderr.txt").read_text()


def test_a_value_outside_the_tolerance_fails(tmp_path: Path, scripted: None):
    receipt = run_custom(tmp_path, expected_output=expected(tolerance=0.001))
    assert not receipt.checks_passed and receipt.metrics == {"value": 2.004}
    assert [check.passed for check in receipt.checks] == [True, False]


def test_a_nan_in_an_output_fails_the_check_and_the_receipt_is_still_written(
    tmp_path: Path, scripted: None
):
    receipt = run_custom(tmp_path, "write-nan")
    assert not receipt.checks_passed and receipt.metrics == {"value": None}
    assert receipts.load_receipt(tmp_path / "out") == receipt
    assert (tmp_path / "out/SHA256SUMS").is_file()


def test_a_receipt_that_cannot_be_serialised_leaves_no_empty_file(smoke_run: Path, tmp_path: Path):
    receipt = receipts.load_receipt(smoke_run)
    broken = receipt.model_copy(update={"wall_seconds": float("inf")})
    with pytest.raises(ValueError, match="JSON compliant"):
        receipts.save(broken, tmp_path)
    assert list(tmp_path.iterdir()) == []
    (tmp_path / "receipt.json").write_text("")
    with pytest.raises(ValueError, match="is not a receipt"):
        receipts.verify(tmp_path)


def test_an_artifact_is_checked_by_its_sha256(tmp_path: Path, scripted: None):
    digest = recipes.hashlib.sha256(json.dumps({"nested": [2.004]}).encode()).hexdigest()
    artifact = {"description": "exact bytes",
                "artifacts": [{"path": "result.json", "sha256": digest}]}
    (tmp_path / "right").mkdir()
    receipt = run_custom(tmp_path / "right", expected_output=artifact)
    assert receipt.checks_passed and receipt.checks[-1].kind == "artifact"
    assert receipt.checks[-1].observed == digest
    result = receipts.verify(tmp_path / "right/out",
                             recipes_dir=tmp_path / "right/checkout/recipes")
    assert statuses(result)["checks"] == "match"
    artifact["artifacts"][0]["sha256"] = "c" * 64
    (tmp_path / "wrong").mkdir()
    receipt = run_custom(tmp_path / "wrong", expected_output=artifact)
    assert not receipt.checks_passed and not receipt.checks[-1].passed
    assert receipt.checks[-1].expected == "c" * 64 and receipt.checks[-1].observed == digest


def test_a_step_that_writes_the_receipt_itself_fails_the_run(tmp_path: Path, scripted: None):
    receipt = run_custom(tmp_path, "write-receipt")
    assert not receipt.checks_passed
    moved = "logs/written-by-a-command-receipt.json"
    assert (tmp_path / "out" / moved).read_text() == "{}"
    assert [check.file for check in receipt.checks if not check.passed] == [moved]
    assert receipts.load_receipt(tmp_path / "out") == receipt
    result = receipts.verify(tmp_path / "out", recipes_dir=tmp_path / "checkout/recipes")
    assert statuses(result)["checks"] == "mismatch" and statuses(result)["run"] == "mismatch"


def test_a_recipe_reads_nested_values_and_never_overwrites_a_run(tmp_path: Path, scripted: None):
    directory = custom_recipes(tmp_path)
    loaded = recipes.get_recipe("custom", directory)
    receipt = receipts.run_recipe(loaded, output=tmp_path / "out", recipes_dir=directory)
    assert receipt.checks_passed and receipt.metrics == {"value": 2.004}
    assert [item.obtained for item in receipt.inputs] == ["repository"]
    assert receipt.recipe.commit is None and receipt.recipe.repository is None
    assert receipt.code == receipts.code_state(receipts._source_root())
    found = statuses(receipts.verify(tmp_path / "out", recipes_dir=directory,
                                     inputs=directory.parent / "data"))
    # The recipe is in no commit, so the receipt is intact but cannot support a promotion.
    assert found.pop("commits") == "not_available" and set(found.values()) == {"match"}
    with pytest.raises(FileExistsError):
        receipts.run_recipe(loaded, output=tmp_path / "out", recipes_dir=directory)
    # Files passed with --inputs are recorded as supplied: nothing was fetched.
    supplied = receipts.run_recipe(loaded, output=tmp_path / "supplied", recipes_dir=directory,
                                   inputs=directory.parent / "data")
    assert [item.obtained for item in supplied.inputs] == ["supplied"]
    assert supplied.inputs[0].source == "repository:data/input.txt"


def git(folder: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", *arguments],
        cwd=folder, check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_code_state_reports_a_dirty_tree(tmp_path: Path):
    git(tmp_path, "init", "--quiet")
    (tmp_path / "tracked.txt").write_text("one\n")
    git(tmp_path, "add", "tracked.txt")
    git(tmp_path, "commit", "--quiet", "-m", "first")
    state = receipts.code_state(tmp_path)
    assert state.commit == git(tmp_path, "rev-parse", "HEAD") and state.dirty is False
    (tmp_path / "tracked.txt").write_text("two\n")
    assert receipts.code_state(tmp_path).dirty is True
    assert receipts.code_state(None).commit is None and receipts.code_state(None).dirty is None


def test_recipe_commit_is_recorded_only_for_the_committed_bytes(tmp_path: Path, scripted: None):
    """The recipes folder and the code can be different checkouts, and a recipe can be edited."""
    directory = custom_recipes(tmp_path)
    root = directory.parent
    git(root, "init", "--quiet")
    git(root, "remote", "add", "origin", "git@github.com:example/recipes.git")
    git(root, "add", "data")
    git(root, "commit", "--quiet", "-m", "data only")

    def run(name: str) -> receipts.Receipt:
        loaded = recipes.get_recipe("custom", directory)
        return receipts.run_recipe(loaded, output=tmp_path / name, recipes_dir=directory)

    untracked = run("untracked")
    assert untracked.recipe.commit is None
    assert untracked.recipe.repository == "https://github.com/example/recipes"
    git(root, "add", "recipes")
    git(root, "commit", "--quiet", "-m", "recipe")
    head = git(root, "rev-parse", "HEAD")
    committed = run("committed")
    assert committed.recipe.commit == head
    # The code commit describes where rewirebench runs from, not the recipes checkout.
    assert committed.code == receipts.code_state(receipts._source_root())
    assert committed.code.commit != head
    recipe = directory / "test/custom.yaml"
    recipe.write_text(recipe.read_text() + "# Edited after the commit.\n")
    edited = run("edited")
    assert edited.recipe.commit is None and edited.recipe.sha256 != committed.recipe.sha256
    # The older receipt still verifies: its recipe is read from the commit it cites.
    result = receipts.verify(tmp_path / "committed", recipes_dir=directory)
    found = statuses(result)
    assert found["recipe"] == found["checks"] == found["inputs"] == "match" and result["ok"]
    assert head in result["results"][3]["detail"]
    # A receipt that cites bytes no commit holds does not.
    rewrite_receipt(tmp_path / "committed",
                    lambda data: data["recipe"].update(sha256="d" * 64))
    result = receipts.verify(tmp_path / "committed", recipes_dir=directory)
    assert statuses(result)["recipe"] == "mismatch" and not result["ok"]


def test_a_failed_command_still_leaves_a_receipt_that_cannot_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], scripted: None
):
    directory = custom_recipes(tmp_path, "fail")
    result, code = run_cli(capsys, "recipe", "run", "custom", "--output", str(tmp_path / "out"),
                           "--recipes-dir", str(directory))
    assert code == 1 and result["checks_passed"] is False
    receipt = receipts.load_receipt(tmp_path / "out")
    assert [command.exit_code for command in receipt.commands] == [3, None]
    assert not receipt.checks_passed and receipt.metrics == {"value": None}
    assert [check.passed for check in receipt.checks] == [False, False]
    result = receipts.verify(tmp_path / "out", recipes_dir=directory)
    assert statuses(result)["checks"] == "mismatch" and not result["ok"]
    # A failed run does not verify when the recipe cannot be found either.
    (tmp_path / "empty").mkdir()
    for missing in (tmp_path / "empty", None):
        result = receipts.verify(tmp_path / "out", recipes_dir=missing)
        found = statuses(result)
        assert found["recipe"] == found["checks"] == "not_available"
        assert found["run"] == "mismatch" and not result["ok"]
    result, code = run_cli(capsys, "receipt", "verify", str(tmp_path / "out"),
                           "--recipes-dir", str(tmp_path / "empty"))
    assert code == 1 and not result["ok"]


def test_a_changed_input_stops_the_run_before_anything_executes(tmp_path: Path, scripted: None):
    directory = custom_recipes(tmp_path)
    loaded = recipes.get_recipe("custom", directory)
    data = yaml.safe_load(loaded.path.read_text())
    data["inputs"][0]["bytes"] = 5
    loaded.path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="is 13 bytes; the recipe pins 5"):
        receipts.run_recipe(recipes.get_recipe("custom", directory), output=tmp_path / "out",
                            recipes_dir=directory)
    assert not (tmp_path / "out").exists()
    (directory.parent / "data/input.txt").write_text("different bytes\n")
    with pytest.raises(ValueError, match="has SHA-256 .* the recipe pins"):
        receipts.run_recipe(loaded, output=tmp_path / "out", recipes_dir=directory)
    assert not (tmp_path / "out").exists()
    supplied = tmp_path / "supplied"
    supplied.mkdir()
    with pytest.raises(ValueError, match="not found"):
        receipts.run_recipe(loaded, output=tmp_path / "out", recipes_dir=directory,
                            inputs=supplied)


def test_a_fetched_input_has_a_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(receipts, "MAX_INPUT_BYTES", 10)
    monkeypatch.setattr(receipts.urllib.request, "urlopen",
                        lambda request, timeout: io.BytesIO(b"x" * 11))
    with pytest.raises(ValueError, match="Input larger than 10 bytes"):
        receipts._fetch("https://example.org/big", tmp_path / "big")
    monkeypatch.setattr(receipts.urllib.request, "urlopen",
                        lambda request, timeout: io.BytesIO(b"x" * 10))
    receipts._fetch("https://example.org/small", tmp_path / "small")
    assert (tmp_path / "small").read_bytes() == b"x" * 10


def test_a_run_is_refused_when_the_machine_or_checkout_does_not_meet_the_recipe(
    tmp_path: Path, scripted: None
):
    directory = custom_recipes(tmp_path)
    recipe = directory / "test/custom.yaml"
    original = yaml.safe_load(recipe.read_text())
    (directory.parent / "requirements.lock").write_text("pydantic==2\n")
    locked = recipes.sha256_file(directory.parent / "requirements.lock")

    def run(environment: dict[str, Any] | None = None, **floor: Any) -> receipts.Receipt:
        data = json.loads(json.dumps(original))
        data["environment"].update(environment or {})
        data["hardware_floor"].update(floor)
        recipe.write_text(yaml.safe_dump(data))
        return receipts.run_recipe(recipes.get_recipe("custom", directory),
                                   output=tmp_path / "out", recipes_dir=directory)

    with pytest.raises(ValueError, match="needs Python >=3.99"):
        run({"python": ">=3.99"})
    with pytest.raises(ValueError, match="needs Python ==3.9"):
        run({"python": "==3.9"})
    with pytest.raises(ValueError, match="requirements.lock has SHA-256"):
        run({"lockfile": {"path": "requirements.lock", "sha256": "0" * 64}})
    with pytest.raises(ValueError, match="missing.lock not found"):
        run({"lockfile": {"path": "missing.lock", "sha256": locked}})
    with pytest.raises(ValueError, match="needs 1000000.0 GB of memory"):
        run(host_memory_gb=1e6)
    assert not (tmp_path / "out").exists()
    minor = ".".join(str(part) for part in sys.version_info[:2])
    assert run({"python": f"=={minor}",
                "lockfile": {"path": "requirements.lock", "sha256": locked}},
               host_memory_gb=0.001).checks_passed
    assert recipes.python_satisfies(">=3.11", (3, 11, 0))
    assert recipes.python_satisfies(">=3.11.4", (3, 12, 0))
    assert not recipes.python_satisfies(">=3.11.4", (3, 11, 3))
    assert not recipes.python_satisfies("==3.11", (3, 12, 1))


def test_only_the_local_backend_runs_here(tmp_path: Path):
    digest = "sha256:" + "a" * 64
    environment = yaml.safe_load(
        (RECIPES / "flip2/flip2-synthetic-smoke.yaml").read_text())["environment"]
    environment["images"]["linux/amd64"] = {"reference": f"ghcr.io/rewire-bio/x@{digest}"}
    basis = {"provider": "runpod", "accelerator": "NVIDIA H100 80GB HBM3",
             "amount_per_hour": 3.99, "currency": "USD", "rate_source": "instance.json",
             "rate_date": "2026-10-09"}
    directory = custom_recipes(
        tmp_path, environment=environment, estimate={"gpu_hours": 1, "price_basis": basis},
        backends={"runpod": {"gpu_types": ["NVIDIA H100 80GB HBM3"], "disk_gb": 50,
                             "max_hours": 1}})
    loaded = recipes.get_recipe("custom", directory)
    with pytest.raises(ValueError, match="does not name the local backend"):
        receipts.run_recipe(loaded, output=tmp_path / "out", recipes_dir=directory)
    assert not (tmp_path / "out").exists()
    with pytest.raises(SystemExit):
        cli.main(["recipe", "run", "custom", "--output", str(tmp_path / "out"),
                  "--backend", "runpod", "--recipes-dir", str(directory)])


def test_cost_is_money_with_a_dated_rate_or_null_with_a_reason():
    assert receipts.owned_hardware_cost().model_dump()["reason"] == "owned hardware"
    paid = {"paid_compute": True, "amount": 1.33, "currency": "USD", "hourly_rate": 3.99,
            "rate_source": "RunPod pod record", "rate_date": "2026-11-20"}
    assert receipts.Cost(**paid).amount == 1.33
    for field in ("amount", "currency", "rate_source", "rate_date"):
        with pytest.raises(ValidationError, match="Paid compute needs"):
            receipts.Cost(**{key: value for key, value in paid.items() if key != field})
    with pytest.raises(ValidationError, match="states its reason"):
        receipts.Cost(paid_compute=False)
    with pytest.raises(ValidationError, match="no money field"):
        receipts.Cost(paid_compute=False, reason="owned hardware", amount=0)


def test_backend_and_cost_must_agree(smoke_run: Path):
    data = json.loads((smoke_run / "receipt.json").read_text())
    paid = {"paid_compute": True, "amount": 1.33, "currency": "USD", "hourly_rate": 3.99,
            "rate_source": "RunPod pod record", "rate_date": "2026-11-20"}
    with pytest.raises(ValidationError, match="owned hardware and has no cost"):
        receipts.Receipt.model_validate({**data, "cost": paid})
    with pytest.raises(ValidationError, match="records what the run cost"):
        receipts.Receipt.model_validate({**data, "backend": "runpod"})
    assert receipts.Receipt.model_validate({**data, "backend": "runpod", "cost": paid})
    with pytest.raises(ValidationError, match="checks_passed must equal"):
        receipts.Receipt.model_validate({**data, "checks_passed": False})
    with pytest.raises(ValidationError, match="recipe pointer or the reason"):
        receipts.Receipt.model_validate({**data, "recipe": None})


def test_peak_memory_units_and_remote_urls(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert receipts.peak_memory_bytes(2048) == 2048 * 1024
    monkeypatch.setattr(sys, "platform", "darwin")
    assert receipts.peak_memory_bytes(2048) == 2048
    expected = "https://github.com/rewire-bio/rewire-benchmarks"
    for remote in ("git@github.com:rewire-bio/rewire-benchmarks.git",
                   "https://x-access-token:secret@github.com/rewire-bio/rewire-benchmarks",
                   "ssh://git@github.com/rewire-bio/rewire-benchmarks.git",
                   "https://github.com/rewire-bio/rewire-benchmarks.git"):
        assert receipts._without_credentials(remote) == expected


def test_receipt_kind_follows_scope_and_ci(monkeypatch: pytest.MonkeyPatch):
    for name in receipts.AGENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CI", "true")
    assert receipts._receipt_kind("synthetic_smoke") == "runner_ci_smoke"
    assert receipts._receipt_kind("full_protocol") == "runner_reproduction"
    assert receipts.default_started_by() == "ci"
    monkeypatch.delenv("CI")
    assert receipts._receipt_kind("synthetic_smoke") == "runner_reproduction"
    assert receipts.default_started_by() == "human"
    monkeypatch.setenv("CLAUDECODE", "1")
    assert receipts.default_started_by() == "agent"


def test_plain_cli_runs_also_write_a_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv(receipts.NESTED_VARIABLE, raising=False)
    options = '{"dataset": "flip2-amylase-one-to-many", "allow_unverified": true, "limit": 3}'
    _, code = run_cli(capsys, "prepare", "flip2-fitness-v1", "--source",
                      str(FIXTURE / "smoke.csv"), "--output", str(tmp_path / "prepared"),
                      "--options", options)
    assert code == 0
    _, code = run_cli(capsys, "run-baselines", "--prepared", str(tmp_path / "prepared"),
                      "--output", str(tmp_path / "baselines"), "--baseline", "training-mean-v1")
    assert code == 0
    receipt = receipts.load_receipt(tmp_path / "baselines")
    assert receipt.recipe is None and receipt.recipe_missing.reason == "inapplicable"
    assert receipt.commands[0].argv == ["rewirebench", "run-baselines", "--prepared",
                                        "<local value>", "--output", "<local value>",
                                        "--baseline", "<local value>"]
    assert str(tmp_path) not in (tmp_path / "baselines/receipt.json").read_text()
    assert receipt.inputs[0].name == "prepared.json" and receipt.inputs[0].obtained == "local"
    assert receipt.metrics["training-mean-v1"]["n"] == 3
    assert receipt.cost.reason == "owned hardware" and receipt.peak_memory_bytes > 0
    result = receipts.verify(tmp_path / "baselines")
    assert result["ok"] and statuses(result)["outputs"] == statuses(result)["run"] == "match"
    # A run without a recipe has nothing to re-check its values against, so --strict fails.
    assert statuses(result)["metrics"] == "not_available" and not result["complete"]

    predictions = tmp_path / "baselines/training-mean-v1/predictions.json"
    result, code = run_cli(capsys, "evaluate", "--prepared", str(tmp_path / "prepared"),
                           "--predictions", str(predictions), "--allow-partial",
                           "--output", str(tmp_path / "evaluated"))
    assert code == 0 and result["receipt"] == str(tmp_path / "evaluated/receipt.json")
    receipt = receipts.load_receipt(tmp_path / "evaluated")
    assert [item.name for item in receipt.inputs] == ["prepared.json", "predictions.json"]
    assert receipt.metrics["n"] == 3

    monkeypatch.setenv(receipts.NESTED_VARIABLE, "1")
    _, code = run_cli(capsys, "run-baselines", "--prepared", str(tmp_path / "prepared"),
                      "--output", str(tmp_path / "nested"), "--baseline", "training-mean-v1")
    assert code == 0 and not (tmp_path / "nested/receipt.json").exists()
    result, code = run_cli(capsys, "evaluate", "--prepared", str(tmp_path / "prepared"),
                           "--predictions", str(predictions), "--allow-partial",
                           "--output", str(tmp_path / "nested-evaluated"))
    assert code == 0 and "receipt" not in result


def test_receipt_dates_are_utc(smoke_run: Path):
    receipt = receipts.load_receipt(smoke_run)
    assert receipt.started_at.tzinfo is not None
    assert receipt.finished_at >= receipt.started_at
    assert receipt.verified_on == receipt.started_at.astimezone(datetime.UTC).date()
