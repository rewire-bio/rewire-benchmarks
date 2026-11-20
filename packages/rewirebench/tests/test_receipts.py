"""Every run writes a receipt, and verify recomputes it instead of trusting it."""
from __future__ import annotations

import datetime
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


def test_verify_passes_on_an_untouched_run(smoke_run: Path, capsys: pytest.CaptureFixture[str]):
    result, code = run_cli(capsys, "receipt", "verify", str(smoke_run), "--strict",
                           "--recipes-dir", str(RECIPES), "--inputs", str(FIXTURE))
    assert code == 0 and result["ok"] and result["complete"]
    assert set(statuses(result).values()) == {"match"}
    result, code = run_cli(capsys, "receipt", "verify", str(smoke_run),
                           "--recipes-dir", str(RECIPES))
    assert code == 0 and statuses(result)["inputs"] == "not_available"
    _, code = run_cli(capsys, "receipt", "verify", str(smoke_run), "--strict",
                      "--recipes-dir", str(RECIPES))
    assert code == 1


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
    sums.unlink()
    result = receipts.verify(run_copy, recipes_dir=RECIPES)
    assert statuses(result)["sha256sums"] == "not_available" and result["ok"]


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


def custom_recipes(tmp_path: Path, argv: list[str], **fields: Any) -> Path:
    """A recipes directory with one recipe that runs ``argv`` on a small input."""
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


WRITE_RESULT = ("import json, pathlib, sys; target = pathlib.Path(sys.argv[1]) / 'result.json'; "
                "target.write_text(json.dumps({'nested': [2.004]}))")


def test_a_recipe_can_run_any_argv_and_read_nested_values(tmp_path: Path):
    directory = custom_recipes(tmp_path, [sys.executable, "-c", WRITE_RESULT, "{output}"])
    loaded = recipes.get_recipe("custom", directory)
    receipt = receipts.run_recipe(loaded, output=tmp_path / "out", recipes_dir=directory)
    assert receipt.checks_passed and receipt.metrics == {"value": 2.004}
    assert receipt.recipe.commit is None and receipt.recipe.repository is None
    assert receipt.code == receipts.code_state(receipts._source_root())
    assert receipts.verify(tmp_path / "out", recipes_dir=directory,
                           inputs=directory.parent / "data")["complete"]
    with pytest.raises(FileExistsError):
        receipts.run_recipe(loaded, output=tmp_path / "out", recipes_dir=directory)


def git(folder: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", *arguments],
        cwd=folder, check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_recipe_commit_is_recorded_only_for_the_committed_bytes(tmp_path: Path):
    """The recipes folder and the code can be different checkouts, and a recipe can be edited."""
    directory = custom_recipes(tmp_path, [sys.executable, "-c", WRITE_RESULT, "{output}"])
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


def test_a_failed_command_still_leaves_a_receipt_that_cannot_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    directory = custom_recipes(tmp_path, [sys.executable, "-c", "raise SystemExit(3)"])
    result, code = run_cli(capsys, "recipe", "run", "custom", "--output", str(tmp_path / "out"),
                           "--recipes-dir", str(directory))
    assert code == 1 and result["checks_passed"] is False
    receipt = receipts.load_receipt(tmp_path / "out")
    assert [command.exit_code for command in receipt.commands] == [3, None]
    assert not receipt.checks_passed and receipt.metrics == {"value": None}
    assert [check.passed for check in receipt.checks] == [False, False]
    result = receipts.verify(tmp_path / "out", recipes_dir=directory)
    assert statuses(result)["checks"] == "mismatch" and not result["ok"]


def test_a_changed_input_stops_the_run_before_anything_executes(tmp_path: Path):
    directory = custom_recipes(tmp_path, [sys.executable, "-c", WRITE_RESULT, "{output}"])
    (directory.parent / "data/input.txt").write_text("different bytes\n")
    loaded = recipes.get_recipe("custom", directory)
    with pytest.raises(ValueError, match="the recipe pins"):
        receipts.run_recipe(loaded, output=tmp_path / "out", recipes_dir=directory)
    assert not (tmp_path / "out").exists()
    supplied = tmp_path / "supplied"
    supplied.mkdir()
    with pytest.raises(ValueError, match="not found"):
        receipts.run_recipe(loaded, output=tmp_path / "out", recipes_dir=directory,
                            inputs=supplied)


def test_only_the_local_backend_runs_here(tmp_path: Path):
    digest = "sha256:" + "a" * 64
    environment = yaml.safe_load(
        (RECIPES / "flip2/flip2-synthetic-smoke.yaml").read_text())["environment"]
    environment["images"]["linux/amd64"] = {"reference": f"ghcr.io/rewire-bio/x@{digest}"}
    directory = custom_recipes(
        tmp_path, [sys.executable, "-c", WRITE_RESULT, "{output}"], environment=environment,
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
    monkeypatch.setenv("CI", "true")
    assert receipts._receipt_kind("synthetic_smoke") == "runner_ci_smoke"
    assert receipts._receipt_kind("full_protocol") == "runner_reproduction"
    assert receipts.default_started_by() == "ci"
    monkeypatch.delenv("CI")
    assert receipts._receipt_kind("synthetic_smoke") == "runner_reproduction"
    assert receipts.default_started_by() == "human"


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
    assert receipt.inputs[0].name == "prepared.json"
    assert receipt.metrics["training-mean-v1"]["n"] == 3
    assert receipt.cost.reason == "owned hardware" and receipt.peak_memory_bytes > 0
    result = receipts.verify(tmp_path / "baselines")
    assert result["ok"] and statuses(result)["outputs"] == "match"

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
