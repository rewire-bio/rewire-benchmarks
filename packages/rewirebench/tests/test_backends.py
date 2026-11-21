"""Backends plan without starting anything. Spark is driven through a fake transport and
RunPod never launches. No test here opens a network connection."""
from __future__ import annotations

import itertools
import json
import platform
import shlex
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest
from recipe_fixtures import (
    ARM64_DIGEST,
    INPUT_URL,
    SMOKE_INPUT,
    gpu_recipe_data,
    write_recipes,
)
from rewirebench import backends, cli, receipts, recipes
from rewirebench.backends import base, runpod, spark

ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "recipes"
HOST = "tim@spark.local"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to connect fails the test."""

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("A backend test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)


def gpu(fixture_recipes: Path) -> recipes.LoadedRecipe:
    return recipes.get_recipe("gpu-fixture", fixture_recipes)


def test_local_plan_states_owned_hardware_and_the_exact_command(tmp_path: Path):
    loaded = recipes.get_recipe("mfass-v2-kmer-baseline", RECIPES)
    plan = backends.get_backend("local").plan(
        loaded, recipes_dir=RECIPES, output=tmp_path / "out", started_by="agent")
    assert plan.backend == "local" and plan.runnable
    assert plan.quote.model_dump(exclude_none=True) == {"paid": False, "reason": "owned hardware"}
    assert not plan.confirmation.required
    assert plan.hardware["accelerator"]["kind"] == "none"
    assert plan.expected_wall_seconds == 21.6
    assert plan.expected_wall_seconds_basis.startswith("hardware_floor.wall_clock_seconds")
    assert plan.command == [
        "rewirebench", "recipe", "run", "mfass-v2-kmer-baseline", "--recipes-dir", str(RECIPES),
        "--output", str(tmp_path / "out"), "--backend", "local", "--started-by", "agent"]
    # The command parses as the real command line, and a plan is the same every time.
    assert cli.parse_strict(plan.command[1:]).recipe == "mfass-v2-kmer-baseline"
    again = backends.get_backend("local").plan(
        loaded, recipes_dir=RECIPES, output=tmp_path / "out", started_by="agent")
    assert again.plan_sha256 == plan.plan_sha256 == base.plan_digest(plan.model_dump(mode="json"))
    assert not (tmp_path / "out").exists()


def test_a_plan_gives_no_wall_time_when_the_recipe_states_none():
    loaded = recipes.get_recipe("flip2-synthetic-smoke", RECIPES)
    plan = backends.get_backend("local").plan(loaded, recipes_dir=RECIPES)
    assert plan.expected_wall_seconds is None
    assert plan.expected_wall_seconds_basis == "The recipe states no wall time."
    assert plan.time_limit_seconds == 7200
    assert "<output folder>" in plan.command


def test_a_changed_plan_no_longer_matches_its_digest(tmp_path: Path):
    loaded = recipes.get_recipe("flip2-synthetic-smoke", RECIPES)
    data = backends.get_backend("local").plan(loaded, recipes_dir=RECIPES).model_dump(mode="json")
    data["command"][-1] = "ci"
    with pytest.raises(ValueError, match="plan_sha256 does not match"):
        base.Plan.model_validate(data)


def test_local_refuses_a_recipe_that_does_not_name_it(fixture_recipes: Path, tmp_path: Path):
    loaded = gpu(fixture_recipes)
    plan = backends.get_backend("local").plan(loaded, recipes_dir=fixture_recipes)
    assert not plan.runnable and "does not name the local backend" in plan.not_runnable_reason
    with pytest.raises(ValueError, match="does not name the local backend"):
        backends.get_backend("local").run(loaded, recipes_dir=fixture_recipes,
                                          output=tmp_path / "out")
    with pytest.raises(ValueError, match="No nextflow backend"):
        backends.get_backend("nextflow")


# RunPod: plan and price only.

def test_runpod_plan_quotes_the_rate_times_the_time_limit(fixture_recipes: Path):
    plan = backends.get_backend("runpod").plan(gpu(fixture_recipes), recipes_dir=fixture_recipes)
    quote = plan.quote
    assert quote.paid and quote.currency == "USD"
    assert quote.max_amount == "11.97"          # 3.99 an hour for at most 3 hours
    assert quote.estimated_amount == "2.00"     # 0.5 GPU-hours, rounded up to the cent
    assert quote.amount_per_hour == "3.99" and quote.max_hours == 3
    assert quote.rate_source == "Made-up rate for tests" and str(quote.rate_date) == "2026-10-14"
    assert plan.confirmation.model_dump() == {
        "required": True, "amount": "11.97", "currency": "USD",
        "how": plan.confirmation.how}
    assert plan.platform == "linux/amd64" and plan.image.endswith("a" * 64)
    assert plan.expected_wall_seconds == 1800 and plan.time_limit_seconds == 3 * 3600
    assert plan.hardware["accelerator"]["model"] == "NVIDIA GB10"
    assert not plan.runnable and "not built" in plan.not_runnable_reason


def test_runpod_pod_specs_follow_the_launcher_in_fallback_order(fixture_recipes: Path):
    plan = backends.get_backend("runpod").plan(gpu(fixture_recipes), recipes_dir=fixture_recipes)
    assert [spec["gpu"]["id"] for spec in plan.pod_specs] == [
        "NVIDIA H100 80GB HBM3", "NVIDIA L40S"]
    assert plan.pod_specs[0] == {
        "name": "rewirebench-gpu-fixture",
        "image": f"ghcr.io/rewire-bio/fixture@sha256:{'a' * 64}",
        "gpu": {"id": "NVIDIA H100 80GB HBM3", "count": 1},
        "cloud": "SECURE",
        "disk": 200,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": runpod.PUBLIC_KEY_PLACEHOLDER, "MAX_SECONDS": "10800"},
    }


def test_runpod_instance_record_reads_the_rate_from_a_created_pod():
    pod = {"id": "abc123", "image": "ghcr.io/rewire-bio/fixture@sha256:" + "a" * 64,
           "gpu": {"id": "NVIDIA H100 80GB HBM3", "count": 1}, "dataCenterId": "EU-RO-1",
           "cost": 3.99, "cudaVersion": "12.8", "ssh": {"direct": {"host": "h", "port": 22}}}
    assert runpod.instance_record(pod, disk_gb=200) == {
        "provider": "runpod", "instance_type": "NVIDIA H100 80GB HBM3",
        "gpu": {"id": "NVIDIA H100 80GB HBM3", "count": 1}, "region": "EU-RO-1",
        "zone": "EU-RO-1", "instance_id": "abc123", "image": pod["image"],
        "purchase_option": "on-demand", "cost_per_hr": 3.99, "disk_gb": 200,
        "host_cuda": "12.8"}
    assert runpod.instance_record({}, disk_gb=1)["cost_per_hr"] is None


@pytest.mark.parametrize(("amount", "currency", "message"), [
    (None, None, "needs confirmed_amount"),
    (None, "USD", "needs confirmed_amount"),
    (11.97, None, "needs confirmed_currency"),
    (11.96, "USD", "is not the plan's amount"),
    ("12", "USD", "is not the plan's amount"),
    (11.97, "EUR", "is not the plan's amount"),
    (True, "USD", "Not an amount"),
    ("a lot", "USD", "Not an amount"),
    ("nan", "USD", "Not an amount"),
    # The right amount is still refused: nothing launches a paid run.
    (11.97, "USD", "not built"),
    ("11.97", "USD", "not built"),
    ("11.970", "USD", "not built"),
])
def test_runpod_launch_always_refuses(
    fixture_recipes: Path, monkeypatch: pytest.MonkeyPatch, amount: Any, currency: str | None,
    message: str,
):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(subprocess, "Popen", None)
    monkeypatch.setattr(subprocess, "run", None)
    backend = backends.get_backend("runpod")
    plan = backend.plan(gpu(fixture_recipes), recipes_dir=fixture_recipes)
    with pytest.raises(backends.LaunchRefused, match=message):
        backend.launch(plan, confirmed_amount=amount, confirmed_currency=currency)


def test_runpod_launch_is_refused_in_ci_whatever_is_confirmed(
    fixture_recipes: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CI", "true")
    backend = backends.get_backend("runpod")
    plan = backend.plan(gpu(fixture_recipes), recipes_dir=fixture_recipes)
    with pytest.raises(backends.LaunchRefused, match="never started in CI"):
        backend.launch(plan, confirmed_amount="11.97", confirmed_currency="USD")


def test_the_runpod_module_has_no_way_to_reach_the_network_or_a_key():
    source = Path(runpod.__file__).read_text()
    for forbidden in ("urllib", "requests", "httpx", "socket", "subprocess", "RUNPOD_API_KEY",
                      "environ.get(\"RUNPOD", "http.client"):
        assert forbidden not in source, forbidden
    assert not hasattr(backends.RunpodBackend, "run")


def test_a_recipe_without_runpod_has_no_runpod_plan():
    loaded = recipes.get_recipe("flip2-synthetic-smoke", RECIPES)
    with pytest.raises(ValueError, match="does not name the runpod backend"):
        backends.get_backend("runpod").plan(loaded, recipes_dir=RECIPES)
    with pytest.raises(ValueError, match="does not name the spark backend"):
        backends.get_backend("spark").plan(loaded, recipes_dir=RECIPES)


# Spark: SSH plus Docker, against a fake transport.

class FakeSpark:
    """Stands in for the host. Its file system is a folder; `docker run` runs the recipe
    in this process with the container's marker variables set, as the image would."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch, *,
                 machine: str = "aarch64", digest: str = ARM64_DIGEST) -> None:
        self.root, self.monkeypatch = root, monkeypatch
        self.machine, self.digest = machine, digest
        self.commands: list[list[str]] = []
        self.pushed: list[tuple[str, str]] = []

    def path(self, remote: str) -> Path:
        return self.root / remote.lstrip("/")

    def run(self, argv: list[str], *, timeout: float) -> spark.Completed:
        # What ssh would hand the remote shell, split back into words.
        argv = shlex.split(spark.ssh_command(HOST, argv)[-1])
        self.commands.append(argv)
        if argv == ["uname", "-m"]:
            return spark.Completed(0, self.machine + "\n", "")
        if argv[0] == "mkdir":
            for folder in (item for item in argv[1:] if item != "-p"):
                self.path(folder).mkdir(parents="-p" in argv, exist_ok="-p" in argv)
            return spark.Completed(0, "", "")
        if argv[:2] == ["docker", "pull"]:
            return spark.Completed(0, "", "")
        if argv[:2] == ["docker", "run"]:
            return self.container(argv)
        raise AssertionError(f"Unexpected remote command: {argv}")

    def container(self, argv: list[str]) -> spark.Completed:
        image = next(index for index, item in enumerate(argv) if "@sha256:" in item)
        options, arguments = argv[2:image], argv[image + 1:]
        mounts = {item.split(":")[1]: self.path(item.split(":")[0])
                  for flag, item in itertools.pairwise(options) if flag == "-v"}
        variables = dict(item.split("=", 1)
                         for flag, item in itertools.pairwise(options) if flag == "-e")
        variables["REWIRE_CONTAINER_DIGEST"] = self.digest
        arguments = [item.replace("/recipes", str(mounts["/recipes"]))
                     .replace("/out", str(mounts["/out"])) for item in arguments]
        with self.monkeypatch.context() as patch:
            for name, value in variables.items():
                patch.setenv(name, value)
            patch.setattr(platform, "machine", lambda: self.machine)
            patch.setattr(receipts, "_fetch", fetch_fixture)
            try:
                cli.main(arguments)
            except SystemExit as stop:
                return spark.Completed(int(stop.code or 0), "", str(stop.code))
        return spark.Completed(0, "", "")

    def push(self, local: Path, remote: str, *, timeout: float) -> None:
        self.pushed.append((str(local), remote))
        shutil.copytree(local, self.path(remote))

    def pull(self, remote: str, local: Path, *, timeout: float) -> None:
        if not self.path(remote).is_dir():
            raise RuntimeError(f"No such folder: {remote}")
        shutil.copytree(self.path(remote), local)


def fetch_fixture(url: str, destination: Path) -> None:
    assert url == INPUT_URL
    shutil.copyfile(SMOKE_INPUT, destination)


def spark_backend(transport: Any = None) -> backends.SparkBackend:
    return backends.SparkBackend(backends.SparkConfig(host=HOST, workdir="/data/runs"),
                                 transport=transport)


def test_spark_dry_run_is_the_default_and_contacts_nothing(
    fixture_recipes: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(subprocess, "run", None)
    fake = FakeSpark(tmp_path / "spark", monkeypatch)
    result = spark_backend(fake).run(gpu(fixture_recipes), recipes_dir=fixture_recipes,
                                     output=tmp_path / "runs/first")
    assert result["dry_run"] is True and fake.commands == [] and fake.pushed == []
    assert not (tmp_path / "runs").exists()
    plan = base.Plan.model_validate(result["plan"])
    assert plan.backend == "spark" and plan.platform == "linux/arm64" and plan.runnable
    assert plan.quote.reason == "owned hardware" and not plan.confirmation.required
    assert [step.name for step in plan.steps] == [
        "check-architecture", "make-work-folder", "make-run-folders", "copy-recipes",
        "pull-image", "run", "copy-results", "verify"]
    run = next(step for step in plan.steps if step.name == "run")
    image = f"ghcr.io/rewire-bio/fixture@{ARM64_DIGEST}"
    assert run.argv == [
        "docker", "run", "--rm", "--platform", "linux/arm64", "--gpus", "all",
        "-e", f"REWIRE_CONTAINER_DIGEST={ARM64_DIGEST}", "-e", "REWIRE_CONTAINER_RUNTIME=docker",
        "-v", "/data/runs/first/recipes:/recipes:ro", "-v", "/data/runs/first/out:/out", image,
        "recipe", "run", "gpu-fixture", "--recipes-dir", "/recipes", "--output", "/out/run",
        "--backend", "spark", "--started-by", "human", "--in-container"]
    # The plan's command is the ssh call for that step, quoted for the remote shell.
    assert plan.command[:5] == ["ssh", "-o", "BatchMode=yes", "--", HOST]
    assert shlex.split(plan.command[5]) == run.argv


def test_spark_runs_the_image_copies_the_folder_back_and_verifies_it(
    fixture_recipes: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake = FakeSpark(tmp_path / "spark", monkeypatch)
    output = tmp_path / "runs/first"
    result = spark_backend(fake).run(
        gpu(fixture_recipes), recipes_dir=fixture_recipes, output=output, started_by="agent",
        kg_release_id="2026-11-20-72316db0764b", dry_run=False)
    assert result["checks_passed"] and result["problems"] == [], result
    assert result["verify"]["ok"] and result["remote_folder"] == f"{HOST}:/data/runs/first"
    assert result["cost"] == {"paid_compute": False, "amount": None, "currency": None,
                              "reason": "owned hardware", "hourly_rate": None,
                              "rate_source": None, "rate_date": None}
    assert [command[:2] for command in fake.commands] == [
        ["uname", "-m"], ["mkdir", "-p"], ["mkdir", "/data/runs/first"],
        ["docker", "pull"], ["docker", "run"]]
    assert fake.pushed == [(str(fixture_recipes), "/data/runs/first/recipes")]
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["backend"] == "spark" and receipt["started_by"] == "agent"
    assert receipt["environment"]["image_used"] == ARM64_DIGEST
    assert receipt["environment"]["container_runtime"] == "docker"
    assert receipt["kg_release_id"] == "2026-11-20-72316db0764b"
    assert receipt["inputs"][0]["obtained"] == "fetched"
    assert receipt["cost"]["reason"] == "owned hardware"
    # The run folder was left on the host, and a second run with the same name is refused.
    assert (tmp_path / "spark/data/runs/first/out/run/receipt.json").is_file()
    with pytest.raises(FileExistsError):
        spark_backend(fake).run(gpu(fixture_recipes), recipes_dir=fixture_recipes,
                                output=output, dry_run=False)


def test_spark_keeps_cuda_visible_only_inside_its_image(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert receipts.step_environment()["CUDA_VISIBLE_DEVICES"] == ""
    assert receipts.step_environment(hide_cuda=False)["CUDA_VISIBLE_DEVICES"] == "0"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert "CUDA_VISIBLE_DEVICES" not in receipts.step_environment(hide_cuda=False)


@pytest.mark.parametrize(("machine", "digest"), [
    ("x86_64", ARM64_DIGEST),          # a GPU machine that is not the Spark
    ("aarch64", "sha256:" + "c" * 64),  # another image
    ("aarch64", ""),                   # no container at all
])
def test_the_in_container_spark_path_is_refused_anywhere_else(
    fixture_recipes: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: str,
    digest: str,
):
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setenv("REWIRE_CONTAINER_DIGEST", digest)
    with pytest.raises(ValueError, match="only inside its linux/arm64 image"):
        receipts.run_recipe(gpu(fixture_recipes), output=tmp_path / "out",
                            recipes_dir=fixture_recipes, backend="spark")
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="does not name the local backend"):
        receipts.run_recipe(gpu(fixture_recipes), output=tmp_path / "out",
                            recipes_dir=fixture_recipes)


def test_spark_stops_on_the_wrong_architecture_before_anything_is_copied(
    fixture_recipes: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake = FakeSpark(tmp_path / "spark", monkeypatch, machine="x86_64")
    with pytest.raises(RuntimeError, match="needs arm64"):
        spark_backend(fake).run(gpu(fixture_recipes), recipes_dir=fixture_recipes,
                                output=tmp_path / "out", dry_run=False)
    assert fake.commands == [["uname", "-m"]] and fake.pushed == []


def test_spark_reports_a_run_that_left_no_folder(
    fixture_recipes: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The container holds another image than the recipe pins, so the run refuses to start.
    fake = FakeSpark(tmp_path / "spark", monkeypatch, digest="sha256:" + "c" * 64)
    with pytest.raises(RuntimeError, match="left no run folder"):
        spark_backend(fake).run(gpu(fixture_recipes), recipes_dir=fixture_recipes,
                                output=tmp_path / "out", dry_run=False)
    assert not (tmp_path / "out").exists()


def test_spark_refuses_inputs_it_cannot_fetch_on_the_host(tmp_path: Path, fixture_recipes: Path):
    data = gpu_recipe_data()
    data["inputs"] = [{"name": "smoke.csv", "repo_path": "workflows/benchmark/examples/smoke.csv",
                       "sha256": recipes.sha256_file(SMOKE_INPUT)}]
    directory = write_recipes(tmp_path / "other", data)
    loaded = recipes.get_recipe("gpu-fixture", directory)
    plan = spark_backend().plan(loaded, recipes_dir=directory, output=tmp_path / "out")
    assert not plan.runnable and "must be URLs" in plan.not_runnable_reason
    with pytest.raises(ValueError, match="must be URLs"):
        spark_backend(object()).run(loaded, recipes_dir=directory, output=tmp_path / "out",
                                    dry_run=False)
    supplied = spark_backend().plan(gpu(fixture_recipes), recipes_dir=fixture_recipes,
                                    inputs=tmp_path)
    assert not supplied.runnable and "--inputs is not supported" in supplied.not_runnable_reason


def test_spark_without_a_host_can_plan_but_not_run(fixture_recipes: Path, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(spark.HOST_VARIABLE, raising=False)
    backend = backends.get_backend("spark")
    plan = backend.plan(gpu(fixture_recipes), recipes_dir=fixture_recipes)
    assert not plan.runnable and "No Spark is named" in plan.not_runnable_reason
    assert plan.command[4] == "<spark host>"
    with pytest.raises(ValueError, match="No Spark is named"):
        backend.run(gpu(fixture_recipes), recipes_dir=fixture_recipes, output=tmp_path / "out",
                    dry_run=False)
    with pytest.raises(ValueError, match="--spark-host"):
        backends.SparkConfig.from_environment()
    monkeypatch.setenv(spark.HOST_VARIABLE, "spark")
    monkeypatch.setenv(spark.WORKDIR_VARIABLE, "/srv/bench")
    assert backends.SparkConfig.from_environment() == backends.SparkConfig("spark", "/srv/bench")


@pytest.mark.parametrize("host", ["-oProxyCommand=sh", "spark;rm", "a b", "", "user@-x", "a@b@c"])
def test_an_ssh_destination_cannot_be_an_option_or_a_command(host: str):
    with pytest.raises(ValueError, match="SSH destination"):
        backends.SparkConfig(host=host)
    with pytest.raises(ValueError, match="SSH destination"):
        spark.SshTransport(host)


@pytest.mark.parametrize("workdir", ["relative/path", "/data/../etc", "/data/$(id)", "/a b"])
def test_the_spark_work_folder_is_a_plain_absolute_path(workdir: str):
    with pytest.raises(ValueError, match="absolute path"):
        backends.SparkConfig(host="spark", workdir=workdir)


def test_the_run_name_on_the_spark_is_plain(fixture_recipes: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="must be plain"):
        spark_backend().plan(gpu(fixture_recipes), recipes_dir=fixture_recipes,
                             output=tmp_path / "$(reboot)")


def test_ssh_transport_quotes_the_remote_command_as_one_argument(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[list[str]] = []

    def record(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["timeout"] == 5
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "aarch64\n", "")

    monkeypatch.setattr(subprocess, "run", record)
    transport = spark.SshTransport(HOST)
    # A JSON option value with quotes, spaces and braces survives the remote shell.
    remote = ["rewirebench", "prepare", "flip2-fitness-v1", "--options",
              '{"dataset": "a b", "note": "it\'s $HOME; `id`"}']
    assert transport.run(remote, timeout=5).stdout == "aarch64\n"
    transport.push(tmp_path, "/data/runs/x/recipes", timeout=5)
    transport.pull("/data/runs/x/out/run", tmp_path / "back", timeout=5)
    assert calls[0][:5] == ["ssh", "-o", "BatchMode=yes", "--", HOST] and len(calls[0]) == 6
    assert shlex.split(calls[0][5]) == remote
    assert calls[1] == ["scp", "-r", "-q", "-o", "BatchMode=yes", "--", str(tmp_path),
                        f"{HOST}:/data/runs/x/recipes"]
    assert calls[2] == ["scp", "-r", "-q", "-o", "BatchMode=yes", "--",
                        f"{HOST}:/data/runs/x/out/run", str(tmp_path / "back")]


def test_cli_plans_every_backend_and_spark_prints_its_steps_unless_told_to_execute(
    fixture_recipes: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(spark.HOST_VARIABLE, raising=False)
    monkeypatch.setattr(subprocess, "run", None)
    common = ["gpu-fixture", "--recipes-dir", str(fixture_recipes)]
    cli.main(["recipe", "plan", *common, "--backend", "runpod"])
    assert json.loads(capsys.readouterr().out)["quote"]["max_amount"] == "11.97"
    cli.main(["recipe", "plan", *common, "--backend", "spark", "--spark-host", HOST])
    assert json.loads(capsys.readouterr().out)["runnable"] is True
    cli.main(["recipe", "run", *common, "--backend", "spark", "--spark-host", HOST,
              "--output", str(tmp_path / "out")])
    printed = json.loads(capsys.readouterr().out)
    assert printed["dry_run"] is True and printed["plan"]["steps"][5]["name"] == "run"
    assert not (tmp_path / "out").exists()
    with pytest.raises(SystemExit):   # --execute needs a named host
        cli.main(["recipe", "run", *common, "--backend", "spark", "--execute",
                  "--output", str(tmp_path / "out")])

    # A host that cannot be reached is an error message and exit code 2, not a traceback.
    def unreachable(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 255, "", "ssh: connect to host: timed out")

    monkeypatch.setattr(subprocess, "run", unreachable)
    capsys.readouterr()
    with pytest.raises(SystemExit) as stop:
        cli.main(["recipe", "run", *common, "--backend", "spark", "--execute",
                  "--spark-host", HOST, "--output", str(tmp_path / "out")])
    assert stop.value.code == 2
    assert "check-architecture failed with exit code 255" in capsys.readouterr().err
    monkeypatch.setattr(subprocess, "run", None)
    with pytest.raises(SystemExit):   # runpod cannot be run from the command line
        cli.main(["recipe", "run", *common, "--backend", "runpod",
                  "--output", str(tmp_path / "out")])
    with pytest.raises(SystemExit):   # --in-container belongs to the spark backend
        cli.main(["recipe", "run", "flip2-synthetic-smoke", "--in-container",
                  "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
