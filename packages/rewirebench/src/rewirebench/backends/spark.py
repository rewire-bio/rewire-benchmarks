"""The spark backend: an owned arm64 GPU host reached by SSH, running Docker.

It copies the recipes to the host, pulls the recipe's linux/arm64 image by digest and
runs ``rewirebench recipe run --backend spark --in-container`` in it, so the receipt is
written on the host and records the host's hardware. The run folder is then copied back
and checked here.

By default nothing is contacted: ``run`` returns the steps it would take. It has been
tested against a fake transport only. No run has been made on a DGX Spark, and no
rewirebench linux/arm64 image is published.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rewirebench import receipts
from rewirebench.backends import base
from rewirebench.receipts import StartedBy
from rewirebench.recipes import LoadedRecipe, Recipe

PLATFORM = "linux/arm64"
HOST_VARIABLE = "REWIREBENCH_SPARK_HOST"
WORKDIR_VARIABLE = "REWIREBENCH_SPARK_WORKDIR"
DEFAULT_WORKDIR = "/var/tmp/rewirebench-runs"
HOST_PLACEHOLDER = "<spark host>"
CONTAINER_RECIPES = "/recipes"
CONTAINER_OUT = "/out"
# Time allowed for the image pull, for each copy and for fetching inputs in the container,
# and the margin on top of the recipe's own time limit for the container to start and stop.
TRANSFER_SECONDS = 3600.0
MARGIN_SECONDS = 300.0
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(@[A-Za-z0-9][A-Za-z0-9._-]*)?")
_WORKDIR = re.compile(r"/[A-Za-z0-9._/-]+")
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class SparkConfig:
    """Which host to use and where on it. ``host`` is an SSH destination or alias."""

    host: str | None = None
    workdir: str = DEFAULT_WORKDIR
    docker: str = "docker"

    def __post_init__(self) -> None:
        # The host becomes an argument of ssh and scp. A value starting with a dash would
        # be read as an option, so only a plain name or user@name is accepted.
        if self.host is not None and not _HOST.fullmatch(self.host):
            raise ValueError(f"Not an SSH destination (name or user@name): {self.host!r}")
        if not _WORKDIR.fullmatch(self.workdir) or ".." in Path(self.workdir).parts:
            raise ValueError(f"The Spark work folder is an absolute path: {self.workdir!r}")

    @classmethod
    def from_environment(
        cls, *, host: str | None = None, workdir: str | None = None, required: bool = True
    ) -> SparkConfig:
        host = host or os.environ.get(HOST_VARIABLE) or None
        if host is None and required:
            raise ValueError(f"Name the Spark with --spark-host or {HOST_VARIABLE}")
        return cls(host=host,
                   workdir=workdir or os.environ.get(WORKDIR_VARIABLE) or DEFAULT_WORKDIR)


@dataclass(frozen=True)
class Completed:
    exit_code: int
    stdout: str
    stderr: str


class Transport(Protocol):
    """How commands and files reach the host. Tests supply a fake."""

    def run(self, argv: list[str], *, timeout: float) -> Completed: ...

    def push(self, local: Path, remote: str, *, timeout: float) -> None: ...

    def pull(self, remote: str, local: Path, *, timeout: float) -> None: ...


def ssh_command(host: str, argv: list[str]) -> list[str]:
    """The local ssh command that runs ``argv`` on the host.

    ssh joins its arguments into one string for the remote shell, so the remote command
    is quoted here as a single argument.
    """
    return ["ssh", "-o", "BatchMode=yes", "--", host, shlex.join(argv)]


def scp_command(source: str, target: str) -> list[str]:
    """The local scp command for one copy. The plan prints it and the transport runs it."""
    return ["scp", "-r", "-q", "-o", "BatchMode=yes", "--", source, target]


def container_name(run_name: str) -> str:
    """The Docker name of a run's container, so it can be found and stopped."""
    return f"rewirebench-{run_name}"


class SshTransport:
    """OpenSSH ``ssh`` and ``scp`` with the caller's own keys and configuration."""

    def __init__(self, host: str) -> None:
        if not _HOST.fullmatch(host):
            raise ValueError(f"Not an SSH destination: {host!r}")
        self.host = host

    def _local(self, argv: list[str], timeout: float) -> Completed:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              check=False, stdin=subprocess.DEVNULL)
        return Completed(done.returncode, done.stdout, done.stderr)

    def run(self, argv: list[str], *, timeout: float) -> Completed:
        return self._local(ssh_command(self.host, argv), timeout)

    def push(self, local: Path, remote: str, *, timeout: float) -> None:
        done = self._local(scp_command(str(local), f"{self.host}:{remote}"), timeout)
        if done.exit_code != 0:
            raise RuntimeError(f"Copy to {self.host}:{remote} failed: {done.stderr.strip()}")

    def pull(self, remote: str, local: Path, *, timeout: float) -> None:
        done = self._local(scp_command(f"{self.host}:{remote}", str(local)), timeout)
        if done.exit_code != 0:
            raise RuntimeError(f"Copy from {self.host}:{remote} failed: {done.stderr.strip()}")


def _image(recipe: Recipe) -> str:
    if recipe.backends.spark is None or recipe.environment is None:
        raise ValueError(f"Recipe {recipe.id} does not name the spark backend")
    reference = recipe.environment.images[PLATFORM].reference
    if reference is None:
        raise ValueError(f"Recipe {recipe.id} has no {PLATFORM} image digest")
    return reference


def container_arguments(
    recipe_id: str, *, started_by: StartedBy, kg_release_id: str | None
) -> list[str]:
    """What the image's entry point (``rewirebench``) is given."""
    argv = base.run_arguments(
        recipe_id, recipes_dir=CONTAINER_RECIPES, output=f"{CONTAINER_OUT}/run",
        backend="spark", started_by=started_by, kg_release_id=kg_release_id,
    )
    return [*argv[1:], "--in-container"]


class SparkBackend:
    name = "spark"

    def __init__(self, config: SparkConfig, transport: Transport | None = None) -> None:
        self.config = config
        self._transport = transport

    def _remote(self, run_name: str) -> str:
        if not _RUN_NAME.fullmatch(run_name):
            raise ValueError(
                f"The output folder's name is used on the Spark and must be plain: {run_name!r}"
            )
        return f"{self.config.workdir}/{run_name}"

    def docker_run(
        self, recipe: Recipe, remote: str, *, started_by: StartedBy,
        kg_release_id: str | None,
    ) -> list[str]:
        image = _image(recipe)
        assert recipe.backends.spark is not None
        argv = [self.config.docker, "run", "--rm", "--name",
                container_name(remote.rsplit("/", 1)[1]), "--platform", PLATFORM]
        if recipe.backends.spark.gpus:
            argv += ["--gpus", "all"]
        argv += [
            "-e", f"REWIRE_CONTAINER_DIGEST={image.rsplit('@', 1)[1]}",
            "-e", "REWIRE_CONTAINER_RUNTIME=docker",
            "-v", f"{remote}/recipes:{CONTAINER_RECIPES}:ro",
            "-v", f"{remote}/out:{CONTAINER_OUT}",
            image,
        ]
        return argv + container_arguments(recipe.id, started_by=started_by,
                                          kg_release_id=kg_release_id)

    def steps(
        self, loaded: LoadedRecipe, *, recipes_dir: Path, output: str | Path | None,
        started_by: StartedBy, kg_release_id: str | None,
    ) -> list[base.PlanStep]:
        recipe = loaded.recipe
        image = _image(recipe)
        run_name = Path(output).name if output is not None else "<run>"
        remote = (self._remote(run_name) if output is not None
                  else f"{self.config.workdir}/{run_name}")
        host = self.config.host or HOST_PLACEHOLDER
        local_output = str(output) if output is not None else base.OUTPUT_PLACEHOLDER
        return [
            base.PlanStep(name="check-architecture", where="remote", argv=["uname", "-m"],
                          note="Must print aarch64 or arm64."),
            base.PlanStep(name="make-work-folder", where="remote",
                          argv=["mkdir", "-p", self.config.workdir]),
            base.PlanStep(name="make-run-folders", where="remote",
                          argv=["mkdir", remote, f"{remote}/out"],
                          note="Fails if the run folder exists, so a run is never mixed "
                               "with an earlier one."),
            base.PlanStep(name="copy-recipes", where="local",
                          argv=scp_command(str(recipes_dir), f"{host}:{remote}/recipes")),
            base.PlanStep(name="pull-image", where="remote",
                          argv=[self.config.docker, "pull", "--platform", PLATFORM, image]),
            base.PlanStep(name="run", where="remote",
                          argv=self.docker_run(recipe, remote, started_by=started_by,
                                               kg_release_id=kg_release_id),
                          note="Writes receipt.json on the Spark."),
            base.PlanStep(name="copy-results", where="local",
                          argv=scp_command(f"{host}:{remote}/out/run", local_output)),
            base.PlanStep(name="verify", where="local",
                          argv=["rewirebench", "receipt", "verify", local_output,
                                "--recipes-dir", str(recipes_dir)]),
        ]

    def plan(
        self, loaded: LoadedRecipe, *, recipes_dir: Path, output: str | Path | None = None,
        inputs: str | Path | None = None, started_by: StartedBy = "human",
        kg_release_id: str | None = None,
    ) -> base.Plan:
        recipe = loaded.recipe
        problem = self.problem(recipe, inputs)
        if recipe.backends.spark is None:
            raise ValueError(f"Recipe {recipe.id} does not name the spark backend")
        steps = self.steps(loaded, recipes_dir=recipes_dir, output=output,
                           started_by=started_by, kg_release_id=kg_release_id)
        hardware, hardware_note = base.hardware_needs(recipe)
        expected, basis = base.expected_wall(recipe)
        run = next(step for step in steps if step.name == "run")
        return base.make_plan(
            recipe=base.recipe_pointer(loaded, recipes_dir),
            backend="spark",
            platform=PLATFORM,
            image=_image(recipe),
            hardware=hardware,
            hardware_note=hardware_note,
            expected_wall_seconds=expected,
            expected_wall_seconds_basis=basis,
            time_limit_seconds=base.time_limit(recipe),
            quote=base.owned_hardware_quote(),
            confirmation=base.Confirmation(required=False),
            command=ssh_command(self.config.host or HOST_PLACEHOLDER, run.argv),
            steps=steps,
            runnable=problem is None and self.config.host is not None,
            not_runnable_reason=problem or (
                None if self.config.host is not None
                else f"No Spark is named. Set --spark-host or {HOST_VARIABLE}."),
            notes=[
                *base.weights_notices(recipe),
                "A dry-run plan: nothing was contacted.",
                "The Spark keeps the run folder after the copy. Remove it there by hand.",
                ("The container is named after the output folder. If this machine loses the "
                 "connection, the container keeps running; stop it there with docker kill."),
                "The image's entry point must be rewirebench, as in containers/Containerfile.",
                "Not yet run on a DGX Spark; tested against a fake transport only.",
            ],
        )

    @staticmethod
    def problem(recipe: Recipe, inputs: str | Path | None) -> str | None:
        """Why this recipe cannot run on the Spark as written, if it cannot."""
        if inputs is not None:
            return "The spark backend fetches inputs on the Spark; --inputs is not supported."
        local_only = [item.name for item in recipe.inputs if item.repo_path is not None]
        if local_only:
            return (f"Inputs {local_only} come from a repository path. The image holds no "
                    "checkout, so a spark recipe's inputs must be URLs.")
        return base.pending_reason(recipe)

    def run(
        self, loaded: LoadedRecipe, *, recipes_dir: Path, output: str | Path,
        inputs: str | Path | None = None, started_by: StartedBy = "human",
        kg_release_id: str | None = None, dry_run: bool = True,
    ) -> dict[str, Any]:
        """Run the recipe on the Spark, or with ``dry_run`` return the plan and stop."""
        recipe = loaded.recipe
        plan = self.plan(loaded, recipes_dir=recipes_dir, output=output, inputs=inputs,
                         started_by=started_by, kg_release_id=kg_release_id)
        if dry_run:
            return {"dry_run": True, "plan": plan.model_dump(mode="json")}
        if not plan.runnable:
            raise ValueError(plan.not_runnable_reason)
        folder = Path(output)
        if folder.exists():
            raise FileExistsError(f"Output folder already exists: {folder}")
        assert self.config.host is not None
        transport = self._transport or SshTransport(self.config.host)
        remote = self._remote(folder.name)

        def remote_step(name: str, timeout: float, *, check: bool = True) -> Completed:
            step = next(item for item in plan.steps if item.name == name)
            done = transport.run(step.argv, timeout=timeout)
            if check and done.exit_code != 0:
                raise RuntimeError(
                    f"Spark step {name} failed with exit code {done.exit_code}: "
                    f"{done.stderr.strip()[-2000:]}"
                )
            return done

        machine = remote_step("check-architecture", 60).stdout.strip()
        if machine not in receipts.ARM64_MACHINES:
            raise RuntimeError(f"The host reports {machine!r}; the spark backend needs arm64")
        remote_step("make-work-folder", 60)
        remote_step("make-run-folders", 60)
        transport.push(recipes_dir, f"{remote}/recipes", timeout=TRANSFER_SECONDS)
        remote_step("pull-image", TRANSFER_SECONDS)
        # The recipe run exits 1 when a check fails and still writes a receipt, so the
        # exit code alone does not decide whether there is something to copy back.
        # Inputs are fetched in the container before the first command, hence the allowance.
        try:
            ran = remote_step(
                "run", plan.time_limit_seconds + TRANSFER_SECONDS + MARGIN_SECONDS, check=False)
        except subprocess.TimeoutExpired as error:
            # The ssh call gave up, but the container goes on without it. Stop it by name.
            name = container_name(folder.name)
            try:
                killed = transport.run([self.config.docker, "kill", name], timeout=60)
                outcome = ("was stopped" if killed.exit_code == 0 else
                           f"could not be stopped ({killed.stderr.strip()[-500:]})")
            except (subprocess.TimeoutExpired, OSError) as problem:
                outcome = f"could not be stopped ({type(problem).__name__})"
            raise RuntimeError(
                f"The run passed its time limit of {error.timeout:g} seconds. Container "
                f"{name} on {self.config.host} {outcome}. If it is still there, stop it "
                f"with `{self.config.docker} kill {name}`."
            ) from error
        folder.parent.mkdir(parents=True, exist_ok=True)
        try:
            transport.pull(f"{remote}/out/run", folder, timeout=TRANSFER_SECONDS)
        except RuntimeError as error:
            raise RuntimeError(
                f"The run on the Spark left no run folder (exit code {ran.exit_code}): "
                f"{ran.stderr.strip()[-2000:]}"
            ) from error
        receipt = receipts.load_receipt(folder)
        image_digest = _image(recipe).rsplit("@", 1)[1]
        problems: list[str] = []
        if receipt.backend != "spark":
            problems.append(f"the receipt's backend is {receipt.backend}")
        if receipt.environment.image_used != image_digest:
            problems.append(f"the receipt's image is {receipt.environment.image_used}")
        if receipt.recipe is None or receipt.recipe.sha256 != loaded.sha256:
            problems.append("the receipt cites another recipe file")
        verified = receipts.verify(folder, recipes_dir=recipes_dir)
        return {
            "dry_run": False,
            "recipe_id": recipe.id,
            "receipt": str(folder / receipts.RECEIPT_FILENAME),
            "checks_passed": receipt.checks_passed and verified["ok"] and not problems,
            "exit_code": ran.exit_code,
            "problems": problems,
            "verify": verified,
            "remote_folder": f"{self.config.host}:{remote}",
            "cost": receipt.cost.model_dump(mode="json"),
        }
