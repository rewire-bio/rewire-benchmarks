"""Run receipts: what ran, on what, from which bytes, and what it produced.

Every run writes ``receipt.json`` into its output folder and a ``SHA256SUMS`` that
covers every file in the folder, the receipt included. ``verify`` recomputes the
digests, re-evaluates the recipe's checks from the output files and compares the
receipt's inputs and metrics with the recipe. A failed run never verifies.

Nothing is signed. Fields that cannot be recomputed (hardware, times, who started the
run) are as trustworthy as the place the folder came from.

The local backend runs only rewirebench's own subcommands, with a reduced environment
and with CUDA devices hidden. It is not a sandbox.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, StringConstraints, model_validator

from rewirebench import sdk
from rewirebench.recipes import (
    PLATFORMS,
    RESERVED_OUTPUTS,
    RUNNER_COMMANDS,
    RUNNER_PROGRAM,
    ImageDigest,
    LoadedRecipe,
    Missing,
    Recipe,
    Revision,
    Scope,
    Sha256,
    Strict,
    Text,
    check_step,
    get_recipe,
    parse_recipe,
    python_satisfies,
    relative_path,
    sha256_file,
)

SCHEMA_VERSION = "1.0"
RECEIPT_FILENAME = "receipt.json"
SUMS_FILENAME = "SHA256SUMS"
NESTED_VARIABLE = "REWIREBENCH_RECIPE_RUN"
STARTED_BY_VARIABLE = "REWIREBENCH_STARTED_BY"
ARM64_MACHINES = ("aarch64", "arm64")
OWNED_HARDWARE = "owned hardware"
CODE_REPOSITORY = "https://github.com/rewire-bio/rewire-benchmarks"
MAX_INPUT_BYTES = 8 * 2**30
MOVED_PREFIX = "logs/written-by-a-command-"
# A recipe step sees these variables and no others, so a cloud or submission key in the
# caller's environment cannot reach a step or its logs.
PASSED_ENVIRONMENT = (
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR", "TEMP",
    "TMP", "SYSTEMROOT", "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHASHSEED", "CI",
    "SOURCE_DATE_EPOCH", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    "REWIRE_CONTAINER_DIGEST", "REWIRE_CONTAINER_RUNTIME", "REWIRE_SIF_SHA256",
)
# Set by agent harnesses. An agent on another harness passes --started-by agent.
AGENT_VARIABLES = ("CLAUDECODE",)

Backend = Literal["local", "spark", "runpod", "nextflow"]
StartedBy = Literal["human", "agent", "ci"]
ReceiptKind = Literal["runner_ci_smoke", "runner_reproduction"]
CheckStatus = Literal["match", "mismatch", "not_available"]
Currency = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]


class RecipePointer(Strict):
    """The same four facts the evidence store keeps: repository, commit, path, SHA-256."""

    id: Text
    repository: Text | None
    commit: Revision | None
    path: Text
    sha256: Sha256


class Code(Strict):
    repository: Text | None
    commit: Revision | None
    dirty: bool | None
    rewirebench_version: Text
    sdk_code_sha256: Sha256


class ReceiptEnvironment(Strict):
    """Image digest per platform from the recipe, and the image this run used, if any."""

    images: dict[Literal["linux/amd64", "linux/arm64"], ImageDigest | None]
    image_used: ImageDigest | None
    container_runtime: Text | None
    python: Text
    os: Text

    @model_validator(mode="after")
    def _both_platforms(self) -> ReceiptEnvironment:
        if set(self.images) != set(PLATFORMS):
            raise ValueError(f"images needs exactly these platforms: {', '.join(PLATFORMS)}")
        return self


class InputRecord(Strict):
    """``source`` is where the recipe says the file comes from; ``obtained`` is how this
    run got it. A file passed with ``--inputs`` is ``supplied``: nothing was fetched."""

    name: Text
    source: Text
    obtained: Literal["fetched", "repository", "supplied", "local"]
    sha256: Sha256
    bytes: int = Field(ge=0)


class Cpu(Strict):
    model: Text | None
    logical_cores: int | None


class Gpu(Strict):
    model: Text
    memory_mib: int | None


class Hardware(Strict):
    architecture: Text
    cpu: Cpu
    memory_bytes: int | None
    gpus: list[Gpu]


class CommandRecord(Strict):
    name: Text
    argv: list[str]
    exit_code: int | None
    wall_seconds: float | None
    peak_memory_bytes: int | None


class OutputRecord(Strict):
    path: Text
    bytes: int = Field(ge=0)
    sha256: Sha256


class CheckRecord(Strict):
    """One recipe check with the value observed in the output files."""

    kind: Literal["output", "value", "artifact"]
    label: Text
    metric_id: Text | None = None
    file: Text
    key: list[str | int] | None = None
    expected: float | str
    tolerance: float | None = None
    observed: float | str | None
    passed: bool


class Cost(Strict):
    """Money only for paid compute, and only against a dated rate source (ADR 0005)."""

    paid_compute: bool
    amount: float | None = Field(default=None, ge=0)
    currency: Currency | None = None
    reason: Text | None = None
    hourly_rate: float | None = Field(default=None, gt=0)
    rate_source: Text | None = None
    rate_date: datetime.date | None = None

    @model_validator(mode="after")
    def _rules(self) -> Cost:
        money = (self.amount, self.currency, self.hourly_rate, self.rate_source, self.rate_date)
        if self.paid_compute:
            if any(value is None for value in money):
                raise ValueError(
                    "Paid compute needs amount, currency, hourly_rate, rate_source and rate_date"
                )
            if self.reason is not None:
                raise ValueError("A paid cost carries no reason for being null")
        else:
            if any(value is not None for value in money):
                raise ValueError("Unpaid compute carries no money field")
            if self.reason is None:
                raise ValueError("A null cost states its reason, for example 'owned hardware'")
        return self


class Receipt(Strict):
    schema_version: Literal["1.0"]
    kind: Literal["rewire_run_receipt"]
    receipt_kind: ReceiptKind
    scope: Scope | None
    verified_on: datetime.date
    started_by: StartedBy
    backend: Backend
    recipe: RecipePointer | None
    recipe_missing: Missing | None = None
    kg_release_id: Text | None = None
    code: Code
    environment: ReceiptEnvironment
    inputs: list[InputRecord]
    hardware: Hardware
    commands: list[CommandRecord] = Field(min_length=1)
    started_at: datetime.datetime
    finished_at: datetime.datetime
    wall_seconds: float = Field(ge=0)
    peak_memory_bytes: int | None
    outputs: list[OutputRecord]
    metrics: dict[str, Any]
    checks: list[CheckRecord]
    checks_passed: bool
    cost: Cost

    @model_validator(mode="after")
    def _rules(self) -> Receipt:
        if (self.recipe is None) == (self.recipe_missing is None):
            raise ValueError("A receipt has a recipe pointer or the reason it has none")
        if self.backend in {"local", "spark"} and self.cost.paid_compute:
            raise ValueError(f"The {self.backend} backend is owned hardware and has no cost")
        if self.backend == "runpod" and not self.cost.paid_compute:
            raise ValueError("A runpod receipt records what the run cost")
        passed = all(command.exit_code == 0 for command in self.commands) and all(
            check.passed for check in self.checks
        )
        if self.checks_passed != passed:
            raise ValueError("checks_passed must equal every command and check passing")
        return self


def owned_hardware_cost() -> Cost:
    return Cost(paid_compute=False, reason=OWNED_HARDWARE)


def _run_text(argv: list[str], cwd: Path | None = None) -> str | None:
    try:
        done = subprocess.run(
            argv, cwd=cwd, check=True, capture_output=True, text=True, timeout=20,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def _cpu_model() -> str | None:
    if sys.platform == "darwin":
        return _run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        return platform.processor() or None
    for line in lines:
        if line.lower().startswith(("model name", "hardware")) and ":" in line:
            return line.split(":", 1)[1].strip() or None
    return platform.processor() or None


def _memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _gpus() -> list[Gpu]:
    """List NVIDIA GPUs through a read-only nvidia-smi query. No GPU work is started."""
    if shutil.which("nvidia-smi") is None:
        return []
    listing = _run_text(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    gpus: list[Gpu] = []
    for line in (listing or "").splitlines():
        name, _, memory = line.partition(",")
        if name.strip():
            gpus.append(Gpu(model=name.strip(),
                            memory_mib=int(memory) if memory.strip().isdigit() else None))
    return gpus


def detect_hardware() -> Hardware:
    return Hardware(
        architecture=platform.machine(),
        cpu=Cpu(model=_cpu_model(), logical_cores=os.cpu_count()),
        memory_bytes=_memory_bytes(),
        gpus=_gpus(),
    )


def _without_credentials(url: str) -> str:
    """Normalise a git remote to an https URL with no user, password or token."""
    if "://" not in url and "@" in url and ":" in url:
        host, _, path = url.split("@", 1)[1].partition(":")
        url = f"https://{host}/{path}"
    parts = urlsplit(url)
    if parts.scheme and parts.hostname:
        host = parts.hostname + (f":{parts.port}" if parts.port else "")
        scheme = "https" if parts.scheme in {"ssh", "git"} else parts.scheme
        url = urlunsplit((scheme, host, parts.path, "", ""))
    return url.removesuffix(".git")


def code_state(root: Path | None) -> Code:
    """Commit and cleanliness of the checkout, when the code runs from one."""
    repository: str | None = None
    commit: str | None = None
    dirty: bool | None = None
    top = _run_text(["git", "rev-parse", "--show-toplevel"], cwd=root) if root else None
    if top is not None:
        commit = _run_text(["git", "rev-parse", "HEAD"], cwd=root)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, capture_output=True, text=True, check=False,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        origin = _run_text(["git", "remote", "get-url", "origin"], cwd=root)
        repository = _without_credentials(origin) if origin else None
    return Code(
        repository=repository,
        commit=commit if commit and len(commit) == 40 else None,
        dirty=dirty,
        rewirebench_version=sdk._version(),
        sdk_code_sha256=sdk._code_digest(),
    )


def recipe_origin(path: Path) -> tuple[str | None, str | None]:
    """Repository and commit of a recipe file, when HEAD holds exactly these bytes.

    A recipe that is untracked or edited since the last commit gets no commit: the
    pointer must never name a commit whose file differs from the one that ran.
    """
    folder = path.resolve().parent
    top = _run_text(["git", "rev-parse", "--show-toplevel"], cwd=folder)
    if top is None:
        return None, None
    origin = _run_text(["git", "remote", "get-url", "origin"], cwd=folder)
    repository = _without_credentials(origin) if origin else None
    relative = path.resolve().relative_to(Path(top).resolve()).as_posix()
    committed = _run_text(["git", "rev-parse", f"HEAD:{relative}"], cwd=folder)
    current = _run_text(["git", "hash-object", "--no-filters", path.name], cwd=folder)
    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=folder)
    if committed is None or committed != current or not commit or len(commit) != 40:
        return repository, None
    return repository, commit


def _source_root() -> Path | None:
    """The checkout this module runs from, or ``None`` for an installed wheel."""
    parents = Path(__file__).resolve().parents
    if len(parents) > 4 and (parents[4] / "packages" / "rewirebench").is_dir():
        return parents[4]
    return None


def _image_used() -> tuple[str | None, str | None]:
    digest = os.environ.get("REWIRE_CONTAINER_DIGEST", "")
    runtime = os.environ.get("REWIRE_CONTAINER_RUNTIME") or None
    valid = digest.startswith("sha256:") and sdk._is_sha256(digest[7:])
    return (digest if valid else None), runtime


def default_started_by() -> StartedBy:
    """``ci`` in CI, ``agent`` under a known agent harness, otherwise ``human``.

    A caller that knows better sets ``REWIREBENCH_STARTED_BY``; the MCP server sets it to
    ``agent`` for every command it starts.
    """
    stated = os.environ.get(STARTED_BY_VARIABLE)
    if stated in ("human", "agent", "ci"):
        return stated  # type: ignore[return-value]
    if os.environ.get("CI"):
        return "ci"
    if any(os.environ.get(name) for name in AGENT_VARIABLES):
        return "agent"
    return "human"


def _receipt_kind(scope: str | None) -> ReceiptKind:
    if os.environ.get("CI") and scope == "synthetic_smoke":
        return "runner_ci_smoke"
    return "runner_reproduction"


def peak_memory_bytes(maxrss: int) -> int:
    """``ru_maxrss`` is bytes on macOS and kibibytes on Linux."""
    return maxrss if sys.platform == "darwin" else maxrss * 1024


def own_peak_memory_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    return peak_memory_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _output_records(folder: Path) -> list[OutputRecord]:
    records: list[OutputRecord] = []
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder).as_posix()
        if path.is_file() and relative not in {RECEIPT_FILENAME, SUMS_FILENAME}:
            records.append(
                OutputRecord(path=relative, bytes=path.stat().st_size, sha256=sha256_file(path))
            )
    return records


def write_sums(folder: Path) -> None:
    lines = [
        f"{sha256_file(path)}  ./{path.relative_to(folder).as_posix()}\n"
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name != SUMS_FILENAME
    ]
    (folder / SUMS_FILENAME).write_text("".join(lines))


def save(receipt: Receipt, folder: Path) -> Path:
    """Write the receipt and SHA256SUMS. Exclusive creation: evidence is never overwritten."""
    path = folder / RECEIPT_FILENAME
    # Serialise before the file is opened, so a value JSON cannot hold leaves no empty file.
    text = json.dumps(receipt.model_dump(mode="json"), indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    write_sums(folder)
    return path


def _lookup(document: Any, key: list[str | int]) -> Any:
    for part in key:
        document = document[part]
    return document


def evaluate_checks(recipe: Recipe, folder: Path) -> list[CheckRecord]:
    """Compare the output files with the recipe's expected output."""
    checks: list[CheckRecord] = []
    for path in recipe.outputs:
        written = (folder / path).is_file() and (folder / path).stat().st_size > 0
        checks.append(CheckRecord(
            kind="output", label=f"{path} written", file=path, expected="present",
            observed="present" if written else "missing", passed=written,
        ))
    for name in RESERVED_OUTPUTS:
        if (folder / f"{MOVED_PREFIX}{name}").is_file():
            checks.append(CheckRecord(
                kind="output", label=f"{name} is written by the runner only",
                file=f"{MOVED_PREFIX}{name}", expected="absent",
                observed="written by a command", passed=False,
            ))
    expected = recipe.expected_output
    if expected is None:
        return checks
    for value in expected.values:
        observed: float | None = None
        try:
            found = _lookup(json.loads((folder / value.file).read_text()), list(value.key))
            # A nan or inf in an output is not a value: the check fails with nothing observed.
            if (isinstance(found, (int, float)) and not isinstance(found, bool)
                    and math.isfinite(found)):
                observed = float(found)
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            observed = None
        checks.append(CheckRecord(
            kind="value", label=value.label, metric_id=value.metric_id, file=value.file,
            key=list(value.key), expected=value.value, tolerance=value.tolerance,
            observed=observed,
            passed=observed is not None and abs(observed - value.value) <= value.tolerance,
        ))
    for artifact in expected.artifacts:
        path = folder / artifact.path
        digest = sha256_file(path) if path.is_file() else None
        checks.append(CheckRecord(
            kind="artifact", label=f"SHA-256 of {artifact.path}", file=artifact.path,
            expected=artifact.sha256, observed=digest, passed=digest == artifact.sha256,
        ))
    return checks


def _fetch(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "rewirebench"})
    with (urllib.request.urlopen(request, timeout=120) as response,
          destination.open("xb") as stream):
        written = 0
        for chunk in iter(lambda: response.read(1 << 20), b""):
            written += len(chunk)
            if written > MAX_INPUT_BYTES:
                raise ValueError(f"Input larger than {MAX_INPUT_BYTES} bytes: {url}")
            stream.write(chunk)


def stage_inputs(
    recipe: Recipe, staging: Path, *, supplied: Path | None, repo_root: Path | None
) -> list[InputRecord]:
    """Place every input in ``staging`` and refuse any whose SHA-256 differs from the pin."""
    staging.mkdir(parents=True, exist_ok=True)
    records: list[InputRecord] = []
    for item in recipe.inputs:
        target = staging / item.name
        obtained: Literal["fetched", "repository", "supplied"]
        if supplied is not None:
            source = supplied / item.name
            if not source.is_file():
                raise ValueError(f"Input {item.name} not found in {supplied}")
            shutil.copyfile(source, target)
            obtained = "supplied"
        elif item.repo_path is not None:
            if repo_root is None:
                raise ValueError(f"Input {item.name} needs a repository checkout or --inputs")
            shutil.copyfile(repo_root / item.repo_path, target)
            obtained = "repository"
        else:
            assert item.url is not None
            _fetch(item.url, target)
            obtained = "fetched"
        digest = sha256_file(target)
        if item.bytes is not None and target.stat().st_size != item.bytes:
            raise ValueError(
                f"Input {item.name} is {target.stat().st_size} bytes; the recipe pins {item.bytes}"
            )
        if digest != item.sha256:
            raise ValueError(
                f"Input {item.name} has SHA-256 {digest}; the recipe pins {item.sha256}"
            )
        records.append(InputRecord(
            name=item.name, source=item.url or f"repository:{item.repo_path}",
            obtained=obtained, sha256=digest, bytes=target.stat().st_size,
        ))
    return records


def _render(argument: str, values: dict[str, str]) -> str:
    for name, value in values.items():
        argument = argument.replace("{" + name + "}", value)
    return argument


def _program(argv: list[str]) -> list[str]:
    """The process for one recipe step: this installation's rewirebench, nothing else."""
    if len(argv) < 2 or argv[0] != RUNNER_PROGRAM or argv[1] not in RUNNER_COMMANDS:
        raise ValueError(f"A recipe step runs only {RUNNER_PROGRAM} data commands: {argv[:2]}")
    # Checked again on the rendered argv, which is what the process receives.
    try:
        check_step(argv)
    except ValueError as error:
        raise ValueError(
            f"A recipe step runs only {RUNNER_PROGRAM} data commands: {error}"
        ) from error
    return [sys.executable, "-m", "rewirebench.cli", *argv[1:]]


def step_environment(*, hide_cuda: bool = True) -> dict[str, str]:
    """Environment of a recipe step: an allowlist, the nested-run marker, no CUDA device.

    Only the spark backend, inside its pinned image, passes ``hide_cuda=False``; it then
    forwards the caller's ``CUDA_VISIBLE_DEVICES`` if one is set.
    """
    environment = {name: os.environ[name] for name in PASSED_ENVIRONMENT if name in os.environ}
    environment[NESTED_VARIABLE] = "1"
    if hide_cuda:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    return environment


# Steps now running. A step has its own session, so stopping this process would leave it
# behind; stop_on_terminate ends the steps first.
_ACTIVE_STEPS: set[subprocess.Popen[bytes]] = set()


def stop_active_steps() -> None:
    for process in list(_ACTIVE_STEPS):
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, PermissionError):
            pass


@contextlib.contextmanager
def stop_on_terminate() -> Iterator[None]:
    """While a recipe runs, SIGTERM stops the running step and then this process.

    The MCP server and a person's ``kill`` both end a run this way. Outside the main
    thread no handler can be set, and the run goes on as before.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def stop(signum: int, frame: Any) -> None:
        stop_active_steps()
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _execute(
    argv: list[str], *, cwd: Path, log: Path, timeout: float, hide_cuda: bool = True
) -> tuple[int | None, float, int | None]:
    """Run one step without a shell. Returns exit code, wall seconds, peak memory.

    A step still running after ``timeout`` seconds is killed; its exit code is then
    negative and the standard error log says why.
    """
    executable = _program(argv)
    if hide_cuda:
        # Hiding CUDA does not hide Apple MPS, so the step is held to the cpu device too.
        check_step(argv, cpu_only=True)
    started = time.perf_counter()
    with (log.with_suffix(".stdout.txt").open("xb") as out,
          log.with_suffix(".stderr.txt").open("xb") as err):
        try:
            # Its own session, so a timeout stops the step's child processes as well.
            process = subprocess.Popen(
                executable, cwd=cwd, env=step_environment(hide_cuda=hide_cuda),
                stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                start_new_session=hasattr(os, "killpg"),
            )
        except OSError as error:
            err.write(f"{type(error).__name__}: {error}\n".encode())
            return None, time.perf_counter() - started, None
        expired = threading.Event()

        def stop() -> None:
            expired.set()
            try:
                if hasattr(os, "killpg"):
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass

        timer = threading.Timer(timeout, stop)
        timer.start()
        peak: int | None = None
        _ACTIVE_STEPS.add(process)
        try:
            if hasattr(os, "wait4"):
                _, status, usage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(status)
                peak = peak_memory_bytes(usage.ru_maxrss)
            else:
                process.wait()
        finally:
            timer.cancel()
            _ACTIVE_STEPS.discard(process)
        if expired.is_set():
            err.write(f"Stopped by rewirebench after {timeout:g} seconds (timeout).\n".encode())
    return process.returncode, time.perf_counter() - started, peak


def check_requirements(recipe: Recipe, repo_root: Path, hardware: Hardware) -> None:
    """Refuse a run this machine or checkout does not meet, before anything is fetched."""
    environment = recipe.environment
    if environment is not None and environment.python is not None:
        running = tuple(sys.version_info[:3])
        if not python_satisfies(environment.python, running):
            raise ValueError(
                f"Recipe {recipe.id} needs Python {environment.python}; "
                f"this is {platform.python_version()}"
            )
    if environment is not None and environment.lockfile is not None:
        lockfile = repo_root / environment.lockfile.path
        if not lockfile.is_file():
            raise ValueError(f"Lock file {environment.lockfile.path} not found in {repo_root}")
        digest = sha256_file(lockfile)
        if digest != environment.lockfile.sha256:
            raise ValueError(
                f"Lock file {environment.lockfile.path} has SHA-256 {digest}; "
                f"the recipe pins {environment.lockfile.sha256}"
            )
    floor = recipe.hardware_floor
    needed = floor.host_memory_gb if floor is not None else None
    if needed and hardware.memory_bytes and hardware.memory_bytes < needed * 1e9:
        raise ValueError(
            f"Recipe {recipe.id} needs {needed} GB of memory; "
            f"this machine has {hardware.memory_bytes / 1e9:.1f} GB"
        )


def inside_spark_image(recipe: Recipe) -> str:
    """The image digest, if this process is in the recipe's linux/arm64 image on arm64.

    The spark backend starts the image with ``REWIRE_CONTAINER_DIGEST`` set to the digest
    it pulled. Anywhere else the spark path is refused, so it cannot be used to run a GPU
    recipe on a machine that merely has a GPU.
    """
    if recipe.backends.spark is None:
        raise ValueError(f"Recipe {recipe.id} does not name the spark backend")
    wanted = recipe.image_digests()["linux/arm64"]
    used, _ = _image_used()
    if wanted is None or used != wanted or platform.machine() not in ARM64_MACHINES:
        raise ValueError(
            f"Recipe {recipe.id} runs on spark only inside its linux/arm64 image ({wanted}) "
            "on an arm64 host. Start it with `rewirebench recipe run --backend spark`, "
            "which uses SSH and Docker."
        )
    return wanted


def run_recipe(
    loaded: LoadedRecipe,
    *,
    output: str | Path,
    recipes_dir: Path,
    inputs: str | Path | None = None,
    started_by: StartedBy | None = None,
    kg_release_id: str | None = None,
    backend: Literal["local", "spark"] = "local",
) -> Receipt:
    """Run a recipe on this machine and write its receipt.

    ``backend="spark"`` is the same run from inside the recipe's pinned image on the
    Spark, where the spark backend starts it. CUDA devices stay visible there if the
    recipe's spark block asks for GPUs. No paid backend runs here.
    """
    recipe = loaded.recipe
    hide_cuda = True
    if backend == "local":
        if recipe.backends.local is None:
            raise ValueError(f"Recipe {recipe.id} does not name the local backend")
    elif backend == "spark":
        inside_spark_image(recipe)
        assert recipe.backends.spark is not None
        hide_cuda = not recipe.backends.spark.gpus
    else:
        raise ValueError(f"The {backend} backend does not run on this machine")
    folder = Path(output)
    if folder.exists():
        raise FileExistsError(f"Output folder already exists: {folder}")
    repo_root = recipes_dir.parent
    hardware = detect_hardware()
    check_requirements(recipe, repo_root, hardware)
    # The code commit comes from where rewirebench is installed, the recipe commit from
    # where the recipe file is. With a wheel and --recipes-dir these are different places.
    code = code_state(_source_root())
    recipe_repository, recipe_commit = recipe_origin(loaded.path)
    started_at = datetime.datetime.now(datetime.UTC)
    started = time.perf_counter()
    commands: list[CommandRecord] = []
    with tempfile.TemporaryDirectory(prefix="rewirebench-recipe-") as scratch:
        work = Path(scratch) / "work"
        staging = Path(scratch) / "inputs"
        work.mkdir()
        input_records = stage_inputs(
            recipe, staging, supplied=Path(inputs) if inputs else None, repo_root=repo_root
        )
        folder.mkdir(parents=True)
        folder = folder.resolve()
        (folder / "logs").mkdir()
        values = {"inputs": str(staging), "work": str(work), "output": str(folder)}
        failed = False
        for index, command in enumerate(recipe.commands, start=1):
            argv = [_render(argument, values) for argument in command.argv]
            if failed:
                commands.append(CommandRecord(name=command.name, argv=list(command.argv),
                                              exit_code=None, wall_seconds=None,
                                              peak_memory_bytes=None))
                continue
            exit_code, seconds, peak = _execute(
                argv, cwd=work, log=folder / "logs" / f"{index:02d}-{command.name}",
                timeout=command.timeout_seconds, hide_cuda=hide_cuda,
            )
            failed = exit_code != 0
            # The recipe's own argv is recorded, so a receipt holds no local paths.
            commands.append(CommandRecord(name=command.name, argv=list(command.argv),
                                          exit_code=exit_code, wall_seconds=round(seconds, 3),
                                          peak_memory_bytes=peak))
    for name in RESERVED_OUTPUTS:
        # A step that wrote one of the runner's own files fails the run; the file is kept.
        if (folder / name).exists():
            (folder / name).rename(folder / f"{MOVED_PREFIX}{name}")
    checks = evaluate_checks(recipe, folder)
    peaks = [command.peak_memory_bytes for command in commands if command.peak_memory_bytes]
    image_used, runtime = _image_used()
    relative = loaded.path.resolve().relative_to(recipes_dir.resolve()).as_posix()
    receipt = Receipt(
        schema_version=SCHEMA_VERSION,
        kind="rewire_run_receipt",
        receipt_kind=_receipt_kind(recipe.scope),
        scope=recipe.scope,
        verified_on=started_at.date(),
        started_by=started_by or default_started_by(),
        backend=backend,
        recipe=RecipePointer(id=recipe.id, repository=recipe_repository, commit=recipe_commit,
                             path=f"recipes/{relative}", sha256=loaded.sha256),
        kg_release_id=kg_release_id,
        code=code,
        environment=ReceiptEnvironment(
            images=recipe.image_digests(), image_used=image_used, container_runtime=runtime,
            python=platform.python_version(), os=platform.platform(),
        ),
        inputs=input_records,
        hardware=hardware,
        commands=commands,
        started_at=started_at,
        finished_at=datetime.datetime.now(datetime.UTC),
        wall_seconds=round(time.perf_counter() - started, 3),
        peak_memory_bytes=max(peaks) if peaks else None,
        outputs=_output_records(folder),
        metrics={check.metric_id: check.observed for check in checks if check.metric_id},
        checks=checks,
        checks_passed=all(c.exit_code == 0 for c in commands) and all(c.passed for c in checks),
        cost=owned_hardware_cost(),
    )
    save(receipt, folder)
    return receipt


# Options of a CLI run whose values are neither a path nor a private name: the registry ID
# of the model and the batch size.
PUBLIC_OPTIONS = ("--model", "--batch-size")
LOCAL_VALUE = "<local value>"


def recorded_argv(argv: list[str]) -> list[str]:
    """The argv a receipt records for a CLI run without a recipe.

    Option values are local paths and private model names, so only the subcommand, the
    option names and the values of ``PUBLIC_OPTIONS`` are kept. Without the model's
    registry ID a receipt would not say which model ran.
    """
    recorded = [argv[0] if argv else "run"]
    keep = False
    for token in argv[1:]:
        if token.startswith("--"):
            name, separator, _ = token.partition("=")
            recorded.append(token if separator and name in PUBLIC_OPTIONS else name)
            keep = not separator and name in PUBLIC_OPTIONS
        else:
            recorded.append(token if keep else LOCAL_VALUE)
            keep = False
    return recorded


def write_adhoc_receipt(
    folder: str | Path,
    *,
    argv: list[str],
    input_files: list[Path],
    metrics: dict[str, Any],
    started_at: datetime.datetime,
    wall_seconds: float,
) -> Receipt | None:
    """Receipt for a CLI run that did not go through a recipe.

    Skipped inside a recipe run, where the recipe's own receipt covers the folder.
    Option values are local paths and private model names, so the recorded command is
    ``recorded_argv``: the subcommand, the option names, and the values of the model ID
    and the batch size.
    """
    if os.environ.get(NESTED_VARIABLE):
        return None
    folder = Path(folder)
    image_used, runtime = _image_used()
    argv = recorded_argv(argv)
    receipt = Receipt(
        schema_version=SCHEMA_VERSION,
        kind="rewire_run_receipt",
        receipt_kind="runner_reproduction",
        scope=None,
        verified_on=started_at.date(),
        started_by=default_started_by(),
        backend="local",
        recipe=None,
        recipe_missing=Missing(reason="inapplicable", note="CLI run without a recipe"),
        code=code_state(_source_root()),
        environment=ReceiptEnvironment(
            images=dict.fromkeys(PLATFORMS), image_used=image_used, container_runtime=runtime,
            python=platform.python_version(), os=platform.platform(),
        ),
        inputs=[
            InputRecord(name=path.name, source="local file", obtained="local",
                        sha256=sha256_file(path), bytes=path.stat().st_size)
            for path in input_files if path.is_file()
        ],
        hardware=detect_hardware(),
        commands=[CommandRecord(name=argv[0], argv=["rewirebench", *argv], exit_code=0,
                                wall_seconds=round(wall_seconds, 3),
                                peak_memory_bytes=own_peak_memory_bytes())],
        started_at=started_at,
        finished_at=datetime.datetime.now(datetime.UTC),
        wall_seconds=round(wall_seconds, 3),
        peak_memory_bytes=own_peak_memory_bytes(),
        outputs=_output_records(folder),
        metrics=metrics,
        checks=[],
        checks_passed=True,
        cost=owned_hardware_cost(),
    )
    save(receipt, folder)
    return receipt


def _folder(path: str | Path) -> Path:
    """The run folder, whether the caller named it or the receipt.json inside it."""
    path = Path(path)
    return path.parent if path.is_file() and path.name == RECEIPT_FILENAME else path


def load_receipt(folder: str | Path) -> Receipt:
    path = _folder(folder) / RECEIPT_FILENAME
    try:
        return Receipt.model_validate(json.loads(path.read_text()))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not a receipt: {error}") from error


def _result(check: str, status: CheckStatus, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "detail": detail}


def _verify_sums(folder: Path) -> list[dict[str, str]]:
    sums = folder / SUMS_FILENAME
    if not sums.is_file():
        # Every run writes one, so its absence is a change to the folder.
        return [_result("sha256sums", "mismatch", f"{SUMS_FILENAME} is not in the folder")]
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, separator, name = line.partition("  ")
        try:
            name = relative_path(name.removeprefix("./"))
        except ValueError:
            return [_result("sha256sums", "mismatch", f"Unsafe path in {SUMS_FILENAME}: {name!r}")]
        if not separator or not sdk._is_sha256(digest):
            return [_result("sha256sums", "mismatch", f"Not a digest line: {line!r}")]
        if name in listed:
            return [_result("sha256sums", "mismatch", f"{name} is listed twice")]
        listed[name] = digest
    present = {
        path.relative_to(folder).as_posix()
        for path in folder.rglob("*") if path.is_file() and path.name != SUMS_FILENAME
    }
    wrong = sorted(name for name in listed if name in present
                   and sha256_file(folder / name) != listed[name])
    problems = [f"changed: {wrong}"] if wrong else []
    if sorted(set(listed) - present):
        problems.append(f"missing: {sorted(set(listed) - present)}")
    if sorted(present - set(listed)):
        problems.append(f"not listed: {sorted(present - set(listed))}")
    if problems:
        return [_result("sha256sums", "mismatch", "; ".join(problems))]
    return [_result("sha256sums", "match", f"{len(listed)} files")]


def _recipe_at_commit(pointer: RecipePointer, recipes_dir: Path) -> Recipe | None:
    """The recipe as committed at the receipt's commit, if this checkout has those bytes."""
    if pointer.commit is None:
        return None
    try:
        done = subprocess.run(
            ["git", "show", f"{pointer.commit}:{relative_path(pointer.path)}"],
            cwd=recipes_dir.parent, check=True, capture_output=True, timeout=20,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if hashlib.sha256(done.stdout).hexdigest() != pointer.sha256:
        return None
    try:
        return parse_recipe(done.stdout)
    except ValueError:
        return None


def _verify_recipe(
    receipt: Receipt, recipes_dir: Path | None
) -> tuple[Recipe | None, dict[str, str]]:
    if receipt.recipe is None:
        return None, _result("recipe", "not_available", "The run did not use a recipe")
    if recipes_dir is None:
        return None, _result("recipe", "not_available", "No recipes directory to compare with")
    loaded: LoadedRecipe | None = None
    problem = ""
    try:
        loaded = get_recipe(receipt.recipe.id, recipes_dir)
    except ValueError as error:
        problem = str(error)
    if loaded is not None and loaded.sha256 == receipt.recipe.sha256:
        return loaded.recipe, _result("recipe", "match", loaded.sha256)
    # The recipe was edited or removed since the run. The receipt's commit may still hold it.
    earlier = _recipe_at_commit(receipt.recipe, recipes_dir)
    if earlier is not None:
        return earlier, _result(
            "recipe", "match",
            f"{receipt.recipe.sha256} at commit {receipt.recipe.commit}; "
            "the file has changed since",
        )
    if loaded is None:
        return None, _result("recipe", "not_available", problem)
    return None, _result(
        "recipe", "mismatch",
        f"The recipe file is {loaded.sha256}; the receipt cites {receipt.recipe.sha256}, and "
        "this checkout does not hold those bytes at the receipt's commit.",
    )


def _verify_inputs(
    receipt: Receipt, recipe: Recipe | None, inputs: str | Path | None
) -> dict[str, str]:
    """Compare the receipt's inputs with the recipe's pins and, if given, with the files."""
    if recipe is None and inputs is None:
        return _result("inputs", "not_available",
                       "Needs the recipe the receipt cites, or --inputs to re-hash the files")
    wrong: list[str] = []
    compared: list[str] = []
    if recipe is not None:
        pinned = {item.name: item for item in recipe.inputs}
        recorded = {record.name: record for record in receipt.inputs}
        wrong += sorted(set(pinned) ^ set(recorded))
        wrong += sorted(
            name for name in set(pinned) & set(recorded)
            if recorded[name].sha256 != pinned[name].sha256
            or (pinned[name].bytes is not None and recorded[name].bytes != pinned[name].bytes)
        )
        compared.append("the recipe's pins")
    if inputs is not None:
        for record in receipt.inputs:
            path = Path(inputs) / record.name
            if not path.is_file() or sha256_file(path) != record.sha256:
                wrong.append(record.name)
        compared.append("the files given")
    if wrong:
        return _result("inputs", "mismatch", f"missing or changed: {sorted(set(wrong))}")
    return _result("inputs", "match",
                   f"{len(receipt.inputs)} inputs equal {' and '.join(compared)}")


def _verify_commits(receipt: Receipt) -> dict[str, str]:
    """Whether the receipt names a clean code commit and a committed recipe.

    A reviewer needs both before a receipt can support a promotion. A run from a wheel,
    a dirty tree or an uncommitted recipe is still a valid run, so this is never a
    mismatch; ``--strict`` turns it into a failure.
    """
    gaps: list[str] = []
    if receipt.code.commit is None:
        gaps.append("code.commit is null")
    if receipt.code.dirty is not False:
        gaps.append("the code tree was not clean")
    if receipt.recipe is None or receipt.recipe.commit is None:
        gaps.append("recipe.commit is null")
    if gaps:
        return _result("commits", "not_available", "; ".join(gaps))
    return _result("commits", "match", f"code {receipt.code.commit}")


def verify(
    folder: str | Path,
    *,
    recipes_dir: Path | None = None,
    inputs: str | Path | None = None,
) -> dict[str, Any]:
    """Re-check a receipt folder. ``ok`` is false when anything differs or the run failed."""
    folder = _folder(folder)
    receipt = load_receipt(folder)
    results = _verify_sums(folder)

    recorded = {record.path: record for record in receipt.outputs}
    present = {record.path: record for record in _output_records(folder)}
    changed = sorted(path for path in recorded if path in present
                     and present[path].sha256 != recorded[path].sha256)
    gone = sorted(set(recorded) - set(present))
    extra = sorted(set(present) - set(recorded))
    if changed or gone or extra:
        results.append(_result("outputs", "mismatch",
                               f"changed: {changed}; missing: {gone}; not in receipt: {extra}"))
    else:
        results.append(_result("outputs", "match", f"{len(recorded)} files"))

    # A failed run never verifies, whether or not the recipe can be found.
    failed = [command.name for command in receipt.commands if command.exit_code != 0]
    failed += [check.label for check in receipt.checks if not check.passed]
    if failed or not receipt.checks_passed:
        results.append(_result("run", "mismatch", f"The run did not pass: {failed}"))
    else:
        results.append(_result(
            "run", "match",
            f"{len(receipt.commands)} commands exited 0, {len(receipt.checks)} checks passed",
        ))

    recipe, outcome = _verify_recipe(receipt, recipes_dir)
    results.append(outcome)
    results.append(_verify_inputs(receipt, recipe, inputs))

    recomputed = evaluate_checks(recipe, folder) if recipe is not None else None
    if recomputed is None:
        results.append(_result("checks", "not_available",
                               "Needs the recipe file the receipt cites"))
    else:
        if recomputed != receipt.checks:
            results.append(_result("checks", "mismatch",
                                   "Checks recomputed from the outputs differ from the receipt"))
        elif not all(check.passed for check in recomputed):
            results.append(_result("checks", "mismatch", "The run did not pass its checks"))
        else:
            results.append(_result("checks", "match", f"{len(recomputed)} checks pass"))

    if receipt.recipe is None:
        results.append(_result("metrics", "not_available",
                               "A run without a recipe has no checks to compare its metrics with"))
    else:
        # With the recipe, the values come from the outputs; without it, the receipt must
        # at least agree with its own checks.
        source = recomputed if recomputed is not None else receipt.checks
        expected = {check.metric_id: check.observed for check in source if check.metric_id}
        if receipt.metrics != expected:
            results.append(_result("metrics", "mismatch",
                                   f"The receipt's metrics differ from the checked values: "
                                   f"{sorted(set(receipt.metrics) | set(expected))}"))
        elif recomputed is None:
            results.append(_result("metrics", "not_available",
                                   "Equal to the receipt's own checks; needs the recipe to "
                                   "re-read them from the outputs"))
        else:
            results.append(_result("metrics", "match", f"{len(expected)} values"))

    results.append(_verify_commits(receipt))
    return {
        "receipt": str(folder / RECEIPT_FILENAME),
        "recipe_id": receipt.recipe.id if receipt.recipe else None,
        "ok": all(result["status"] != "mismatch" for result in results),
        "complete": all(result["status"] == "match" for result in results),
        "results": results,
    }
