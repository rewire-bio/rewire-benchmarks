"""Run receipts: what ran, on what, from which bytes, and what it produced.

Every run writes ``receipt.json`` into its output folder and a ``SHA256SUMS`` that
covers every file in the folder, the receipt included. ``verify`` recomputes the
digests and re-evaluates the recipe's checks from the output files; it never trusts
the values the receipt recorded.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, StringConstraints, model_validator

from rewirebench import sdk
from rewirebench.recipes import (
    PLATFORMS,
    ImageDigest,
    LoadedRecipe,
    Missing,
    Recipe,
    Revision,
    Scope,
    Sha256,
    Strict,
    Text,
    get_recipe,
    relative_path,
    sha256_file,
)

SCHEMA_VERSION = "1.0"
RECEIPT_FILENAME = "receipt.json"
SUMS_FILENAME = "SHA256SUMS"
NESTED_VARIABLE = "REWIREBENCH_RECIPE_RUN"
OWNED_HARDWARE = "owned hardware"
CODE_REPOSITORY = "https://github.com/rewire-bio/rewire-benchmarks"
MAX_INPUT_BYTES = 8 * 2**30

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
    name: Text
    source: Text
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
    return "ci" if os.environ.get("CI") else "human"


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
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt.model_dump(mode="json"), indent=2, allow_nan=False) + "\n")
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
    expected = recipe.expected_output
    if expected is None:
        return checks
    for value in expected.values:
        observed: float | None = None
        try:
            found = _lookup(json.loads((folder / value.file).read_text()), list(value.key))
            if isinstance(found, (int, float)) and not isinstance(found, bool):
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
        if supplied is not None:
            source = supplied / item.name
            if not source.is_file():
                raise ValueError(f"Input {item.name} not found in {supplied}")
            shutil.copyfile(source, target)
        elif item.repo_path is not None:
            if repo_root is None:
                raise ValueError(f"Input {item.name} needs a repository checkout or --inputs")
            shutil.copyfile(repo_root / item.repo_path, target)
        else:
            assert item.url is not None
            _fetch(item.url, target)
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
            sha256=digest, bytes=target.stat().st_size,
        ))
    return records


def _render(argument: str, values: dict[str, str]) -> str:
    for name, value in values.items():
        argument = argument.replace("{" + name + "}", value)
    return argument


def _execute(argv: list[str], *, cwd: Path, log: Path) -> tuple[int | None, float, int | None]:
    """Run one command without a shell. Returns exit code, wall seconds, peak memory."""
    executable = [sys.executable, "-m", "rewirebench.cli", *argv[1:]] \
        if argv[0] == "rewirebench" else argv
    environment = {**os.environ, NESTED_VARIABLE: "1"}
    started = time.perf_counter()
    with (log.with_suffix(".stdout.txt").open("xb") as out,
          log.with_suffix(".stderr.txt").open("xb") as err):
        try:
            process = subprocess.Popen(executable, cwd=cwd, env=environment, stdout=out, stderr=err)
        except OSError as error:
            err.write(f"{type(error).__name__}: {error}\n".encode())
            return None, time.perf_counter() - started, None
        peak: int | None = None
        if hasattr(os, "wait4"):
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            peak = peak_memory_bytes(usage.ru_maxrss)
        else:
            process.wait()
    return process.returncode, time.perf_counter() - started, peak


def run_recipe(
    loaded: LoadedRecipe,
    *,
    output: str | Path,
    recipes_dir: Path,
    inputs: str | Path | None = None,
    started_by: StartedBy | None = None,
    kg_release_id: str | None = None,
) -> Receipt:
    """Run a recipe on this machine and write its receipt. No other backend runs here."""
    recipe = loaded.recipe
    if recipe.backends.local is None:
        raise ValueError(f"Recipe {recipe.id} does not name the local backend")
    folder = Path(output)
    if folder.exists():
        raise FileExistsError(f"Output folder already exists: {folder}")
    repo_root = recipes_dir.parent
    code = code_state(repo_root)
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
                argv, cwd=work, log=folder / "logs" / f"{index:02d}-{command.name}"
            )
            failed = exit_code != 0
            # The recipe's own argv is recorded, so a receipt holds no local paths.
            commands.append(CommandRecord(name=command.name, argv=list(command.argv),
                                          exit_code=exit_code, wall_seconds=round(seconds, 3),
                                          peak_memory_bytes=peak))
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
        backend="local",
        recipe=RecipePointer(id=recipe.id, repository=code.repository, commit=code.commit,
                             path=f"recipes/{relative}", sha256=loaded.sha256),
        kg_release_id=kg_release_id,
        code=code,
        environment=ReceiptEnvironment(
            images=recipe.image_digests(), image_used=image_used, container_runtime=runtime,
            python=platform.python_version(), os=platform.platform(),
        ),
        inputs=input_records,
        hardware=detect_hardware(),
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
    Option values are local paths and private model names, so only the subcommand
    and the option names are recorded, as the baseline manifest already does.
    """
    if os.environ.get(NESTED_VARIABLE):
        return None
    folder = Path(folder)
    image_used, runtime = _image_used()
    name = argv[0] if argv else "run"
    argv = [name, *(token.split("=", 1)[0] if token.startswith("--") else "<local value>"
                    for token in argv[1:])]
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
            InputRecord(name=path.name, source="local file", sha256=sha256_file(path),
                        bytes=path.stat().st_size)
            for path in input_files if path.is_file()
        ],
        hardware=detect_hardware(),
        commands=[CommandRecord(name=name, argv=["rewirebench", *argv], exit_code=0,
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


def load_receipt(folder: str | Path) -> Receipt:
    path = Path(folder)
    path = path / RECEIPT_FILENAME if path.is_dir() else path
    return Receipt.model_validate(json.loads(path.read_text()))


def _result(check: str, status: CheckStatus, detail: str) -> dict[str, str]:
    return {"check": check, "status": status, "detail": detail}


def _verify_sums(folder: Path) -> list[dict[str, str]]:
    sums = folder / SUMS_FILENAME
    if not sums.is_file():
        return [_result("sha256sums", "not_available", f"{SUMS_FILENAME} is not in the folder")]
    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, _, name = line.partition("  ")
        try:
            listed[relative_path(name.removeprefix("./"))] = digest
        except ValueError:
            return [_result("sha256sums", "mismatch", f"Unsafe path in {SUMS_FILENAME}: {name!r}")]
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


def verify(
    folder: str | Path,
    *,
    recipes_dir: Path | None = None,
    inputs: str | Path | None = None,
) -> dict[str, Any]:
    """Re-check a receipt folder. ``ok`` is false when anything recomputed differs."""
    folder = Path(folder)
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

    recipe: Recipe | None = None
    if receipt.recipe is None:
        results.append(_result("recipe", "not_available", "The run did not use a recipe"))
    elif recipes_dir is None:
        results.append(_result("recipe", "not_available", "No recipes directory to compare with"))
    else:
        try:
            loaded = get_recipe(receipt.recipe.id, recipes_dir)
        except ValueError as error:
            results.append(_result("recipe", "not_available", str(error)))
        else:
            if loaded.sha256 == receipt.recipe.sha256:
                recipe = loaded.recipe
                results.append(_result("recipe", "match", loaded.sha256))
            else:
                results.append(_result(
                    "recipe", "mismatch",
                    f"The recipe file is {loaded.sha256}; the receipt cites "
                    f"{receipt.recipe.sha256}. Check out the receipt's commit to compare.",
                ))

    if inputs is None:
        results.append(_result("inputs", "not_available", "Pass --inputs to re-hash the inputs"))
    else:
        wrong = []
        for record in receipt.inputs:
            path = Path(inputs) / record.name
            if not path.is_file() or sha256_file(path) != record.sha256:
                wrong.append(record.name)
        results.append(_result("inputs", "mismatch", f"missing or changed: {wrong}") if wrong
                       else _result("inputs", "match", f"{len(receipt.inputs)} files"))

    if recipe is None:
        results.append(_result("checks", "not_available",
                               "Needs the recipe file the receipt cites"))
    else:
        recomputed = evaluate_checks(recipe, folder)
        commands_passed = all(command.exit_code == 0 for command in receipt.commands)
        passed = commands_passed and all(check.passed for check in recomputed)
        if recomputed != receipt.checks:
            results.append(_result("checks", "mismatch",
                                   "Checks recomputed from the outputs differ from the receipt"))
        elif not passed:
            results.append(_result("checks", "mismatch", "The run did not pass its checks"))
        else:
            results.append(_result("checks", "match", f"{len(recomputed)} checks pass"))

    return {
        "receipt": str(folder / RECEIPT_FILENAME),
        "recipe_id": receipt.recipe.id if receipt.recipe else None,
        "ok": all(result["status"] != "mismatch" for result in results),
        "complete": all(result["status"] == "match" for result in results),
        "results": results,
    }
