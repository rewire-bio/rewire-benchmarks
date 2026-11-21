"""Run recipes: validated YAML files under ``recipes/`` that the runner executes.

A recipe pins what a run needs (inputs, environment, weights) and states how to tell
that it worked (expected output). It carries no status. The evidence store holds a
pointer to the file (repository, commit, path, SHA-256) and the verified status, so
promoting a recipe never changes the bytes a receipt cites.

Schema 1.0 lets a recipe run only five subcommands of rewirebench itself: four data
commands, and ``run`` with a model from the registry in ``rewirebench.models``. A recipe
cannot name a shell, another program or an arbitrary Python callable, so it cannot start
a pod or submit a result. The one model and the one baseline that start an executable
named in their options (FIMO) are refused in a recipe for that reason.

Field names follow ADR 0005 of rewire-benchmark-data where the two overlap; see
``docs/adr-c-recipes-and-receipts.md``.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    ValidationError,
    model_validator,
)

SCHEMA_VERSION = "1.0"
PLATFORMS = ("linux/amd64", "linux/arm64")
REQUIREMENT_BLOCKS = ("weights_pin", "environment", "hardware_floor", "expected_output")
PLACEHOLDERS = ("inputs", "work", "output")
RECIPES_DIR_VARIABLE = "REWIREBENCH_RECIPES_DIR"
SCHEMA_FILENAME = "recipe.schema.json"
# The only program a recipe may name, and the subcommands it may use. These prepare and
# score data on the machine. Left out on purpose: `submit`, `submit-queue` and `enqueue`
# (a recipe never sends anything), `recipe` and `receipt` (it never starts another recipe).
# `run` is accepted only with --model and a registered model: its --adapter option imports
# and calls any installed Python callable, for example subprocess.Popen. See check_run_step.
RUNNER_PROGRAM = "rewirebench"
DATA_COMMANDS = ("prepare", "evaluate", "run-baselines", "prepare-baseline-artifact")
MODEL_COMMAND = "run"
BASELINES_COMMAND = "run-baselines"
RUNNER_COMMANDS = (*DATA_COMMANDS, MODEL_COMMAND)
# Names the runner writes into a run folder. A recipe cannot list them as outputs.
RESERVED_OUTPUTS = ("receipt.json", "SHA256SUMS")
RESERVED_FOLDER = "logs"
DEFAULT_TIMEOUT_SECONDS = 3600
MAX_TIMEOUT_SECONDS = 24 * 3600
MAX_RUNPOD_HOURS = 24
RUNPOD_PROVIDER = "runpod"

Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
Revision = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{40}$")]
ImageDigest = Annotated[str, StringConstraints(pattern=r"^sha256:[a-f0-9]{64}$")]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$")]
Text = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
# Strict numbers: YAML `"3"` or `true` is an error, not 3 or 1, and inf and nan are refused.
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PythonSpec = Annotated[str, StringConstraints(pattern=r"^(>=|==)3\.\d+(\.\d+)?$")]
Scope = Literal["synthetic_smoke", "selected_assay", "full_protocol"]
BackendName = Literal["local", "spark", "runpod", "nextflow"]
MissingReason = Literal["inapplicable", "unavailable", "unextracted", "unreported"]

_PLACEHOLDER = re.compile(r"\{([A-Za-z_]+)\}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RecipeError(ValueError):
    """A recipe file that cannot be used, with a message that names the field."""


def relative_path(value: str, inside: str = "the run folder") -> str:
    """Reject absolute paths and parent traversal in a path written in a recipe."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Not a relative path inside {inside}: {value!r}")
    return str(path)


def python_satisfies(spec: str, version: tuple[int, ...]) -> bool:
    """Whether a Python version meets a recipe's ``>=3.11`` or ``==3.11`` style bound."""
    wanted = tuple(int(part) for part in spec[2:].split("."))
    if spec.startswith("=="):
        return tuple(version[: len(wanted)]) == wanted
    return tuple(version[:3]) >= wanted


def check_run_step(argv: list[str], *, cpu_only: bool = False) -> None:
    """Refuse a ``rewirebench run`` step that could call anything but a registered model.

    The step is parsed by the real command-line parser, which accepts no abbreviated
    option for ``run``, so ``--adapter=x`` and ``--adap x`` are seen for what they are.
    It must name its model with ``--model``, the model must be one a recipe may use, and
    the adapter options must be ones the model declares, of the declared type. With
    ``cpu_only`` (the local backend) a ``device`` other than ``cpu`` is refused too:
    hiding CUDA devices does not hide Apple MPS.
    """
    from rewirebench import cli, models

    args = cli.parse_strict(argv[1:])
    if args.command != MODEL_COMMAND:
        raise ValueError(f"Not a {MODEL_COMMAND} step: {argv[:2]}")
    if args.adapter is not None or args.model is None:
        raise ValueError(
            "A recipe run step names a registered model with --model, never --adapter"
        )
    spec = models.get_model(args.model)
    if not spec.recipe_allowed:
        raise ValueError(f"Model {spec.id} starts another program, so a recipe cannot use it")
    try:
        options = json.loads(args.adapter_options)
    except json.JSONDecodeError as error:
        raise ValueError(f"--adapter-options is not JSON: {error}") from error
    models.check_options(spec, options)
    if cpu_only:
        models.check_cpu(spec, options)


def check_baselines_step(argv: list[str]) -> None:
    """Refuse a ``run-baselines`` step that would start an executable it names.

    The FIMO baseline takes the path of an executable in its options and starts it. It
    is refused whether it is selected with ``--baseline`` or named in
    ``--baseline-options``, and so is any ``fimo`` option under another name.
    """
    from rewirebench import baselines, cli

    args = cli.parse_strict(argv[1:])
    if args.command != BASELINES_COMMAND:
        raise ValueError(f"Not a {BASELINES_COMMAND} step: {argv[:2]}")
    try:
        options = json.loads(args.baseline_options)
    except json.JSONDecodeError as error:
        raise ValueError(f"--baseline-options is not JSON: {error}") from error
    if not isinstance(options, dict):
        raise ValueError("--baseline-options is a JSON object")  # noqa: TRY004
    named = [*(args.baseline or []), *options]
    takes_program = any(isinstance(value, dict) and baselines.PROGRAM_OPTION in value
                        for value in options.values())
    barred = sorted(set(named) & set(baselines.PROGRAM_BASELINES))
    if barred or takes_program:
        raise ValueError(
            f"Baseline {barred[0] if barred else baselines.PROGRAM_BASELINES[0]} starts "
            "another program, so a recipe cannot use it"
        )


def check_step(argv: list[str], *, cpu_only: bool = False) -> None:
    """The content rules of one step, by subcommand. Used at validation and at run time."""
    if len(argv) > 1 and argv[1] == MODEL_COMMAND:
        check_run_step(argv, cpu_only=cpu_only)
    elif len(argv) > 1 and argv[1] == BASELINES_COMMAND:
        check_baselines_step(argv)


class Strict(BaseModel):
    """Unknown keys are errors, so a misspelt field cannot pass as an absent one."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Missing(Strict):
    """Why a block is absent, in the store's missingness vocabulary."""

    reason: MissingReason
    note: Text | None = None


class Licence(Strict):
    """Free text, an SPDX identifier where one applies."""

    code: Text
    weights: Text
    data: Text


class PlatformImage(Strict):
    """A container image by digest for one platform, or the reason there is none."""

    reference: Text | None = None
    missing: Missing | None = None

    @model_validator(mode="after")
    def _one_of(self) -> PlatformImage:
        if (self.reference is None) == (self.missing is None):
            raise ValueError("Give exactly one of reference or missing")
        if self.reference is not None and not re.fullmatch(
            r"[^\s@]+@sha256:[a-f0-9]{64}", self.reference
        ):
            raise ValueError("An image is referenced by digest (name@sha256:...), never by tag")
        return self

    @property
    def digest(self) -> str | None:
        return None if self.reference is None else self.reference.rsplit("@", 1)[1]


class Lockfile(Strict):
    """A lock file in the recipe's repository. A run refuses a file with other bytes."""

    path: Text
    sha256: Sha256

    @model_validator(mode="after")
    def _relative(self) -> Lockfile:
        relative_path(self.path, "the repository")
        return self


class Environment(Strict):
    """ADR 0005 ``environment``: runtime, Python and one image digest per platform."""

    runtime: Literal["python", "docker", "podman", "apptainer"]
    python: PythonSpec | None = None
    images: dict[Literal["linux/amd64", "linux/arm64"], PlatformImage]
    lockfile: Lockfile | None = None

    @model_validator(mode="after")
    def _both_platforms(self) -> Environment:
        if set(self.images) != set(PLATFORMS):
            raise ValueError(f"images needs exactly these platforms: {', '.join(PLATFORMS)}")
        return self


class WeightsFile(Strict):
    path: Text
    sha256: Sha256

    @model_validator(mode="after")
    def _relative(self) -> WeightsFile:
        relative_path(self.path, "the weights repository")
        return self


class WeightsPin(Strict):
    """ADR 0005 ``weights_pin``: where the weights are and which bytes they are."""

    host: Literal["huggingface", "github", "zenodo", "other"]
    repository: Text
    revision: Revision | None = None
    files: list[WeightsFile] = Field(min_length=1)
    licence: Text
    gated: StrictBool
    resolved_on: datetime.date

    @model_validator(mode="after")
    def _revision_on_git_hosts(self) -> WeightsPin:
        if self.host in {"huggingface", "github"} and not self.revision:
            raise ValueError("A Hugging Face or GitHub weights pin needs a 40-hex revision")
        return self


class RecipeInput(Strict):
    """One input file, pinned by SHA-256, from a URL or from this repository."""

    name: Text
    sha256: Sha256
    url: Text | None = None
    repo_path: Text | None = None
    bytes: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _one_source(self) -> RecipeInput:
        if (self.url is None) == (self.repo_path is None):
            raise ValueError("An input has exactly one of url or repo_path")
        if self.url is not None:
            parts = urlsplit(self.url)
            if parts.scheme != "https" or not parts.hostname:
                raise ValueError("Input URLs must use https")
            if parts.username is not None or parts.password is not None:
                # The URL is copied into the receipt, so it must not carry a credential.
                raise ValueError("Input URLs must not contain a user name or password")
        if self.repo_path is not None:
            relative_path(self.repo_path, "the repository")
        if relative_path(self.name) != self.name or "/" in self.name:
            raise ValueError("An input name is a plain file name")
        return self


class Accelerator(Strict):
    kind: Literal["none", "gpu"]
    model: Text | None = None
    count: StrictInt | None = Field(default=None, ge=1)
    memory_gb: Number | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _cpu_has_no_gpu_fields(self) -> Accelerator:
        if self.kind == "none" and (self.model or self.count or self.memory_gb):
            raise ValueError("A recipe with no accelerator cannot name a GPU model or memory")
        return self


class HardwareFloor(Strict):
    """ADR 0005 ``hardware_floor``. Every figure shares the stated basis."""

    accelerator: Accelerator
    host_memory_gb: Number | None = Field(default=None, gt=0)
    disk_gb: Number | None = Field(default=None, gt=0)
    wall_clock_seconds: Number | None = Field(default=None, gt=0)
    basis: Literal["source_stated", "measured_by_rewire"]
    note: Text | None = None


def _step_schema(schema: dict[str, Any]) -> None:
    """Add the shape of the step rules to the JSON Schema of a command."""
    schema["properties"]["argv"]["prefixItems"] = [
        {"const": RUNNER_PROGRAM}, {"enum": list(RUNNER_COMMANDS)}]
    schema["if"] = {"properties": {"argv": {"prefixItems": [{}, {"const": MODEL_COMMAND}]}}}
    schema["then"] = {"properties": {"argv": {
        "contains": {"pattern": "^--model(=.+)?$"},
        "not": {"contains": {"pattern": "^--adapter(=.*)?$"}},
    }}}


class Command(Strict):
    """One rewirebench step as an argv list. No shell is involved.

    The first element is ``rewirebench`` and the second one of ``RUNNER_COMMANDS``. A
    ``run`` step names a registered model with ``--model``. A step that runs longer than
    ``timeout_seconds`` is stopped and the run fails.

    The JSON Schema states the shape of these rules: the program, the subcommand, and
    for ``run`` a ``--model`` option and no ``--adapter``. Whether the model is
    registered, may be used in a recipe and is given options it declares is checked by
    ``rewirebench recipe validate`` only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True,
                              json_schema_extra=lambda schema: _step_schema(schema))

    name: Identifier
    argv: list[Text] = Field(min_length=2)
    timeout_seconds: StrictInt = Field(
        default=DEFAULT_TIMEOUT_SECONDS, gt=0, le=MAX_TIMEOUT_SECONDS
    )

    @model_validator(mode="after")
    def _known_placeholders(self) -> Command:
        if self.argv[0] != RUNNER_PROGRAM or self.argv[1] not in RUNNER_COMMANDS:
            raise ValueError(
                f"Command {self.name}: a recipe runs only `{RUNNER_PROGRAM}` followed by one "
                f"of {', '.join(RUNNER_COMMANDS)}. No shell and no other program."
            )
        try:
            check_step(list(self.argv))
        except ValueError as error:
            raise ValueError(f"Command {self.name}: {error}") from error
        for argument in self.argv:
            unknown = sorted(set(_PLACEHOLDER.findall(argument)) - set(PLACEHOLDERS))
            if unknown:
                raise ValueError(
                    f"Command {self.name}: unknown placeholder {{{unknown[0]}}}. "
                    "Use {inputs}, {work} or {output}."
                )
        return self


class ExpectedValue(Strict):
    """A value read from a JSON output file and compared within a tolerance."""

    label: Text
    metric_id: Text
    file: Text
    key: list[Text | int] = Field(min_length=1)
    value: Number
    tolerance: Number = Field(ge=0)
    unit: Text | None = None

    @model_validator(mode="after")
    def _relative_file(self) -> ExpectedValue:
        relative_path(self.file)
        # A tolerance wider than the value accepts zero and a sign change, so it checks
        # nothing. For an expected value of zero the limit is 1.
        limit = abs(self.value) if self.value else 1.0
        if self.tolerance > limit:
            raise ValueError(
                f"{self.metric_id}: tolerance {self.tolerance} is wider than the check allows "
                f"(at most {limit} for an expected value of {self.value})"
            )
        return self


class ExpectedArtifact(Strict):
    """An output file that is bit-reproducible, checked by SHA-256."""

    path: Text
    sha256: Sha256

    @model_validator(mode="after")
    def _relative(self) -> ExpectedArtifact:
        relative_path(self.path)
        return self


class ExpectedOutput(Strict):
    """ADR 0005 ``expected_output``: how a reader can tell the run worked."""

    description: Text
    values: list[ExpectedValue] = []
    artifacts: list[ExpectedArtifact] = []

    @model_validator(mode="after")
    def _something_to_check(self) -> ExpectedOutput:
        if not self.values and not self.artifacts:
            raise ValueError("expected_output needs at least one value or artifact")
        identifiers = [value.metric_id for value in self.values]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("expected_output metric_id values must be unique")
        return self


class PriceBasis(Strict):
    """A dated price observation. Without one, a recipe states no money figure."""

    provider: Text
    accelerator: Text
    amount_per_hour: Number = Field(gt=0)
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
    rate_source: Text
    rate_date: datetime.date


class Estimate(Strict):
    gpu_hours: Number | None = Field(default=None, ge=0)
    wall_clock_seconds: Number | None = Field(default=None, gt=0)
    price_basis: PriceBasis | None = None
    note: Text | None = None


class LocalBackend(Strict):
    pass


class SparkBackend(Strict):
    gpus: StrictBool = True


class RunpodBackend(Strict):
    """Pod shape and the longest the pod may live. The quote is built from these."""

    gpu_types: list[Text] = Field(min_length=1)
    cloud: Literal["SECURE", "COMMUNITY"] = "SECURE"
    disk_gb: StrictInt = Field(gt=0)
    max_hours: Number = Field(gt=0, le=MAX_RUNPOD_HOURS)


class NextflowBackend(Strict):
    profile: Text


class Backends(Strict):
    """Where the recipe may run. A backend that is not named here must refuse it."""

    local: LocalBackend | None = None
    spark: SparkBackend | None = None
    runpod: RunpodBackend | None = None
    nextflow: NextflowBackend | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> Backends:
        if not self.names():
            raise ValueError("A recipe names at least one backend")
        return self

    def names(self) -> list[str]:
        return [name for name in type(self).model_fields if getattr(self, name) is not None]


class Recipe(Strict):
    schema_version: Literal["1.0"]
    id: Identifier
    title: Text
    summary: Text
    kind: Literal["model_run", "benchmark_run", "data_prep"]
    purpose: Literal[
        "rescore_predictions", "generate_and_evaluate", "obtain_and_run", "harness_run"
    ]
    scope: Scope
    rewirebench_protocol: Text | None = None
    kg_record_id: Text | None = None
    store_recipe_id: Text | None = None
    licence: Licence
    environment: Environment | None = None
    weights_pin: list[WeightsPin] | None = None
    hardware_floor: HardwareFloor | None = None
    expected_output: ExpectedOutput | None = None
    missing: dict[
        Literal["weights_pin", "environment", "hardware_floor", "expected_output"], Missing
    ] = {}
    inputs: list[RecipeInput] = Field(min_length=1)
    commands: list[Command] = Field(min_length=1)
    outputs: list[Text] = Field(min_length=1)
    estimate: Estimate | None = None
    backends: Backends
    limitations: list[Text] = []

    @model_validator(mode="after")
    def _rules(self) -> Recipe:
        for block in REQUIREMENT_BLOCKS:
            value = getattr(self, block)
            present = value is not None and value != []
            if present == (block in self.missing):
                raise ValueError(
                    f"{block} must be either present or named once in missing with a reason"
                )
        if self.purpose == "harness_run" and not self.rewirebench_protocol:
            raise ValueError("A harness_run recipe names its rewirebench_protocol")
        for label, names in (
            ("input", [item.name for item in self.inputs]),
            ("command", [command.name for command in self.commands]),
            ("output", self.outputs),
        ):
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate {label} names")
        for path in self.outputs:
            parts = PurePosixPath(relative_path(path)).parts
            if path in RESERVED_OUTPUTS or parts[0] == RESERVED_FOLDER:
                raise ValueError(
                    f"Output {path!r} uses a name the runner writes itself "
                    f"({', '.join(RESERVED_OUTPUTS)} and {RESERVED_FOLDER}/)"
                )
        floor = self.hardware_floor
        if self.backends.local is not None and floor and floor.accelerator.kind == "gpu":
            raise ValueError(
                "The local backend is CPU only. A recipe that needs a GPU names spark or runpod"
            )
        if self.backends.local is not None:
            for command in self.commands:
                try:
                    check_step(list(command.argv), cpu_only=True)
                except ValueError as error:
                    raise ValueError(
                        f"Command {command.name}: {error}. A recipe that names the local "
                        "backend runs on CPU"
                    ) from error
        if self.backends.runpod is not None:
            image = self.environment.images["linux/amd64"] if self.environment else None
            if image is None or image.digest is None:
                raise ValueError("The runpod backend needs a linux/amd64 image digest")
            if self.estimate is None or self.estimate.price_basis is None:
                # Without a dated hourly rate there is no amount to quote and confirm.
                raise ValueError("The runpod backend needs estimate.price_basis")
            basis = self.estimate.price_basis
            priced = self.backends.runpod.gpu_types[0]
            if basis.provider != RUNPOD_PROVIDER or basis.accelerator != priced:
                # The quote multiplies this rate, so it must be the rate of the pod asked for.
                raise ValueError(
                    f"estimate.price_basis must price the first runpod GPU type: provider "
                    f"{RUNPOD_PROVIDER!r} and accelerator {priced!r}, not "
                    f"{basis.provider!r} and {basis.accelerator!r}"
                )
        if self.backends.spark is not None:
            image = self.environment.images["linux/arm64"] if self.environment else None
            if image is None or image.digest is None:
                raise ValueError("The spark backend needs a linux/arm64 image digest")
        return self

    def image_digests(self) -> dict[str, str | None]:
        """Image digest per platform; ``None`` where the recipe records none."""
        if self.environment is None:
            return dict.fromkeys(PLATFORMS)
        return {platform: self.environment.images[platform].digest for platform in PLATFORMS}


@dataclass(frozen=True)
class LoadedRecipe:
    """A parsed recipe with the file it came from and the SHA-256 of its bytes."""

    recipe: Recipe
    path: Path
    sha256: str


class _UniqueKeyLoader(yaml.SafeLoader):
    """PyYAML keeps the last of two equal keys. A recipe with a repeated key is rejected."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        line = key_node.start_mark.line + 1
        try:
            hash(key)
        except TypeError:
            raise ValueError(f"A key at line {line} is a list or mapping, not text") from None
        if key in seen:
            raise ValueError(f"Duplicate key {key!r} at line {line}")
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _problems(error: ValidationError) -> str:
    """One line per problem: the field, then the message. No dump of the recipe, no URL."""
    lines: list[str] = []
    for problem in error.errors(include_url=False, include_input=False):
        where = ".".join(str(part) for part in problem["loc"]) or "recipe"
        lines.append(f"{where}: {problem['msg'].removeprefix('Value error, ')}")
    return "\n".join(lines)


def parse_recipe(text: str | bytes) -> Recipe:
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise RecipeError(f"Not valid YAML: {error}") from error
    except ValueError as error:
        raise RecipeError(str(error)) from error
    if not isinstance(data, dict):
        raise RecipeError("A recipe is a YAML mapping")
    try:
        recipe = Recipe.model_validate(data)
    except ValidationError as error:
        raise RecipeError(_problems(error)) from error
    if recipe.rewirebench_protocol is not None:
        from rewirebench import sdk

        if recipe.rewirebench_protocol not in sdk.PROTOCOLS:
            raise RecipeError(
                f"rewirebench_protocol: Unknown rewirebench protocol: {recipe.rewirebench_protocol}"
            )
    return recipe


def load_recipe(path: str | Path) -> LoadedRecipe:
    path = Path(path)
    content = path.read_bytes()
    if path.suffix != ".yaml":
        raise RecipeError(f"Recipe files end in .yaml, not {path.suffix or 'nothing'}")
    recipe = parse_recipe(content)
    if path.stem != recipe.id:
        raise RecipeError(f"Recipe {recipe.id} must be in a file named {recipe.id}.yaml")
    return LoadedRecipe(recipe=recipe, path=path, sha256=hashlib.sha256(content).hexdigest())


def find_recipes_dir(explicit: str | Path | None = None) -> Path | None:
    """Locate ``recipes/``: the argument, the environment, the checkout, the working tree.

    The checkout rewirebench runs from comes before the working directory, so running
    from inside another folder that has its own ``recipes/`` does not swap the recipes.
    The wheel does not ship recipes; an installed wheel looks at the working directory
    and its parents, and returns ``None`` when none has a ``recipes/``.
    """
    named = explicit or os.environ.get(RECIPES_DIR_VARIABLE)
    if named:
        path = Path(named)
        if not path.is_dir():
            raise ValueError(f"Recipes directory not found: {path}")
        return path.resolve()
    parents = Path(__file__).resolve().parents
    if len(parents) > 4 and (parents[4] / "recipes" / SCHEMA_FILENAME).is_file():
        return parents[4] / "recipes"
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / "recipes" / SCHEMA_FILENAME).is_file():
            return folder / "recipes"
    return None


def recipe_files(directory: Path) -> list[Path]:
    """Every YAML file under the directory. A ``.yml`` file is listed so it is reported."""
    return sorted([*directory.rglob("*.yaml"), *directory.rglob("*.yml")])


def validate_files(paths: list[Path]) -> list[dict[str, Any]]:
    """Validate each file and report per file, so one bad recipe does not hide another."""
    results: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for path in paths:
        entry: dict[str, Any] = {"path": str(path), "valid": False}
        try:
            loaded = load_recipe(path)
            if loaded.recipe.id in seen:
                raise ValueError(f"Recipe ID already used by {seen[loaded.recipe.id]}")
            seen[loaded.recipe.id] = path
            entry.update(valid=True, id=loaded.recipe.id, sha256=loaded.sha256)
        except (ValueError, OSError) as error:
            entry["error"] = str(error)
        results.append(entry)
    return results


def list_recipes(directory: Path) -> tuple[list[LoadedRecipe], list[dict[str, Any]]]:
    """The valid recipes, and a report for each file that is not one."""
    loaded: list[LoadedRecipe] = []
    invalid: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for path in recipe_files(directory):
        try:
            item = load_recipe(path)
            if item.recipe.id in seen:
                raise RecipeError(f"Recipe ID already used by {seen[item.recipe.id]}")
        except (ValueError, OSError) as error:
            invalid.append({"path": str(path), "error": str(error)})
            continue
        seen[item.recipe.id] = path
        loaded.append(item)
    return loaded, invalid


def get_recipe(identifier: str, directory: Path) -> LoadedRecipe:
    matches = [path for path in recipe_files(directory) if path.stem == identifier]
    if len(matches) != 1:
        raise ValueError(
            f"Unknown recipe: {identifier}" if not matches else f"Duplicate recipe: {identifier}"
        )
    if matches[0].suffix != ".yaml":
        raise RecipeError(f"Recipe files end in .yaml: rename {matches[0].name}")
    return load_recipe(matches[0])


def summary(loaded: LoadedRecipe, directory: Path) -> dict[str, Any]:
    recipe = loaded.recipe
    return {
        "id": recipe.id,
        "title": recipe.title,
        "kind": recipe.kind,
        "scope": recipe.scope,
        "rewirebench_protocol": recipe.rewirebench_protocol,
        "backends": recipe.backends.names(),
        "path": str(PurePosixPath("recipes", *loaded.path.relative_to(directory).parts)),
        "sha256": loaded.sha256,
    }


def json_schema() -> dict[str, Any]:
    schema = Recipe.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "rewirebench run recipe"
    return schema
