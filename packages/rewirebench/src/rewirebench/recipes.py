"""Run recipes: validated YAML files under ``recipes/`` that the runner executes.

A recipe pins what a run needs (inputs, environment, weights) and states how to tell
that it worked (expected output). It carries no status. The evidence store holds a
pointer to the file (repository, commit, path, SHA-256) and the verified status, so
promoting a recipe never changes the bytes a receipt cites.

Field names follow ADR 0005 of rewire-benchmark-data where the two overlap; see
``docs/adr-c-recipes-and-receipts.md``.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SCHEMA_VERSION = "1.0"
PLATFORMS = ("linux/amd64", "linux/arm64")
REQUIREMENT_BLOCKS = ("weights_pin", "environment", "hardware_floor", "expected_output")
PLACEHOLDERS = ("inputs", "work", "output")
RECIPES_DIR_VARIABLE = "REWIREBENCH_RECIPES_DIR"
SCHEMA_FILENAME = "recipe.schema.json"

Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
Revision = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{40}$")]
ImageDigest = Annotated[str, StringConstraints(pattern=r"^sha256:[a-f0-9]{64}$")]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$")]
Text = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
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


def relative_path(value: str) -> str:
    """Reject absolute paths and parent traversal in a path written in a recipe."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Not a relative path inside the run folder: {value!r}")
    return str(path)


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
    path: Text
    sha256: Sha256


class Environment(Strict):
    """ADR 0005 ``environment``: runtime, Python and one image digest per platform."""

    runtime: Literal["python", "docker", "podman", "apptainer"]
    python: Text | None = None
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


class WeightsPin(Strict):
    """ADR 0005 ``weights_pin``: where the weights are and which bytes they are."""

    host: Text
    repository: Text
    revision: Revision | None = None
    files: list[WeightsFile] = Field(min_length=1)
    licence: Text
    gated: bool
    resolved_on: datetime.date

    @model_validator(mode="after")
    def _revision_on_git_hosts(self) -> WeightsPin:
        if self.host.lower() in {"huggingface", "hugging face", "github"} and not self.revision:
            raise ValueError("A Hugging Face or GitHub weights pin needs a 40-hex revision")
        return self


class RecipeInput(Strict):
    """One input file, pinned by SHA-256, from a URL or from this repository."""

    name: Text
    sha256: Sha256
    url: Text | None = None
    repo_path: Text | None = None
    bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _one_source(self) -> RecipeInput:
        if (self.url is None) == (self.repo_path is None):
            raise ValueError("An input has exactly one of url or repo_path")
        if self.url is not None and not self.url.startswith("https://"):
            raise ValueError("Input URLs must use https")
        if self.repo_path is not None:
            relative_path(self.repo_path)
        if relative_path(self.name) != self.name or "/" in self.name:
            raise ValueError("An input name is a plain file name")
        return self


class Accelerator(Strict):
    kind: Literal["none", "gpu"]
    model: Text | None = None
    count: int | None = Field(default=None, ge=1)
    memory_gb: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _cpu_has_no_gpu_fields(self) -> Accelerator:
        if self.kind == "none" and (self.model or self.count or self.memory_gb):
            raise ValueError("A recipe with no accelerator cannot name a GPU model or memory")
        return self


class HardwareFloor(Strict):
    """ADR 0005 ``hardware_floor``. Every figure shares the stated basis."""

    accelerator: Accelerator
    host_memory_gb: float | None = Field(default=None, gt=0)
    disk_gb: float | None = Field(default=None, gt=0)
    wall_clock_seconds: float | None = Field(default=None, gt=0)
    basis: Literal["source_stated", "measured_by_rewire"]
    note: Text | None = None


class Command(Strict):
    """One step as an argv list. No shell is involved, so no quoting rules apply."""

    name: Identifier
    argv: list[Text] = Field(min_length=1)

    @model_validator(mode="after")
    def _known_placeholders(self) -> Command:
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
    value: float
    tolerance: float = Field(ge=0)
    unit: Text | None = None

    @model_validator(mode="after")
    def _relative_file(self) -> ExpectedValue:
        relative_path(self.file)
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
    amount_per_hour: float = Field(gt=0)
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
    rate_source: Text
    rate_date: datetime.date


class Estimate(Strict):
    gpu_hours: float | None = Field(default=None, ge=0)
    wall_clock_seconds: float | None = Field(default=None, gt=0)
    price_basis: PriceBasis | None = None
    note: Text | None = None


class LocalBackend(Strict):
    pass


class SparkBackend(Strict):
    gpus: bool = True


class RunpodBackend(Strict):
    gpu_types: list[Text] = Field(min_length=1)
    cloud: Literal["SECURE", "COMMUNITY"] = "SECURE"
    disk_gb: int = Field(gt=0)
    max_hours: float = Field(gt=0)


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
    protocol_id: Text | None = None
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
        if self.purpose == "harness_run" and not self.protocol_id:
            raise ValueError("A harness_run recipe names its rewirebench protocol_id")
        for label, names in (
            ("input", [item.name for item in self.inputs]),
            ("command", [command.name for command in self.commands]),
            ("output", self.outputs),
        ):
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate {label} names")
        for path in self.outputs:
            relative_path(path)
        if self.backends.runpod is not None:
            image = self.environment.images["linux/amd64"] if self.environment else None
            if image is None or image.digest is None:
                raise ValueError("The runpod backend needs a linux/amd64 image digest")
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
        if key in seen:
            raise ValueError(f"Duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def parse_recipe(text: str | bytes) -> Recipe:
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("A recipe is a YAML mapping")  # noqa: TRY004 - CLI reports ValueError
    recipe = Recipe.model_validate(data)
    if recipe.protocol_id is not None and recipe.purpose == "harness_run":
        from rewirebench import sdk

        if recipe.protocol_id not in sdk.PROTOCOLS:
            raise ValueError(f"Unknown rewirebench protocol: {recipe.protocol_id}")
    return recipe


def load_recipe(path: str | Path) -> LoadedRecipe:
    path = Path(path)
    content = path.read_bytes()
    recipe = parse_recipe(content)
    if path.stem != recipe.id or path.suffix != ".yaml":
        raise ValueError(f"Recipe {recipe.id} must be in a file named {recipe.id}.yaml")
    return LoadedRecipe(recipe=recipe, path=path, sha256=hashlib.sha256(content).hexdigest())


def find_recipes_dir(explicit: str | Path | None = None) -> Path | None:
    """Locate ``recipes/``: the argument, the environment, the working tree, the checkout.

    The wheel does not ship recipes, so this returns ``None`` outside a checkout
    unless the caller or ``REWIREBENCH_RECIPES_DIR`` names a directory.
    """
    named = explicit or os.environ.get(RECIPES_DIR_VARIABLE)
    if named:
        path = Path(named)
        if not path.is_dir():
            raise ValueError(f"Recipes directory not found: {path}")
        return path.resolve()
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / "recipes" / SCHEMA_FILENAME).is_file():
            return folder / "recipes"
    parents = Path(__file__).resolve().parents
    if len(parents) > 4 and (parents[4] / "recipes" / SCHEMA_FILENAME).is_file():
        return parents[4] / "recipes"
    return None


def recipe_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.yaml"))


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


def list_recipes(directory: Path) -> list[LoadedRecipe]:
    loaded = [load_recipe(path) for path in recipe_files(directory)]
    identifiers = [item.recipe.id for item in loaded]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate recipe IDs")
    return loaded


def get_recipe(identifier: str, directory: Path) -> LoadedRecipe:
    matches = [path for path in recipe_files(directory) if path.stem == identifier]
    if len(matches) != 1:
        raise ValueError(
            f"Unknown recipe: {identifier}" if not matches else f"Duplicate recipe: {identifier}"
        )
    return load_recipe(matches[0])


def summary(loaded: LoadedRecipe, directory: Path) -> dict[str, Any]:
    recipe = loaded.recipe
    return {
        "id": recipe.id,
        "title": recipe.title,
        "kind": recipe.kind,
        "scope": recipe.scope,
        "protocol_id": recipe.protocol_id,
        "backends": recipe.backends.names(),
        "path": str(PurePosixPath("recipes", *loaded.path.relative_to(directory).parts)),
        "sha256": loaded.sha256,
    }


def json_schema() -> dict[str, Any]:
    schema = Recipe.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "rewirebench run recipe"
    return schema
