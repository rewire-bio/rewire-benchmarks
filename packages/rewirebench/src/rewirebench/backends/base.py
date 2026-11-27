"""The plan every backend returns, and the interface a backend implements."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from rewirebench.receipts import OWNED_HARDWARE, StartedBy
from rewirebench.recipes import LoadedRecipe, Recipe, Sha256, Strict, Text

PLAN_SCHEMA_VERSION = "1.0"
OUTPUT_PLACEHOLDER = "<output folder>"
BackendName = Literal["local", "spark", "runpod"]


class Quote(Strict):
    """What a run costs. Owned hardware has no price, and says so with a reason.

    Amounts are decimal strings so that the figure a person confirms is compared exactly.
    ``max_amount`` is the hourly rate per accelerator, times ``accelerator_count``, times
    the longest the pod may live; ``estimated_amount`` is the rate times the recipe's
    GPU-hour estimate, when it gives one. ``limit_enforced`` says whether anything stops
    the pod at ``max_hours``. Where it is false, ``max_amount`` is not a ceiling.
    """

    paid: bool
    reason: Text | None = None
    currency: Text | None = None
    max_amount: Text | None = None
    estimated_amount: Text | None = None
    amount_per_hour: Text | None = None
    max_hours: float | None = None
    provider: Text | None = None
    accelerator: Text | None = None
    accelerator_count: int | None = None
    limit_enforced: bool | None = None
    rate_source: Text | None = None
    rate_date: datetime.date | None = None
    note: Text | None = None

    @model_validator(mode="after")
    def _rules(self) -> Quote:
        money = (self.currency, self.max_amount, self.amount_per_hour, self.max_hours,
                 self.rate_source, self.rate_date)
        if self.paid and any(value is None for value in money):
            raise ValueError("A paid quote needs a currency, an amount, a rate and its source")
        if not self.paid and (self.reason is None or self.max_amount is not None):
            raise ValueError("An unpaid quote states its reason and no amount")
        return self


class Confirmation(Strict):
    """What a person must confirm before the run starts. Nothing for owned hardware."""

    required: bool
    amount: Text | None = None
    currency: Text | None = None
    how: Text | None = None


class PlanStep(Strict):
    """One command the backend would run, and on which machine."""

    name: Text
    where: Literal["local", "remote"]
    argv: list[str]
    note: Text | None = None


class PlanRecipe(Strict):
    id: Text
    path: Text
    sha256: Sha256


class Plan(Strict):
    """What a run needs and costs. Making a plan starts nothing and contacts nothing."""

    schema_version: Literal["1.0"]
    kind: Literal["rewire_run_plan"]
    recipe: PlanRecipe
    backend: BackendName
    platform: Text | None
    image: Text | None
    hardware: dict[str, Any] | None
    hardware_note: Text | None = None
    expected_wall_seconds: float | None
    expected_wall_seconds_basis: Text
    time_limit_seconds: float
    quote: Quote
    confirmation: Confirmation
    command: list[str]
    steps: list[PlanStep] = Field(default_factory=list)
    pod_specs: list[dict[str, Any]] = Field(default_factory=list)
    runnable: bool
    not_runnable_reason: Text | None = None
    notes: list[Text] = Field(default_factory=list)
    plan_sha256: Sha256

    @model_validator(mode="after")
    def _digest(self) -> Plan:
        if self.plan_sha256 != plan_digest(self.model_dump(mode="json")):
            raise ValueError("plan_sha256 does not match the plan")
        return self


def plan_digest(plan: dict[str, Any]) -> str:
    """SHA-256 of the plan's canonical JSON, without the digest field itself."""
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def make_plan(**fields: Any) -> Plan:
    """Build a plan and stamp its digest."""
    draft = Plan.model_construct(
        schema_version=PLAN_SCHEMA_VERSION, kind="rewire_run_plan", plan_sha256="0" * 64,
        **fields,
    )
    data = draft.model_dump(mode="json")
    data["plan_sha256"] = plan_digest(data)
    return Plan.model_validate(data)


PENDING_SHA256 = "0" * 64


def pending_inputs(recipe: Recipe) -> list[str]:
    """Inputs whose digest is the all-zero placeholder: not pinned yet, so no run passes."""
    return [item.name for item in recipe.inputs if item.sha256 == PENDING_SHA256]


NON_COMMERCIAL = "These weights are licensed for non-commercial use only: {licence}"


def weights_notices(recipe: Recipe) -> list[str]:
    """One plain line for each model a run step names whose weights forbid commercial use."""
    from rewirebench import cli, models

    lines = []
    for command in recipe.commands:
        if command.argv[1] != "run":
            continue
        spec = models.get_model(cli.parse_strict(command.argv[1:]).model)
        if spec.commercial_use is False:
            lines.append(NON_COMMERCIAL.format(licence=spec.weights_licence))
    return lines


def pending_reason(recipe: Recipe) -> str | None:
    pending = pending_inputs(recipe)
    if not pending:
        return None
    return (f"Recipe {recipe.id} has no SHA-256 pinned yet for {', '.join(pending)}: the "
            "all-zero digest marks it pending, so a run would stop at the input check.")


def owned_hardware_quote() -> Quote:
    return Quote(paid=False, reason=OWNED_HARDWARE)


def recipe_pointer(loaded: LoadedRecipe, recipes_dir: Path) -> PlanRecipe:
    relative = loaded.path.resolve().relative_to(recipes_dir.resolve())
    return PlanRecipe(id=loaded.recipe.id, sha256=loaded.sha256,
                      path=str(PurePosixPath("recipes", *relative.parts)))


def hardware_needs(recipe: Recipe) -> tuple[dict[str, Any] | None, str | None]:
    """The recipe's hardware floor, or the reason it records none."""
    if recipe.hardware_floor is not None:
        return recipe.hardware_floor.model_dump(mode="json"), None
    missing = recipe.missing.get("hardware_floor")
    reason = f"hardware_floor is {missing.reason}" if missing else "no hardware_floor"
    return None, f"The recipe states no hardware floor ({reason})."


def expected_wall(recipe: Recipe) -> tuple[float | None, str]:
    """Expected wall time from the recipe. Null with a reason, never a guess."""
    estimate = recipe.estimate
    if estimate is not None and estimate.wall_clock_seconds is not None:
        return estimate.wall_clock_seconds, "estimate.wall_clock_seconds"
    floor = recipe.hardware_floor
    if floor is not None and floor.wall_clock_seconds is not None:
        return floor.wall_clock_seconds, f"hardware_floor.wall_clock_seconds ({floor.basis})"
    return None, "The recipe states no wall time."


def time_limit(recipe: Recipe) -> float:
    """The longest the recipe's commands may run: the sum of their timeouts."""
    return float(sum(command.timeout_seconds for command in recipe.commands))


def run_arguments(
    recipe_id: str, *, recipes_dir: str, output: str, backend: str,
    started_by: StartedBy, inputs: str | None = None, kg_release_id: str | None = None,
) -> list[str]:
    """The ``rewirebench recipe run`` argv, shared by the plan and by whoever runs it."""
    argv = ["rewirebench", "recipe", "run", recipe_id, "--recipes-dir", recipes_dir,
            "--output", output, "--backend", backend, "--started-by", started_by]
    if inputs is not None:
        argv += ["--inputs", inputs]
    if kg_release_id is not None:
        argv += ["--kg-release-id", kg_release_id]
    return argv


class Backend(Protocol):
    """A place a recipe can run."""

    name: str

    def plan(
        self, loaded: LoadedRecipe, *, recipes_dir: Path, output: str | Path | None = None,
        inputs: str | Path | None = None, started_by: StartedBy = "human",
        kg_release_id: str | None = None,
    ) -> Plan: ...
