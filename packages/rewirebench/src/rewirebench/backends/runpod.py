"""The RunPod backend: a pod spec and a price quote. It launches nothing.

Ported from ``cloud/runpod/launch.sh`` in rewire-bio/evo2-cloud-operations: the pod
request body per GPU type in fallback order (REST v2, Secure Cloud, one SSH port, a
lifetime limit in the pod's environment) and the instance record read from a created
pod, which is where that script learns the hourly rate. That script asks no price
before it creates a pod, so the quote here comes from the recipe's dated
``estimate.price_basis``, not from a live call.

This module makes no network call and reads no API key. ``launch`` exists so that the
confirmation rule has one place to live: it refuses without a confirmed amount, refuses
a different amount, and then refuses anyway because launching is not built.
"""

from __future__ import annotations

import os
from decimal import ROUND_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from rewirebench.backends import base
from rewirebench.receipts import StartedBy
from rewirebench.recipes import LoadedRecipe, Recipe

API = "https://api.runpod.io/v2"
PLATFORM = "linux/amd64"
SSH_PORT = "22/tcp"
PUBLIC_KEY_PLACEHOLDER = "<public key generated at launch>"
CENT = Decimal("0.01")
NOT_BUILT = (
    "RunPod launch is not built in this version of rewirebench. The plan and the pod "
    "spec are for review only. Do not create the pod with another tool: the spec has no "
    "start script, so nothing would delete the pod at the time limit and the quote would "
    "not bound its cost."
)


class LaunchRefused(ValueError):
    """A paid launch that did not go ahead, with the reason."""


def _decimal(value: Any) -> Decimal:
    """An exact amount from a number or text. A float goes through its shortest repr."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise LaunchRefused(f"Not an amount: {value!r}")
    try:
        amount = Decimal(str(value).strip())
    except InvalidOperation:
        raise LaunchRefused(f"Not an amount: {value!r}") from None
    if not amount.is_finite():
        raise LaunchRefused(f"Not an amount: {value!r}")
    return amount


def _money(amount: Decimal) -> str:
    """Rounded up to the cent, so a quote never understates."""
    return str(amount.quantize(CENT, rounding=ROUND_UP))


def gpu_count(recipe: Recipe) -> int:
    """How many accelerators the pod asks for: the hardware floor's count, at least one."""
    floor = recipe.hardware_floor
    return floor.accelerator.count if floor and floor.accelerator.count else 1


def quote(recipe: Recipe) -> base.Quote:
    """GPU time at the dated hourly rate, for every GPU asked for, up to the time limit.

    The rate is per accelerator and prices the first GPU type only, which the recipe
    schema ties to the price basis. ``gpu_hours`` in the estimate already counts every
    GPU. Nothing deletes the pod at the limit yet, so the amount is not a ceiling.
    """
    pod = recipe.backends.runpod
    basis = recipe.estimate.price_basis if recipe.estimate else None
    if pod is None or basis is None:
        raise ValueError(f"Recipe {recipe.id} has no runpod backend with a price basis")
    if basis.accelerator != pod.gpu_types[0]:
        raise ValueError(f"Recipe {recipe.id}: the price basis is not for {pod.gpu_types[0]}")
    rate = Decimal(str(basis.amount_per_hour))
    count = gpu_count(recipe)
    gpu_hours = recipe.estimate.gpu_hours if recipe.estimate else None
    return base.Quote(
        paid=True,
        currency=basis.currency,
        max_amount=_money(rate * count * Decimal(str(pod.max_hours))),
        estimated_amount=_money(rate * Decimal(str(gpu_hours))) if gpu_hours else None,
        amount_per_hour=str(rate),
        max_hours=pod.max_hours,
        provider=basis.provider,
        accelerator=basis.accelerator,
        accelerator_count=count,
        limit_enforced=False,
        rate_source=basis.rate_source,
        rate_date=basis.rate_date,
        note="The rate is the recipe's dated observation per GPU, not a live price. The "
             "amount is that rate for every GPU in the pod for max_hours. It leaves out "
             "storage and network charges. It is not a ceiling yet: no start script "
             "deletes the pod at the time limit, so a pod left running costs more.",
    )


def pod_body(
    *, name: str, image: str, gpu_type: str, gpu_count: int, disk_gb: int,
    max_seconds: int, cloud: str = "SECURE", public_key: str = PUBLIC_KEY_PLACEHOLDER,
) -> dict[str, Any]:
    """The ``POST /pods`` body ``launch.sh`` sends, for one GPU type.

    ``launch.sh`` also sets ``cmd`` to its image's start script, which installs the key,
    starts sshd and deletes the pod after ``MAX_SECONDS``. No rewirebench image has such a
    script yet, so ``cmd`` is left out and the plan says so.
    """
    return {
        "name": name,
        "image": image,
        "gpu": {"id": gpu_type, "count": gpu_count},
        "cloud": cloud,
        "disk": disk_gb,
        "ports": [SSH_PORT],
        "env": {"PUBLIC_KEY": public_key, "MAX_SECONDS": str(max_seconds)},
    }


def pod_bodies(recipe: Recipe, *, name: str) -> list[dict[str, Any]]:
    """The body for the GPU type the quote prices: the first in the recipe's list.

    ``launch.sh`` tries every type in order. The recipe's later types have no dated rate,
    so no body is written for them; the plan names them as unpriced.
    """
    pod = recipe.backends.runpod
    environment = recipe.environment
    if pod is None or environment is None:
        raise ValueError(f"Recipe {recipe.id} does not name the runpod backend")
    image = environment.images[PLATFORM].reference
    if image is None:
        raise ValueError(f"Recipe {recipe.id} has no {PLATFORM} image digest")
    return [
        pod_body(name=name, image=image, gpu_type=pod.gpu_types[0],
                 gpu_count=gpu_count(recipe), disk_gb=pod.disk_gb,
                 max_seconds=int(pod.max_hours * 3600), cloud=pod.cloud)
    ]


def instance_record(pod: dict[str, Any], *, disk_gb: int) -> dict[str, Any]:
    """The instance facts ``launch.sh`` keeps from a created pod, hourly rate included."""
    gpu = pod.get("gpu") if isinstance(pod.get("gpu"), dict) else None
    return {
        "provider": "runpod",
        "instance_type": gpu.get("id") if gpu else None,
        "gpu": gpu,
        "region": pod.get("dataCenterId"),
        "zone": pod.get("dataCenterId"),
        "instance_id": pod.get("id"),
        "image": pod.get("image"),
        "purchase_option": "on-demand",
        "cost_per_hr": pod.get("cost"),
        "disk_gb": disk_gb,
        "host_cuda": pod.get("cudaVersion"),
    }


def check_confirmation(
    plan: base.Plan, *, confirmed_amount: Any, confirmed_currency: str | None
) -> None:
    """Refuse unless the caller passes exactly the plan's amount and currency.

    The amount is text or a number, never a boolean. This checks a value, not who
    supplied it: an agent can pass the figure it read in the plan. A real launch needs
    a confirmation that only a person can give (a terminal prompt, or MCP elicitation).
    """
    quoted = plan.quote
    if not quoted.paid or quoted.max_amount is None or quoted.currency is None:
        raise LaunchRefused("This plan has no price to confirm")
    if os.environ.get("CI"):
        raise LaunchRefused("A paid run is never started in CI or by a schedule")
    if confirmed_amount is None:
        raise LaunchRefused(
            "A paid run needs confirmed_amount: the amount a person stated after seeing "
            "the plan's quote. Show the person the quote and ask. Do not fill it in from "
            "the plan yourself."
        )
    if confirmed_currency is None:
        raise LaunchRefused("A paid run needs confirmed_currency, as the person stated it.")
    if (_decimal(confirmed_amount) != Decimal(quoted.max_amount)
            or confirmed_currency != quoted.currency):
        raise LaunchRefused(
            f"The confirmed amount ({confirmed_amount} {confirmed_currency}) is not the "
            "plan's amount. Nothing was started. Call plan_run and show the person its quote."
        )


class RunpodBackend:
    name = "runpod"

    def plan(
        self, loaded: LoadedRecipe, *, recipes_dir: Path, output: str | Path | None = None,
        inputs: str | Path | None = None, started_by: StartedBy = "human",
        kg_release_id: str | None = None,
    ) -> base.Plan:
        recipe = loaded.recipe
        if recipe.backends.runpod is None:
            raise ValueError(f"Recipe {recipe.id} does not name the runpod backend")
        if inputs is not None:
            raise ValueError("The runpod backend fetches the recipe's inputs in the pod")
        quoted = quote(recipe)
        hardware, hardware_note = base.hardware_needs(recipe)
        expected, basis = base.expected_wall(recipe)
        assert recipe.environment is not None
        fallbacks = list(recipe.backends.runpod.gpu_types[1:])
        note = (f"Fallback GPU types {fallbacks} have no dated rate in the recipe, so "
                "they are not priced and have no pod spec.")
        unpriced = [note] if fallbacks else []
        return base.make_plan(
            recipe=base.recipe_pointer(loaded, recipes_dir),
            backend="runpod",
            platform=PLATFORM,
            image=recipe.environment.images[PLATFORM].reference,
            hardware=hardware,
            hardware_note=hardware_note,
            expected_wall_seconds=expected,
            expected_wall_seconds_basis=basis,
            time_limit_seconds=recipe.backends.runpod.max_hours * 3600,
            quote=quoted,
            confirmation=base.Confirmation(
                required=True, amount=quoted.max_amount, currency=quoted.currency,
                how="Show a person the quote. Pass the amount and currency they state as "
                    "confirmed_amount and confirmed_currency, with this plan's plan_sha256.",
            ),
            command=base.run_arguments(
                recipe.id, recipes_dir="/recipes", output="/out/run", backend="runpod",
                started_by=started_by, kg_release_id=kg_release_id,
            ),
            pod_specs=pod_bodies(recipe, name=f"rewirebench-{recipe.id}"),
            runnable=False,
            not_runnable_reason=NOT_BUILT,
            notes=[
                *base.weights_notices(recipe),
                (f"The pod spec is the body for POST {API}/pods, for review only. Do not "
                 "send it with another tool."),
                *unpriced,
                ("No start script exists in a rewirebench image yet. Before any launch the "
                 "image needs one that installs the SSH key, starts sshd and deletes the "
                 "pod after MAX_SECONDS. Until then nothing bounds the pod's lifetime."),
                ("The command is what would run in the pod. No rewirebench release accepts "
                 "--backend runpod yet."),
            ],
        )

    def launch(
        self, plan: base.Plan, *, confirmed_amount: Any = None,
        confirmed_currency: str | None = None,
    ) -> None:
        """Always refuses. The confirmation is still checked first, in the order a real
        launch will need: no amount, a wrong amount, and only then anything else."""
        check_confirmation(plan, confirmed_amount=confirmed_amount,
                           confirmed_currency=confirmed_currency)
        raise LaunchRefused(NOT_BUILT)
