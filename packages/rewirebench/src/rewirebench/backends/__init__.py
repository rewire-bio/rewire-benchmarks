"""Where a recipe runs: this machine, an owned arm64 GPU host, or a paid pod.

Every backend answers ``plan`` without starting anything: what the run needs, how long
the recipe says it takes, what it costs and the exact command. ``local`` runs here.
``spark`` runs the recipe's linux/arm64 image over SSH and Docker and prints its steps
unless told to execute. ``runpod`` only plans and quotes; its launch refuses.
"""

from __future__ import annotations

from typing import Any

from rewirebench.backends.base import (
    Backend,
    Confirmation,
    Plan,
    PlanStep,
    Quote,
    owned_hardware_quote,
    plan_digest,
)
from rewirebench.backends.local import LocalBackend
from rewirebench.backends.runpod import LaunchRefused, RunpodBackend
from rewirebench.backends.spark import SparkBackend, SparkConfig

BACKENDS = ("local", "spark", "runpod")
PAID_BACKENDS = ("runpod",)

__all__ = [
    "BACKENDS",
    "PAID_BACKENDS",
    "Backend",
    "Confirmation",
    "LaunchRefused",
    "LocalBackend",
    "Plan",
    "PlanStep",
    "Quote",
    "RunpodBackend",
    "SparkBackend",
    "SparkConfig",
    "get_backend",
    "owned_hardware_quote",
    "plan_digest",
]


def get_backend(name: str, *, spark: SparkConfig | None = None, transport: Any = None) -> Any:
    """The backend of that name. ``nextflow`` is a recipe field with no backend here yet."""
    if name == "local":
        return LocalBackend()
    if name == "spark":
        return SparkBackend(spark or SparkConfig.from_environment(required=False),
                            transport=transport)
    if name == "runpod":
        return RunpodBackend()
    raise ValueError(f"No {name} backend. Choose one of {', '.join(BACKENDS)}")
