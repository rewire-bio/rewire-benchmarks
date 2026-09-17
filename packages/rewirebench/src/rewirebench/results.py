"""Result schema.

The house rules in the repository README are enforced here. A result that omits
coverage against the original denominator, end-to-end timing, the independent-group
count, or the contamination statement for a pretrained model is refused rather than
written. A leaderboard is only as trustworthy as the weakest row in it.
"""
from __future__ import annotations

import json
import math
import pathlib
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class BenchmarkResult:
    """One method, one run, one split.

    `contamination` is required for any method with pretrained weights. State what
    was checked and what remains unknown. "unknown" is an acceptable value; omitting
    it is not, because a reader cannot tell the difference between unchecked and
    clean.
    """

    benchmark: str                 # e.g. "mfass-v1"
    method: str                    # e.g. "spliceai-1.3.1"
    family: str                    # "trivial baseline" | "specialist" | "pretrained encoder"
    description: str
    split: str                     # path or id of the split manifest
    metrics: dict[str, Any]
    coverage: dict[str, int]       # scored, unscored, denominator
    timing_seconds: dict[str, float]
    independent_groups: int
    config: dict[str, Any] = field(default_factory=dict)
    contamination: str | None = None
    pretrained: bool = False
    notes: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    created_utc: str = ""
    git_revision: str | None = None

    def validate(self) -> None:
        missing = [k for k in ("scored", "unscored", "denominator") if k not in self.coverage]
        if missing:
            raise ValueError(f"coverage must report {missing}")
        if any(type(self.coverage[k]) is not int or self.coverage[k] < 0
               for k in ("scored", "unscored", "denominator")):
            raise ValueError("coverage counts must be nonnegative integers")
        if self.coverage["scored"] + self.coverage["unscored"] != self.coverage["denominator"]:
            raise ValueError(
                "coverage does not reconcile: scored + unscored must equal the original "
                "denominator. A method that cannot score an input has a coverage problem, "
                "not a negative prediction."
            )
        if not self.timing_seconds:
            raise ValueError("timing_seconds is required: throughput is reported beside accuracy")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or v < 0 for v in self.timing_seconds.values()):
            raise ValueError("timing_seconds must contain finite nonnegative values")
        def check_metrics(value):
            if isinstance(value, dict):
                for child in value.values():
                    check_metrics(child)
            elif isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("metrics cannot contain nonfinite values")
        check_metrics(self.metrics)
        smoke = self.config.get("scope") == "smoke" and self.benchmark.endswith("-smoke")
        if type(self.independent_groups) is not int or self.independent_groups < 0:
            raise ValueError("independent_groups must be a nonnegative integer")
        if self.independent_groups < 2 and not smoke:
            raise ValueError("independent_groups must be at least 2 for any interval to mean anything")
        if self.pretrained and not self.contamination:
            raise ValueError(
                "contamination is required for a pretrained method. State what was checked "
                "and what is unknown; 'unknown' is acceptable, silence is not."
            )
        if self.pretrained and not self.config:
            raise ValueError(
                "config is required for a pretrained method: checkpoint revision, pooling, "
                "context in bases and tokens, precision and batch size"
            )

    def finalise(self) -> "BenchmarkResult":
        self.validate()
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat()
        if self.git_revision is None:
            self.git_revision = _git_revision()
        if not self.environment:
            self.environment = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor() or None,
            }
        return self


def write_result(result: BenchmarkResult, path: str | pathlib.Path) -> pathlib.Path:
    """Validate, stamp and write. Refuses to write an invalid result."""
    result = result.finalise()
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as fh:
        json.dump(asdict(result), fh, indent=2, sort_keys=False, allow_nan=False)
    return out
