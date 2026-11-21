"""Recipes built for tests. They carry made-up image digests and prices, so they live
here and never under recipes/."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from rewirebench import recipes

ROOT = Path(__file__).resolve().parents[3]
SMOKE_RECIPE = ROOT / "recipes/flip2/flip2-synthetic-smoke.yaml"
SMOKE_INPUT = ROOT / "workflows/benchmark/examples/smoke.csv"
AMD64_DIGEST = "sha256:" + "a" * 64
ARM64_DIGEST = "sha256:" + "b" * 64
INPUT_URL = "https://data.invalid/smoke.csv"
PRICE_BASIS = {
    "provider": "runpod", "accelerator": "NVIDIA H100 80GB HBM3", "amount_per_hour": 3.99,
    "currency": "USD", "rate_source": "Made-up rate for tests", "rate_date": "2026-10-14",
}


def gpu_recipe_data() -> dict[str, Any]:
    """The synthetic FLIP2 recipe, rewritten to need a GPU and name spark and runpod."""
    data = yaml.safe_load(SMOKE_RECIPE.read_text())
    data["id"] = "gpu-fixture"
    data["environment"]["images"] = {
        "linux/amd64": {"reference": f"ghcr.io/rewire-bio/fixture@{AMD64_DIGEST}"},
        "linux/arm64": {"reference": f"ghcr.io/rewire-bio/fixture@{ARM64_DIGEST}"},
    }
    data["hardware_floor"] = {
        "accelerator": {"kind": "gpu", "model": "NVIDIA GB10", "count": 1, "memory_gb": 16.0},
        "host_memory_gb": 1.0, "basis": "source_stated",
    }
    data["inputs"] = [{"name": "smoke.csv", "url": INPUT_URL,
                       "sha256": recipes.sha256_file(SMOKE_INPUT)}]
    data["estimate"] = {"gpu_hours": 0.5, "wall_clock_seconds": 1800.0,
                        "price_basis": dict(PRICE_BASIS)}
    data["backends"] = {
        "spark": {},
        "runpod": {"gpu_types": ["NVIDIA H100 80GB HBM3", "NVIDIA L40S"], "disk_gb": 200,
                   "max_hours": 3},
    }
    return data


def model_recipe_data() -> dict[str, Any]:
    """The synthetic FLIP2 recipe with a `run --model` step in place of run-baselines."""
    data = yaml.safe_load(SMOKE_RECIPE.read_text())
    data["id"] = "model-step"
    data["commands"][1] = {"name": "run-model", "argv": [
        "rewirebench", "run", "--prepared", "{work}/prepared", "--output", "{output}/run",
        "--model", "train-mean"]}
    data["outputs"] = ["run/report.json", "run/predictions.json"]
    data["expected_output"]["values"] = [
        {"label": "Scored rows", "metric_id": "n", "file": "run/report.json",
         "key": ["metrics", "n"], "value": 3, "tolerance": 0}]
    return data


def write_recipes(root: Path, *documents: dict[str, Any]) -> Path:
    """A checkout-shaped folder: recipes/test/<id>.yaml and the fixture the recipes pin."""
    folder = root / "recipes/test"
    folder.mkdir(parents=True)
    target = root / "workflows/benchmark/examples"
    target.mkdir(parents=True)
    (target / "smoke.csv").write_bytes(SMOKE_INPUT.read_bytes())
    for document in documents:
        (folder / f"{document['id']}.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    return root / "recipes"

