"""docs/spark.md must keep up with what it describes: the models, the pins and the locks."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

from rewirebench import models

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "docs/spark.md").read_text()
TEXT = " ".join(PAGE.split())


def test_every_registered_model_is_in_the_readiness_table():
    missing = [model for model in models.REGISTRY if f"`{model}`" not in PAGE]
    assert not missing, f"Add these models to the table in docs/spark.md: {missing}"


def test_the_pins_the_page_quotes_are_the_pins_in_the_package():
    project = tomllib.loads((ROOT / "packages/rewirebench/pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]
    for extra in ("esm", "dnabert2"):
        for requirement in extras[extra]:
            if requirement.startswith(("torch", "fair-esm", "transformers", "einops")):
                assert f"`{requirement}`" in PAGE, f"docs/spark.md does not quote {requirement}"
    generator = (ROOT / "containers/generate_cpu_locks.py").read_text()
    wheel = re.search(r"torch-[0-9.]+%2Bcpu-cp311-cp311-linux_x86_64\.whl", generator)
    assert wheel and wheel.group(0).replace("%2B", "+") in PAGE


def test_every_amd64_pin_in_the_repository_is_listed():
    assert "linux/amd64" in (ROOT / ".github/workflows/sdk.yml").read_text()
    assert "linux/amd64" in (ROOT / ".github/workflows/sdk-release.yml").read_text()
    assert "x86-64" in (ROOT / "containers/Containerfile").read_text()
    for place in ("containers/Containerfile", ".github/workflows/sdk.yml", "sdk-release.yml",
                  "containers/esm.lock.txt", "dnabert2.lock.txt", "docs/hpc.md",
                  "workflows/benchmark/nextflow.config"):
        assert f"`{place}`" in PAGE, place
    for lock in (ROOT / "containers").glob("*.lock.txt"):
        assert f"{lock.name}`" in PAGE, lock.name


def test_no_recipe_names_spark_until_an_arm64_image_exists():
    """The page says no recipe names the spark backend. Keep that true, or update the page."""
    import yaml

    for path in (ROOT / "recipes").rglob("*.yaml"):
        recipe = yaml.safe_load(path.read_text())
        arm64 = recipe["environment"]["images"]["linux/arm64"]
        if "spark" in recipe["backends"] or "digest" in arm64:
            assert "No `linux/arm64` image exists" not in TEXT, path.name


def test_the_page_states_the_publication_rule_and_that_nothing_has_run():
    assert "No run has been made on a Spark" in TEXT
    assert "Until both legs exist for a model, its Spark numbers are not published" in TEXT
    assert "No launch is built" in TEXT
