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
    containerfile = (ROOT / "containers/Containerfile").read_text()
    assert "x86-64" in containerfile and "linux/arm64" in containerfile
    for place in ("containers/Containerfile", ".github/workflows/sdk.yml", "sdk-release.yml",
                  "containers/esm.lock.txt", "dnabert2.lock.txt", "docs/hpc.md",
                  "workflows/benchmark/nextflow.config"):
        assert f"`{place}`" in PAGE, place
    for lock in (ROOT / "containers").glob("*.lock.txt"):
        assert f"{lock.name}`" in PAGE, lock.name


NO_ARM64 = ("No `linux/arm64` image exists, so no recipe in `recipes/` names the `spark` "
            "backend")


def spark_recipes(recipes_dir: Path) -> list[str]:
    """Recipes that name the spark backend or pin a linux/arm64 image."""
    import yaml

    found = []
    for path in sorted(recipes_dir.rglob("*.yaml")):
        recipe = yaml.safe_load(path.read_text())
        arm64 = recipe["environment"]["images"]["linux/arm64"]
        if "spark" in recipe["backends"] or "missing" not in arm64:
            found.append(path.name)
    return found


def test_the_page_says_whether_a_recipe_names_spark():
    """The page says no recipe names the spark backend. It must say so exactly while that
    is true, and stop saying so when it is not."""
    recipes = sorted((ROOT / "recipes").rglob("*.yaml"))
    assert len(recipes) >= 2
    if spark_recipes(ROOT / "recipes"):
        assert NO_ARM64 not in TEXT, spark_recipes(ROOT / "recipes")
    else:
        assert NO_ARM64 in TEXT


def test_a_recipe_that_names_spark_is_found(tmp_path: Path):
    """The check above reads the real recipes. This shows it tells a recipe with neither a
    spark backend nor a linux/arm64 image from one with either."""
    import yaml

    source = ROOT / "recipes/mfass/mfass-v2-kmer-baseline.yaml"
    recipe = yaml.safe_load(source.read_text())
    missing = {"missing": {"reason": "unavailable", "note": "No image."}}
    recipe["environment"]["images"]["linux/arm64"] = missing
    recipe["backends"].pop("spark", None)
    (tmp_path / "plain.yaml").write_text(yaml.safe_dump(recipe))
    assert spark_recipes(tmp_path) == []
    recipe["backends"]["spark"] = {"gpus": True}
    (tmp_path / "spark.yaml").write_text(yaml.safe_dump(recipe))
    recipe["backends"].pop("spark")
    recipe["environment"]["images"]["linux/arm64"] = {
        "reference": "ghcr.io/rewire-bio/rewirebench@sha256:" + "a" * 64}
    (tmp_path / "image.yaml").write_text(yaml.safe_dump(recipe))
    assert spark_recipes(tmp_path) == ["image.yaml", "spark.yaml"]


def test_the_page_states_the_publication_rule_and_that_nothing_has_run():
    assert "No run has been made on a Spark" in TEXT
    assert "Until both legs exist for a model, its Spark numbers are not published" in TEXT
    assert "No launch is built" in TEXT


def test_the_page_names_the_registry_the_workflow_pushes_to():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/image.yml").read_text())
    image = workflow["env"]["IMAGE"]
    assert image == "ghcr.io/rewire-bio/rewirebench"
    readme = (ROOT / "README.md").read_text()
    for page in (PAGE, readme):
        assert f"{image}@sha256:" in page
    assert "`.github/workflows/image.yml`" in PAGE and "scripts/set_recipe_image.py" in PAGE
    assert "Change visibility" in PAGE and "No GPU image" in TEXT
