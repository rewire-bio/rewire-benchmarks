"""scripts/set_recipe_image.py writes a published image digest into a recipe and nothing else."""
from __future__ import annotations

import difflib
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest
import yaml
from rewirebench.recipes import load_recipe

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("set_recipe_image", ROOT / "scripts/set_recipe_image.py")
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)

RECIPE = "mfass-v2-kmer-baseline"
AMD64 = f"{script.IMAGE}@sha256:{'a' * 64}"
ARM64 = f"{script.IMAGE}@sha256:{'b' * 64}"


def without_images(text: str) -> str:
    """A recipe as it was before any image existed: both platforms missing."""
    for platform in ("linux/amd64", "linux/arm64"):
        text = re.sub(
            rf"(    {platform}:\n)      reference: \S+\n",
            rf"\1      missing:\n        reason: unavailable\n        note: No {platform} image has been built.\n",
            text)
    assert "reference:" not in text
    return text


@pytest.fixture
def folders(tmp_path: Path) -> tuple[Path, Path]:
    recipes, copy = tmp_path / "recipes", tmp_path / "plugin/recipes"
    shutil.copytree(ROOT / "recipes", recipes)
    for path in recipes.rglob("*.yaml"):
        path.write_text(without_images(path.read_text()))
    shutil.copytree(recipes, copy)
    return recipes, copy


def run(folders: tuple[Path, Path], *arguments: str) -> int:
    recipes, copy = folders
    return script.main([RECIPE, "--recipes-dir", str(recipes), "--copy-dir", str(copy), *arguments])


def test_one_platform_changes_only_that_entry(folders):
    recipes, copy = folders
    path = recipes / "mfass" / f"{RECIPE}.yaml"
    before = path.read_text()
    assert run(folders, "--platform", "linux/arm64", "--reference", ARM64) == 0
    after = path.read_text()
    diff = [line for line in difflib.unified_diff(before.splitlines(), after.splitlines(), n=0, lineterm="")
            if line[:1] in "+-" and line[:3] not in ("+++", "---")]
    assert diff == ["-      missing:", "-        reason: unavailable",
                    "-        note: No linux/arm64 image has been built.",
                    f"+      reference: {ARM64}"]
    loaded = load_recipe(path)
    assert loaded.recipe.image_digests() == {"linux/amd64": None, "linux/arm64": "sha256:" + "b" * 64}
    assert (copy / "mfass" / f"{RECIPE}.yaml").read_bytes() == path.read_bytes()
    # Comments survive: the file is edited as text, not dumped from YAML.
    assert "# The published value has three decimals" in after


def test_both_platforms_from_the_workflow_digest_file(folders, tmp_path, capsys):
    recipes, _ = folders
    digests = tmp_path / "image-digests.json"
    digests.write_text(json.dumps({"core": {
        "image": script.IMAGE, "commit": "c" * 40, "manifest": "sha256:" + "d" * 64,
        "platforms": {"linux/amd64": "sha256:" + "a" * 64, "linux/arm64": "sha256:" + "b" * 64}}}))
    assert run(folders, "--digests", str(digests), "--environment", "core") == 0
    images = yaml.safe_load((recipes / "mfass" / f"{RECIPE}.yaml").read_text())["environment"]["images"]
    assert images == {"linux/amd64": {"reference": AMD64}, "linux/arm64": {"reference": ARM64}}
    assert "written before an image existed" not in capsys.readouterr().out
    path = recipes / "mfass" / f"{RECIPE}.yaml"
    path.write_text(path.read_text() + "# no container image is published\n")
    shutil.copyfile(path, tmp_path / "plugin/recipes/mfass" / f"{RECIPE}.yaml")
    assert run(folders, "--digests", str(digests), "--environment", "core") == 0
    # A line written before an image existed is pointed at, for a person to edit.
    assert "written before an image existed" in capsys.readouterr().out
    assert run(folders, "--digests", str(digests), "--environment", "sequence") == 1


def test_a_second_run_replaces_the_reference_and_is_idempotent(folders):
    recipes, _ = folders
    path = recipes / "mfass" / f"{RECIPE}.yaml"
    assert run(folders, "--platform", "linux/amd64", "--reference", ARM64) == 0
    assert run(folders, "--platform", "linux/amd64", "--reference", AMD64) == 0
    once = path.read_text()
    assert run(folders, "--platform", "linux/amd64", "--reference", AMD64) == 0
    assert path.read_text() == once and once.count("reference:") == 1 and AMD64 in once


@pytest.mark.parametrize("reference", [
    f"{script.IMAGE}:core",
    f"{script.IMAGE}@sha256:{'a' * 63}",
    f"{script.IMAGE}@sha256:{'A' * 64}",
    f"docker.io/someone/rewirebench@sha256:{'a' * 64}",
])
def test_a_tag_a_short_digest_or_another_registry_is_refused(folders, reference):
    recipes, _ = folders
    path = recipes / "mfass" / f"{RECIPE}.yaml"
    before = path.read_bytes()
    assert run(folders, "--platform", "linux/amd64", "--reference", reference) == 1
    assert path.read_bytes() == before


def test_nothing_is_written_when_the_copy_has_drifted_or_the_recipe_is_unknown(folders):
    recipes, copy = folders
    twin = copy / "mfass" / f"{RECIPE}.yaml"
    twin.write_text(twin.read_text() + "# drift\n")
    before = (recipes / "mfass" / f"{RECIPE}.yaml").read_bytes()
    assert run(folders, "--platform", "linux/amd64", "--reference", AMD64) == 1
    assert (recipes / "mfass" / f"{RECIPE}.yaml").read_bytes() == before
    assert script.main(["no-such-recipe", "--recipes-dir", str(recipes), "--copy-dir", str(copy),
                        "--platform", "linux/amd64", "--reference", AMD64]) == 1


def test_every_recipe_on_main_can_take_both_digests():
    for path in sorted((ROOT / "recipes").rglob("*.yaml")):
        for start in (path.read_text(), without_images(path.read_text())):
            text = script.set_image(script.set_image(start, "linux/amd64", AMD64), "linux/arm64", ARM64)
            images = yaml.safe_load(text)["environment"]["images"]
            assert images == {"linux/amd64": {"reference": AMD64}, "linux/arm64": {"reference": ARM64}}


def test_the_script_names_the_image_the_workflow_pushes():
    workflow = yaml.safe_load((ROOT / ".github/workflows/image.yml").read_text())
    assert workflow["env"]["IMAGE"] == script.IMAGE
