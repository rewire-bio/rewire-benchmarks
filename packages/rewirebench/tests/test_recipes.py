"""Recipes are validated files: a misspelt, unpinned or status-carrying recipe is refused."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from rewirebench import cli, recipes, sdk
from rewirebench.protocols import flip2

ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "recipes"
DIGEST = "sha256:" + "a" * 64
BASIS = {"provider": "runpod", "accelerator": "NVIDIA H100 80GB HBM3", "amount_per_hour": 3.99,
         "currency": "USD", "rate_source": "instance.json", "rate_date": "2026-10-09"}


def document(name: str = "flip2/flip2-synthetic-smoke.yaml") -> dict[str, Any]:
    return yaml.safe_load((RECIPES / name).read_text())


def parse(data: dict[str, Any]) -> recipes.Recipe:
    return recipes.parse_recipe(yaml.safe_dump(data))


def changed(**fields: Any) -> dict[str, Any]:
    data = copy.deepcopy(document())
    data.update(fields)
    return data


def without_images(data: dict[str, Any]) -> dict[str, Any]:
    """The recipe with no image for either platform, as before any image was published."""
    data["environment"]["images"] = {
        platform: {"missing": {"reason": "unavailable", "note": "No image."}}
        for platform in ("linux/amd64", "linux/arm64")}
    return data


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[Any, int]:
    code = 0
    try:
        cli.main(list(argv))
    except SystemExit as stop:
        code = int(stop.code or 0)
    out = capsys.readouterr().out
    return (json.loads(out) if out.strip() else None), code


def test_every_committed_recipe_validates_and_sits_at_its_own_name():
    files = recipes.recipe_files(RECIPES)
    assert files, "recipes/ holds no recipe"
    for path in files:
        loaded = recipes.load_recipe(path)
        assert path.relative_to(RECIPES).parts == (path.parent.name, f"{loaded.recipe.id}.yaml")
        assert loaded.sha256 == recipes.sha256_file(path)
    assert "mfass-v2-kmer-baseline" in {path.stem for path in files}


def test_committed_schema_matches_the_models():
    committed = json.loads((RECIPES / recipes.SCHEMA_FILENAME).read_text())
    assert committed == recipes.json_schema(), (
        "Regenerate with: rewirebench recipe schema > recipes/recipe.schema.json"
    )


def test_mfass_recipe_pins_the_published_inputs_and_values():
    recipe = recipes.load_recipe(RECIPES / "mfass/mfass-v2-kmer-baseline.yaml").recipe
    assert recipe.rewirebench_protocol == "mfass-v2" and recipe.scope == "full_protocol"
    assert {item.name: item.sha256 for item in recipe.inputs} == {
        "snv_data_clean.txt": "a637ca0e307e66ff48811ec7efa22b9ce453bc7883b04f0cacb867f7283132d8",
        "snv_func_annot.txt": "71a857fe647c4e68acbb41ca61e959c47e1176de89b1442bd6ca1772aa60d5a1",
    }
    assert all("9a8e4f27106be52aeb11acad27f95f5cded663a8" in item.url for item in recipe.inputs)
    expected = {value.metric_id: value.value for value in recipe.expected_output.values}
    assert expected == {"n": 8324, "positives": 315, "precision_at_capacity": 0.61,
                        "average_precision_sklearn": 0.286, "auroc": 0.778}
    # Both platforms name the published core image by digest.
    assert set(recipe.image_digests()) == {"linux/amd64", "linux/arm64"}
    for image in recipe.environment.images.values():
        assert image.reference.startswith("ghcr.io/rewire-bio/rewirebench@sha256:")
    assert len(set(recipe.image_digests().values())) == 2
    assert recipe.backends.names() == ["local"]


@pytest.mark.parametrize("key", ["status", "verified", "unknown_field"])
def test_status_and_unknown_keys_are_rejected(key: str):
    with pytest.raises(recipes.RecipeError, match="Extra inputs are not permitted"):
        parse(changed(**{key: "executed"}))


def test_a_block_is_present_or_named_missing_never_both_or_neither():
    data = changed()
    del data["missing"]["weights_pin"]
    with pytest.raises(recipes.RecipeError, match="weights_pin must be either present"):
        parse(data)
    data = changed()
    data["missing"]["hardware_floor"] = {"reason": "unreported"}
    with pytest.raises(recipes.RecipeError, match="hardware_floor must be either present"):
        parse(data)
    data = changed()
    del data["expected_output"]
    data["missing"]["expected_output"] = {"reason": "unreported"}
    assert parse(data).expected_output is None
    with pytest.raises(recipes.RecipeError):
        parse(changed(missing={"weights_pin": {"reason": "not measured yet"}}))


def test_images_are_by_digest_for_both_platforms():
    data = without_images(changed())
    data["environment"]["images"]["linux/amd64"] = {"reference": "ghcr.io/rewire-bio/x:latest"}
    with pytest.raises(recipes.RecipeError, match="by digest"):
        parse(data)
    data["environment"]["images"]["linux/amd64"] = {"reference": f"ghcr.io/rewire-bio/x@{DIGEST}"}
    assert parse(data).image_digests() == {"linux/amd64": DIGEST, "linux/arm64": None}
    del data["environment"]["images"]["linux/arm64"]
    with pytest.raises(recipes.RecipeError, match="exactly these platforms"):
        parse(data)
    data = without_images(changed())
    data["environment"]["images"]["linux/arm64"] = {}
    with pytest.raises(recipes.RecipeError, match="exactly one of reference or missing"):
        parse(data)


def test_gpu_backends_need_the_image_for_their_platform():
    runpod = {"gpu_types": ["NVIDIA H100 80GB HBM3"], "disk_gb": 200, "max_hours": 3}
    data = without_images(changed(backends={"runpod": runpod}))
    with pytest.raises(recipes.RecipeError, match="linux/amd64 image digest"):
        parse(data)
    data["environment"]["images"]["linux/amd64"] = {"reference": f"ghcr.io/rewire-bio/x@{DIGEST}"}
    # A paid backend also needs a dated hourly rate, or there is no amount to quote.
    with pytest.raises(recipes.RecipeError, match="runpod backend needs estimate.price_basis"):
        parse(data)
    data["estimate"] = {"gpu_hours": 1}
    with pytest.raises(recipes.RecipeError, match="runpod backend needs estimate.price_basis"):
        parse(data)
    data["estimate"] = {"gpu_hours": 1, "price_basis": BASIS}
    assert parse(data).backends.names() == ["runpod"]
    data["backends"]["runpod"]["max_hours"] = 1e6
    with pytest.raises(recipes.RecipeError, match="backends.runpod.max_hours"):
        parse(data)
    data["backends"] = {"spark": {}}
    with pytest.raises(recipes.RecipeError, match="linux/arm64 image digest"):
        parse(data)
    with pytest.raises(recipes.RecipeError, match="at least one backend"):
        parse(changed(backends={}))


def test_the_local_backend_is_cpu_only():
    data = changed()
    data["hardware_floor"]["accelerator"] = {"kind": "gpu", "model": "NVIDIA GB10", "count": 1}
    with pytest.raises(recipes.RecipeError, match="local backend is CPU only"):
        parse(data)
    data["environment"]["images"]["linux/arm64"] = {"reference": f"ghcr.io/rewire-bio/x@{DIGEST}"}
    data["backends"] = {"spark": {}}
    assert parse(data).hardware_floor.accelerator.kind == "gpu"


@pytest.mark.parametrize("argv", [
    ["sh", "-c", "runpodctl create pod"],
    ["runpodctl", "create", "pod"],
    ["python", "-c", "print(1)"],
    ["rewirebench"],
    ["rewirebench", "submit", "--bundle", "x"],
    # `run --adapter` calls any installed callable, so it would run another program.
    ["rewirebench", "run", "--prepared", "{work}/prepared", "--output", "{output}/run",
     "--adapter", "subprocess:Popen", "--adapter-options", '{"args": ["runpodctl"]}'],
    ["rewirebench", "submit-queue", "--send"],
    ["rewirebench", "enqueue"],
    ["rewirebench", "recipe", "run", "other"],
    ["rewirebench", "receipt", "verify", "x"],
    ["/usr/bin/env", "rewirebench", "prepare"],
])
def test_a_recipe_runs_only_rewirebench_data_commands(argv: list[str]):
    data = changed()
    data["commands"][0]["argv"] = argv
    with pytest.raises(recipes.RecipeError):
        parse(data)
    for path in recipes.recipe_files(RECIPES):
        for command in recipes.load_recipe(path).recipe.commands:
            assert command.argv[0] == "rewirebench"
            assert command.argv[1] in recipes.RUNNER_COMMANDS


def test_a_command_has_a_bounded_timeout():
    data = changed()
    assert parse(data).commands[0].timeout_seconds == 3600
    for seconds in (0, 86401, "60", 1.5):
        data["commands"][0]["timeout_seconds"] = seconds
        with pytest.raises(recipes.RecipeError, match="timeout_seconds"):
            parse(data)
    data["commands"][0]["timeout_seconds"] = 86400
    assert parse(data).commands[0].timeout_seconds == 86400


@pytest.mark.parametrize(("field", "value", "message"), [
    ("tolerance", float("inf"), "finite number"),
    ("tolerance", float("nan"), "finite number"),
    ("value", float("nan"), "finite number"),
    ("value", float("inf"), "finite number"),
    ("tolerance", 1e9, "wider than the check allows"),
    ("tolerance", 3.5, "wider than the check allows"),
    ("tolerance", -0.1, "greater than or equal to 0"),
    ("value", "3", "valid number"),
    ("value", True, "valid number"),
])
def test_expected_values_are_finite_numbers_with_a_tolerance_that_checks_something(
    field: str, value: Any, message: str
):
    data = changed()
    data["expected_output"]["values"][0][field] = value
    with pytest.raises(recipes.RecipeError, match=message):
        parse(data)


def test_a_zero_expected_value_allows_a_tolerance_up_to_one():
    data = changed()
    data["expected_output"]["values"][0].update(value=0, tolerance=1)
    assert parse(data).expected_output.values[0].tolerance == 1
    data["expected_output"]["values"][0]["tolerance"] = 1.01
    with pytest.raises(recipes.RecipeError, match="wider than the check allows"):
        parse(data)


def test_fields_do_not_coerce_and_stated_requirements_have_a_checkable_form():
    data = changed()
    data["inputs"][0]["bytes"] = True
    with pytest.raises(recipes.RecipeError, match="inputs.0.bytes"):
        parse(data)
    data = changed()
    data["environment"]["python"] = "banana"
    with pytest.raises(recipes.RecipeError, match="environment.python"):
        parse(data)
    data["environment"]["python"] = "==3.12.1"
    assert parse(data).environment.python == "==3.12.1"
    data["environment"]["lockfile"] = {"path": "../uv.lock", "sha256": "0" * 64}
    with pytest.raises(recipes.RecipeError, match="relative path inside the repository"):
        parse(data)
    for output in ("receipt.json", "SHA256SUMS", "logs/01-prepare.stdout.txt"):
        with pytest.raises(recipes.RecipeError, match="the runner writes itself"):
            parse(changed(outputs=[output]))


def test_inputs_are_pinned_and_come_from_one_place():
    data = changed()
    data["inputs"][0]["sha256"] = "not-a-digest"
    with pytest.raises(recipes.RecipeError):
        parse(data)
    data = changed()
    data["inputs"][0]["url"] = "https://example.org/smoke.csv"
    with pytest.raises(recipes.RecipeError, match="exactly one of url or repo_path"):
        parse(data)
    del data["inputs"][0]["repo_path"]
    data["inputs"][0]["url"] = "http://example.org/smoke.csv"
    with pytest.raises(recipes.RecipeError, match="https"):
        parse(data)
    # The URL is copied into the receipt, so it carries no credential.
    for url in ("https://user:token@example.org/smoke.csv", "https://token@example.org/x"):
        data["inputs"][0]["url"] = url
        with pytest.raises(recipes.RecipeError, match="user name or password"):
            parse(data)
    data["inputs"][0]["url"] = "https://"
    with pytest.raises(recipes.RecipeError, match="https"):
        parse(data)
    data = changed()
    data["inputs"][0]["repo_path"] = "../outside.csv"
    with pytest.raises(recipes.RecipeError, match="relative path inside the repository"):
        parse(data)


def test_commands_are_argv_lists_with_known_placeholders():
    data = changed()
    data["commands"][0]["argv"] = "rewirebench prepare"
    with pytest.raises(recipes.RecipeError):
        parse(data)
    data = changed()
    data["commands"][0]["argv"][-3] = "{outputs}/prepared"
    with pytest.raises(recipes.RecipeError, match="unknown placeholder"):
        parse(data)
    data = changed(outputs=["/etc/passwd"])
    with pytest.raises(recipes.RecipeError, match="relative path inside the run folder") as error:
        parse(data)
    # The message names the problem without a dump of the recipe or a documentation link.
    assert "input_value" not in str(error.value) and "pydantic" not in str(error.value)
    data = changed()
    data["commands"][0]["argv"] = "rewirebench prepare"
    with pytest.raises(recipes.RecipeError, match=r"^commands\.0\.argv: "):
        parse(data)


def test_weights_pin_needs_a_revision_and_file_digests():
    pin = {"host": "huggingface", "repository": "arcinstitute/evo2_7b", "licence": "Apache-2.0",
           "gated": False, "resolved_on": "2026-11-20",
           "files": [{"path": "evo2_7b.pt", "sha256": "c" * 64}]}
    data = changed(weights_pin=[pin])
    del data["missing"]["weights_pin"]
    with pytest.raises(recipes.RecipeError, match="40-hex revision"):
        parse(data)
    pin["revision"] = "b" * 40
    assert parse(data).weights_pin[0].revision == "b" * 40
    for field, value in (("host", "HuggingFace Hub"), ("gated", "yes"),
                         ("files", [{"path": "../../m.bin", "sha256": "c" * 64}])):
        with pytest.raises(recipes.RecipeError):
            parse(changed(weights_pin=[{**pin, field: value}],
                          missing={key: value for key, value in data["missing"].items()}))
    del pin["revision"]
    pin["host"] = "zenodo"
    assert parse(data).weights_pin[0].revision is None


def test_a_price_needs_a_dated_source():
    basis = {key: value for key, value in BASIS.items() if key != "rate_date"}
    with pytest.raises(recipes.RecipeError, match="estimate.price_basis.rate_date"):
        parse(changed(estimate={"gpu_hours": 1, "price_basis": basis}))
    basis["rate_date"] = "2026-10-09"
    assert parse(changed(estimate={"gpu_hours": 1, "price_basis": basis})).estimate.gpu_hours == 1


def test_harness_recipes_name_a_real_protocol():
    with pytest.raises(ValueError, match="Unknown rewirebench protocol"):
        parse(changed(rewirebench_protocol="no-such-protocol"))
    with pytest.raises(recipes.RecipeError, match="rewirebench_protocol"):
        parse(changed(rewirebench_protocol=None))
    with pytest.raises(recipes.RecipeError, match="Extra inputs"):
        parse(changed(protocol_id="rewire-mfass-v2"))


def test_duplicate_yaml_keys_are_rejected():
    text = (RECIPES / "flip2/flip2-synthetic-smoke.yaml").read_text()
    with pytest.raises(ValueError, match="Duplicate key 'scope'"):
        recipes.parse_recipe(text + "scope: full_protocol\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        recipes.parse_recipe("- a list\n")
    with pytest.raises(recipes.RecipeError, match="line 1 is a list or mapping"):
        recipes.parse_recipe("? [a, b]\n: 1\n")


def test_file_name_must_equal_the_recipe_id(tmp_path: Path):
    target = tmp_path / "other-name.yaml"
    target.write_text((RECIPES / "flip2/flip2-synthetic-smoke.yaml").read_text())
    with pytest.raises(ValueError, match="flip2-synthetic-smoke.yaml"):
        recipes.load_recipe(target)


def test_cli_lists_shows_and_validates(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    listing, code = run_cli(capsys, "recipe", "list", "--recipes-dir", str(RECIPES))
    assert code == 0
    by_id = {entry["id"]: entry for entry in listing["recipes"]}
    entry = by_id["mfass-v2-kmer-baseline"]
    assert entry["path"] == "recipes/mfass/mfass-v2-kmer-baseline.yaml"
    assert entry["sha256"] == recipes.sha256_file(RECIPES / "mfass/mfass-v2-kmer-baseline.yaml")
    shown, code = run_cli(capsys, "recipe", "show", "mfass-v2-kmer-baseline",
                          "--recipes-dir", str(RECIPES))
    assert code == 0 and shown["recipe"]["inputs"][0]["name"] == "snv_data_clean.txt"
    assert "status" not in shown["recipe"]
    results, code = run_cli(capsys, "recipe", "validate", "--recipes-dir", str(RECIPES))
    assert code == 0 and all(entry["valid"] for entry in results)
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(changed(id="bad", status="executed")))
    results, code = run_cli(capsys, "recipe", "validate", str(bad),
                            str(RECIPES / "flip2/flip2-synthetic-smoke.yaml"))
    assert code == 1
    assert [entry["valid"] for entry in results] == [False, True]
    assert "status" in results[0]["error"]
    with pytest.raises(SystemExit) as stop:
        cli.main(["recipe", "show", "no-such-recipe", "--recipes-dir", str(RECIPES)])
    assert stop.value.code == 2


def test_a_bad_file_is_reported_and_does_not_hide_the_others(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    directory = tmp_path / "recipes"
    (directory / "flip2").mkdir(parents=True)
    good = (RECIPES / "flip2/flip2-synthetic-smoke.yaml").read_text()
    (directory / "flip2/flip2-synthetic-smoke.yaml").write_text(good)
    (directory / "flip2/unhashable.yaml").write_text("? [a, b]\n: 1\n")
    (directory / "flip2/other.yml").write_text(good.replace("flip2-synthetic-smoke", "other"))
    (directory / "flip2/copy.yaml").write_text(good)
    listing, code = run_cli(capsys, "recipe", "list", "--recipes-dir", str(directory))
    assert code == 1
    assert [entry["id"] for entry in listing["recipes"]] == ["flip2-synthetic-smoke"]
    problems = {Path(entry["path"]).name: entry["error"] for entry in listing["invalid"]}
    assert set(problems) == {"unhashable.yaml", "other.yml", "copy.yaml"}
    assert "end in .yaml" in problems["other.yml"]
    assert "list or mapping" in problems["unhashable.yaml"]
    results, code = run_cli(capsys, "recipe", "validate", "--recipes-dir", str(directory))
    assert code == 1
    assert {Path(entry["path"]).name for entry in results if not entry["valid"]} == set(problems)
    with pytest.raises(SystemExit) as stop:
        cli.main(["recipe", "show", "other", "--recipes-dir", str(directory)])
    assert stop.value.code == 2 and "end in .yaml" in capsys.readouterr().err


def test_the_checkout_the_code_runs_from_comes_before_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Running from a folder that has its own recipes/ must not swap the recipes."""
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes" / recipes.SCHEMA_FILENAME).write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(recipes.RECIPES_DIR_VARIABLE, raising=False)
    assert recipes.find_recipes_dir() == RECIPES
    # An installed wheel has no checkout, so it falls back to the working directory.
    monkeypatch.setattr(recipes, "__file__", str(tmp_path / "site-packages/rewirebench/recipes.py"))
    assert recipes.find_recipes_dir() == (tmp_path / "recipes").resolve()


def test_cli_works_where_no_recipes_directory_exists(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The wheel ships no recipes, so listing outside a checkout is empty, not an error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(recipes.RECIPES_DIR_VARIABLE, raising=False)
    monkeypatch.setattr(recipes, "__file__", str(tmp_path / "site-packages/rewirebench/recipes.py"))
    assert recipes.find_recipes_dir() is None
    listing, code = run_cli(capsys, "recipe", "list")
    assert code == 0 and listing["recipes"] == [] and listing["recipes_dir"] is None
    schema, code = run_cli(capsys, "recipe", "schema")
    assert code == 0 and schema["properties"]["schema_version"]
    monkeypatch.setenv(recipes.RECIPES_DIR_VARIABLE, str(RECIPES))
    assert recipes.find_recipes_dir() == RECIPES


def test_packaged_recipes_fallback_and_wheel_config(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Installed wheels fall back to packaged recipes when no repo or cwd recipes exist."""
    import shutil
    import tomllib

    pyproject = tomllib.loads((ROOT / "packages/rewirebench/pyproject.toml").read_text())
    assert pyproject["tool"]["hatch"]["build"]["hooks"]["custom"]["path"] == "hatch_build.py"
    assert (ROOT / "packages/rewirebench/hatch_build.py").is_file()

    pkg_root = tmp_path / "site-packages/rewirebench"
    pkg_recipes = pkg_root / "resources/recipes"
    shutil.copytree(RECIPES, pkg_recipes)
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    monkeypatch.delenv(recipes.RECIPES_DIR_VARIABLE, raising=False)
    monkeypatch.setattr(recipes, "__file__", str(pkg_root / "recipes.py"))
    assert recipes.find_recipes_dir() == pkg_recipes.resolve()
    listing, code = run_cli(capsys, "recipe", "list")
    assert code == 0 and listing["recipes"]

    cwd_recipes = empty_cwd / "recipes"
    cwd_recipes.mkdir()
    (cwd_recipes / recipes.SCHEMA_FILENAME).write_text((RECIPES / recipes.SCHEMA_FILENAME).read_text())
    assert recipes.find_recipes_dir() == cwd_recipes.resolve()


PREPARE = ["rewirebench", "prepare", "flip2-fitness-v1"]
FLIP2 = [*PREPARE, "--source", "{inputs}/x", "--output", "{work}/p"]
MFASS = ["rewirebench", "prepare", "mfass-v2", "--source", "{inputs}", "--output", "{work}/p"]
PROTEINGYM = ["rewirebench", "prepare", "proteingym-v1.3-dms-substitutions", "--source",
              "{inputs}", "--output", "{work}/p"]
EVALUATE = ["rewirebench", "evaluate", "--prepared", "{work}/prepared", "--output",
            "{output}/scored", "--predictions", "{inputs}/p.json"]


def swapped(argv: list[str], old: str, new: str) -> list[str]:
    return [new if item == old else item for item in argv]


@pytest.mark.parametrize("argv", [
    # The two steps of issue 80.
    [*PREPARE, "--source", "{inputs}/x", "--output", "/etc/x"],
    ["rewirebench", "evaluate", "--prepared", "/etc/passwd", "--output", "/etc/x",
     "--predictions", "p"],
    [*PREPARE, "--source", "/etc/passwd", "--output", "{work}/prepared"],
    [*PREPARE, "--source", "{inputs}/x", "--output=/etc/x"],
    # `prepare` accepts abbreviated options, so the parsed value is what is checked.
    [*PREPARE, "--source", "{inputs}/x", "--outp", "/etc/x"],
    [*PREPARE, "--source", "{inputs}/x", "--output", "{work}/../../etc/x"],
    [*PREPARE, "--source", "{inputs}/x", "--output", "{work}/.."],
    [*PREPARE, "--source", "{inputs}/x", "--output", "prepared"],
    [*PREPARE, "--source", "{inputs}/x", "--output", "x/{work}/prepared"],
    [*PREPARE, "--source", "{inputs}/x", "--output", "{work}x/prepared"],
    [*PREPARE, "--source", "{inputs}/x", "--output", "{work}\\..\\x"],
    # A step writes to {work} and {output} only.
    [*PREPARE, "--source", "{inputs}/x", "--output", "{inputs}/x"],
    swapped(EVALUATE, "{output}/scored", "{inputs}/scored"),
    ["rewirebench", "run", "--prepared", "{work}/p", "--output", "{inputs}/run", "--model",
     "train-mean"],
    ["rewirebench", "run-baselines", "--prepared", "{work}/p", "--output", "{inputs}/b"],
    ["rewirebench", "prepare-baseline-artifact", "proteingym-evcouplings-independent-v1",
     "--manifest", "{inputs}/m.json", "--output", "{inputs}/a.json"],
    swapped(EVALUATE, "{inputs}/p.json", "~/p.json"),
    swapped(EVALUATE, "--predictions", "--embeddings")[:-1] + ["/etc/e.npz"],
    ["rewirebench", "run-baselines", "--prepared", "{work}/prepared", "--output", "/etc/x"],
    ["rewirebench", "run", "--prepared", "/etc", "--output", "{output}/run", "--model",
     "train-mean"],
    ["rewirebench", "prepare-baseline-artifact", "proteingym-evcouplings-independent-v1",
     "--manifest", "/etc/m.json", "--output", "{output}/a.json"],
    ["rewirebench", "prepare-baseline-artifact", "proteingym-evcouplings-independent-v1",
     "--manifest", "{inputs}/m.json", "--output", "/etc/a.json"],
])
def test_a_step_names_paths_only_under_the_three_placeholders(argv: list[str]):
    with pytest.raises(ValueError, match="must be a path under"):
        recipes.check_step_paths(argv)
    data = changed()
    data["commands"][0]["argv"] = argv
    with pytest.raises(recipes.RecipeError, match=r"^commands\.0: Command prepare: "):
        parse(data)


@pytest.mark.parametrize(("argv", "where"), [
    # MFASS opens `split` as a file and ProteinGym opens `reference`, so there the value
    # is a path whatever it looks like.
    ([*MFASS, "--options", '{"split": "/etc/passwd"}'], "--options 'split'"),
    ([*MFASS, "--options", '{"split": "split.tsv"}'], "--options 'split'"),
    (["rewirebench", "prepare", "mfass-v2-frozen-encoder", "--source", "{inputs}", "--output",
      "{work}/p", "--options", '{"split": "split.tsv"}'], "--options 'split'"),
    ([*PROTEINGYM, "--options", '{"reference": "reference.csv"}'], "--options 'reference'"),
    # FLIP2 reads `split` as a name, but a value that could be used as a path is still held
    # to the rule, as it is under any key.
    ([*FLIP2, "--options", '{"split": "/etc/passwd"}'], "--options 'split'"),
    # Each of the three signs alone: absolute, a .. part, a placeholder.
    ([*FLIP2, "--options", '{"other": {"deep": "~/x"}}'], "--options 'other' 'deep'"),
    ([*FLIP2, "--options", '{"other": ["a/../b"]}'], "--options 'other'"),
    ([*FLIP2, "--options", '{"other": "x{work}/y"}'], "--options 'other'"),
    ([*FLIP2, "--options", '{"other": "{inputs}x"}'], "--options 'other'"),
    # Everything under a path key is a path, however deep.
    (["rewirebench", "run", "--prepared", "{work}/p", "--output", "{output}/run", "--model",
      "esm2-embeddings", "--adapter-options", '{"checkpoint": {"path": "x.pt"}}'],
     "--adapter-options 'checkpoint' 'path'"),
    (["rewirebench", "run", "--prepared", "{work}/p", "--output", "{output}/run", "--model",
      "esm2-embeddings", "--adapter-options", '{"checkpoint": ["x.pt"]}'],
     "--adapter-options 'checkpoint'"),
    (["rewirebench", "run", "--prepared", "{work}/p", "--output", "{output}/run", "--model",
      "esm2-embeddings", "--adapter-options", '{"checkpoint": "/home/me/x.pt"}'],
     "--adapter-options 'checkpoint'"),
    (["rewirebench", "run-baselines", "--prepared", "{work}/p", "--output", "{output}/b",
      "--baseline-options",
      '{"proteingym-evcouplings-independent-v1": {"artifact": "/etc/a.json"}}'],
     "--baseline-options 'proteingym-evcouplings-independent-v1' 'artifact'"),
])
def test_paths_inside_the_json_options_follow_the_same_rule(argv: list[str], where: str):
    with pytest.raises(ValueError, match="must be a path under") as error:
        recipes.check_step_paths(argv)
    assert str(error.value).startswith(where)


def test_steps_that_stay_in_the_run_folders_pass():
    for argv in (
        EVALUATE,
        [*PREPARE, "--source", "{inputs}", "--output", "{output}"],
        [*MFASS, "--options", '{"split": "{inputs}/split-v2.tsv", "limit": 3}'],
        [*PROTEINGYM, "--options", '{"reference": "{inputs}/reference.csv"}'],
        # A FLIP2 split is a name, not a file.
        [*FLIP2, "--options", '{"dataset": "amylase", "split": "one_to_many", "limit": 3}'],
        [*FLIP2, "--options", '{"dataset": "flip2-amylase-one-to-many", "note": "a/b.c"}'],
        ["rewirebench", "run", "--prepared", "{work}/p", "--output", "{output}/run", "--model",
         "esm2-embeddings", "--adapter-options", '{"checkpoint": "{inputs}/x.pt"}'],
        ["rewirebench", "run-baselines", "--prepared", "{work}/p", "--output", "{output}/b",
         "--baseline-options",
         '{"proteingym-evcouplings-independent-v1": {"artifact": "{work}/a.json"}}'],
    ):
        recipes.check_step_paths(argv)


def test_a_flip2_split_name_is_not_taken_for_a_path():
    """`split` names a FLIP2 split, and the same key is a file for MFASS. The committed
    FLIP2 recipe still validates with the split named, as it did before the path rule."""
    data = yaml.safe_load((RECIPES / "flip2/flip2-synthetic-smoke.yaml").read_text())
    argv = data["commands"][0]["argv"]
    options = json.loads(argv[argv.index("--options") + 1])
    entry = flip2.DATASETS[options["dataset"]]
    argv[argv.index("--options") + 1] = json.dumps({**options, "split": entry["split"]})
    parse(data)
    assert flip2._selection({**options, "split": entry["split"]}) == entry
    assert flip2._selection({"dataset": entry["dataset"], "split": entry["split"]}) == entry
    recipes.check_step_paths([*FLIP2, "--options", json.dumps(
        {"dataset": entry["dataset"], "split": entry["split"]})])


@pytest.mark.parametrize("output", ["{output}/receipt.json", "{output}/SHA256SUMS",
                                    "{output}/logs", "{output}/logs/01-prepare.stdout.txt"])
def test_a_step_cannot_write_to_a_name_the_runner_writes(output: str, tmp_path: Path):
    step = ["rewirebench", "prepare-baseline-artifact", "proteingym-evcouplings-independent-v1",
            "--manifest", "{inputs}/m.json", "--output", output]
    with pytest.raises(ValueError, match="--output uses a name the runner writes itself"):
        recipes.check_step_paths(step)
    values = folders(tmp_path)
    with pytest.raises(ValueError, match="--output resolves to a name the runner writes "
                                         "itself") as error:
        recipes.check_step_paths(rendered(step, values), values)
    assert str(tmp_path) not in str(error.value)
    # The names are reserved at the top of the run folder and for writing only.
    for fine in ("{output}/a/receipt.json", "{work}/receipt.json", "{work}/logs/x",
                 "{output}/logs.json"):
        recipes.check_step_paths(swapped(step, output, fine))
        recipes.check_step_paths(rendered(swapped(step, output, fine), values), values)
    read = swapped(EVALUATE, "{inputs}/p.json", "{output}/receipt.json")
    recipes.check_step_paths(read)
    recipes.check_step_paths(rendered(read, values), values)


def test_a_link_to_a_name_the_runner_writes_is_followed(tmp_path: Path):
    values = folders(tmp_path)
    run = Path(values["output"])
    (run / "logs").mkdir()
    (run / "alias").symlink_to(run / "logs", target_is_directory=True)
    step = [*FLIP2[:-1], "{output}/alias/x"]
    recipes.check_step_paths(step)
    with pytest.raises(ValueError, match="resolves to a name the runner writes itself"):
        recipes.check_step_paths(rendered(step, values), values)


def test_an_option_that_is_not_classified_is_refused(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(recipes.STEP_VALUES, "evaluate", ("model_name", "training_overlap"))
    with pytest.raises(ValueError, match="A recipe step cannot use the option --allow-partial"):
        recipes.check_step_paths(EVALUATE)


def test_a_step_that_does_not_parse_is_refused():
    with pytest.raises(ValueError):
        recipes.check_step_paths(["rewirebench", "evaluate", "write", "{output}"])
    with pytest.raises(ValueError, match="--options is not JSON"):
        recipes.check_step_paths([*PREPARE, "--source", "{inputs}", "--output", "{work}/p",
                                  "--options", "{"])
    with pytest.raises(ValueError, match="Not a recipe step"):
        recipes.check_step_paths(["rewirebench", "export", "--report", "{output}/r",
                                  "--output", "{output}/b"])


def test_every_option_of_a_step_command_is_classified():
    """A new option of a step command is a path, JSON or a plain value. Until it is listed
    as one of them a recipe cannot use it, and this test says so first."""
    subcommands = next(action for action in cli.build_parser()._actions
                       if isinstance(action.choices, dict)).choices
    assert set(recipes.STEP_PATHS) == set(recipes.STEP_VALUES) == set(recipes.RUNNER_COMMANDS)
    for name in recipes.RUNNER_COMMANDS:
        options = {action.dest for action in subcommands[name]._actions} - {"help"}
        listed = {*recipes.STEP_PATHS[name], *recipes.STEP_VALUES[name]}
        if name in recipes.STEP_JSON:
            listed.add(recipes.STEP_JSON[name])
        assert options == listed, name


def test_every_text_option_of_a_model_is_a_path_key_or_known_not_to_be_one():
    """A model option that holds a path has to be in MODEL_PATH_KEYS, or a recipe could
    name any file with it. A new text option fails here until it is put on one side."""
    from rewirebench import models

    not_paths = {"alphabet", "artifact_sha256", "device", "field", "fimo_sha256", "model_name"}
    text = {name for spec in models.REGISTRY.values() for name in spec.options or {}
            if spec.option_type(name) == "text"}
    assert text - not_paths == set(recipes.MODEL_PATH_KEYS)


def test_every_option_of_a_baseline_is_a_path_key_or_known_not_to_be_one():
    """The same for --baseline-options, whose keys are the artifact options of a baseline."""
    from rewirebench import baselines

    not_paths = {"artifact_sha256", "fimo_sha256"}
    names = {name for protocol in sdk.PROTOCOLS
             for entry in baselines.describe_baselines(protocol)["baselines"]
             for name in entry.get("artifact_options", {})}
    assert names == {"artifact", "artifact_sha256", "fimo", "fimo_sha256", "motifs"}
    assert names - not_paths <= set(recipes.MODEL_PATH_KEYS)


# Every key a protocol's prepare reads from --options, and whether it opens it as a file.
PROTOCOL_OPTIONS = {
    "dart_eval": {"dataset": False, "limit": False, "source_sha256": False},
    "flip2": {"allow_unverified": False, "dataset": False, "limit": False, "split": False},
    "genomic_benchmarks": {"dataset": False, "limit": False},
    "mfass": {"frozen_encoder": False, "limit": False, "protocol_id": False, "split": True},
    "mrnabench": {"dataset": False, "limit": False, "require_official": False, "seed": False,
                  "target": False},
    "proteingym": {"assay_ids": False, "expected_hashes": False, "limit": False,
                   "reference": True},
    "tdc_admet": {"dataset": False, "limit": False},
}


def option_keys(module: str) -> set[str]:
    """The keys a protocol module reads from the options of prepare, from its source."""
    tree = ast.parse((Path(sdk.__file__).parent / "protocols" / f"{module}.py").read_text())
    keys = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop", "setdefault")):
            holder, key = node.func.value, node.args[0] if node.args else None
        elif isinstance(node, ast.Subscript):
            holder, key = node.value, node.slice
        else:
            continue
        if isinstance(holder, ast.Name) and holder.id == "options":
            assert isinstance(key, ast.Constant), f"{module}: an option key that is not a literal"
            keys.add(key.value)
    return keys


def test_every_option_of_a_protocol_is_a_path_key_or_known_not_to_be_one():
    """`split` is a file for MFASS and a name for FLIP2, so the path keys of --options are
    per protocol. A key a protocol starts to read fails here until it is put on one side,
    and a protocol that passes its options on whole fails too."""
    assert set(PROTOCOL_OPTIONS) == set(sdk.PROTOCOLS.values())
    for module, expected in PROTOCOL_OPTIONS.items():
        assert option_keys(module) == set(expected), module
        source = (Path(sdk.__file__).parent / "protocols" / f"{module}.py").read_text()
        assert "**options" not in source.replace("def prepare(source: Path, **options)", ""), module
        paths = {key for key, is_path in expected.items() if is_path}
        assert paths == set(recipes.PREPARE_PATH_KEYS.get(module, ())), module
    assert set(recipes.PREPARE_PATH_KEYS) <= set(PROTOCOL_OPTIONS)


def test_an_unknown_protocol_holds_every_prepare_path_key():
    assert recipes._path_keys("prepare", {"protocol": "no-such"}) == {"reference", "split"}
    assert recipes._path_keys("prepare", {"protocol": "flip2-fitness-v1"}) == frozenset()
    assert recipes._path_keys("prepare", {"protocol": "mfass-v2"}) == {"split"}
    assert recipes._path_keys("run", {}) == set(recipes.MODEL_PATH_KEYS)
    assert recipes._path_keys("run-baselines", {}) == set(recipes.MODEL_PATH_KEYS)


def folders(tmp_path: Path) -> dict[str, str]:
    for name in ("scratch/inputs", "scratch/work", "run", "outside"):
        (tmp_path / name).mkdir(parents=True)
    return {"inputs": str(tmp_path / "scratch/inputs"), "work": str(tmp_path / "scratch/work"),
            "output": str(tmp_path / "run")}


def rendered(argv: list[str], values: dict[str, str]) -> list[str]:
    for name, value in values.items():
        argv = [item.replace("{" + name + "}", value) for item in argv]
    return argv


def test_rendered_paths_are_resolved_and_refused_outside_the_run_folders(tmp_path: Path):
    values = folders(tmp_path)
    outside = tmp_path / "outside"
    step = [*PREPARE, "--source", "{inputs}", "--output", "{work}/prepared"]
    recipes.check_step_paths(rendered(step, values), values)
    recipes.check_step_paths(rendered(EVALUATE, values), values)
    for output in (str(outside / "x"), "{work}/../../outside/x", "{inputs}/x", "../x"):
        with pytest.raises(ValueError, match="--output resolves to a path outside") as error:
            recipes.check_step_paths(rendered(swapped(step, "{work}/prepared", output), values),
                                     values)
        # The rendered value is a local path, so the message does not repeat it.
        assert str(tmp_path) not in str(error.value)
    # A relative path is taken from {work}, where the step runs.
    recipes.check_step_paths(rendered(swapped(step, "{work}/prepared", "prepared"), values),
                             values)


def test_a_link_out_of_the_run_folders_is_followed_and_refused(tmp_path: Path):
    values = folders(tmp_path)
    outside = tmp_path / "outside"
    (outside / "secret.json").write_text("{}")
    work, inputs = Path(values["work"]), Path(values["inputs"])
    (work / "folder").symlink_to(outside, target_is_directory=True)
    (work / "dangling").symlink_to(outside / "not-there-yet")
    (inputs / "p.json").symlink_to(outside / "secret.json")
    (work / "inside").symlink_to(Path(values["output"]), target_is_directory=True)
    step = [*PREPARE, "--source", "{inputs}", "--output", "{work}/prepared"]
    for output in ("{work}/folder/x", "{work}/folder", "{work}/dangling",
                   "{work}/inside/../../outside/x"):
        with pytest.raises(ValueError, match="--output resolves to a path outside"):
            recipes.check_step_paths(rendered(swapped(step, "{work}/prepared", output), values),
                                     values)
    with pytest.raises(ValueError, match="--predictions resolves to a path outside"):
        recipes.check_step_paths(rendered(EVALUATE, values), values)
    options = [*MFASS, "--options", '{"split": "{work}/folder/secret.json"}']
    with pytest.raises(ValueError, match="--options 'split' resolves to a path outside"):
        recipes.check_step_paths(rendered(options, values), values)
    # A link that stays inside the run folders is fine.
    recipes.check_step_paths(rendered(swapped(step, "{work}/prepared", "{work}/inside/x"),
                                      values), values)


def test_the_run_folders_may_themselves_sit_behind_a_link(tmp_path: Path):
    """The temporary folder is behind a link on macOS, so both sides are resolved."""
    values = folders(tmp_path / "real")
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    aliased = {name: value.replace(str(tmp_path / "real"), str(tmp_path / "alias"))
               for name, value in values.items()}
    step = [*PREPARE, "--source", "{inputs}", "--output", "{work}/prepared"]
    recipes.check_step_paths(rendered(step, aliased), aliased)
    recipes.check_step_paths(rendered(step, values), aliased)
