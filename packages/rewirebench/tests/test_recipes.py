"""Recipes are validated files: a misspelt, unpinned or status-carrying recipe is refused."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from rewirebench import cli, recipes

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
    assert recipe.image_digests() == {"linux/amd64": None, "linux/arm64": None}
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
    data = changed()
    data["environment"]["images"]["linux/amd64"] = {"reference": "ghcr.io/rewire-bio/x:latest"}
    with pytest.raises(recipes.RecipeError, match="by digest"):
        parse(data)
    data["environment"]["images"]["linux/amd64"] = {"reference": f"ghcr.io/rewire-bio/x@{DIGEST}"}
    assert parse(data).image_digests() == {"linux/amd64": DIGEST, "linux/arm64": None}
    del data["environment"]["images"]["linux/arm64"]
    with pytest.raises(recipes.RecipeError, match="exactly these platforms"):
        parse(data)
    data = changed()
    data["environment"]["images"]["linux/arm64"] = {}
    with pytest.raises(recipes.RecipeError, match="exactly one of reference or missing"):
        parse(data)


def test_gpu_backends_need_the_image_for_their_platform():
    runpod = {"gpu_types": ["NVIDIA H100 80GB HBM3"], "disk_gb": 200, "max_hours": 3}
    data = changed(backends={"runpod": runpod})
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
