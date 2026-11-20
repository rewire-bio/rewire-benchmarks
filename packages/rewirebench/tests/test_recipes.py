"""Recipes are validated files: a misspelt, unpinned or status-carrying recipe is refused."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from rewirebench import cli, recipes

ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "recipes"
DIGEST = "sha256:" + "a" * 64


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
    assert recipe.protocol_id == "mfass-v2" and recipe.scope == "full_protocol"
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
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        parse(changed(**{key: "executed"}))


def test_a_block_is_present_or_named_missing_never_both_or_neither():
    data = changed()
    del data["missing"]["weights_pin"]
    with pytest.raises(ValidationError, match="weights_pin must be either present"):
        parse(data)
    data = changed()
    data["missing"]["hardware_floor"] = {"reason": "unreported"}
    with pytest.raises(ValidationError, match="hardware_floor must be either present"):
        parse(data)
    data = changed()
    del data["expected_output"]
    data["missing"]["expected_output"] = {"reason": "unreported"}
    assert parse(data).expected_output is None
    with pytest.raises(ValidationError):
        parse(changed(missing={"weights_pin": {"reason": "not measured yet"}}))


def test_images_are_by_digest_for_both_platforms():
    data = changed()
    data["environment"]["images"]["linux/amd64"] = {"reference": "ghcr.io/rewire-bio/x:latest"}
    with pytest.raises(ValidationError, match="by digest"):
        parse(data)
    data["environment"]["images"]["linux/amd64"] = {"reference": f"ghcr.io/rewire-bio/x@{DIGEST}"}
    assert parse(data).image_digests() == {"linux/amd64": DIGEST, "linux/arm64": None}
    del data["environment"]["images"]["linux/arm64"]
    with pytest.raises(ValidationError, match="exactly these platforms"):
        parse(data)
    data = changed()
    data["environment"]["images"]["linux/arm64"] = {}
    with pytest.raises(ValidationError, match="exactly one of reference or missing"):
        parse(data)


def test_gpu_backends_need_the_image_for_their_platform():
    runpod = {"gpu_types": ["NVIDIA H100 80GB HBM3"], "disk_gb": 200, "max_hours": 3}
    data = changed(backends={"runpod": runpod})
    with pytest.raises(ValidationError, match="linux/amd64 image digest"):
        parse(data)
    data["environment"]["images"]["linux/amd64"] = {"reference": f"ghcr.io/rewire-bio/x@{DIGEST}"}
    assert parse(data).backends.names() == ["runpod"]
    data["backends"] = {"spark": {}}
    with pytest.raises(ValidationError, match="linux/arm64 image digest"):
        parse(data)
    with pytest.raises(ValidationError, match="at least one backend"):
        parse(changed(backends={}))


def test_inputs_are_pinned_and_come_from_one_place():
    data = changed()
    data["inputs"][0]["sha256"] = "not-a-digest"
    with pytest.raises(ValidationError):
        parse(data)
    data = changed()
    data["inputs"][0]["url"] = "https://example.org/smoke.csv"
    with pytest.raises(ValidationError, match="exactly one of url or repo_path"):
        parse(data)
    del data["inputs"][0]["repo_path"]
    data["inputs"][0]["url"] = "http://example.org/smoke.csv"
    with pytest.raises(ValidationError, match="https"):
        parse(data)
    data = changed()
    data["inputs"][0]["repo_path"] = "../outside.csv"
    with pytest.raises(ValidationError, match="relative path"):
        parse(data)


def test_commands_are_argv_lists_with_known_placeholders():
    data = changed()
    data["commands"][0]["argv"] = "rewirebench prepare"
    with pytest.raises(ValidationError):
        parse(data)
    data = changed()
    data["commands"][0]["argv"][-3] = "{outputs}/prepared"
    with pytest.raises(ValidationError, match="unknown placeholder"):
        parse(data)
    data = changed(outputs=["/etc/passwd"])
    with pytest.raises(ValidationError, match="relative path"):
        parse(data)


def test_weights_pin_needs_a_revision_and_file_digests():
    pin = {"host": "huggingface", "repository": "arcinstitute/evo2_7b", "licence": "Apache-2.0",
           "gated": False, "resolved_on": "2026-11-20",
           "files": [{"path": "evo2_7b.pt", "sha256": "c" * 64}]}
    data = changed(weights_pin=[pin])
    del data["missing"]["weights_pin"]
    with pytest.raises(ValidationError, match="40-hex revision"):
        parse(data)
    pin["revision"] = "b" * 40
    assert parse(data).weights_pin[0].revision == "b" * 40


def test_a_price_needs_a_dated_source():
    basis = {"provider": "runpod", "accelerator": "NVIDIA H100 80GB HBM3",
             "amount_per_hour": 3.99, "currency": "USD", "rate_source": "instance.json"}
    with pytest.raises(ValidationError, match="rate_date"):
        parse(changed(estimate={"gpu_hours": 1, "price_basis": basis}))
    basis["rate_date"] = "2026-10-09"
    assert parse(changed(estimate={"gpu_hours": 1, "price_basis": basis})).estimate.gpu_hours == 1


def test_harness_recipes_name_a_real_protocol():
    with pytest.raises(ValueError, match="Unknown rewirebench protocol"):
        parse(changed(protocol_id="no-such-protocol"))
    with pytest.raises(ValidationError, match="protocol_id"):
        parse(changed(protocol_id=None))


def test_duplicate_yaml_keys_are_rejected():
    text = (RECIPES / "flip2/flip2-synthetic-smoke.yaml").read_text()
    with pytest.raises(ValueError, match="Duplicate key 'scope'"):
        recipes.parse_recipe(text + "scope: full_protocol\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        recipes.parse_recipe("- a list\n")


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
