"""The model registry is the allowlist for `run --model` and for recipe run steps."""
from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from recipe_fixtures import model_recipe_data
from rewirebench import cli, models, receipts, recipes, sdk


def parse(data: dict[str, Any]) -> recipes.Recipe:
    return recipes.parse_recipe(json.dumps(data))


def with_step(*arguments: str) -> dict[str, Any]:
    data = model_recipe_data()
    data["commands"][1]["argv"] = ["rewirebench", "run", *arguments]
    return data


STEP = ("--prepared", "{work}/prepared", "--output", "{output}/run")


def test_every_model_names_a_rewirebench_adapter_with_the_declared_interface():
    assert len(models.REGISTRY) == len(models._MODELS) >= 10
    for spec in models.REGISTRY.values():
        module, _, name = spec.adapter.partition(":")
        assert module.startswith("rewirebench.adapters.")
        adapter = getattr(importlib.import_module(module), name)
        for method in spec.interfaces:
            assert callable(getattr(adapter, method)), (spec.id, method)
        parameters = inspect.signature(adapter.__init__).parameters
        accepted = {key for key in parameters if key != "self"}
        assert set(spec.options or {}) <= accepted, spec.id
        required = {key for key, value in parameters.items()
                    if key != "self" and value.default is inspect.Parameter.empty
                    and value.kind in (value.POSITIONAL_OR_KEYWORD, value.KEYWORD_ONLY)}
        assert required == set(spec.required_options), spec.id


def test_every_model_fits_the_rules_of_the_protocols_it_lists():
    for spec in models.REGISTRY.values():
        assert spec.protocols, spec.id
        for protocol in spec.protocols:
            rules = sdk.capabilities(protocol)
            assert spec.prediction_type in rules["prediction_types"], (spec.id, protocol)
            if "fit" in spec.interfaces:
                assert rules["allow_fit"], (spec.id, protocol)


def test_the_interface_protocols_describe_the_adapters():
    from rewirebench.adapters.sequence import SequenceComposition, TrainMean

    assert isinstance(TrainMean(), models.FittedModel)
    assert isinstance(TrainMean(), models.ScoreModel)
    assert isinstance(SequenceComposition(), models.EmbeddingModel)
    assert not isinstance(SequenceComposition(), models.ScoreModel)


def test_list_models_filters_by_protocol_and_rejects_an_unknown_one():
    listed = {entry["id"] for entry in models.list_models("mfass-v2")}
    assert listed == {"training-prior", "mfass-kmer-position"}
    assert len(models.list_models()) == len(models.REGISTRY)
    with pytest.raises(ValueError, match="Unknown protocol"):
        models.list_models("no-such-protocol")
    with pytest.raises(ValueError, match="Unknown model"):
        models.get_model("subprocess:Popen")


def test_options_are_limited_to_the_ones_a_model_declares():
    spec = models.get_model("seeded-random")
    assert models.check_options(spec, {"seed": 3}) == {"seed": 3}
    with pytest.raises(ValueError, match="has no option 'args'"):
        models.check_options(spec, {"args": ["runpodctl"]})
    with pytest.raises(ValueError, match="needs the option 'checkpoint'"):
        models.check_options(models.get_model("esm2-embeddings"), {})
    with pytest.raises(ValueError, match="JSON object"):
        models.check_options(spec, ["seed"])
    assert models.load(spec, {"seed": 3}).seed == 3


def test_cli_lists_models_and_runs_one_by_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    cli.main(["models", "--protocol", "flip2-fitness-v1"])
    listed = json.loads(capsys.readouterr().out)
    assert "train-mean" in {entry["id"] for entry in listed}
    source = Path(__file__).resolve().parents[3] / "workflows/benchmark/examples/smoke.csv"
    cli.main(["prepare", "flip2-fitness-v1", "--source", str(source),
              "--output", str(tmp_path / "prepared"), "--options",
              '{"dataset": "flip2-amylase-one-to-many", "allow_unverified": true, "limit": 3}'])
    capsys.readouterr()
    cli.main(["run", "--prepared", str(tmp_path / "prepared"), "--output", str(tmp_path / "run"),
              "--model", "train-mean"])
    assert json.loads(capsys.readouterr().out)["metrics"]["n"] == 3
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["model"]["name"] == "Training target mean"
    with pytest.raises(SystemExit):
        cli.main(["run", "--prepared", str(tmp_path / "prepared"),
                  "--output", str(tmp_path / "other"), "--model", "train-mean",
                  "--adapter", "rewirebench.adapters.sequence:TrainMean"])


def test_a_recipe_run_step_names_a_registered_model():
    recipe = parse(model_recipe_data())
    assert recipe.commands[1].argv[1] == "run"
    assert parse(with_step(*STEP, "--model", "seeded-random", "--adapter-options",
                           '{"seed": 1}')).commands[1].name == "run-model"


@pytest.mark.parametrize(("arguments", "message"), [
    # --adapter imports any installed callable, in every spelling argparse accepts.
    ((*STEP, "--adapter", "subprocess:Popen"), "never --adapter"),
    ((*STEP, "--adapter=subprocess:Popen"), "never --adapter"),
    ((*STEP, "--adapter", "rewirebench.adapters.sequence:TrainMean"), "never --adapter"),
    ((*STEP, "--model", "train-mean", "--adapter", "subprocess:Popen"), "not allowed with"),
    # An abbreviation is not read as the option it shortens.
    ((*STEP, "--adap", "subprocess:Popen"), "required"),
    ((*STEP, "--model", "train-mean", "--adapter-opt", '{"args": 1}'), "unrecognized"),
    ((*STEP, "--mod", "train-mean"), "required"),
    ((*STEP, "--model", "subprocess:Popen"), "Unknown model"),
    ((*STEP,), "required"),
    ((*STEP, "--model", "train-mean", "--adapter-options", '{"args": ["x"]}'), "no option"),
    ((*STEP, "--model", "train-mean", "--adapter-options", "{not json"), "not JSON"),
    # Asking for help text would print to standard output, which the MCP server speaks on.
    ((*STEP, "--model", "train-mean", "--help"), "asks for help text"),
    (("-h",), "asks for help text"),
    # A model that starts an executable named in its options stays out of recipes.
    ((*STEP, "--model", "fimo-h12core-hit-count", "--adapter-options",
      '{"fimo": "/bin/sh", "motifs": "m", "fimo_sha256": "0"}'), "starts another program"),
])
def test_a_recipe_run_step_cannot_reach_any_other_callable(
    arguments: tuple[str, ...], message: str, capsys: pytest.CaptureFixture[str]
):
    with pytest.raises(recipes.RecipeError, match=message):
        parse(with_step(*arguments))
    # The runner applies the same rule to the argv it is about to start.
    with pytest.raises(ValueError, match="runs only rewirebench data commands"):
        receipts._program(["rewirebench", "run", *arguments])
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_a_recipe_with_a_model_step_runs_and_verifies(fixture_recipes: Path, tmp_path: Path):
    loaded = recipes.get_recipe("model-step", fixture_recipes)
    receipt = receipts.run_recipe(loaded, output=tmp_path / "out", recipes_dir=fixture_recipes,
                                  started_by="agent")
    assert receipt.checks_passed, receipt.checks
    assert receipt.commands[1].argv[:2] == ["rewirebench", "run"]
    assert receipt.metrics == {"n": 3.0}
    # The step's own receipt is not written inside a recipe run: one receipt per folder.
    assert not (tmp_path / "out/run/receipt.json").exists()
    result = receipts.verify(tmp_path / "out", recipes_dir=fixture_recipes)
    assert result["ok"], result
