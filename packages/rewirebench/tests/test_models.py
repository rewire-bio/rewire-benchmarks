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


def test_options_have_the_declared_json_type():
    seeded = models.get_model("seeded-random")
    for wrong in ({"a": 1}, "3", 3.5, True, None):
        with pytest.raises(ValueError, match="option 'seed' must be integer"):
            models.check_options(seeded, {"seed": wrong})
    prior = models.get_model("training-prior")
    assert models.check_options(prior, {"multiclass": True}) == {"multiclass": True}
    for wrong in (1, "true"):
        with pytest.raises(ValueError, match="must be boolean"):
            models.check_options(prior, {"multiclass": wrong})
    esm = models.get_model("esm2-embeddings")
    with pytest.raises(ValueError, match="option 'checkpoint' must be text"):
        models.check_options(esm, {"checkpoint": ["x.pt"]})
    for spec in models.REGISTRY.values():
        described = spec.describe()
        # Every declared type belongs to a declared option.
        assert set(spec.option_types or {}) <= set(spec.options or {}), spec.id
        assert set(described["option_types"]) == set(described["options"])


@pytest.mark.parametrize("device", ["mps", "cuda", "cuda:0"])
def test_a_local_recipe_cannot_ask_a_model_for_an_accelerator(device: str):
    """Hiding CUDA devices does not hide Apple MPS, so the device itself is refused."""
    step = (*STEP, "--model", "esm2-embeddings", "--adapter-options",
            json.dumps({"checkpoint": "{inputs}/x.pt", "device": device}))
    with pytest.raises(recipes.RecipeError, match="device must be cpu"):
        parse(with_step(*step))
    cpu = (*STEP, "--model", "esm2-embeddings", "--adapter-options",
           '{"checkpoint": "{inputs}/x.pt", "device": "cpu"}')
    assert parse(with_step(*cpu)).backends.local is not None
    # The step rule alone allows it, because a spark recipe needs cuda in its image.
    argv = ["rewirebench", "run", *step]
    recipes.check_run_step(argv)
    with pytest.raises(ValueError, match="device must be cpu"):
        recipes.check_run_step(argv, cpu_only=True)


def test_the_runner_refuses_an_accelerator_wherever_cuda_is_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    started: list[Any] = []
    monkeypatch.setattr(receipts.subprocess, "Popen",
                        lambda *args, **kwargs: started.append(args))
    argv = ["rewirebench", "run", *STEP, "--model", "esm2-embeddings", "--adapter-options",
            '{"checkpoint": "{inputs}/x.pt", "device": "mps"}']
    with pytest.raises(ValueError, match="device must be cpu"):
        receipts._execute(argv, cwd=tmp_path, log=tmp_path / "step", timeout=5)
    assert started == []


@pytest.mark.parametrize("value", ["0", "-5", "many"])
def test_a_batch_size_is_a_positive_integer(value: str):
    with pytest.raises(recipes.RecipeError, match="batch-size"):
        parse(with_step(*STEP, "--model", "train-mean", "--batch-size", value))


def test_run_refuses_a_model_on_data_of_a_protocol_it_is_not_registered_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    source = Path(__file__).resolve().parents[3] / "workflows/benchmark/examples/smoke.csv"
    cli.main(["prepare", "flip2-fitness-v1", "--source", str(source),
              "--output", str(tmp_path / "prepared"), "--options",
              '{"dataset": "flip2-amylase-one-to-many", "allow_unverified": true, "limit": 3}'])
    capsys.readouterr()
    # seeded-random is registered for ProteinGym and DART, not for FLIP2.
    assert "flip2-fitness-v1" not in models.get_model("seeded-random").protocols
    with pytest.raises(SystemExit) as stop:
        cli.main(["run", "--prepared", str(tmp_path / "prepared"),
                  "--output", str(tmp_path / "run"), "--model", "seeded-random"])
    assert stop.value.code == 2
    assert "not registered for protocol flip2-fitness-v1" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
    # Your own adapter is still yours to point at any data.
    cli.main(["run", "--prepared", str(tmp_path / "prepared"), "--output", str(tmp_path / "own"),
              "--adapter", "rewirebench.adapters.sequence:SeededRandomScore"])
    assert (tmp_path / "own/report.json").is_file()


BASELINES = ("--prepared", "{work}/prepared", "--output", "{output}/baselines")


@pytest.mark.parametrize("arguments", [
    # Named in the options, selected, both, abbreviated, and under another baseline's name.
    (*BASELINES, "--baseline-options",
     '{"dart-h12core-fimo-hit-count-v1": {"fimo": "/bin/sh", "motifs": "m"}}'),
    (*BASELINES, "--baseline", "dart-h12core-fimo-hit-count-v1"),
    (*BASELINES, "--baseline", "training-mean-v1", "--baseline",
     "dart-h12core-fimo-hit-count-v1", "--baseline-options",
     '{"dart-h12core-fimo-hit-count-v1": {"fimo": "/bin/sh"}}'),
    (*BASELINES, "--baseline-o", '{"dart-h12core-fimo-hit-count-v1": {"fimo": "/bin/sh"}}'),
    (*BASELINES, "--baseline-options=" + '{"training-mean-v1": {"fimo": "/bin/sh"}}'),
])
def test_a_recipe_cannot_start_a_program_through_a_baseline(arguments: tuple[str, ...]):
    data = model_recipe_data()
    data["commands"][1]["argv"] = ["rewirebench", "run-baselines", *arguments]
    with pytest.raises(recipes.RecipeError, match="starts another program"):
        parse(data)
    with pytest.raises(ValueError, match="starts another program"):
        receipts._program(["rewirebench", "run-baselines", *arguments])
    # The baseline itself is still there for a person at the command line.
    from rewirebench import baselines
    assert baselines.PROGRAM_BASELINES == ("dart-h12core-fimo-hit-count-v1",)
    assert baselines.PROGRAM_OPTION in models.get_model("fimo-h12core-hit-count").options


def test_other_baseline_steps_are_still_accepted():
    data = model_recipe_data()
    data["commands"][1]["argv"] = [
        "rewirebench", "run-baselines", *BASELINES, "--baseline", "training-mean-v1"]
    assert parse(data).commands[1].argv[1] == "run-baselines"
    data["commands"][1]["argv"] += ["--baseline-options", "{not json"]
    with pytest.raises(recipes.RecipeError, match="not JSON"):
        parse(data)


def test_the_published_json_schema_carries_the_shape_of_the_step_rules():
    """Someone who validates with recipe.schema.json alone gets the program, the
    subcommand and the --model rule. The registry checks need rewirebench itself."""
    import jsonschema

    root = Path(__file__).resolve().parents[3]
    schema = json.loads((root / "recipes/recipe.schema.json").read_text())
    assert schema == json.loads(json.dumps(recipes.json_schema()))
    validator = jsonschema.Draft202012Validator(schema)

    def errors(*argv: str) -> list[str]:
        data = model_recipe_data()
        data["commands"][1]["argv"] = list(argv)
        return [error.message for error in validator.iter_errors(data)]

    assert errors("rewirebench", "run", *STEP, "--model", "train-mean") == []
    assert errors("rewirebench", "run", *STEP, "--model=train-mean") == []
    assert errors("rewirebench", "prepare", "flip2-fitness-v1") == []
    for bad in (("rewirebench", "run", *STEP, "--adapter", "subprocess:Popen"),
                ("rewirebench", "run", *STEP, "--adapter=subprocess:Popen"),
                ("rewirebench", "run", *STEP, "--model", "train-mean", "--adapter", "x:y"),
                ("rewirebench", "run", *STEP),
                ("rewirebench", "submit", "--bundle", "x"),
                ("sh", "run", "--model", "train-mean"),
                ("runpodctl", "create", "pod")):
        assert errors(*bad), bad
        with pytest.raises(recipes.RecipeError):
            parse({**model_recipe_data(),
                   "commands": [{"name": "step", "argv": list(bad)}]})
    # Every recipe in the repository passes the published schema.
    import yaml
    for path in sorted((root / "recipes").rglob("*.yaml")):
        assert list(validator.iter_errors(yaml.safe_load(path.read_text()))) == [], path


def test_an_adhoc_receipt_records_the_model_id_and_no_local_value():
    recorded = receipts.recorded_argv([
        "run", "--prepared", "/home/me/prepared", "--output=/home/me/out", "--model",
        "train-mean", "--model-name", "My private model", "--adapter-options",
        '{"checkpoint": "/home/me/x.pt"}', "--batch-size", "8", "--allow-partial"])
    assert recorded == [
        "run", "--prepared", receipts.LOCAL_VALUE, "--output", "--model", "train-mean",
        "--model-name", receipts.LOCAL_VALUE, "--adapter-options", receipts.LOCAL_VALUE,
        "--batch-size", "8", "--allow-partial"]
    assert receipts.recorded_argv(["run", "--model=esm2-embeddings", "--batch-size=4"]) == [
        "run", "--model=esm2-embeddings", "--batch-size=4"]
    # An adapter path is the caller's own code and stays out.
    assert receipts.recorded_argv(["run", "--adapter", "mine:Model", "positional"]) == [
        "run", "--adapter", receipts.LOCAL_VALUE, receipts.LOCAL_VALUE]
    assert "/home/me" not in json.dumps(recorded)
