"""The benchmarks MCP server: ``rewirebench mcp``, on stdio, on the user's machine.

It lets an agent list protocols, recipes and models, prepare data, plan a run, run it on
the local backend, and read and verify the receipt. It wraps the command line: every
command that prepares, runs or scores is started as ``python -m rewirebench.cli ...`` in
its own process, with ``started_by`` set to ``agent`` and CUDA devices hidden.

What it will not do:

- start anything on the ``spark`` backend (it returns the dry-run plan);
- start a paid run. ``run`` on ``runpod`` refuses without a ``confirmed_amount`` that
  equals the plan's quote, and refuses with one too, because no launch is built;
- read a cloud API key, or send a result anywhere.

It is not a sandbox. It runs with the user's rights, like the command line it wraps.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP

from rewirebench import backends, models, receipts, recipes, sdk
from rewirebench.backends import runpod

SERVER_NAME = "rewirebench"
TOOLS = (
    "list_protocols", "describe_protocol", "list_recipes", "describe_recipe", "list_models",
    "prepare", "plan_run", "run", "run_status", "evaluate", "get_receipt", "verify_receipt",
)
INSTRUCTIONS = (
    "Runs biological model benchmarks with rewirebench on this machine. Start with "
    "list_recipes or list_protocols. Call plan_run before run: it returns the hardware "
    "needs, the expected wall time, the price or 'owned hardware', and the exact command. "
    "run starts work on the local backend only and returns at once; poll run_status, then "
    "call get_receipt and verify_receipt. Nothing here starts a GPU or paid run. A paid "
    "backend is refused unless a person has confirmed the plan's amount."
)
LOG_TAIL_BYTES = 4000
NO_RECIPES = (
    "No recipes directory found. Start the server from a rewire-benchmarks checkout, or "
    f"pass --recipes-dir or set {recipes.RECIPES_DIR_VARIABLE}."
)


@dataclass
class Run:
    """One started command. Its logs live in the server's state folder, never in the run
    folder, which holds only what the receipt covers."""

    id: str
    kind: str
    argv: list[str]
    output: Path
    stdout: Path
    stderr: Path
    process: subprocess.Popen[bytes]
    started_at: datetime.datetime
    recipe_id: str | None = None
    plan_sha256: str | None = None
    finished_at: datetime.datetime | None = None


@dataclass
class State:
    recipes_dir: Path | None
    folder: Path
    runs: dict[str, Run] = field(default_factory=dict)


def child_environment() -> dict[str, str]:
    """Environment of every command the server starts: an agent started it, on CPU."""
    return {**os.environ, receipts.STARTED_BY_VARIABLE: "agent", "CUDA_VISIBLE_DEVICES": ""}


def child_argv(arguments: list[str]) -> list[str]:
    """This installation's command line, so a tool runs the code that serves it."""
    return [sys.executable, "-m", "rewirebench.cli", *arguments]


def _tail(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-LOG_TAIL_BYTES:].decode("utf-8", "replace").strip()


def _command(state: State, arguments: list[str]) -> Any:
    """Run one command to completion and return the JSON it printed."""
    token = uuid.uuid4().hex[:12]
    out, err = state.folder / f"{token}.stdout.txt", state.folder / f"{token}.stderr.txt"
    with out.open("xb") as stdout, err.open("xb") as stderr:
        done = subprocess.run(
            child_argv(arguments), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            env=child_environment(), check=False,
        )
    if done.returncode != 0:
        raise ValueError(f"rewirebench {arguments[0]} failed with exit code "
                         f"{done.returncode}: {_tail(err) or _tail(out)}")
    return json.loads(out.read_text())


def _recipes_dir(state: State) -> Path:
    if state.recipes_dir is None:
        raise ValueError(NO_RECIPES)
    return state.recipes_dir


def _loaded(state: State, recipe: str) -> recipes.LoadedRecipe:
    return recipes.get_recipe(recipe, _recipes_dir(state))


def _new_output(output: str) -> Path:
    folder = Path(output).expanduser()
    if not folder.is_absolute():
        raise ValueError(f"Give output as an absolute path: {output!r}")
    if folder.exists():
        raise ValueError(f"Output folder already exists: {folder}")
    return folder


def _plan(state: State, recipe: str, backend: str, output: str | None,
          inputs: str | None, kg_release_id: str | None) -> backends.Plan:
    if backend not in backends.BACKENDS:
        raise ValueError(f"No {backend} backend. Choose one of {', '.join(backends.BACKENDS)}")
    loaded = _loaded(state, recipe)
    return backends.get_backend(backend).plan(
        loaded, recipes_dir=_recipes_dir(state), output=output, inputs=inputs,
        started_by="agent", kg_release_id=kg_release_id,
    )


def _start(state: State, *, kind: str, arguments: list[str], output: Path,
           recipe_id: str | None = None, plan_sha256: str | None = None) -> Run:
    running = [run.id for run in state.runs.values() if run.process.poll() is None]
    if running:
        raise ValueError(f"Run {running[0]} is still going. One run at a time; poll "
                         "run_status until it finishes.")
    identifier = f"run-{uuid.uuid4().hex[:12]}"
    stdout = state.folder / f"{identifier}.stdout.txt"
    stderr = state.folder / f"{identifier}.stderr.txt"
    argv = child_argv(arguments)
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=child_environment(),
        )
    run = Run(id=identifier, kind=kind, argv=argv, output=output, stdout=stdout,
              stderr=stderr, process=process, recipe_id=recipe_id, plan_sha256=plan_sha256,
              started_at=datetime.datetime.now(datetime.UTC))
    state.runs[identifier] = run
    return run


def _status(run: Run) -> dict[str, Any]:
    exit_code = run.process.poll()
    if exit_code is not None and run.finished_at is None:
        run.finished_at = datetime.datetime.now(datetime.UTC)
    receipt_path = run.output / receipts.RECEIPT_FILENAME
    status: dict[str, Any] = {
        "run_id": run.id,
        "kind": run.kind,
        "recipe_id": run.recipe_id,
        "plan_sha256": run.plan_sha256,
        "state": "running" if exit_code is None else "finished",
        "exit_code": exit_code,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "output": str(run.output),
        "command": run.argv,
        "receipt": None,
        "checks_passed": None,
        "result": None,
        "stderr_tail": None,
    }
    if exit_code is None:
        return status
    if receipt_path.is_file():
        status["receipt"] = str(receipt_path)
        try:
            status["checks_passed"] = receipts.load_receipt(run.output).checks_passed
        except ValueError as error:
            status["stderr_tail"] = f"The receipt cannot be read: {error}"
    try:
        status["result"] = json.loads(run.stdout.read_text())
    except (OSError, ValueError):
        status["result"] = None
    if exit_code != 0:
        status["stderr_tail"] = status["stderr_tail"] or _tail(run.stderr)
    return status


def _receipt_folder(state: State, run_id: str | None, folder: str | None) -> Path:
    if (run_id is None) == (folder is None):
        raise ValueError("Give exactly one of run_id or folder")
    if run_id is not None:
        if run_id not in state.runs:
            raise ValueError(f"Unknown run: {run_id}. Runs are known for the life of the "
                             "server; pass the run folder as folder instead.")
        return state.runs[run_id].output
    assert folder is not None
    return Path(folder).expanduser()


def build_server(recipes_dir: str | Path | None = None,
                 state_dir: str | Path | None = None) -> FastMCP:
    """The server with its twelve tools. ``state_dir`` holds logs of started commands."""
    state = State(
        recipes_dir=recipes.find_recipes_dir(recipes_dir),
        folder=Path(state_dir) if state_dir else Path(tempfile.mkdtemp(prefix="rewirebench-mcp-")),
    )
    state.folder.mkdir(parents=True, exist_ok=True)
    server = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)

    @server.tool()
    def list_protocols() -> dict[str, Any]:
        """List the benchmark protocols rewirebench implements, with the recipes for each."""
        by_protocol: dict[str, list[str]] = {}
        if state.recipes_dir is not None:
            for item in recipes.list_recipes(state.recipes_dir)[0]:
                if item.recipe.rewirebench_protocol:
                    by_protocol.setdefault(item.recipe.rewirebench_protocol, []).append(
                        item.recipe.id)
        return {"protocols": [
            {"id": protocol, "recipes": by_protocol.get(protocol, []),
             "models": [spec.id for spec in models.REGISTRY.values()
                        if protocol in spec.protocols]}
            for protocol in sdk.PROTOCOLS
        ]}

    @server.tool()
    def describe_protocol(protocol: str) -> dict[str, Any]:
        """Describe one protocol: inputs, datasets, metrics, execution rules, reference
        baselines and the registered models it accepts. Nothing is downloaded or run."""
        from rewirebench.baselines import describe_baselines

        description = sdk.describe(protocol)
        description["baselines"] = describe_baselines(protocol)
        description["models"] = models.list_models(protocol)
        return json.loads(json.dumps(description, default=str))

    @server.tool()
    def list_recipes() -> dict[str, Any]:
        """List the run recipes with their SHA-256, scope and backends. A file that is not
        a valid recipe is reported under invalid."""
        if state.recipes_dir is None:
            return {"recipes_dir": None, "recipes": [], "invalid": [], "note": NO_RECIPES}
        valid, invalid = recipes.list_recipes(state.recipes_dir)
        return {"recipes_dir": str(state.recipes_dir),
                "recipes": [recipes.summary(item, state.recipes_dir) for item in valid],
                "invalid": invalid}

    @server.tool()
    def describe_recipe(recipe: str) -> dict[str, Any]:
        """Return one recipe in full: pinned inputs, environment, commands, expected
        output, estimate and backends."""
        loaded = _loaded(state, recipe)
        return {**recipes.summary(loaded, _recipes_dir(state)),
                "recipe": loaded.recipe.model_dump(mode="json", by_alias=True)}

    @server.tool()
    def list_models(protocol: str | None = None) -> dict[str, Any]:
        """List the registered model adapters, or those one protocol accepts. Each entry
        gives its interface (predict, embed, fit), options and install extra."""
        return {"models": models.list_models(protocol)}

    @server.tool()
    async def prepare(protocol: str, source: str, output: str,
                      options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate local upstream files and prepare a protocol's data into a new folder.
        source is a file or folder already on this machine; nothing is downloaded."""
        if protocol not in sdk.PROTOCOLS:
            raise ValueError(f"Unknown protocol: {protocol}")
        folder = _new_output(output)
        arguments = ["prepare", protocol, "--source", source, "--output", str(folder),
                     "--options", json.dumps(options or {})]
        return await anyio.to_thread.run_sync(_command, state, arguments)

    @server.tool()
    def plan_run(recipe: str, backend: str = "local", output: str | None = None,
                 inputs: str | None = None, kg_release_id: str | None = None) -> dict[str, Any]:
        """Say what running a recipe on a backend needs, without starting or contacting
        anything: hardware, expected wall time, the price quote or 'owned hardware', the
        exact command and a plan_sha256. Backends: local, spark, runpod."""
        return _plan(state, recipe, backend, output, inputs, kg_release_id).model_dump(mode="json")

    @server.tool()
    def run(
        output: str,
        recipe: str | None = None,
        backend: str = "local",
        inputs: str | None = None,
        kg_release_id: str | None = None,
        plan_sha256: str | None = None,
        confirmed_amount: float | str | None = None,
        confirmed_currency: str | None = None,
        prepared: str | None = None,
        model: str | None = None,
        model_options: dict[str, Any] | None = None,
        model_name: str | None = None,
        training_overlap: str | None = None,
        batch_size: int = 32,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        """Start a run on the local backend and return its run_id at once. Poll run_status.

        Give either recipe (a recipe run, which writes a receipt checked against the
        recipe), or prepared and model (a registered model on prepared data). output is a
        new absolute folder. If plan_sha256 is given it must equal the current plan.

        The spark backend is not started from here; the reply is its dry-run plan. A paid
        backend is refused unless confirmed_amount and confirmed_currency equal the plan's
        quote, and no paid launch is built yet, so it is refused then as well.
        """
        folder = _new_output(output)
        if (recipe is None) == (prepared is None and model is None):
            raise ValueError("Give either recipe, or prepared and model")
        if recipe is None:
            if backend != "local":
                raise ValueError("A model run outside a recipe is local only")
            if prepared is None or model is None:
                raise ValueError("A model run needs both prepared and model")
            spec = models.get_model(model)
            options = models.check_options(spec, model_options or {})
            if options.get("device", "cpu") != "cpu":
                raise ValueError("The local backend is CPU only: device must be cpu")
            arguments = ["run", "--prepared", prepared, "--output", str(folder),
                         "--model", spec.id, "--adapter-options", json.dumps(options),
                         "--batch-size", str(batch_size)]
            if model_name is not None:
                arguments += ["--model-name", model_name]
            if training_overlap is not None:
                arguments += ["--training-overlap", training_overlap]
            if allow_partial:
                arguments.append("--allow-partial")
            started = _start(state, kind="model_run", arguments=arguments, output=folder)
            return _status(started)
        plan = _plan(state, recipe, backend, str(folder), inputs, kg_release_id)
        if plan_sha256 is not None and plan_sha256 != plan.plan_sha256:
            raise ValueError(
                f"plan_sha256 does not match. The plan for these arguments is "
                f"{plan.plan_sha256}; call plan_run again and review it."
            )
        if backend in backends.PAID_BACKENDS:
            # Raises in every case: without an amount, with a wrong one, and with the
            # right one, because nothing in this version launches a paid run.
            runpod.RunpodBackend().launch(
                plan, confirmed_amount=confirmed_amount, confirmed_currency=confirmed_currency)
            raise ValueError(runpod.NOT_BUILT)
        if backend != "local":
            return {"started": False, "dry_run": True, "plan": plan.model_dump(mode="json"),
                    "note": "The MCP server does not start runs on the spark backend. A "
                            "person can run `rewirebench recipe run ... --backend spark "
                            "--execute` at a terminal."}
        if not plan.runnable:
            raise ValueError(plan.not_runnable_reason or "This plan cannot run")
        started = _start(state, kind="recipe_run", arguments=plan.command[1:], output=folder,
                         recipe_id=recipe, plan_sha256=plan.plan_sha256)
        return _status(started)

    @server.tool()
    def run_status(run_id: str) -> dict[str, Any]:
        """Report a started run: running or finished, exit code, the receipt path and
        whether its checks passed. Exit code 1 with a receipt means a check failed."""
        if run_id not in state.runs:
            raise ValueError(f"Unknown run: {run_id}")
        return _status(state.runs[run_id])

    @server.tool()
    async def evaluate(
        prepared: str,
        output: str,
        predictions: str | None = None,
        embeddings: str | None = None,
        model_name: str | None = None,
        training_overlap: str | None = None,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        """Score a predictions file (or an embeddings file) against prepared data with the
        protocol's own metrics, into a new folder. No model code runs."""
        if (predictions is None) == (embeddings is None):
            raise ValueError("Give exactly one of predictions or embeddings")
        folder = _new_output(output)
        arguments = ["evaluate", "--prepared", prepared, "--output", str(folder)]
        arguments += (["--predictions", predictions] if predictions is not None
                      else ["--embeddings", str(embeddings)])
        if model_name is not None:
            arguments += ["--model-name", model_name]
        if training_overlap is not None:
            arguments += ["--training-overlap", training_overlap]
        if allow_partial:
            arguments.append("--allow-partial")
        return await anyio.to_thread.run_sync(_command, state, arguments)

    @server.tool()
    def get_receipt(run_id: str | None = None, folder: str | None = None) -> dict[str, Any]:
        """Return receipt.json of a run, by run_id or by run folder."""
        location = _receipt_folder(state, run_id, folder)
        if not (receipts._folder(location) / receipts.RECEIPT_FILENAME).is_file():
            raise ValueError(f"No receipt.json in {location}. A run that is still going, or "
                             "that stopped before writing one, has no receipt.")
        return receipts.load_receipt(location).model_dump(mode="json")

    @server.tool()
    async def verify_receipt(run_id: str | None = None, folder: str | None = None,
                             inputs: str | None = None, strict: bool = False) -> dict[str, Any]:
        """Recompute a receipt folder's digests and the recipe's checks. passed is false on
        any mismatch; with strict it is also false when a check could not be made."""
        location = _receipt_folder(state, run_id, folder)

        def check() -> dict[str, Any]:
            result = receipts.verify(location, inputs=inputs, recipes_dir=state.recipes_dir)
            return {**result, "strict": strict,
                    "passed": result["complete" if strict else "ok"]}

        return await anyio.to_thread.run_sync(check)

    return server


def serve(recipes_dir: str | Path | None = None) -> None:
    """Serve on stdio until the client closes the stream."""
    build_server(recipes_dir).run("stdio")
