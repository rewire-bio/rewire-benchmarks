"""The benchmarks MCP server: ``rewirebench mcp``, on stdio, on the user's machine.

It lets an agent list protocols, recipes and models, prepare data, plan a run, run it on
the local backend, stop it, and read and verify the receipt. It wraps the command line:
every command that prepares, runs or scores is started as ``python -m rewirebench.cli
...`` in its own process, with ``started_by`` set to ``agent`` and CUDA devices hidden.

What it will not do:

- start anything on the ``spark`` backend (it returns the dry-run plan);
- start a paid run. ``run`` on ``runpod`` refuses without the plan's ``plan_sha256`` and
  a ``confirmed_amount`` that equals the plan's quote, and refuses with them too,
  because no launch is built;
- start a program other than rewirebench: a model that starts an executable named in
  its options is refused, as it is in a recipe;
- accept an argument a tool does not declare. A misspelt name is an error, not a default;
- read a cloud API key, or send a result anywhere.

A started run has a time limit, can be stopped with ``cancel_run``, and is stopped when
the server exits, so closing the client leaves nothing running. A server that is killed
outright (SIGKILL) cannot do that.

It is not a sandbox. It runs with the user's rights, like the command line it wraps.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import Tool
from pydantic import Field

from rewirebench import backends, models, receipts, recipes, sdk
from rewirebench.backends import runpod

SERVER_NAME = "rewirebench"
TOOLS = (
    "list_protocols", "describe_protocol", "list_recipes", "describe_recipe", "list_models",
    "prepare", "plan_run", "run", "run_status", "cancel_run", "evaluate", "get_receipt",
    "verify_receipt",
)
INSTRUCTIONS = (
    "Runs biological model benchmarks with rewirebench on this machine. Start with "
    "list_recipes or list_protocols. Call plan_run before run: it returns the hardware "
    "needs, the expected wall time, the price or 'owned hardware', the exact command and a "
    "plan_sha256, which a recipe run requires. Show the person the plan before run. "
    "run starts work on the local backend only and returns at once; poll run_status, then "
    "call get_receipt and verify_receipt. cancel_run stops a run. Nothing here starts a GPU "
    "or paid run, or a program other than rewirebench. A paid backend needs an amount that "
    "a person stated after seeing the quote: ask them, and never copy the amount from the "
    "plan yourself. No paid launch is built, so it is refused in every case. "
    "verify_receipt shows that a run folder is consistent with its receipt and recipe. "
    "Nothing is signed, so it does not show where the folder came from, and the "
    "predictions are not scored again."
)
LOG_TAIL_BYTES = 4000
# A model run outside a recipe has no step timeouts of its own, so the server sets one.
DEFAULT_RUN_SECONDS = 3600.0
DEFAULT_BATCH_SIZE = 32
MAX_RUN_SECONDS = 24 * 3600.0
# On top of a recipe's own step timeouts: time to fetch its inputs.
RECIPE_MARGIN_SECONDS = 3600.0
STOP_GRACE_SECONDS = 5.0
# When the server itself is closing, its client may kill it within a couple of seconds.
CLOSE_GRACE_SECONDS = 1.0
VERIFY_NOTE = (
    "Verification shows the folder is consistent with its receipt and, for a recipe run, "
    "with the recipe. Nothing is signed: hardware, times and who started the run are as "
    "trustworthy as the place the folder came from, and a metric the recipe does not "
    "check can be changed together with its digests. Checked values are read again from "
    "the output files; predictions are not scored again, because the prepared data is not "
    "in the folder. The commits check reads the commit the receipt states and does not "
    "look it up."
)
# A release ID is recorded in the receipt as the caller gave it, so it is held to the
# shape of an identifier: no spaces, no free text.
RELEASE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
# For a protocol whose own description is only its capabilities.
BARE_PROTOCOL_NOTE = (
    "This protocol describes only its capabilities here. Its inputs, sizes and licence are "
    "in a recipe for it (describe_recipe), where one exists, and in its documentation."
)

Backend = Literal["local", "spark", "runpod"]
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
    time_limit_seconds: float
    recipe_id: str | None = None
    plan_sha256: str | None = None
    finished_at: datetime.datetime | None = None
    stopped: str | None = None
    timer: threading.Timer | None = None


@dataclass
class State:
    recipes_dir: Path | None
    folder: Path
    owns_folder: bool = False
    runs: dict[str, Run] = field(default_factory=dict)

    def close(self) -> None:
        """Stop every run still going and remove the log folder the server made."""
        for run in list(self.runs.values()):
            stop_run(run, "the server closed", grace=CLOSE_GRACE_SECONDS)
        if self.owns_folder:
            shutil.rmtree(self.folder, ignore_errors=True)


def _signal(run: Run, signum: int) -> None:
    """Signal the run's process group: the command and anything it started itself."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        if hasattr(os, "killpg"):
            os.killpg(run.process.pid, signum)
        elif signum == signal.SIGTERM:
            run.process.terminate()
        else:
            run.process.kill()


def stop_run(run: Run, reason: str, *, grace: float = STOP_GRACE_SECONDS) -> bool:
    """Stop a run that is still going. Returns whether it was.

    SIGTERM first: a recipe run then stops its current step, which has its own session,
    before it exits. Whatever is left after a short wait is killed.
    """
    if run.timer is not None:
        run.timer.cancel()
    if run.process.poll() is not None:
        return False
    run.stopped = run.stopped or reason
    _signal(run, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        run.process.wait(timeout=grace)
    _signal(run, getattr(signal, "SIGKILL", signal.SIGTERM))
    with contextlib.suppress(subprocess.TimeoutExpired):
        run.process.wait(timeout=grace)
    return True


class RewirebenchServer(FastMCP):
    """FastMCP that refuses an argument a tool does not declare.

    FastMCP drops unknown arguments, so ``bakend="runpod"`` would run on the default
    backend and a misspelt ``confirmed_amount`` would be read as absent.
    """

    state: State

    async def list_tools(self) -> list[Tool]:
        return [
            tool.model_copy(update={
                "inputSchema": {**tool.inputSchema, "additionalProperties": False}})
            for tool in await super().list_tools()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        declared = {tool.name: set(tool.inputSchema.get("properties", {}))
                    for tool in await super().list_tools()}
        unknown = sorted(set(arguments or {}) - declared.get(name, set()))
        if name in declared and unknown:
            raise ToolError(
                f"{name} has no argument {unknown[0]!r}. Its arguments are: "
                f"{', '.join(sorted(declared[name])) or 'none'}. Nothing was started."
            )
        return await super().call_tool(name, arguments)

    def close(self) -> None:
        self.state.close()


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
           time_limit_seconds: float, recipe_id: str | None = None,
           plan_sha256: str | None = None) -> Run:
    running = [run.id for run in state.runs.values() if run.process.poll() is None]
    if running:
        raise ValueError(f"Run {running[0]} is still going. One run at a time; poll "
                         "run_status until it finishes.")
    identifier = f"run-{uuid.uuid4().hex[:12]}"
    stdout = state.folder / f"{identifier}.stdout.txt"
    stderr = state.folder / f"{identifier}.stderr.txt"
    argv = child_argv(arguments)
    with stdout.open("xb") as out, stderr.open("xb") as err:
        # Its own session, so the run and what it starts can be stopped as one group.
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=child_environment(),
            start_new_session=hasattr(os, "killpg"),
        )
    run = Run(id=identifier, kind=kind, argv=argv, output=output, stdout=stdout,
              stderr=stderr, process=process, recipe_id=recipe_id, plan_sha256=plan_sha256,
              time_limit_seconds=time_limit_seconds,
              started_at=datetime.datetime.now(datetime.UTC))
    run.timer = threading.Timer(
        time_limit_seconds, stop_run,
        args=(run, f"it passed its time limit of {time_limit_seconds:g} seconds"))
    run.timer.daemon = True
    run.timer.start()
    state.runs[identifier] = run
    return run


def _status(run: Run) -> dict[str, Any]:
    exit_code = run.process.poll()
    if exit_code is not None and run.finished_at is None:
        run.finished_at = datetime.datetime.now(datetime.UTC)
        if run.timer is not None:
            run.timer.cancel()
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
        "time_limit_seconds": run.time_limit_seconds,
        "stopped": run.stopped if exit_code is not None else None,
        "output": str(run.output),
        "command": run.argv,
        "logs": {"stdout": str(run.stdout), "stderr": str(run.stderr)},
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


def _arg(description: str, **rules: Any) -> Any:
    return Field(description=description, **rules)


RecipeArg = Annotated[str, _arg("Recipe ID, from list_recipes.")]
BackendArg = Annotated[Backend, _arg(
    "Where to run. local runs on this machine on CPU. spark and runpod are planned only.")]
NewFolder = Annotated[str, _arg("Absolute path of a folder that does not exist yet.")]
PreparedArg = Annotated[str, _arg("Folder written by prepare, or its prepared.json.")]
InputsArg = Annotated[str | None, _arg(
    "Folder that already holds the recipe's inputs. Default: fetch them and check each "
    "against its pinned SHA-256.")]
ReleaseArg = Annotated[str | None, _arg(
    "Knowledge graph release ID to record in the receipt, if the run came from a "
    "knowledge graph query. Recorded as given: it is not checked against the graph.",
    pattern=RELEASE_ID_PATTERN)]
ModelName = Annotated[str | None, _arg(
    "Free label for the model in the report. It does not choose what runs.")]
OverlapArg = Annotated[str | None, _arg(
    "What is known about overlap between the model's training data and this benchmark. "
    "Default: unreported.")]
PartialArg = Annotated[bool, _arg(
    "Score even when some inputs have no prediction. The report states the coverage.")]
RunId = Annotated[str, _arg("run_id returned by run.")]
RunIdOrNone = Annotated[str | None, _arg(
    "run_id returned by run. Give this or folder, not both.")]
FolderOrNone = Annotated[str | None, _arg(
    "A run folder holding receipt.json. Give this or run_id, not both.")]


def build_server(recipes_dir: str | Path | None = None,
                 state_dir: str | Path | None = None) -> RewirebenchServer:
    """The server with its tools. ``state_dir`` holds logs of started commands; without
    one the server makes a temporary folder and removes it when it closes."""
    state = State(
        recipes_dir=recipes.find_recipes_dir(recipes_dir),
        folder=Path(state_dir) if state_dir else Path(tempfile.mkdtemp(prefix="rewirebench-mcp-")),
        owns_folder=state_dir is None,
    )
    state.folder.mkdir(parents=True, exist_ok=True)
    server = RewirebenchServer(SERVER_NAME, instructions=INSTRUCTIONS)
    server.state = state

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
    def describe_protocol(
        protocol: Annotated[str, _arg("Protocol ID, from list_protocols.")],
    ) -> dict[str, Any]:
        """Describe one protocol: what it can do (capabilities), its reference baselines,
        the registered models it accepts and the recipes for it. Some protocols also give
        their datasets, metrics and data access; others give capabilities only, and the
        reply then says so in note. Inputs with sizes and licence are in describe_recipe.
        Nothing is downloaded or run."""
        from rewirebench.baselines import describe_baselines

        description = sdk.describe(protocol)
        if set(description) <= {"protocol_id", "capabilities"}:
            description["note"] = BARE_PROTOCOL_NOTE
        description["baselines"] = describe_baselines(protocol)
        description["models"] = models.list_models(protocol)
        description["recipes"] = (
            [item.recipe.id for item in recipes.list_recipes(state.recipes_dir)[0]
             if item.recipe.rewirebench_protocol == protocol]
            if state.recipes_dir is not None else [])
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
    def describe_recipe(recipe: RecipeArg) -> dict[str, Any]:
        """Return one recipe in full: pinned inputs, environment, commands, expected
        output, estimate and backends."""
        loaded = _loaded(state, recipe)
        return {**recipes.summary(loaded, _recipes_dir(state)),
                "recipe": loaded.recipe.model_dump(mode="json", by_alias=True)}

    @server.tool()
    def list_models(
        protocol: Annotated[str | None, _arg(
            "Protocol ID. Default: every registered model.")] = None,
    ) -> dict[str, Any]:
        """List the registered model adapters, or those registered for one protocol. run
        refuses a model on data of a protocol it is not registered for. Each entry gives
        its interface (predict, embed, fit), options with their types, and install extra.
        A model with runs_program true cannot be run from here."""
        return {"models": models.list_models(protocol)}

    @server.tool()
    async def prepare(
        protocol: Annotated[str, _arg("Protocol ID, from list_protocols.")],
        source: Annotated[str, _arg(
            "File or folder of upstream data already on this machine.")],
        output: NewFolder,
        options: Annotated[dict[str, Any] | None, _arg(
            "Protocol options, as describe_protocol lists them.")] = None,
    ) -> dict[str, Any]:
        """Validate local upstream files and prepare a protocol's data into a new folder.
        source is a file or folder already on this machine; nothing is downloaded."""
        if protocol not in sdk.PROTOCOLS:
            raise ValueError(f"Unknown protocol: {protocol}")
        folder = _new_output(output)
        arguments = ["prepare", protocol, "--source", source, "--output", str(folder),
                     "--options", json.dumps(options or {})]
        return await anyio.to_thread.run_sync(_command, state, arguments)

    @server.tool()
    def plan_run(
        recipe: RecipeArg,
        backend: BackendArg = "local",
        output: Annotated[str | None, _arg(
            "Absolute path of the new folder the run would write. Give the one you will "
            "pass to run, or its plan_sha256 will differ.")] = None,
        inputs: InputsArg = None,
        kg_release_id: ReleaseArg = None,
    ) -> dict[str, Any]:
        """Say what running a recipe on a backend needs, without starting or contacting
        anything: hardware, expected wall time, the price quote or 'owned hardware', the
        exact command and a plan_sha256. A runpod plan is for review only: its pod spec
        must not be sent to RunPod with another tool, because nothing in it stops the
        pod and the quote would not bound the cost."""
        return _plan(state, recipe, backend, output, inputs, kg_release_id).model_dump(mode="json")

    @server.tool()
    def run(
        output: NewFolder,
        recipe: Annotated[str | None, _arg(
            "Recipe ID for a recipe run. Give this, or prepared and model.")] = None,
        backend: BackendArg = "local",
        inputs: InputsArg = None,
        kg_release_id: ReleaseArg = None,
        plan_sha256: Annotated[str | None, _arg(
            "For a recipe run, required: plan_sha256 of the plan_run reply for the same "
            "arguments. The run is refused without it, or if the plan has changed.")] = None,
        confirmed_amount: Annotated[str | None, _arg(
            "Paid backends only. The amount a person stated after reading the plan's "
            "quote, as decimal text such as \"11.97\". Ask the person. Never copy it "
            "from the plan or from an error message.")] = None,
        confirmed_currency: Annotated[str | None, _arg(
            "Paid backends only. The currency the person stated, such as USD.")] = None,
        prepared: Annotated[str | None, _arg(
            "For a model run: folder written by prepare, or its prepared.json.")] = None,
        model: Annotated[str | None, _arg(
            "For a model run: a model ID from list_models. Never a Python path.")] = None,
        model_options: Annotated[dict[str, Any] | None, _arg(
            "For a model run: options the model declares in list_models, of the declared "
            "types. device must be cpu.")] = None,
        model_name: ModelName = None,
        training_overlap: OverlapArg = None,
        batch_size: Annotated[int | None, _arg(
            f"For a model run: inputs per model call. Default {DEFAULT_BATCH_SIZE}.",
            ge=1)] = None,
        allow_partial: Annotated[bool | None, _arg(
            "For a model run: score even when some inputs have no prediction. The report "
            "states the coverage. Default false.")] = None,
        timeout_seconds: Annotated[float | None, _arg(
            "For a model run: seconds after which the run is stopped. Default "
            f"{DEFAULT_RUN_SECONDS:g}. A recipe run uses the recipe's own step timeouts "
            "and refuses this argument.", gt=0, le=MAX_RUN_SECONDS,
        )] = None,
    ) -> dict[str, Any]:
        """Start a run on the local backend and return its run_id at once. Poll run_status;
        stop it with cancel_run. One run at a time.

        Give either recipe (a recipe run, which writes a receipt checked against the
        recipe), or prepared and model (a registered model on prepared data of a protocol
        the model is registered for). Only rewirebench is started: a model that starts
        another program is refused. An argument this tool does not declare is an error,
        and so is an argument of the other kind of run.

        A recipe run on local or on a paid backend needs plan_sha256 from plan_run, so
        that what starts (the downloads included) is what the plan showed.

        The spark backend is not started from here; the reply is its dry-run plan. A paid
        backend is refused without plan_sha256, confirmed_amount and confirmed_currency
        that equal the plan, and no paid launch is built yet, so it is refused then as
        well. The amount is a statement that a person confirmed it, so ask the person.
        """
        folder = _new_output(output)
        if (recipe is None) == (prepared is None and model is None):
            raise ValueError("Give either recipe, or prepared and model")
        if recipe is None:
            if backend != "local":
                raise ValueError("A model run outside a recipe is local only")
            if prepared is None or model is None:
                raise ValueError("A model run needs both prepared and model")
            recipe_only = {"inputs": inputs, "kg_release_id": kg_release_id,
                           "plan_sha256": plan_sha256, "confirmed_amount": confirmed_amount,
                           "confirmed_currency": confirmed_currency}
            given = sorted(name for name, value in recipe_only.items() if value is not None)
            if given:
                raise ValueError(f"{given[0]} applies to a recipe run, not to a model run")
            spec = models.get_model(model)
            if spec.runs_program:
                raise ValueError(
                    f"Model {spec.id} starts another program, so it cannot run through the "
                    "MCP server. A person can run it at a terminal."
                )
            options = models.check_options(spec, model_options or {})
            models.check_cpu(spec, options)
            arguments = ["run", "--prepared", prepared, "--output", str(folder),
                         "--model", spec.id, "--adapter-options", json.dumps(options),
                         "--batch-size", str(batch_size or DEFAULT_BATCH_SIZE)]
            if model_name is not None:
                arguments += ["--model-name", model_name]
            if training_overlap is not None:
                arguments += ["--training-overlap", training_overlap]
            if allow_partial:
                arguments.append("--allow-partial")
            started = _start(state, kind="model_run", arguments=arguments, output=folder,
                             time_limit_seconds=timeout_seconds or DEFAULT_RUN_SECONDS)
            return _status(started)
        model_only = {"model_options": model_options, "model_name": model_name,
                      "training_overlap": training_overlap, "batch_size": batch_size,
                      "allow_partial": allow_partial, "timeout_seconds": timeout_seconds}
        given = sorted(name for name, value in model_only.items() if value is not None)
        if given:
            raise ValueError(f"{given[0]} applies to a model run, not to a recipe run. "
                             "Nothing was started.")
        plan = _plan(state, recipe, backend, str(folder), inputs, kg_release_id)
        if plan_sha256 is None and backend != "spark":
            if backend == "local" and not plan.runnable:
                raise ValueError(plan.not_runnable_reason or "This plan cannot run")
            raise ValueError(
                "A recipe run needs plan_sha256 from plan_run for the same arguments, so "
                "that what starts is the plan a person was shown. Nothing was started."
            )
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
        if confirmed_amount is not None or confirmed_currency is not None:
            raise ValueError(f"The {backend} backend is owned hardware; there is no amount "
                             "to confirm")
        if backend != "local":
            return {"started": False, "dry_run": True, "plan": plan.model_dump(mode="json"),
                    "note": "The MCP server does not start runs on the spark backend. A "
                            "person can run `rewirebench recipe run ... --backend spark "
                            "--execute` at a terminal."}
        if not plan.runnable:
            raise ValueError(plan.not_runnable_reason or "This plan cannot run")
        started = _start(state, kind="recipe_run", arguments=plan.command[1:], output=folder,
                         recipe_id=recipe, plan_sha256=plan.plan_sha256,
                         time_limit_seconds=plan.time_limit_seconds + RECIPE_MARGIN_SECONDS)
        return _status(started)

    @server.tool()
    def run_status(run_id: RunId) -> dict[str, Any]:
        """Report a started run: running or finished, exit code, the receipt path and
        whether its checks passed. Exit code 1 with a receipt means a check failed.
        stopped says why a run was ended early (cancelled, time limit, server closed)."""
        if run_id not in state.runs:
            raise ValueError(f"Unknown run: {run_id}")
        return _status(state.runs[run_id])

    @server.tool()
    async def cancel_run(run_id: RunId) -> dict[str, Any]:
        """Stop a run that is still going, with everything it started, and report its
        status. A run that has already finished is left as it is. A stopped run has no
        receipt; its output folder may hold partial files."""
        if run_id not in state.runs:
            raise ValueError(f"Unknown run: {run_id}")
        target = state.runs[run_id]
        cancelled = await anyio.to_thread.run_sync(stop_run, target, "cancel_run was called")
        return {**_status(target), "cancelled": cancelled}

    @server.tool()
    async def evaluate(
        prepared: PreparedArg,
        output: NewFolder,
        predictions: Annotated[str | None, _arg(
            "Predictions file. Give this or embeddings, not both.")] = None,
        embeddings: Annotated[str | None, _arg(
            "Embeddings file. Give this or predictions, not both.")] = None,
        model_name: ModelName = None,
        training_overlap: OverlapArg = None,
        allow_partial: PartialArg = False,
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
    def get_receipt(run_id: RunIdOrNone = None, folder: FolderOrNone = None) -> dict[str, Any]:
        """Return receipt.json of a run, by run_id or by run folder."""
        location = _receipt_folder(state, run_id, folder)
        if not (receipts._folder(location) / receipts.RECEIPT_FILENAME).is_file():
            raise ValueError(f"No receipt.json in {location}. A run that is still going, or "
                             "that stopped before writing one, has no receipt.")
        return receipts.load_receipt(location).model_dump(mode="json")

    @server.tool()
    async def verify_receipt(
        run_id: RunIdOrNone = None,
        folder: FolderOrNone = None,
        inputs: Annotated[str | None, _arg(
            "Folder holding the input files, to compare with the receipt's digests.")] = None,
        strict: Annotated[bool, _arg(
            "Also fail when a check could not be made. Passes only for a recipe run "
            "from a clean commit; a model run outside a recipe never passes it.")] = False,
    ) -> dict[str, Any]:
        """Check that a run folder is consistent with its receipt: recompute the digests
        and, for a recipe run, compare inputs and re-read the checked values against the
        recipe. passed is false on any mismatch; with strict it is also false when a
        check could not be made, and not_checked lists those. A model run outside a
        recipe has no recipe checks, so only its digests are verified. Nothing is signed:
        this does not prove where the folder came from or what hardware it ran on."""
        location = _receipt_folder(state, run_id, folder)

        def check() -> dict[str, Any]:
            result = receipts.verify(location, inputs=inputs, recipes_dir=state.recipes_dir)
            return {**result, "strict": strict,
                    "passed": result["complete" if strict else "ok"],
                    "not_checked": [item["check"] for item in result["results"]
                                    if item["status"] == "not_available"],
                    "note": VERIFY_NOTE}

        return await anyio.to_thread.run_sync(check)

    return server


def serve(recipes_dir: str | Path | None = None) -> None:
    """Serve on stdio until the client closes the stream, then stop any run still going.

    SIGTERM ends the server the same way. A server that is killed outright cannot stop
    its run.
    """
    server = build_server(recipes_dir)

    def terminate(signum: int, frame: Any) -> None:
        # A second signal must not cut the cleanup short. The process then leaves at
        # once: unwinding would wait for the thread that is reading standard input.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        server.close()
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        server.run("stdio")
    finally:
        server.close()
