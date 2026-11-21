"""The benchmarks MCP server, driven through an in-process MCP client.

One test also starts `rewirebench mcp` as a real process and speaks to it over stdio,
because a stray line on stdout breaks that transport and an in-process client cannot see it.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from rewirebench import cli, mcp_server, models, receipts, recipes, sdk

ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "recipes"
SMOKE = "flip2-synthetic-smoke"
SMOKE_INPUT = ROOT / "workflows/benchmark/examples/smoke.csv"
FLIP2_OPTIONS = {"dataset": "flip2-amylase-one-to-many", "allow_unverified": True, "limit": 3}


def call(server: FastMCP, tool: str, **arguments: Any) -> Any:
    """Call a tool through an MCP client session and return its JSON result."""

    async def go() -> Any:
        async with create_connected_server_and_client_session(server) as session:
            return await session.call_tool(tool, arguments)

    result = anyio.run(go)
    text = "\n".join(item.text for item in result.content if hasattr(item, "text"))
    if result.isError:
        raise RuntimeError(text)
    return json.loads(text)


def wait(server: FastMCP, run_id: str, seconds: float = 120) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = call(server, "run_status", run_id=run_id)
        if status["state"] == "finished":
            return status
        time.sleep(0.2)
    raise AssertionError(f"Run {run_id} did not finish: {status}")


@pytest.fixture
def server(tmp_path: Path) -> FastMCP:
    return mcp_server.build_server(RECIPES, state_dir=tmp_path / "state")


@pytest.fixture
def gpu_server(fixture_recipes: Path, tmp_path: Path) -> FastMCP:
    return mcp_server.build_server(fixture_recipes, state_dir=tmp_path / "state")


def test_the_server_offers_exactly_the_documented_tools(server: FastMCP):
    async def go() -> Any:
        async with create_connected_server_and_client_session(server) as session:
            return await session.list_tools()

    tools = anyio.run(go).tools
    assert sorted(tool.name for tool in tools) == sorted(mcp_server.TOOLS)
    assert len(mcp_server.TOOLS) == 13
    assert all(tool.description for tool in tools)
    for tool in tools:
        # Every argument says what it is, and no tool takes an argument it does not name.
        assert tool.inputSchema["additionalProperties"] is False, tool.name
        for name, schema in tool.inputSchema.get("properties", {}).items():
            assert schema.get("description"), f"{tool.name}.{name}"
    run = next(tool for tool in tools if tool.name == "run").inputSchema
    assert {"confirmed_amount", "confirmed_currency", "plan_sha256"} <= set(run["properties"])
    assert run["properties"]["backend"]["enum"] == ["local", "spark", "runpod"]
    assert "Ask the person" in run["properties"]["confirmed_amount"]["description"]
    # The amount is text: a JSON number or true is not an amount a person typed.
    assert {option.get("type") for option in run["properties"]["confirmed_amount"]["anyOf"]} \
        == {"string", "null"}


@pytest.mark.parametrize(("tool", "arguments", "unknown"), [
    ("plan_run", {"recipe": SMOKE, "bakend": "runpod"}, "bakend"),
    ("run", {"recipe": SMOKE, "output": "/tmp/unused", "bakend": "spark"}, "bakend"),
    ("run", {"recipe": SMOKE, "output": "/tmp/unused", "confirmed_ammount": "1"},
     "confirmed_ammount"),
    ("run", {"recipe": SMOKE, "output": "/tmp/unused", "plan_sha": "zz"}, "plan_sha"),
    ("list_protocols", {"nonsense": 1}, "nonsense"),
])
def test_a_misspelt_argument_is_an_error_and_never_a_default(
    server: FastMCP, tmp_path: Path, tool: str, arguments: dict[str, Any], unknown: str
):
    if "output" in arguments:
        arguments = {**arguments, "output": str(tmp_path / "out")}
    with pytest.raises(RuntimeError, match=f"{tool} has no argument '{unknown}'"):
        call(server, tool, **arguments)
    assert not (tmp_path / "out").exists()
    assert server.state.runs == {}


def test_protocols_recipes_and_models_can_be_listed_and_described(server: FastMCP):
    protocols = call(server, "list_protocols")["protocols"]
    assert [entry["id"] for entry in protocols] == list(sdk.PROTOCOLS)
    by_id = {entry["id"]: entry for entry in protocols}
    assert by_id["mfass-v2"]["recipes"] == ["mfass-v2-kmer-baseline"]
    assert "mfass-kmer-position" in by_id["mfass-v2"]["models"]
    described = call(server, "describe_protocol", protocol="mfass-v2")
    assert described["baselines"]["protocol_id"] == "mfass-v2"
    assert {entry["id"] for entry in described["models"]} == {"training-prior",
                                                              "mfass-kmer-position"}
    listed = call(server, "list_recipes")
    assert {entry["id"] for entry in listed["recipes"]} >= {SMOKE, "mfass-v2-kmer-baseline"}
    assert listed["invalid"] == [] and listed["recipes_dir"] == str(RECIPES)
    recipe = call(server, "describe_recipe", recipe="mfass-v2-kmer-baseline")
    assert recipe["sha256"] == recipes.get_recipe("mfass-v2-kmer-baseline", RECIPES).sha256
    assert recipe["recipe"]["inputs"][0]["sha256"].startswith("a637ca0e")
    models = call(server, "list_models", protocol="flip2-fitness-v1")["models"]
    assert "train-mean" in {entry["id"] for entry in models}
    assert len(call(server, "list_models")["models"]) > len(models)
    for tool, arguments in (("describe_protocol", {"protocol": "nope"}),
                            ("describe_recipe", {"recipe": "nope"}),
                            ("list_models", {"protocol": "nope"})):
        with pytest.raises(RuntimeError, match="Unknown|Unsupported"):
            call(server, tool, **arguments)


def test_a_recipe_runs_from_plan_to_verified_receipt_through_mcp_only(
    server: FastMCP, tmp_path: Path
):
    output = tmp_path / "runs/smoke"
    plan = call(server, "plan_run", recipe=SMOKE, backend="local", output=str(output))
    assert plan["backend"] == "local" and plan["runnable"]
    assert plan["quote"]["reason"] == "owned hardware" and not plan["quote"]["paid"]
    assert plan["hardware"]["accelerator"]["kind"] == "none"
    assert plan["expected_wall_seconds"] is None and plan["expected_wall_seconds_basis"]
    assert not output.exists(), "A plan starts nothing"

    with pytest.raises(RuntimeError, match="plan_sha256 does not match"):
        call(server, "run", recipe=SMOKE, output=str(output), plan_sha256="0" * 64)
    started = call(server, "run", recipe=SMOKE, backend="local", output=str(output),
                   plan_sha256=plan["plan_sha256"])
    assert started["kind"] == "recipe_run" and started["plan_sha256"] == plan["plan_sha256"]
    # What ran is the plan's command, with this installation's Python in front.
    assert started["command"] == [sys.executable, "-m", "rewirebench.cli", *plan["command"][1:]]
    with pytest.raises(RuntimeError, match="Output folder already exists"):
        wait(server, started["run_id"])
        call(server, "run", recipe=SMOKE, output=str(output))

    status = wait(server, started["run_id"])
    assert status["exit_code"] == 0 and status["checks_passed"] is True, status
    assert status["receipt"] == str(output / "receipt.json")
    assert status["result"]["recipe_id"] == SMOKE

    receipt = call(server, "get_receipt", run_id=started["run_id"])
    assert receipt["backend"] == "local" and receipt["started_by"] == "agent"
    assert receipt["recipe"]["id"] == SMOKE and receipt["checks_passed"]
    assert receipt["cost"]["paid_compute"] is False
    assert receipt["cost"]["reason"] == "owned hardware" and receipt["cost"]["amount"] is None
    assert receipt["hardware"]["architecture"] and receipt["wall_seconds"] > 0
    assert receipt == call(server, "get_receipt", folder=str(output))

    # The server keeps its own logs elsewhere, so polling leaves the evidence folder intact.
    call(server, "run_status", run_id=started["run_id"])
    verified = call(server, "verify_receipt", run_id=started["run_id"])
    assert verified["ok"] and verified["passed"], verified
    assert "Nothing is signed" in verified["note"]
    assert {entry["check"]: entry["status"] for entry in verified["results"]}["sha256sums"] \
        == "match"
    listed = {line.split("  ./")[1] for line in (output / "SHA256SUMS").read_text().splitlines()}
    present = {path.relative_to(output).as_posix() for path in output.rglob("*")
               if path.is_file() and path.name != "SHA256SUMS"}
    assert listed == present

    (output / "baselines/training-mean-v1/report.json").write_text("{}")
    changed = call(server, "verify_receipt", folder=str(output))
    assert not changed["ok"] and not changed["passed"]


def test_prepare_run_a_model_and_evaluate_through_mcp_only(server: FastMCP, tmp_path: Path):
    prepared = call(server, "prepare", protocol="flip2-fitness-v1", source=str(SMOKE_INPUT),
                    output=str(tmp_path / "prepared"), options=FLIP2_OPTIONS)
    assert prepared["protocol_id"] == "flip2-fitness-v1"
    assert Path(prepared["prepared"]).is_file()
    with pytest.raises(RuntimeError, match="already exists"):
        call(server, "prepare", protocol="flip2-fitness-v1", source=str(SMOKE_INPUT),
             output=str(tmp_path / "prepared"), options=FLIP2_OPTIONS)

    started = call(server, "run", prepared=str(tmp_path / "prepared"), model="train-mean",
                   output=str(tmp_path / "model-run"))
    assert started["kind"] == "model_run" and "--model" in started["command"]
    status = wait(server, started["run_id"])
    assert status["exit_code"] == 0 and status["result"]["metrics"]["n"] == 3, status
    receipt = call(server, "get_receipt", run_id=started["run_id"])
    # A run outside a recipe has no recipe pointer; it still says an agent started it.
    assert receipt["recipe"] is None and receipt["started_by"] == "agent"
    # The receipt names the registered model that ran; paths and options stay out of it.
    command = receipt["commands"][0]["argv"]
    assert command[command.index("--model") + 1] == "train-mean"
    assert command[command.index("--batch-size") + 1] == "32"
    assert command[command.index("--prepared") + 1] == receipts.LOCAL_VALUE
    assert command[command.index("--adapter-options") + 1] == receipts.LOCAL_VALUE
    # Its digests verify. It has no recipe, so the recipe checks are named as not made
    # and strict, which needs every check, does not pass.
    verified = call(server, "verify_receipt", run_id=started["run_id"])
    assert verified["ok"] and verified["passed"] and not verified["complete"]
    assert {"recipe", "checks", "metrics"} <= set(verified["not_checked"])
    strict = call(server, "verify_receipt", run_id=started["run_id"], strict=True)
    assert strict["ok"] and strict["strict"] and strict["passed"] is False

    # A model registered for other protocols is refused on this data, with no receipt.
    wrong = call(server, "run", prepared=str(tmp_path / "prepared"), model="seeded-random",
                 output=str(tmp_path / "wrong-model"))
    status = wait(server, wrong["run_id"])
    assert status["exit_code"] == 2 and status["receipt"] is None
    assert "not registered for protocol flip2-fitness-v1" in status["stderr_tail"]
    assert not (tmp_path / "wrong-model/receipt.json").exists()

    scored = call(server, "evaluate", prepared=str(tmp_path / "prepared"),
                  predictions=str(tmp_path / "model-run/predictions.json"),
                  output=str(tmp_path / "scored"), model_name="Training mean, rescored")
    assert scored["metrics"]["n"] == 3 and Path(scored["receipt"]).is_file()
    assert call(server, "get_receipt", folder=str(tmp_path / "scored"))["started_by"] == "agent"
    with pytest.raises(RuntimeError, match="exactly one of predictions or embeddings"):
        call(server, "evaluate", prepared=str(tmp_path / "prepared"),
             output=str(tmp_path / "again"))


def test_run_takes_a_registered_model_and_never_a_python_path(server: FastMCP, tmp_path: Path):
    arguments = {"prepared": str(tmp_path / "prepared"), "output": str(tmp_path / "out")}
    for model, options, message in (
        ("subprocess:Popen", None, "Unknown model"),
        ("rewirebench.adapters.sequence:TrainMean", None, "Unknown model"),
        ("seeded-random", {"args": ["runpodctl"]}, "has no option"),
        ("esm2-embeddings", {"checkpoint": "x.pt", "device": "cuda"}, "device must be cpu"),
        ("esm2-embeddings", {"checkpoint": "x.pt", "device": "mps"}, "device must be cpu"),
        ("seeded-random", {"seed": {"a": 1}}, "must be integer"),
        ("seeded-random", {"seed": True}, "must be integer"),
    ):
        with pytest.raises(RuntimeError, match=message):
            call(server, "run", model=model, model_options=options, **arguments)
    for extra, message in (({"batch_size": 0}, "greater than or equal to 1"),
                           ({"timeout_seconds": 0}, "greater than 0"),
                           ({"timeout_seconds": 10**9}, "less than or equal to"),
                           ({"plan_sha256": "0" * 64}, "applies to a recipe run"),
                           ({"confirmed_amount": "1"}, "applies to a recipe run")):
        with pytest.raises(RuntimeError, match=message):
            call(server, "run", model="train-mean", **arguments, **extra)
    for bad in ({"output": str(tmp_path / "out")},
                {"recipe": SMOKE, "model": "train-mean", "output": str(tmp_path / "out")},
                {"model": "train-mean", "output": str(tmp_path / "out")},
                {"recipe": SMOKE, "output": "relative/folder"}):
        with pytest.raises(RuntimeError, match="Give either|needs both|absolute path"):
            call(server, "run", **bad)
    assert not (tmp_path / "out").exists()
    assert mcp_server.child_environment()["CUDA_VISIBLE_DEVICES"] == ""
    assert mcp_server.child_environment()[receipts.STARTED_BY_VARIABLE] == "agent"
    assert server.state.runs == {}


def test_run_never_starts_a_program_named_by_the_caller(server: FastMCP, tmp_path: Path):
    """The FIMO model starts the executable named in its options. Not through MCP."""
    marker = tmp_path / "RAN"
    program = tmp_path / "program.sh"
    program.write_text(f"#!/bin/sh\ntouch {marker}\necho 5.5.9\n")
    program.chmod(0o755)
    listed = {entry["id"]: entry for entry in call(server, "list_models")["models"]}
    assert listed["fimo-h12core-hit-count"]["runs_program"] is True
    with pytest.raises(RuntimeError, match="starts another program"):
        call(server, "run", prepared=str(tmp_path / "missing"), output=str(tmp_path / "out"),
             model="fimo-h12core-hit-count",
             model_options={"fimo": str(program), "motifs": "x",
                            "fimo_sha256": hashlib.sha256(program.read_bytes()).hexdigest()})
    time.sleep(0.5)
    assert not marker.exists() and not (tmp_path / "out").exists()
    assert server.state.runs == {}
    # Every model that starts a program is refused, whichever it is.
    for spec in models.REGISTRY.values():
        if spec.runs_program:
            with pytest.raises(RuntimeError, match="starts another program"):
                call(server, "run", prepared="x", output=str(tmp_path / "out"), model=spec.id,
                     model_options=dict.fromkeys(spec.required_options, "x"))


SLOW = ("import subprocess, sys, time; "
        "child = subprocess.Popen(['sleep', '300']); "
        "open(sys.argv[1], 'w').write(f'{child.pid}'); time.sleep(300)")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A stopped child of this process stays in the table until it is reaped.
    done = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                          text=True, check=False)
    return done.returncode == 0 and not done.stdout.strip().startswith("Z")


def gone(pid: int, seconds: float = 15) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def slow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Every command the server starts hangs, with a child process of its own. Returns
    the file that receives that child's process ID."""
    record = tmp_path / "grandchild.pid"
    monkeypatch.setattr(mcp_server, "child_argv",
                        lambda arguments: [sys.executable, "-c", SLOW, str(record)])
    return record


def started_child(record: Path, seconds: float = 30) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if record.is_file() and record.read_text().strip():
            return int(record.read_text())
        time.sleep(0.05)
    raise AssertionError("The slow command did not start")


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="needs process groups")
def test_cancel_run_stops_the_run_and_what_it_started_and_frees_the_server(
    server: FastMCP, tmp_path: Path, slow: Path
):
    first = call(server, "run", prepared="p", model="train-mean", output=str(tmp_path / "one"))
    assert first["state"] == "running" and first["time_limit_seconds"] == 3600
    assert Path(first["logs"]["stderr"]).is_file()
    grandchild = started_child(slow)
    with pytest.raises(RuntimeError, match=f"Run {first['run_id']} is still going"):
        call(server, "run", prepared="p", model="train-mean", output=str(tmp_path / "two"))
    stopped = call(server, "cancel_run", run_id=first["run_id"])
    assert stopped["cancelled"] is True and stopped["state"] == "finished"
    assert stopped["exit_code"] != 0 and stopped["stopped"] == "cancel_run was called"
    assert stopped["receipt"] is None
    assert gone(grandchild), "cancel_run left a process the run had started"
    assert gone(server.state.runs[first["run_id"]].process.pid)
    # Cancelling again changes nothing, and the server takes the next run.
    again = call(server, "cancel_run", run_id=first["run_id"])
    assert again["cancelled"] is False and again["stopped"] == "cancel_run was called"
    with pytest.raises(RuntimeError, match="Unknown run"):
        call(server, "cancel_run", run_id="run-unknown")
    slow.unlink()
    second = call(server, "run", prepared="p", model="train-mean", output=str(tmp_path / "two"))
    assert second["state"] == "running"
    other = started_child(slow)
    # Closing the server stops what is still going.
    server.close()
    assert gone(other) and call(server, "run_status", run_id=second["run_id"])["stopped"] \
        == "the server closed"


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="needs process groups")
def test_a_run_is_stopped_at_its_time_limit(server: FastMCP, tmp_path: Path, slow: Path):
    started = call(server, "run", prepared="p", model="train-mean", timeout_seconds=4,
                   output=str(tmp_path / "one"))
    grandchild = started_child(slow)
    status = wait(server, started["run_id"], seconds=30)
    assert status["exit_code"] != 0 and "time limit of 4 seconds" in status["stopped"]
    assert gone(grandchild)


def test_a_recipe_run_is_limited_by_its_own_timeouts(server: FastMCP, tmp_path: Path,
                                                    slow: Path):
    plan = call(server, "plan_run", recipe=SMOKE, output=str(tmp_path / "out"))
    started = call(server, "run", recipe=SMOKE, output=str(tmp_path / "out"))
    assert started["time_limit_seconds"] == (plan["time_limit_seconds"]
                                             + mcp_server.RECIPE_MARGIN_SECONDS)
    assert call(server, "cancel_run", run_id=started["run_id"])["cancelled"] is True


def test_the_server_removes_only_a_log_folder_it_made(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mcp_server.tempfile, "mkdtemp",
                        lambda prefix: str(Path(tmp_path / "made").resolve()))
    made = mcp_server.build_server(RECIPES)
    assert made.state.folder.is_dir() and made.state.owns_folder
    made.close()
    assert not made.state.folder.exists()
    given = mcp_server.build_server(RECIPES, state_dir=tmp_path / "given")
    given.close()
    assert (tmp_path / "given").is_dir()


def test_a_failed_run_reports_its_exit_code_and_has_no_receipt(server: FastMCP, tmp_path: Path):
    started = call(server, "run", prepared=str(tmp_path / "missing"), model="train-mean",
                   output=str(tmp_path / "out"))
    status = wait(server, started["run_id"])
    assert status["exit_code"] == 2 and status["receipt"] is None and status["stderr_tail"]
    with pytest.raises(RuntimeError, match="No receipt.json"):
        call(server, "get_receipt", run_id=started["run_id"])
    for arguments in ({}, {"run_id": "run-x", "folder": "/tmp/x"}, {"run_id": "run-unknown"}):
        with pytest.raises(RuntimeError, match="exactly one of|Unknown run"):
            call(server, "get_receipt", **arguments)
    with pytest.raises(RuntimeError, match="Unknown run"):
        call(server, "run_status", run_id="run-unknown")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Fail on any connection or new process, and return the list of attempts."""
    attempts: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> None:
        attempts.append(args)
        raise AssertionError("A paid-backend test tried to start something")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    return attempts


def test_a_paid_backend_is_planned_and_priced_but_never_started(
    gpu_server: FastMCP, tmp_path: Path, offline: list[Any]
):
    output = str(tmp_path / "paid")
    plan = call(gpu_server, "plan_run", recipe="gpu-fixture", backend="runpod", output=output)
    assert plan["quote"]["paid"] and plan["quote"]["max_amount"] == "11.97"
    assert plan["quote"]["currency"] == "USD" and plan["quote"]["rate_date"] == "2026-10-14"
    assert plan["confirmation"] == {"required": True, "amount": "11.97", "currency": "USD",
                                    "how": plan["confirmation"]["how"]}
    assert plan["hardware"]["accelerator"]["kind"] == "gpu"
    assert plan["expected_wall_seconds"] == 1800
    assert plan["pod_specs"][0]["gpu"]["id"] == "NVIDIA H100 80GB HBM3"
    assert plan["command"][:3] == ["rewirebench", "recipe", "run"]

    run = {"recipe": "gpu-fixture", "backend": "runpod", "output": output}
    assert len(plan["pod_specs"]) == 1 and plan["quote"]["limit_enforced"] is False

    run = {"recipe": "gpu-fixture", "backend": "runpod", "output": output}
    # Without the plan's digest nothing else is looked at, the amount included.
    for arguments in ({}, {"confirmed_amount": "11.97", "confirmed_currency": "USD"}):
        with pytest.raises(RuntimeError, match="A paid run needs plan_sha256"):
            call(gpu_server, "run", **run, **arguments)
    with pytest.raises(RuntimeError, match="plan_sha256 does not match"):
        call(gpu_server, "run", **run, plan_sha256="0" * 64, confirmed_amount="11.97",
             confirmed_currency="USD")
    run["plan_sha256"] = plan["plan_sha256"]
    with pytest.raises(RuntimeError, match="needs confirmed_amount") as refused:
        call(gpu_server, "run", **run)
    # The refusal does not hand back the figure to send.
    assert "11.97" not in str(refused.value) and "Do not fill it in" in str(refused.value)
    with pytest.raises(RuntimeError, match="is not the plan's amount") as refused:
        call(gpu_server, "run", **run, confirmed_amount="11.00", confirmed_currency="USD")
    assert "11.97" not in str(refused.value)
    with pytest.raises(RuntimeError, match="is not the plan's amount"):
        call(gpu_server, "run", **run, confirmed_amount="11.97", confirmed_currency="GBP")
    with pytest.raises(RuntimeError, match="needs confirmed_currency"):
        call(gpu_server, "run", **run, confirmed_amount="11.97")
    # The amount is text a person typed. A number, true or a word is not one.
    for amount in (11.97, True, 1):
        with pytest.raises(RuntimeError, match="valid string"):
            call(gpu_server, "run", **run, confirmed_amount=amount, confirmed_currency="USD")
    for amount in ("true", "a lot"):
        with pytest.raises(RuntimeError, match="Not an amount"):
            call(gpu_server, "run", **run, confirmed_amount=amount, confirmed_currency="USD")
    # The matching amount is refused as well: this version launches nothing.
    with pytest.raises(RuntimeError, match="not built") as refused:
        call(gpu_server, "run", **run, confirmed_amount="11.97", confirmed_currency="USD")
    assert "Do not create the pod with another tool" in str(refused.value)
    assert offline == [] and not Path(output).exists()
    assert gpu_server.state.runs == {}


def test_a_paid_backend_is_refused_in_ci(gpu_server: FastMCP, tmp_path: Path,
                                         offline: list[Any], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CI", "true")
    with pytest.raises(RuntimeError, match="never started in CI"):
        call(gpu_server, "run", recipe="gpu-fixture", backend="runpod",
             output=str(tmp_path / "paid"), confirmed_amount="11.97", confirmed_currency="USD",
             plan_sha256=call(gpu_server, "plan_run", recipe="gpu-fixture", backend="runpod",
                              output=str(tmp_path / "paid"))["plan_sha256"])
    assert offline == []


def test_spark_is_planned_through_mcp_and_never_started(
    gpu_server: FastMCP, tmp_path: Path, offline: list[Any], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("REWIREBENCH_SPARK_HOST", "user@spark.example")
    output = str(tmp_path / "first")
    plan = call(gpu_server, "plan_run", recipe="gpu-fixture", backend="spark", output=output)
    assert plan["platform"] == "linux/arm64" and plan["quote"]["reason"] == "owned hardware"
    assert plan["command"][:5] == ["ssh", "-o", "BatchMode=yes", "--", "user@spark.example"]
    assert [step["name"] for step in plan["steps"]][5] == "run"
    reply = call(gpu_server, "run", recipe="gpu-fixture", backend="spark", output=output)
    assert reply["started"] is False and reply["dry_run"] is True
    assert reply["plan"]["plan_sha256"] == plan["plan_sha256"]
    # A GPU recipe cannot be moved to the local backend, and an unknown backend is an error.
    with pytest.raises(RuntimeError, match="does not name the local backend"):
        call(gpu_server, "run", recipe="gpu-fixture", backend="local", output=output)
    with pytest.raises(RuntimeError, match="'local', 'spark' or 'runpod'"):
        call(gpu_server, "plan_run", recipe="gpu-fixture", backend="nextflow")
    # Owned hardware has nothing to confirm, so an amount there is a mistake.
    with pytest.raises(RuntimeError, match="there is no amount to confirm"):
        call(gpu_server, "run", recipe="gpu-fixture", backend="spark", output=output,
             confirmed_amount="1.00")
    assert offline == [] and not Path(output).exists()


def test_without_a_recipes_directory_the_server_still_lists_protocols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(recipes, "find_recipes_dir", lambda explicit=None: None)
    server = mcp_server.build_server(state_dir=tmp_path / "state")
    assert call(server, "list_recipes")["recipes"] == []
    assert "No recipes directory" in call(server, "list_recipes")["note"]
    assert len(call(server, "list_protocols")["protocols"]) == len(sdk.PROTOCOLS)
    with pytest.raises(RuntimeError, match="No recipes directory"):
        call(server, "plan_run", recipe=SMOKE)


# Loaded by every Python the stdio test starts. Only a command started by the server
# carries REWIREBENCH_STARTED_BY=agent; that one records its process ID and hangs.
HANG = """
import os, time
if os.environ.get("REWIREBENCH_STARTED_BY") == "agent":
    with open(os.environ["HANG_RECORD"], "w") as record:
        record.write(str(os.getpid()))
    time.sleep(300)
"""


def test_rewirebench_mcp_speaks_mcp_over_real_stdio(tmp_path: Path):
    """Start the command as a process: initialise, list the tools, call some of them, and
    close the client while a run is going. Nothing the server started may outlive it."""
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(HANG)
    record = tmp_path / "run.pid"
    path = os.pathsep.join(filter(None, [str(hook), os.environ.get("PYTHONPATH")]))
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": path,
                   "HANG_RECORD": str(record)}
    environment.pop(receipts.STARTED_BY_VARIABLE, None)
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "rewirebench.cli", "mcp", "--recipes-dir", str(RECIPES)],
        env=environment, cwd=str(tmp_path),
    )

    async def go() -> tuple[Any, ...]:
        async with (stdio_client(parameters) as (read, write),
                    ClientSession(read, write) as session):
            started = await session.initialize()
            tools = await session.list_tools()
            protocols = await session.call_tool("list_protocols", {})
            plan = await session.call_tool(
                "plan_run", {"recipe": SMOKE, "output": str(tmp_path / "out")})
            misspelt = await session.call_tool(
                "plan_run", {"recipe": SMOKE, "bakend": "runpod"})
            run = await session.call_tool(
                "run", {"recipe": SMOKE, "output": str(tmp_path / "out")})
            with anyio.fail_after(60):
                while not (record.is_file() and record.read_text()):
                    await anyio.sleep(0.05)
            return started, tools, protocols, plan, misspelt, run

    started, tools, protocols, plan, misspelt, run = anyio.run(go)
    assert misspelt.isError and "no argument 'bakend'" in misspelt.content[0].text
    status = json.loads(run.content[0].text)
    assert not run.isError and status["state"] == "running"
    if hasattr(os, "killpg"):
        # The client has closed. The run it left is stopped and the log folder is gone.
        assert gone(int(record.read_text())), "The run outlived the server"
        assert not Path(status["logs"]["stdout"]).parent.exists()
    assert started.serverInfo.name == "rewirebench"
    assert sorted(tool.name for tool in tools.tools) == sorted(mcp_server.TOOLS)
    assert not protocols.isError and not plan.isError
    listed = json.loads(protocols.content[0].text)["protocols"]
    assert [entry["id"] for entry in listed] == list(sdk.PROTOCOLS)
    assert json.loads(plan.content[0].text)["quote"]["reason"] == "owned hardware"


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="needs process groups")
def test_a_terminated_server_stops_its_run_and_removes_its_logs(tmp_path: Path):
    """SIGTERM to a real `rewirebench mcp` process while a run is going."""
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(HANG)
    record = tmp_path / "run.pid"
    path = os.pathsep.join(filter(None, [str(hook), os.environ.get("PYTHONPATH")]))
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": path,
                   "HANG_RECORD": str(record)}
    environment.pop(receipts.STARTED_BY_VARIABLE, None)
    served = subprocess.Popen(
        [sys.executable, "-m", "rewirebench.cli", "mcp", "--recipes-dir", str(RECIPES)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env=environment, cwd=tmp_path, text=True, bufsize=1,
    )
    assert served.stdin is not None and served.stdout is not None

    def send(message: dict[str, Any]) -> None:
        served.stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        served.stdin.flush()

    def reply(identifier: int) -> dict[str, Any]:
        while True:
            message = json.loads(served.stdout.readline())
            if message.get("id") == identifier:
                return message

    try:
        send({"id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}}})
        assert reply(1)["result"]["serverInfo"]["name"] == "rewirebench"
        send({"method": "notifications/initialized"})
        send({"id": 2, "method": "tools/call", "params": {
            "name": "run", "arguments": {"recipe": SMOKE, "output": str(tmp_path / "out")}}})
        status = json.loads(reply(2)["result"]["content"][0]["text"])
        assert status["state"] == "running"
        run = started_child(record)
        served.send_signal(signal.SIGTERM)
        # Sent again at once, as a client that is in a hurry would. It changes nothing.
        served.send_signal(signal.SIGTERM)
        assert served.wait(timeout=20) == 128 + signal.SIGTERM
        assert gone(run), "The run outlived the server"
        assert not Path(status["logs"]["stdout"]).parent.exists()
    finally:
        served.kill()
        served.wait()


def test_the_command_line_says_how_to_install_the_server_when_mcp_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    import rewirebench

    # As on an installation without the extra: the SDK cannot be imported.
    monkeypatch.delitem(sys.modules, "rewirebench.mcp_server", raising=False)
    monkeypatch.delattr(rewirebench, "mcp_server", raising=False)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    with pytest.raises(SystemExit) as stop:
        cli.main(["mcp"])
    assert stop.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "rewirebench[mcp]" in captured.err
    # The other commands never import the MCP SDK.
    done = subprocess.run(
        [sys.executable, "-c",
         ("import sys; from rewirebench import cli; cli.main(['protocols']); "
          "cli.main(['recipe', 'list', '--recipes-dir', sys.argv[1]]); "
          "assert 'mcp' not in sys.modules and 'anyio' not in sys.modules"),
         str(RECIPES)],
        capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, done.stderr
