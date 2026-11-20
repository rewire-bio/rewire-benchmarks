"""The benchmarks MCP server, driven through an in-process MCP client.

One test also starts `rewirebench mcp` as a real process and speaks to it over stdio,
because a stray line on stdout breaks that transport and an in-process client cannot see it.
"""
from __future__ import annotations

import json
import os
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
from rewirebench import cli, mcp_server, receipts, recipes, sdk

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
    assert len(mcp_server.TOOLS) == 12
    assert all(tool.description for tool in tools)
    run = next(tool for tool in tools if tool.name == "run")
    assert {"confirmed_amount", "confirmed_currency", "plan_sha256"} <= set(
        run.inputSchema["properties"])


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
    with pytest.raises(RuntimeError, match="still going|already exists"):
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
    assert call(server, "verify_receipt", run_id=started["run_id"])["ok"]

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
    ):
        with pytest.raises(RuntimeError, match=message):
            call(server, "run", model=model, model_options=options, **arguments)
    for bad in ({"output": str(tmp_path / "out")},
                {"recipe": SMOKE, "model": "train-mean", "output": str(tmp_path / "out")},
                {"model": "train-mean", "output": str(tmp_path / "out")},
                {"recipe": SMOKE, "output": "relative/folder"}):
        with pytest.raises(RuntimeError, match="Give either|needs both|absolute path"):
            call(server, "run", **bad)
    assert not (tmp_path / "out").exists()
    assert mcp_server.child_environment()["CUDA_VISIBLE_DEVICES"] == ""
    assert mcp_server.child_environment()[receipts.STARTED_BY_VARIABLE] == "agent"


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
    with pytest.raises(RuntimeError, match="needs confirmed_amount"):
        call(gpu_server, "run", **run)
    with pytest.raises(RuntimeError, match="needs confirmed_amount"):
        call(gpu_server, "run", **run, plan_sha256=plan["plan_sha256"])
    with pytest.raises(RuntimeError, match="is not the plan's amount"):
        call(gpu_server, "run", **run, confirmed_amount=11.0, confirmed_currency="USD")
    with pytest.raises(RuntimeError, match="is not the plan's amount"):
        call(gpu_server, "run", **run, confirmed_amount="11.97", confirmed_currency="GBP")
    with pytest.raises(RuntimeError, match="needs confirmed_currency"):
        call(gpu_server, "run", **run, confirmed_amount=11.97)
    # The matching amount is refused as well: this version launches nothing.
    for amount in (11.97, "11.97"):
        with pytest.raises(RuntimeError, match="not built"):
            call(gpu_server, "run", **run, confirmed_amount=amount, confirmed_currency="USD",
                 plan_sha256=plan["plan_sha256"])
    assert offline == [] and not Path(output).exists()


def test_a_paid_backend_is_refused_in_ci(gpu_server: FastMCP, tmp_path: Path,
                                         offline: list[Any], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CI", "true")
    with pytest.raises(RuntimeError, match="never started in CI"):
        call(gpu_server, "run", recipe="gpu-fixture", backend="runpod",
             output=str(tmp_path / "paid"), confirmed_amount="11.97", confirmed_currency="USD")
    assert offline == []


def test_spark_is_planned_through_mcp_and_never_started(
    gpu_server: FastMCP, tmp_path: Path, offline: list[Any], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("REWIREBENCH_SPARK_HOST", "tim@spark.local")
    output = str(tmp_path / "first")
    plan = call(gpu_server, "plan_run", recipe="gpu-fixture", backend="spark", output=output)
    assert plan["platform"] == "linux/arm64" and plan["quote"]["reason"] == "owned hardware"
    assert plan["command"][:5] == ["ssh", "-o", "BatchMode=yes", "--", "tim@spark.local"]
    assert [step["name"] for step in plan["steps"]][5] == "run"
    reply = call(gpu_server, "run", recipe="gpu-fixture", backend="spark", output=output)
    assert reply["started"] is False and reply["dry_run"] is True
    assert reply["plan"]["plan_sha256"] == plan["plan_sha256"]
    # A GPU recipe cannot be moved to the local backend, and an unknown backend is an error.
    with pytest.raises(RuntimeError, match="does not name the local backend"):
        call(gpu_server, "run", recipe="gpu-fixture", backend="local", output=output)
    with pytest.raises(RuntimeError, match="No nextflow backend"):
        call(gpu_server, "plan_run", recipe="gpu-fixture", backend="nextflow")
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


def test_rewirebench_mcp_speaks_mcp_over_real_stdio(tmp_path: Path):
    """Start the command as a process: initialise, list the tools and call two of them."""
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "rewirebench.cli", "mcp", "--recipes-dir", str(RECIPES)],
        env={**os.environ, "PYTHONUNBUFFERED": "1"}, cwd=str(tmp_path),
    )

    async def go() -> tuple[Any, Any, Any, Any]:
        async with (stdio_client(parameters) as (read, write),
                    ClientSession(read, write) as session):
            started = await session.initialize()
            tools = await session.list_tools()
            protocols = await session.call_tool("list_protocols", {})
            plan = await session.call_tool(
                "plan_run", {"recipe": SMOKE, "output": str(tmp_path / "out")})
            return started, tools, protocols, plan

    started, tools, protocols, plan = anyio.run(go)
    assert started.serverInfo.name == "rewirebench"
    assert sorted(tool.name for tool in tools.tools) == sorted(mcp_server.TOOLS)
    assert not protocols.isError and not plan.isError
    listed = json.loads(protocols.content[0].text)["protocols"]
    assert [entry["id"] for entry in listed] == list(sdk.PROTOCOLS)
    assert json.loads(plan.content[0].text)["quote"]["reason"] == "owned hardware"


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
