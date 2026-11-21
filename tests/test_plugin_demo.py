"""The demo in docs/demo/: its client, its committed record, and the offline fixture run.

The recorded MFASS v2 run needs the network, so it is not repeated here. This test runs the
three-row synthetic fixture through the same client code and the same MCP path: a
`rewirebench mcp` process on stdio, reading the plugin's copy of the recipes.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import anyio
import pytest
from rewirebench import mcp_server, receipts

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins/bio-model-benchmarks"
DEMO = ROOT / "docs/demo"
RUN = DEMO / "mfass-v2-kmer-baseline"
SMOKE = "flip2-synthetic-smoke"
RECIPE_FLOW = ["list_recipes", "describe_recipe", "plan_run", "run", "run_status",
               "get_receipt", "verify_receipt"]
# The published corrected baseline, as the task and benchmarks/mfass/README.md state it.
PUBLISHED = {"precision_at_capacity": 0.61, "average_precision_sklearn": 0.2864,
             "auroc": 0.7779}


def load_client() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mcp_demo", ROOT / "scripts/demo/mcp_demo.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def client() -> ModuleType:
    return load_client()


def collapsed(tools: list[str]) -> list[str]:
    """The sequence of tools with repeated polls of one tool counted once."""
    return [tool for index, tool in enumerate(tools) if index == 0 or tools[index - 1] != tool]


def test_three_row_fixture_runs_through_the_mcp_path_offline(client: ModuleType, tmp_path: Path):
    transcript = client.Transcript()
    transcript.label(tmp_path, "<test folder>")
    command, args = client.runner_command(str(PLUGIN / "recipes"))
    output = tmp_path / "run"

    async def go() -> dict[str, Any]:
        async with client.connect(transcript, client.RUNNER, command, args,
                                  env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                  cwd=str(tmp_path)) as session:
            return await client.run_recipe(session, transcript, recipe=SMOKE, output=output,
                                           kg_release_id="test-release")

    result = anyio.run(go)
    receipt = result["receipt"]
    assert receipt["recipe"]["id"] == SMOKE and receipt["scope"] == "synthetic_smoke"
    assert receipt["metrics"] == {"n": 3.0, "coverage_scored": 3.0}
    assert receipt["checks_passed"] is True
    assert receipt["started_by"] == "agent" and receipt["backend"] == "local"
    assert receipt["kg_release_id"] == "test-release"
    assert receipt["cost"]["paid_compute"] is False
    assert receipt["cost"]["amount"] is None and receipt["cost"]["reason"] == "owned hardware"
    assert receipt["hardware"]["gpus"] == []
    assert [item["obtained"] for item in receipt["inputs"]] == ["repository"]
    assert result["verified"]["passed"] is True
    assert (output / "receipt.json").is_file() and (output / "SHA256SUMS").is_file()

    document = transcript.document()
    calls = document["calls"]
    assert collapsed([call["tool"] for call in calls]) == RECIPE_FLOW
    assert not any(call["is_error"] for call in calls)
    assert {call["server"] for call in calls} == {"rewirebench"}
    assert sorted(document["servers"]["rewirebench"]["tools"]) == sorted(mcp_server.TOOLS)
    # The run started with the plan it was shown, and local folders are labelled.
    plan = next(call for call in calls if call["tool"] == "plan_run")
    run = next(call for call in calls if call["tool"] == "run")
    assert run["arguments"]["plan_sha256"] == plan["result"]["plan_sha256"]
    assert str(tmp_path) not in json.dumps(document)
    assert "<test folder>/run" in json.dumps(document)


def test_the_client_stops_when_a_tool_call_fails(client: ModuleType, tmp_path: Path):
    transcript = client.Transcript()
    command, args = client.runner_command(str(PLUGIN / "recipes"))

    async def go() -> None:
        async with client.connect(transcript, client.RUNNER, command, args,
                                  env={**os.environ}, cwd=str(tmp_path)) as session:
            await client.run_recipe(session, transcript, recipe="no-such-recipe",
                                    output=tmp_path / "run")

    with pytest.raises(client.DemoError, match="does not offer no-such-recipe"):
        anyio.run(go)
    assert not (tmp_path / "run").exists()


def test_typed_literals_from_the_graph_are_read_as_values(client: ModuleType):
    assert client._plain('"0.61"^^xsd:decimal') == "0.61"
    assert client._plain('"Average precision"@en') == "Average precision"
    assert client._plain("status:reproduced") == "status:reproduced"
    published = {"values": {"precision-at-100": {"value": 0.61}, "auroc": {"value": 0.5}}}
    rows = client.compare(published, {"metrics": {"precision_at_capacity": 0.61, "auroc": 0.7}})
    assert [(row["metric"], row["agrees"]) for row in rows] == [
        ("auroc", False), ("precision-at-100", True)]


def recorded(name: str) -> dict[str, Any]:
    return json.loads((DEMO / name).read_text())


def test_the_recorded_run_folder_is_intact():
    """The committed folder still matches its own digests and the recipe's checks."""
    result = receipts.verify(RUN, recipes_dir=ROOT / "recipes")
    statuses = {item["check"]: item["status"] for item in result["results"]}
    for check in ("sha256sums", "outputs", "run", "checks", "metrics", "commits"):
        assert statuses[check] == "match", (check, result["results"])
    assert "mismatch" not in statuses.values(), result["results"]


def test_the_recorded_run_gives_the_published_values():
    receipt = json.loads((RUN / "receipt.json").read_text())
    assert receipt["recipe"]["id"] == "mfass-v2-kmer-baseline"
    assert receipt["scope"] == "full_protocol" and receipt["checks_passed"] is True
    assert receipt["metrics"]["n"] == 8324 and receipt["metrics"]["positives"] == 315
    for key, value in PUBLISHED.items():
        assert receipt["metrics"][key] == pytest.approx(value, abs=5e-5)
    assert receipt["code"]["dirty"] is False
    assert receipt["code"]["commit"] == receipt["recipe"]["commit"]
    assert receipt["cost"] == {
        "paid_compute": False, "amount": None, "currency": None, "reason": "owned hardware",
        "hourly_rate": None, "rate_source": None, "rate_date": None}
    assert receipt["hardware"]["gpus"] == []
    assert set(receipt["environment"]["images"]) == {"linux/amd64", "linux/arm64"}


def test_the_transcript_is_the_run_in_the_folder_and_uses_only_mcp_tools():
    transcript = recorded("transcript.json")
    receipt = json.loads((RUN / "receipt.json").read_text())
    calls = transcript["calls"]
    assert not any(call["is_error"] for call in calls)
    for call in calls:
        assert call["tool"] in transcript["servers"][call["server"]]["tools"]
    graph = [call["tool"] for call in calls if call["server"] == "rewire-benchmark-graph"]
    runner = [call["tool"] for call in calls if call["server"] == "rewirebench"]
    assert graph == ["dataset_info", "find_entity", "sparql_query"]
    assert collapsed(runner) == RECIPE_FLOW
    assert next(call for call in calls if call["tool"] == "get_receipt")["result"] == receipt
    verified = next(call for call in calls if call["tool"] == "verify_receipt")
    assert verified["arguments"]["strict"] is True and verified["result"]["passed"] is True
    assert verified["result"]["not_checked"] == []
    summary = transcript["summary"]
    assert summary["kg_release_id"] == receipt["kg_release_id"] is not None
    assert summary["kg_release_id"] == calls[0]["result"]["version"]
    assert [row["agrees"] for row in summary["comparison"]] == [True, True, True]
    assert {row["receipt_key"] for row in summary["comparison"]} == set(PUBLISHED)


def test_the_plugin_pins_the_code_the_recorded_run_used():
    """The run in docs/demo/ ran the commit the plugin installs rewirebench from."""
    receipt = json.loads((RUN / "receipt.json").read_text())
    args = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]["rewirebench"]["args"]
    assert f"@{receipt['code']['commit']}#" in args[args.index("--from") + 1]


def test_the_server_check_was_recorded_with_the_launch_lines_the_plugin_ships():
    """Raising a pin means recording docs/demo/plugin-servers.json again."""
    check = recorded("plugin-servers.json")
    declared = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]
    assert sorted(check["servers"]) == sorted(declared)
    for name, server in declared.items():
        assert check["servers"][name]["command"] == [server["command"], *server["args"]]
    assert [(call["server"], call["is_error"]) for call in check["calls"]] == [
        (name, False) for name in declared]


@pytest.mark.parametrize("name", ["transcript.json", "plugin-servers.json"])
def test_recorded_transcripts_name_no_local_folder(name: str):
    text = (DEMO / name).read_text()
    for marker in ("/Users/", "/home/", "/private/", "/var/folders/"):
        assert marker not in text, marker
