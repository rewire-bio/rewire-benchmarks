"""The demo in docs/demo/: its client, its committed record, and the offline fixture run.

The recorded MFASS v2 run needs the network, so it is not repeated here. This test runs the
three-row synthetic fixture through the same client code and the same MCP path: a
`rewirebench mcp` process on stdio, reading the plugin's copy of the recipes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
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


def test_the_client_labels_a_temporary_folder_that_is_not_its_own(client: ModuleType):
    """A server, or a step it starts, can print another temporary folder than the client's."""
    transcript = client.Transcript()
    transcript.calls.append({"result": {
        "a": "/var/folders/ab/c_d0000gn/T/rewirebench-recipe-x/work/prepared.json",
        "b": "wrote /private/var/folders/ab/c_d0000gn/T/x and left"}})
    cleaned = transcript.document()["calls"][0]["result"]
    assert cleaned == {"a": "<temp>/rewirebench-recipe-x/work/prepared.json",
                       "b": "wrote <temp>/x and left"}
    assert "<temp>/rewirebench-recipe-x" in transcript.text()


@pytest.mark.parametrize("path", ["/Users/someone/runs/out", "/home/someone/out",
                                  "/private/tmp/work/out", "/var/folders/zz"])
def test_the_client_refuses_to_write_a_transcript_that_names_a_local_folder(
    client: ModuleType, tmp_path: Path, path: str
):
    """A recording must not need editing by hand, so a path that was not labelled stops it."""
    transcript = client.Transcript()
    transcript.calls.append({"result": {"output": path}})
    with pytest.raises(client.DemoError, match="still names a local folder"):
        transcript.write(tmp_path / "transcript.json")
    assert not (tmp_path / "transcript.json").exists()
    transcript.label(path, "<run folder>")
    transcript.write(tmp_path / "transcript.json")
    assert json.loads((tmp_path / "transcript.json").read_text())["calls"][0]["result"] == {
        "output": "<run folder>"}


def recorded(name: str) -> dict[str, Any]:
    return json.loads((DEMO / name).read_text())


def one(calls: list[dict[str, Any]], tool: str) -> dict[str, Any]:
    [found] = [call for call in calls if call["tool"] == tool]
    return found


def table(page: str, heading: str) -> list[list[str]]:
    """Body rows of the first table under a heading of a Markdown page, as cells."""
    section = page.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("|")]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


def check_recorded_run(transcript: dict[str, Any], client: ModuleType) -> dict[str, Any]:
    """What must hold in any recorded recipe run. Every value in the summary is worked out
    again from the recorded calls, so the summary cannot be edited apart from them."""
    calls = transcript["calls"]
    assert [call["number"] for call in calls] == list(range(1, len(calls) + 1))
    assert not any(call["is_error"] for call in calls)
    for call in calls:
        assert call["tool"] in transcript["servers"][call["server"]]["tools"]
    graph = [call["tool"] for call in calls if call["server"] == "rewire-benchmark-graph"]
    runner = [call["tool"] for call in calls if call["server"] == "rewirebench"]
    assert graph == ["dataset_info", "find_entity", "sparql_query"]
    assert collapsed(runner) == RECIPE_FLOW
    assert [call["server"] for call in calls] == (
        ["rewire-benchmark-graph"] * 3 + ["rewirebench"] * len(runner))

    # The plan the run started from: local, unpaid, runnable, and the one that was shown.
    plan, run = one(calls, "plan_run"), one(calls, "run")
    assert plan["result"]["backend"] == "local" and plan["result"]["runnable"] is True
    assert plan["result"]["quote"]["paid"] is False
    assert plan["result"]["quote"]["reason"] == "owned hardware"
    assert plan["result"]["quote"]["max_amount"] is None
    assert plan["result"]["confirmation"]["required"] is False
    assert plan["result"]["pod_specs"] == []
    digest = plan["result"]["plan_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", digest) and digest != "0" * 64
    assert run["arguments"] == {**plan["arguments"], "plan_sha256": digest}
    assert "confirmed_amount" not in run["arguments"]
    polls = [call for call in calls if call["tool"] == "run_status"]
    for status in [run, *polls]:
        assert status["result"]["plan_sha256"] == digest
        assert status["result"]["recipe_id"] == run["arguments"]["recipe"]
    assert {call["arguments"]["run_id"] for call in polls} == {run["result"]["run_id"]}
    assert polls[-1]["result"]["state"] == "finished"
    assert polls[-1]["result"]["exit_code"] == 0 and polls[-1]["result"]["checks_passed"] is True

    receipt = one(calls, "get_receipt")["result"]
    verified = one(calls, "verify_receipt")
    release = one(calls, "dataset_info")["result"]["version"]
    assert receipt["kg_release_id"] == run["arguments"]["kg_release_id"] == release
    assert receipt["started_by"] == "agent" and receipt["backend"] == "local"
    assert receipt["cost"]["paid_compute"] is False and receipt["hardware"]["gpus"] == []
    recipe = ROOT / receipt["recipe"]["path"]
    assert receipt["recipe"]["sha256"] == hashlib.sha256(recipe.read_bytes()).hexdigest()
    assert receipt["recipe"]["sha256"] == plan["result"]["recipe"]["sha256"]

    # The comparison, worked out again from the graph's recorded reply and the receipt.
    values = client.read_published(one(calls, "sparql_query")["result"])
    assert set(values) == {"auroc", "average-precision", "precision-at-100"}
    for entry in values.values():
        assert (entry["scored"], entry["eligible"]) == (8324, 8324)
    rows = client.compare({"values": values}, receipt)
    assert [row["agrees"] for row in rows] == [True, True, True]
    assert {row["receipt_key"] for row in rows} == set(PUBLISHED)
    summary = transcript["summary"]
    assert summary["comparison"] == rows
    assert summary["published"]["values"] == values
    assert summary["published"]["kg_release_id"] == summary["kg_release_id"] == release
    assert summary["plan_sha256"] == digest
    assert summary["metrics"] == receipt["metrics"]
    assert summary["code_commit"] == receipt["code"]["commit"]
    assert summary["recipe_sha256"] == receipt["recipe"]["sha256"]
    assert summary["checks_passed"] is receipt["checks_passed"] is True
    assert summary["cost"] == receipt["cost"]
    assert summary["verify"] == {key: verified["result"][key]
                                 for key in ("strict", "passed", "not_checked")}
    assert verified["arguments"]["strict"] is verified["result"]["strict"]
    assert verified["result"]["passed"] is True
    return {"receipt": receipt, "verified": verified["result"], "rows": rows, "calls": calls}


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
    assert receipt["code"]["dirty"] is False and receipt["code"]["commit_source"] == "checkout"
    assert receipt["code"]["commit"] == receipt["recipe"]["commit"]
    assert receipt["cost"] == {
        "paid_compute": False, "amount": None, "currency": None, "reason": "owned hardware",
        "hourly_rate": None, "rate_source": None, "rate_date": None}
    assert receipt["hardware"]["gpus"] == []
    assert set(receipt["environment"]["images"]) == {"linux/amd64", "linux/arm64"}


def test_the_transcript_is_the_run_in_the_folder_and_uses_only_mcp_tools(client: ModuleType):
    transcript = recorded("transcript.json")
    found = check_recorded_run(transcript, client)
    assert found["receipt"] == json.loads((RUN / "receipt.json").read_text())
    assert found["verified"]["strict"] is True and found["verified"]["not_checked"] == []
    assert all(row["difference"] == 0 for row in found["rows"])
    graph = "rewire-benchmark-graph"
    assert transcript["servers"][graph]["command"] == declared(graph)


def declared(name: str) -> list[str]:
    server = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"][name]
    return [server["command"], *server["args"]]


def test_the_plugin_run_was_made_with_the_launch_lines_the_plugin_ships(client: ModuleType):
    """docs/demo/plugin-run.json: the same run through the plugin as installed. Its receipt
    names the pinned commit, read from install metadata. It verifies, and it cannot pass
    strict, because no tree was checked and the plugin's recipes are in no commit. Raising
    the rewirebench or graph pin means recording it again."""
    transcript = recorded("plugin-run.json")
    found = check_recorded_run(transcript, client)
    for name in ("rewire-benchmark-graph", "rewirebench"):
        assert transcript["servers"][name]["command"] == declared(name)
    assert one(found["calls"], "list_recipes")["result"]["recipes_dir"] == (
        "${CLAUDE_PLUGIN_ROOT}/recipes")
    receipt, verified = found["receipt"], found["verified"]
    source = declared("rewirebench")[declared("rewirebench").index("--from") + 1]
    assert f"@{receipt['code']['commit']}#" in source
    assert receipt["code"]["commit_source"] == "install_metadata"
    assert receipt["code"]["dirty"] is None and receipt["recipe"]["commit"] is None
    assert verified["strict"] is False and verified["not_checked"] == ["commits"]
    statuses = {item["check"]: item["status"] for item in verified["results"]}
    assert statuses.pop("commits") == "not_available"
    [commits] = [item for item in verified["results"] if item["check"] == "commits"]
    assert "install metadata" in commits["detail"] and "recipe.commit is null" in commits["detail"]
    assert set(statuses.values()) == {"match"} and len(statuses) == 7
    # Unlocked packages may move the last digits; nothing more.
    assert all(row["difference"] <= 1e-12 for row in found["rows"])


def test_both_recorded_runs_used_the_same_code_recipe_and_predictions():
    """The checkout run names a commit; nothing here can look that commit up. What ties it to
    the plugin's pin is the wheel uvx built from the pinned commit: the plugin run, made with
    that wheel, reports the same digest of the SDK code, the same recipe bytes and the same
    predictions as the checkout run."""
    checkout = json.loads((RUN / "receipt.json").read_text())
    plugin = one(recorded("plugin-run.json")["calls"], "get_receipt")["result"]
    args = declared("rewirebench")
    assert f"@{checkout['code']['commit']}#" in args[args.index("--from") + 1]
    assert checkout["code"]["commit"] == plugin["code"]["commit"]
    assert checkout["code"]["sdk_code_sha256"] == plugin["code"]["sdk_code_sha256"]
    assert checkout["code"]["rewirebench_version"] == plugin["code"]["rewirebench_version"]
    assert checkout["recipe"]["sha256"] == plugin["recipe"]["sha256"]
    assert checkout["inputs"] == [{**item, "obtained": "fetched"} for item in plugin["inputs"]]
    predictions = "baselines/mfass-kmer-position-v2/predictions.json"
    digests = [{item["path"]: item["sha256"] for item in receipt["outputs"]}[predictions]
               for receipt in (checkout, plugin)]
    assert digests[0] == digests[1]
    assert digests[0] == hashlib.sha256((RUN / predictions).read_bytes()).hexdigest()
    report = json.loads((RUN / "baselines/mfass-kmer-position-v2/report.json").read_text())
    assert json.dumps(report).count(digests[0]) >= 1


def test_the_demo_page_reports_what_the_transcripts_hold():
    """The tables in docs/demo/README.md are read against the recordings."""
    page = (DEMO / "README.md").read_text()
    transcript, plugin = recorded("transcript.json"), recorded("plugin-run.json")
    steps = table(page, "What ran")
    calls = transcript["calls"]
    verify = one(calls, "verify_receipt")
    assert steps[-1][0] == str(verify["number"]) == str(len(calls))
    assert steps[-1][2].startswith("`verify_receipt`") and steps[-1][3].startswith("Passed")
    polls = [call["number"] for call in calls if call["tool"] == "run_status"]
    assert f"{polls[0]} to {polls[-1]}" in [row[0] for row in steps]
    for row in steps:
        if row[0].isdigit():
            assert f"`{calls[int(row[0]) - 1]['tool']}`" in row[2], row
    assert f"Release `{transcript['summary']['kg_release_id']}`" in steps[0][3]

    names = {"Precision at 100": "precision-at-100", "Average precision": "average-precision",
             "AUROC": "auroc"}
    rows = {row["metric"]: row for row in transcript["summary"]["comparison"]}
    results = table(page, "Result")
    assert [row[0] for row in results] == [*names, "Scored variants"]
    for label, published, observed, difference in results[:3]:
        row = rows[names[label]]
        assert (published, observed) == (repr(row["published"]), repr(row["observed"]))
        assert difference == "0" and row["difference"] == 0
    receipt = one(calls, "get_receipt")["result"]
    assert results[3][1:3] == ["8324 of 8324", str(int(receipt["metrics"]["n"]))]
    assert f"SHA-256 `{receipt['recipe']['sha256']}`" in page
    assert f"commit `{receipt['code']['commit']}`" in page
    assert f"wall time {receipt['wall_seconds']:.2f} seconds" in page

    side = {row[0]: row[1:] for row in table(page, "The same run through the plugin as shipped")}
    other = one(plugin["calls"], "get_receipt")["result"]
    keys = {"Precision at 100": "precision_at_capacity",
            "Average precision": "average_precision_sklearn", "AUROC": "auroc"}
    for label, key in keys.items():
        assert side[label] == [repr(receipt["metrics"][key]), repr(other["metrics"][key])]
    short = f"`{receipt['code']['commit'][:7]}`"
    assert side["`code.commit`"] == [short, f"`{other['code']['commit'][:7]}`"] == [short, short]
    assert side["`code.commit_source`"] == [
        f"`{receipt['code']['commit_source']}`", f"`{other['code']['commit_source']}`"]
    assert side["`code.dirty`"] == ["false", "null"]
    assert (receipt["code"]["dirty"], other["code"]["dirty"]) == (False, None)
    assert side["`recipe.commit`"] == [short, "null"] and other["recipe"]["commit"] is None
    assert side["Wall time"][0] == f"{receipt['wall_seconds']:.2f} s"
    assert side["Wall time"][1].startswith(f"{other['wall_seconds']:.2f} s")
    assert side["`sdk_code_sha256`"] == [f"`{receipt['code']['sdk_code_sha256'][:8]}...`"] * 2
    graph = [call for call in plugin["calls"] if call["server"] == "rewire-benchmark-graph"]
    assert (f"the same {len(plugin['calls'])} calls ({len(graph)} to the graph, "
            f"{len(plugin['calls']) - len(graph)} to rewirebench)") in " ".join(page.split())


def test_the_server_check_was_recorded_with_the_launch_lines_the_plugin_ships():
    """Raising a pin means recording docs/demo/plugin-servers.json again."""
    check = recorded("plugin-servers.json")
    declared = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]
    assert sorted(check["servers"]) == sorted(declared)
    for name, server in declared.items():
        assert check["servers"][name]["command"] == [server["command"], *server["args"]]
    assert [(call["server"], call["is_error"]) for call in check["calls"]] == [
        (name, False) for name in declared]


@pytest.mark.parametrize("name", ["transcript.json", "plugin-servers.json", "plugin-run.json"])
def test_recorded_transcripts_name_no_local_folder(name: str, client: ModuleType):
    text = (DEMO / name).read_text()
    for marker in client.MACHINE_PATHS:
        assert marker not in text, marker


def test_the_recorded_run_folder_names_no_local_folder():
    """The run folder cannot be edited without breaking its digests, so it must be clean as
    recorded. Step logs write {work} for the scratch folder. Nothing in the folder names a
    user, a home folder or a temporary folder."""
    found = []
    for path in sorted(RUN.rglob("*")):
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for marker in ("/Users/", "/home/", "/private/", "/var/folders/", "/tmp/"):
                if marker in line:
                    found.append((path.relative_to(RUN).as_posix(), number, marker))
    assert found == []
    line = (RUN / "logs/01-prepare.stdout.txt").read_text().splitlines()[3]
    assert line.strip() == '"prepared": "{work}/prepared/prepared.json"'
