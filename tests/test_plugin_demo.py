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
import shutil
import subprocess
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


def section(page: str, heading: str) -> str:
    """The text under one second-level heading of a Markdown page."""
    return page.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]


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
    """docs/demo/plugin-run.json: the same run through the plugin as installed. Running from
    the PyPI wheel records code.rewirebench_version with null code.commit. It verifies, and
    cannot pass strict because no tree was checked and the plugin's recipes are in no commit.
    Raising the rewirebench or graph pin means recording it again."""
    transcript = recorded("plugin-run.json")
    found = check_recorded_run(transcript, client)
    for name in ("rewire-benchmark-graph", "rewirebench"):
        assert transcript["servers"][name]["command"] == declared(name)
    assert one(found["calls"], "list_recipes")["result"]["recipes_dir"] == (
        "${CLAUDE_PLUGIN_ROOT}/recipes")
    receipt, verified = found["receipt"], found["verified"]
    source = declared("rewirebench")[declared("rewirebench").index("--from") + 1]
    assert source == f"rewirebench[mcp]=={receipt['code']['rewirebench_version']}"
    assert receipt["code"]["commit"] is None
    assert receipt["code"]["commit_source"] is None
    assert receipt["code"]["dirty"] is None and receipt["recipe"]["commit"] is None
    assert verified["strict"] is False and verified["not_checked"] == ["commits"]
    statuses = {item["check"]: item["status"] for item in verified["results"]}
    assert statuses.pop("commits") == "not_available"
    [commits] = [item for item in verified["results"] if item["check"] == "commits"]
    assert "code.commit is null" in commits["detail"] and "recipe.commit is null" in commits["detail"]
    assert set(statuses.values()) == {"match"} and len(statuses) == 7
    # Unlocked packages may move the last digits; nothing more.
    assert all(row["difference"] <= 1e-12 for row in found["rows"])


def test_both_recorded_runs_used_the_same_code_recipe_and_predictions():
    """The checkout run names the release commit and sdk_code_sha256. The plugin run installs
    the matching PyPI wheel version and reports the same digest of the SDK code, the same
    recipe bytes and the same predictions as the checkout run."""
    checkout = json.loads((RUN / "receipt.json").read_text())
    plugin = one(recorded("plugin-run.json")["calls"], "get_receipt")["result"]
    args = declared("rewirebench")
    assert args[args.index("--from") + 1] == f"rewirebench[mcp]=={checkout['code']['rewirebench_version']}"
    assert checkout["code"]["commit"]
    assert plugin["code"]["commit"] is None and plugin["code"]["commit_source"] is None
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


def test_the_pages_say_what_the_recorded_plugin_receipt_carries():
    """The README, docs/mcp.md, the run skill and the agent page each say what a receipt from
    an installed plugin holds."""
    code = one(recorded("plugin-run.json")["calls"], "get_receipt")["result"]["code"]
    assert code["commit"] is None and code["commit_source"] is None
    assert code["dirty"] is None and code["rewirebench_version"] == pinned_version()
    pages = {name: " ".join((ROOT / name).read_text().split()) for name in (
        "README.md", "docs/mcp.md", "docs/demo/README.md", "skills/run-benchmark/SKILL.md",
        "plugins/bio-model-benchmarks/skills/run-benchmark/SKILL.md",
        "plugins/bio-model-benchmarks/agents/benchmarking-agent.md")}
    for name, text in pages.items():
        assert re.search(r"strict", text) and re.search(r"wheel|PyPI", text), name
    section = pages["README.md"].split("What an installed plugin cannot do yet:")[1]
    section = section.split("## ")[0]
    assert f"rewirebench[mcp]=={pinned_version()}" in section
    assert "commits as not checked" in section
    for name in ("docs/mcp.md", "docs/demo/README.md", "skills/run-benchmark/SKILL.md",
                 "plugins/bio-model-benchmarks/skills/run-benchmark/SKILL.md"):
        # recipe.commit is not always null for a plugin: it depends on the folder.
        assert "inside a git work tree whose HEAD holds" in pages[name], name
        assert "`code.dirty` and `recipe.commit` are `null`" not in pages[name], name


def test_a_plugin_recipe_gets_a_commit_only_inside_a_work_tree_that_holds_it(tmp_path: Path):
    """What the pages above say about recipe.commit, run against the plugin's recipes."""
    plain = Path(shutil.copytree(PLUGIN / "recipes", tmp_path / "plain/recipes"))
    name = "mfass/mfass-v2-kmer-baseline.yaml"
    assert (plain / name).is_file()
    assert receipts.recipe_origin(plain / name) == (None, None)

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", *arguments],
            cwd=tmp_path / "plain", check=True, capture_output=True, text=True).stdout.strip()

    git("init", "--quiet")
    assert receipts.recipe_origin(plain / name) == (None, None)
    git("add", "recipes")
    git("commit", "--quiet", "-m", "a copy of the plugin")
    assert receipts.recipe_origin(plain / name) == (None, git("rev-parse", "HEAD"))
    (plain / name).write_text((plain / name).read_text() + "# Edited.\n")
    assert receipts.recipe_origin(plain / name) == (None, None)


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
    listed = " ".join(section(page, "Result").split("From the receipt:")[1].split())
    assert (f"code and recipe commit `{receipt['code']['commit']}` on `main`, clean tree, "
            f"`commit_source` `{receipt['code']['commit_source']}`") in listed
    assert receipt["recipe"]["commit"] == receipt["code"]["commit"]
    assert (f"wall time {receipt['wall_seconds']:.2f} seconds including the download, peak "
            f"memory {receipt['peak_memory_bytes'] / 1e6:.0f} MB") in listed
    assert page.startswith(f"# Recorded demo: MFASS v2 baseline, rerun through MCP tools only"
                           f"\n\nOn {receipt['verified_on']} ")

    side = {row[0]: row[1:] for row in table(page, "The same run through the plugin as shipped")}
    other = one(plugin["calls"], "get_receipt")["result"]
    keys = {"Precision at 100": "precision_at_capacity",
            "Average precision": "average_precision_sklearn", "AUROC": "auroc"}
    for label, key in keys.items():
        assert side[label] == [repr(receipt["metrics"][key]), repr(other["metrics"][key])]
    short = f"`{receipt['code']['commit'][:7]}`"
    assert side["`code.commit`"] == [short, "null"]
    assert side["`code.commit_source`"] == [f"`{receipt['code']['commit_source']}`", "null"]
    assert side["`code.dirty`"] == ["false", "null"]
    assert (receipt["code"]["dirty"], other["code"]["dirty"]) == (False, None)
    assert side["`recipe.commit`"] == [short, "null"] and other["recipe"]["commit"] is None
    assert side["Wall time"][0] == f"{receipt['wall_seconds']:.2f} s"
    assert side["Wall time"][1].startswith(f"{other['wall_seconds']:.2f} s")
    assert side["`sdk_code_sha256`"] == [f"`{receipt['code']['sdk_code_sha256'][:8]}...`"] * 2
    graph = [call for call in plugin["calls"] if call["server"] == "rewire-benchmark-graph"]
    assert (f"the same {len(plugin['calls'])} calls ({len(graph)} to the graph, "
            f"{len(plugin['calls']) - len(graph)} to rewirebench)") in " ".join(page.split())
    shipped = " ".join(section(page, "The same run through the plugin as shipped").split())
    counts = [sum(call["tool"] == "run_status" for call in run["calls"])
              for run in (plugin, transcript)]
    assert (f"It makes {counts[0]} `run_status` calls where the checkout run made "
            f"{counts[1]}") in shipped
    assert f"The checkout run made {len(calls)}." in shipped
    assert f"It was recorded on {other['verified_on']} " in shipped
    checked = [one(run["calls"], "verify_receipt")["result"] for run in (transcript, plugin)]
    assert [(result["strict"], result["passed"], result["not_checked"]) for result in checked] == [
        (True, True, []), (False, True, ["commits"])]
    assert side["`verify_receipt`"] == [
        "passed with `strict`", "passed; `commits` not checked, so `strict` does not pass"]


def pinned_version() -> str:
    source = declared("rewirebench")[declared("rewirebench").index("--from") + 1]
    found = re.fullmatch(r"rewirebench\[mcp\]==([0-9]+\.[0-9]+\.[0-9]+)", source)
    assert found, source
    return found.group(1)


def test_the_demo_page_says_which_pin_the_agent_session_is_from():
    """docs/demo/agent-session.md names the pin it was recorded at. While that is not the pin
    in the plugin's .mcp.json, the demo page must say so in both places that mention it, and
    must give the plugin's present version and pin."""
    session = " ".join((DEMO / "agent-session.md").read_text().split())
    found = re.search(r"`\.mcp\.json` pins rewirebench at `([0-9a-f]{40})`", session)
    assert found, "agent-session.md no longer names the pin it was recorded at"
    current = one(recorded("transcript.json")["calls"], "get_receipt")["result"]["code"]
    old, now = found.group(1), current["commit"]
    assert current["rewirebench_version"] == pinned_version()
    digest = re.search(r"`sdk_code_sha256`, `([0-9a-f]{64})`", session)
    assert digest and session.count(digest.group(1)) >= 2
    page = (DEMO / "README.md").read_text()
    version = json.loads((PLUGIN / ".claude-plugin/plugin.json").read_text())["version"]
    about = " ".join(section(page, "An agent session").split())
    assert f"The plugin is now {version} and pins `rewirebench[mcp]=={pinned_version()}` (`v{pinned_version()}`, commit `{now[:7]}`)." in about
    if old == now:
        assert digest.group(1) == current["sdk_code_sha256"]
        return
    assert digest.group(1) != current["sdk_code_sha256"]
    assert f"pins rewirebench at `{old[:7]}`" in about
    assert f"`sdk_code_sha256` `{digest.group(1)[:8]}...`" in about
    assert f"has not been recorded again: not at `{now[:7]}`" in about
    assert f"What follows describes the recording made at `{old[:7]}`." in about
    line = " ".join(section(page, "The command line at the pinned commit").split())
    assert (f"The agent session in [`agent-session.md`](agent-session.md) is from `{old[:7]}` "
            "and has not been recorded again.") in line
    whole = " ".join(page.split())
    assert "None has been recorded at the current pin." in whole
    assert f"recorded at an older pin (`{old[:7]}`)" in whole
    assert not re.search(r"(?<!not been )recorded again at", whole)


def test_the_command_line_section_is_of_the_pinned_commit():
    """The command line run's folder is not committed, so its times cannot be checked here.
    Its commit must be the release commit matching the plugin's pin, and its metrics and two
    digests the recorded ones."""
    page = (DEMO / "README.md").read_text()
    text = " ".join(section(page, "The command line at the pinned commit").split())
    receipt = json.loads((RUN / "receipt.json").read_text())
    pin = receipt["code"]["commit"]
    assert receipt["code"]["rewirebench_version"] == pinned_version()
    assert f"clone of tag `v{pinned_version()}` at commit `{pin[:7]}`, the release the plugin now pins" in text
    assert f"The `commits` check read code commit `{pin}` from the receipt." in text
    metrics = receipt["metrics"]
    assert (f"{int(metrics['n'])} variants, {int(metrics['positives'])} positives, precision at "
            f"100 of {metrics['precision_at_capacity']!r}, average precision "
            f"{metrics['average_precision_sklearn']!r} and AUROC {metrics['auroc']!r}") in text
    assert "`receipt verify --strict` passed: all eight checks match" in text
    outputs = {item["path"]: item["sha256"] for item in receipt["outputs"]}
    predictions = outputs["baselines/mfass-kmer-position-v2/predictions.json"]
    assert f"`predictions.json` had the SHA-256 `{predictions[:8]}...`" in text
    assert f"`sdk_code_sha256` was `{receipt['code']['sdk_code_sha256'][:8]}...`" in text
    assert "That run folder is not committed." in text
    assert f"needs commit `{pin[:7]}` in your clone's history" in " ".join(page.split())


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
