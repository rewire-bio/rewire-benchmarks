"""The Claude Code plugin: its skills, commands, agent, MCP pins and copied recipes.

The plugin must ship the same skills that are reviewed in skills/ and the same recipes
that are validated in recipes/.
"""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from rewirebench import sdk

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
SKILL = SKILLS / "bio-model-benchmarks"
PLUGIN = ROOT / "plugins/bio-model-benchmarks"
SKILL_NAMES = ["author-benchmark", "bio-model-benchmarks", "find-model-for-use-case",
               "investigate-discrepancy", "prepare-benchmark-data", "run-benchmark",
               "verify-published-result"]
COMMANDS = ["author", "data", "demo", "discrepancy", "find", "plan", "run", "verify"]
PLUGIN_ROOT = "${CLAUDE_PLUGIN_ROOT}"


def files(root):
    return {
        p.relative_to(root): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def frontmatter(path: Path) -> dict[str, str]:
    header = path.read_text().split("---")[1]
    return dict(line.split(": ", 1) for line in header.strip().splitlines())


def test_plugin_skills_are_byte_identical_copies():
    source, copy = files(SKILLS), files(PLUGIN / "skills")
    assert sorted(source) == sorted(copy), "Copy skills/ into the plugin"
    changed = [str(path) for path in source if source[path] != copy[path]]
    assert not changed, f"Plugin skills differ from skills/: {changed}"


def test_the_skills_are_the_listed_ones():
    assert sorted(path.name for path in SKILLS.iterdir() if path.is_dir()) == SKILL_NAMES


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_skill_frontmatter_and_length(name: str):
    text = (SKILLS / name / "SKILL.md").read_text()
    fields = frontmatter(SKILLS / name / "SKILL.md")
    assert fields["name"] == name
    assert fields["description"]
    assert len(text.splitlines()) <= 200


def test_plugin_recipes_are_byte_identical_copies():
    """The plugin's rewirebench server reads the plugin's own recipes/, so an installed
    plugin needs no checkout. They must be the recipes that CI validates and runs."""
    source, copy = files(ROOT / "recipes"), files(PLUGIN / "recipes")
    assert sorted(source) == sorted(copy), "Copy recipes/ into the plugin"
    changed = [str(path) for path in source if source[path] != copy[path]]
    assert not changed, f"Plugin recipes differ from recipes/: {changed}"


def test_plugin_holds_every_repository_input_its_recipes_name():
    """A recipe input with repo_path is read from beside recipes/, here the plugin root."""
    named = set()
    for path in (PLUGIN / "recipes").rglob("*.yaml"):
        for item in yaml.safe_load(path.read_text())["inputs"]:
            if "repo_path" in item:
                named.add(item["repo_path"])
                assert (PLUGIN / item["repo_path"]).read_bytes() == (
                    ROOT / item["repo_path"]).read_bytes(), item["repo_path"]
    assert named == {"workflows/benchmark/examples/smoke.csv"}


def test_commands_and_agent_have_descriptions():
    assert sorted(path.stem for path in (PLUGIN / "commands").glob("*.md")) == COMMANDS
    for name in COMMANDS:
        fields = frontmatter(PLUGIN / "commands" / f"{name}.md")
        assert fields["description"] and "$ARGUMENTS" in (
            PLUGIN / "commands" / f"{name}.md").read_text()
    [agent] = (PLUGIN / "agents").glob("*.md")
    fields = frontmatter(agent)
    assert fields["name"] == agent.stem == "benchmarking-agent"
    assert fields["description"]


def test_the_agent_can_use_the_three_servers_tools_and_nothing_else():
    """The agent's limits are its tools list, not its prompt. Without the list a plugin
    agent inherits every tool of the session: a shell, and any RunPod tool installed.
    Claude Code names a plugin's MCP tool mcp__plugin_<plugin>_<server>__<tool>."""
    fields = frontmatter(PLUGIN / "agents/benchmarking-agent.md")
    assert set(fields) == {"name", "description", "tools"}
    allowed = fields["tools"].split(", ")
    assert len(allowed) == len(set(allowed))
    expected = {f"mcp__plugin_bio-model-benchmarks_{server}__{tool}"
                for server, tools in recorded_tools().items() for tool in tools}
    assert set(allowed) == expected
    assert len(allowed) == len(GRAPH_TOOLS) + len(GENOMICS_TOOLS) + len(REWIREBENCH_TOOLS)
    servers = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]
    for tool in allowed:
        assert re.fullmatch(r"mcp__plugin_bio-model-benchmarks_([a-z-]+)__([a-z_]+)", tool), tool
        assert tool.split("__")[1].removeprefix("plugin_bio-model-benchmarks_") in servers
    body = (PLUGIN / "agents/benchmarking-agent.md").read_text().split("\n---\n", 1)[1]
    assert "You have no shell" in body and "you cannot load skills" in body


def test_marketplace_lists_the_plugin():
    marketplace = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    plugin = json.loads((PLUGIN / ".claude-plugin/plugin.json").read_text())
    [entry] = marketplace["plugins"]
    assert entry["name"] == plugin["name"] == "bio-model-benchmarks"
    assert (ROOT / entry["source"]).resolve() == PLUGIN
    assert entry["version"] == plugin["version"]
    assert entry["description"] == plugin["description"]


def test_marketplace_describes_what_the_plugin_does_today():
    """The plugin reruns registered CPU baselines. It does not reproduce a paper's result
    independently, and a GPU run gets a quote and is not started."""
    marketplace = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    description = marketplace["metadata"]["description"]
    assert "find published results" in description
    assert "rerun registered CPU baselines" in description
    assert "A GPU run is quoted, not launched." in description
    assert "reproduc" not in description.lower()


GRAPH_SERVER = "https://github.com/rewire-bio/rdf-kg-mcp"
# v0.1.0 has no search, list_queries or run_query, which the product page describes.
# v0.4.0 has all eight read tools and was read at its tag when this floor was set.
MINIMUM_GRAPH_SERVER = (0, 4, 0)
# The read tools in src/rdf_kg_mcp/server.py at tag v0.4.0, read on 2026-11-20.
GRAPH_TOOLS = {"dataset_info", "describe_schema", "find_entity", "search", "describe_resource",
               "sparql_query", "list_queries", "run_query"}
SKIP_FRESHNESS = "REWIRE_SKIP_PIN_FRESHNESS"
RELEASE_TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)")


def version_tuple(tag: str) -> tuple[int, int, int]:
    match = RELEASE_TAG.fullmatch(tag)
    assert match, f"Not a release tag: {tag}"
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def pinned_tag() -> str:
    servers = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]
    server = servers["rewire-benchmark-graph"]
    assert server["command"] == "uvx"
    source = server["args"][server["args"].index("--from") + 1]
    prefix = f"git+{GRAPH_SERVER}@"
    assert source.startswith(prefix)
    assert server["args"][-2:] == ["serve", "https://benchmarks.rewirebio.io/agents/kg.tar.gz"]
    return source.removeprefix(prefix)


def newest_release(ls_remote_output: str) -> str:
    """Newest vX.Y.Z tag in `git ls-remote --tags --refs` output, compared as integers."""
    tags = [line.rsplit("refs/tags/", 1)[1] for line in ls_remote_output.splitlines()
            if "refs/tags/" in line]
    releases = [tag for tag in tags if RELEASE_TAG.fullmatch(tag)]
    assert releases, "No release tags found"
    return max(releases, key=version_tuple)


def test_mcp_server_is_pinned():
    assert version_tuple(pinned_tag()) >= MINIMUM_GRAPH_SERVER


def test_newest_release_sorts_numerically_and_ignores_other_tags():
    listing = "\n".join(f"{'a' * 40}\trefs/tags/{tag}"
                        for tag in ("v0.9.0", "v0.10.0", "v0.2.1", "nightly", "v1.0.0rc1"))
    assert newest_release(listing) == "v0.10.0"


def test_readme_shows_the_plugin_pin():
    assert f"git+{GRAPH_SERVER}@{pinned_tag()}" in (ROOT / "README.md").read_text()


COMMIT = re.compile(r"[0-9a-f]{40}")


def server_args(name: str) -> list[str]:
    server = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"][name]
    assert server["command"] == "uvx"
    return server["args"]


def test_plugin_declares_three_servers():
    servers = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]
    assert list(servers) == ["rewire-benchmark-graph", "genomics-mcp", "rewirebench"]


def test_genomics_server_is_pinned_to_one_commit():
    """No genomics-mcp release has the truth-set tools yet, so the pin is a commit. The
    build constraints must come from the same commit as the code."""
    args = server_args("genomics-mcp")
    source = args[args.index("--from") + 1]
    prefix = "git+https://github.com/rewire-bio/genomics-mcp@"
    assert source.startswith(prefix)
    commit = source.removeprefix(prefix)
    assert COMMIT.fullmatch(commit)
    constraints = args[args.index("--build-constraints") + 1]
    assert constraints == ("https://raw.githubusercontent.com/rewire-bio/genomics-mcp/"
                           f"{commit}/packaging/build-constraints.txt")
    # Without a source build pyBigWig cannot read remote files (genomics-mcp docs/install.md).
    assert args[args.index("--no-binary-package") + 1] == "pybigwig"
    assert args[-1] == "genomics-mcp"
    assert commit in (ROOT / "README.md").read_text()


def test_rewirebench_server_is_pinned_and_reads_the_plugin_recipes():
    """The plugin pins the released rewirebench PyPI version matching packages/rewirebench."""
    import tomllib

    version = tomllib.loads((ROOT / "packages/rewirebench/pyproject.toml").read_text())["project"]["version"]
    args = server_args("rewirebench")
    source = args[args.index("--from") + 1]
    assert source == f"rewirebench[mcp]=={version}", source
    assert args[args.index("--from") + 2:] == [
        "rewirebench", "mcp", "--recipes-dir", f"{PLUGIN_ROOT}/recipes"]
    readme = (ROOT / "README.md").read_text()
    assert f"v{version}" in readme and f"rewirebench[mcp]=={version}" in readme


# Every backticked lower-case word in the new skills, the commands and the agent is either
# a tool of one of the three servers or one of these: fields, arguments and values. A new
# word must be added here by hand, so a tool that does not exist cannot be named unnoticed.
NOT_TOOLS = {
    "allow_partial", "applies_to", "average_precision_sklearn", "batch_size", "benchmark",
    "benchmark_run", "budget_bytes", "configuration", "confirmed_amount",
    "contested_results_for_use_case", "dataset_subset", "discrepancies_for_system",
    "discrepancy_history", "discrepancy_review", "evaluation", "expected_wall_seconds",
    "full_protocol", "gaps", "harness_run", "install_metadata", "kg_record_id", "kg_release_id",
    "kind", "limit_enforced", "model_name", "open_discrepancies", "plan_sha256", "pod_specs",
    "precision_at_capacity", "provenance", "purpose", "rank_for_use_case", "recipe_for",
    "reproduced", "result", "results_for_model", "rewire_run", "run_id", "sha256", "source_checked",
    "started_by", "status", "store_recipe_id", "synthetic_smoke", "timeout_seconds",
    "training_overlap", "variant_type",
    "20", "assembly", "backend", "chr20", "command", "commits", "contig", "cost", "cpu",
    "device", "eligible", "finished", "hardware", "limitations", "local", "method", "model",
    "null", "offline", "output", "planned", "prepared", "protocol", "quote", "recipe",
    "rewirebench", "runnable", "runpod", "sample", "scope", "scored", "setting", "source",
    "spark", "state", "strict", "version",
}
WORD = r"`([a-z0-9]+(?:_[a-z0-9]+)*)`"
# The tools of genomics-mcp at the commit the plugin pins: the values of the Operation enum
# in src/genomics_mcp/registry.py, which server.py registers one tool for, read at 3b395a9 on
# 2026-11-21. And of rewirebench at the commit it pins (mcp_server.TOOLS at 6d98f09). A
# recording that lists another tool was not made with the pinned servers. Update these when
# a pin is raised.
GENOMICS_TOOLS = {
    "cancel_transfer", "compare_samples", "convert_sra_run", "describe_dataset",
    "describe_synthetic_protocol", "fetch_file", "get_coverage", "get_features", "get_pileup",
    "get_reads", "get_sample_metadata", "get_sequence", "get_signal", "get_transfer_status",
    "get_truth_set", "get_variants", "inspect_locus", "list_files", "list_samples",
    "list_sources", "list_synthetic_protocols", "lookup_gene", "lookup_protein",
    "lookup_variant", "normalize_variant", "resolve_identifier", "search_datasets",
}
REWIREBENCH_TOOLS = {
    "list_protocols", "describe_protocol", "list_recipes", "describe_recipe", "list_models",
    "prepare", "plan_run", "run", "run_status", "cancel_run", "evaluate", "get_receipt",
    "verify_receipt",
}


def recorded_tools() -> dict[str, set[str]]:
    """Tool names each server gave when it was started as the plugin declares it."""
    check = json.loads((ROOT / "docs/demo/plugin-servers.json").read_text())
    return {name: set(server["tools"]) for name, server in check["servers"].items()}


def test_recorded_tool_lists_are_the_tools_of_the_pinned_servers():
    """The recording is what the pinned servers offer. It is compared with lists read at the
    pinned commits, not with the server code in this checkout, which may gain a tool before
    the plugin's pin is raised. A tool added to the recording by hand fails here."""
    assert recorded_tools() == {"rewire-benchmark-graph": GRAPH_TOOLS,
                                "genomics-mcp": GENOMICS_TOOLS,
                                "rewirebench": REWIREBENCH_TOOLS}
    # Both recorded runs listed the same tools for the servers they started.
    for name in ("transcript.json", "plugin-run.json"):
        servers = json.loads((ROOT / "docs/demo" / name).read_text())["servers"]
        assert {server: set(entry["tools"]) for server, entry in servers.items()} == {
            "rewire-benchmark-graph": GRAPH_TOOLS, "rewirebench": REWIREBENCH_TOOLS}


def test_the_recorded_server_check_holds_real_answers():
    """One read-only call per server, and each answer is read: an error, or a reply that is
    not what the tool returns, fails."""
    check = json.loads((ROOT / "docs/demo/plugin-servers.json").read_text())
    calls = {call["server"]: call for call in check["calls"]}
    assert [(call["server"], call["tool"]) for call in check["calls"]] == [
        ("rewire-benchmark-graph", "dataset_info"), ("genomics-mcp", "get_truth_set"),
        ("rewirebench", "list_recipes")]
    assert not any(call["is_error"] for call in check["calls"])
    assert check["summary"] == {
        name: {"version": server["version"], "tools": len(server["tools"])}
        for name, server in check["servers"].items()}
    info = calls["rewire-benchmark-graph"]["result"]
    assert info["dataset"] == "rewire-benchmark-data"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[0-9a-f]{12}", info["version"])
    truth = calls["genomics-mcp"]["result"]
    assert truth["operation"] == "get_truth_set" and truth["status"] == "ok"
    assert truth["data"]["records"][0]["id"] == "giab-hg002-v4.2.1-grch38"
    listed = calls["rewirebench"]["result"]
    assert listed["recipes_dir"] == f"{PLUGIN_ROOT}/recipes" and listed["invalid"] == []
    shipped = {yaml.safe_load(path.read_text())["id"]: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (PLUGIN / "recipes").rglob("*.yaml")}
    assert {item["id"]: item["sha256"] for item in listed["recipes"]} == shipped


def pages() -> list[Path]:
    """The new skills, the commands and the agent: what an agent reads to choose a tool."""
    found = [SKILLS / name / "SKILL.md" for name in SKILL_NAMES if name != SKILL.name]
    return found + sorted((PLUGIN / "commands").glob("*.md")) + sorted(
        (PLUGIN / "agents").glob("*.md"))


def body(page: Path) -> str:
    """A page without its front matter, where the agent's tools list is."""
    return page.read_text().split("\n---\n", 1)[1]


def test_skills_commands_and_agent_name_only_tools_that_exist():
    tools = set().union(*recorded_tools().values())
    assert not tools & NOT_TOOLS
    for page in pages():
        unknown = sorted(set(re.findall(WORD, body(page))) - tools - NOT_TOOLS)
        assert not unknown, f"{page.relative_to(ROOT)} names {unknown}: not a tool of any server"
        # A numbered step that starts with a backticked word starts with a tool.
        for step in re.findall(r"^\s*\d+\. `([a-z0-9_]+)`", body(page), flags=re.MULTILINE):
            assert step in tools, f"{page.relative_to(ROOT)}: step starts with {step}"
    used = set().union(*(set(re.findall(WORD, body(page))) for page in pages()))
    # The flows the plugin promises are all there, single-word tools included.
    assert {"dataset_info", "run_query", "plan_run", "run", "run_status", "get_receipt",
            "verify_receipt", "get_truth_set", "describe_protocol", "prepare",
            "evaluate"} <= used
    assert NOT_TOOLS <= used, sorted(NOT_TOOLS - used)


def sentences(page: Path) -> list[str]:
    return re.split(r"(?<=[.?!])\s+", " ".join(body(page).split()))


def test_no_page_tells_an_agent_to_confirm_an_amount_or_send_a_pod_spec():
    """Skills, commands and the agent. Every sentence that names `confirmed_amount` or
    `pod_specs` forbids its use, so an instruction to pass one cannot be added beside the
    sentence that forbids it."""
    named = []
    for page in pages() + [SKILL / "SKILL.md"]:
        for sentence in sentences(page):
            if "confirmed_amount" in sentence or "pod_specs" in sentence:
                named.append(page.name)
                assert re.search(r"\b[Nn]ever\b", sentence), (
                    f"{page.relative_to(ROOT)}: {sentence!r}")
            if re.search(r"max_amount|quote", sentence) and re.search(
                    r"\b(copy|pass|send|confirm)\b", sentence, flags=re.IGNORECASE):
                assert re.search(r"\b[Nn]ever\b|\bshow\b", sentence), (
                    f"{page.relative_to(ROOT)}: {sentence!r}")
    assert named.count("benchmarking-agent.md") >= 1 and named.count("run.md") >= 1
    skill = " ".join(body(SKILLS / "run-benchmark/SKILL.md").split())
    assert "Never copy an amount from the plan into `confirmed_amount`" in skill
    assert "Never send a plan's `pod_specs` to RunPod with another tool" in skill
    agent = " ".join(body(PLUGIN / "agents/benchmarking-agent.md").split())
    assert "You never start a GPU run or a paid run" in agent
    assert "You never pass `confirmed_amount`" in agent
    assert "Never call `run` without a `plan_sha256`" in agent
    run = " ".join(body(PLUGIN / "commands/run.md").split())
    assert "Never pass `confirmed_amount`" in run


def test_no_page_says_describe_protocol_gives_a_protocols_inputs():
    """For five of the eight protocols the tool returns capabilities only. A page that
    promised inputs from it would leave an agent to fill them in from memory."""
    bare = [protocol for protocol in sdk.PROTOCOLS
            if set(sdk.describe(protocol)) <= {"protocol_id", "capabilities"}]
    assert len(bare) == 5 and "mfass-v2" in bare
    fuller = sorted(set(sdk.PROTOCOLS) - set(bare))
    data = " ".join(body(SKILLS / "prepare-benchmark-data/SKILL.md").split())
    for protocol in fuller:
        assert f"`{protocol}`" in (SKILLS / "prepare-benchmark-data/SKILL.md").read_text()
    assert "The other five, `mfass-v2` among them, return capabilities only" in data
    assert "Do not list inputs, sizes or licences from memory" in data
    for page in pages():
        for sentence in sentences(page):
            if "`describe_protocol`" in sentence and re.search(
                    r"\binputs?\b|\blicen[cs]e\b|\bsizes?\b|\bdatasets\b", sentence):
                assert re.search(r"capabilities|no inputs|[Dd]o not", sentence), (
                    f"{page.relative_to(ROOT)}: {sentence!r}")
    command = " ".join(body(PLUGIN / "commands/data.md").split())
    assert "it gives no inputs" in command and "Do not fill them in from memory" in command
    assert "`describe_protocol` gives the protocol's capabilities, baselines and models" in command
    for page in pages():
        text = " ".join(body(page).split())
        assert not re.search(r"`describe_protocol`[^.]{0,40}\b(lists?|returns?|gives?|shows?)"
                             r"( each| the| its)? inputs?\b", text), page.name


# How each protocol module is named in prose.
PROTOCOL_NAMES = {
    "dart_eval": "DART-Eval", "flip2": "FLIP2", "genomic_benchmarks": "Genomic Benchmarks",
    "mfass": "MFASS v2", "mrnabench": "mRNABench", "proteingym": "ProteinGym",
    "tdc_admet": "TDC ADMET",
}
RANDOM_SOURCES = ("secrets", "uuid", "urandom", "import random", "default_rng", "time.time")


def test_the_pages_name_the_protocols_that_give_rows_random_ids():
    """Why two runs differ depends on the protocol. The pages once said prepare gives rows
    random IDs for every protocol; it does for four of the seven modules. The list in the
    skill and the docs is read against the protocol code."""
    import rewirebench.protocols as protocols

    folder = Path(protocols.__file__).parent
    modules = {path.stem: path.read_text() for path in folder.glob("*.py")
               if path.stem != "__init__"}
    assert set(modules) == set(PROTOCOL_NAMES)
    random = sorted(name for name, text in modules.items() if "secrets.token_hex(" in text)
    fixed = sorted(set(modules) - set(random))
    assert random == ["dart_eval", "flip2", "genomic_benchmarks", "mrnabench"]
    for name in fixed:
        for source in RANDOM_SOURCES:
            assert source not in modules[name], f"{name} draws on {source}"
    listed = "FLIP2, DART-Eval, mRNABench and Genomic Benchmarks"
    assert sorted(listed.replace(" and ", ", ").split(", ")) == sorted(
        PROTOCOL_NAMES[name] for name in random)
    stable = "MFASS v2, ProteinGym and TDC ADMET"
    assert sorted(stable.replace(" and ", ", ").split(", ")) == sorted(
        PROTOCOL_NAMES[name] for name in fixed)
    texts = {name: " ".join((ROOT / name).read_text().split()) for name in (
        "skills/run-benchmark/SKILL.md", "docs/mcp.md", "docs/spark.md", "docs/demo/README.md")}
    for name in ("skills/run-benchmark/SKILL.md", "docs/mcp.md", "docs/spark.md"):
        assert listed in texts[name], name
    for name in ("docs/mcp.md", "docs/spark.md"):
        assert f"{stable} keep their rows' own IDs" in texts[name], name
    for name, text in texts.items():
        assert "because `prepare` gives rows random IDs" not in text, name
        assert "`prepare` gives every row a random opaque ID, so" not in text, name
    assert "byte-identical between runs on one machine" in texts["skills/run-benchmark/SKILL.md"]
    assert "do not compare digests across machines" in texts["skills/run-benchmark/SKILL.md"]


def test_the_pages_say_what_a_verified_receipt_does_not_show():
    """A pass, strict included, reads the checked values from the report and the commit from
    the receipt. The skill, the agent and the docs say so wherever they describe a pass."""
    skill = " ".join(body(SKILLS / "run-benchmark/SKILL.md").split())
    assert "It does not score the predictions again" in skill
    assert ("A pass, `strict` included, does not show that the metrics follow from the "
            "predictions") in skill
    assert "reads the commit the receipt states and does not look it up" in skill
    agent = " ".join(body(PLUGIN / "agents/benchmarking-agent.md").split())
    assert "without scoring the predictions again" in agent
    for name in ("docs/mcp.md", "docs/recipes.md", "docs/demo/README.md"):
        text = " ".join((ROOT / name).read_text().split())
        assert "predictions are not scored again" in text or (
            "predictions were not scored again" in text), name
        assert "not look" in text and "up" in text, name


def test_the_values_the_pages_quote_are_the_recorded_ones():
    """/demo and the verify skill tell an agent what to expect. Those figures are read
    against the recorded run and the graph's recorded reply."""
    transcript = json.loads((ROOT / "docs/demo/transcript.json").read_text())
    summary = transcript["summary"]
    values = {name: entry["value"] for name, entry in summary["published"]["values"].items()}
    demo = " ".join(body(PLUGIN / "commands/demo.md").split())
    quoted = re.search(r"precision at 100 should be ([0-9.]+), average precision ([0-9.]+) "
                       r"and AUROC ([0-9.]+), on ([0-9,]+) of ([0-9,]+) variants", demo)
    assert quoted, "commands/demo.md no longer states the expected values"
    assert [float(quoted.group(index)) for index in (1, 2, 3)] == [
        round(summary["metrics"]["precision_at_capacity"], 4),
        round(summary["metrics"]["average_precision_sklearn"], 4),
        round(summary["metrics"]["auroc"], 4)]
    scored = f"{int(summary['metrics']['n']):,}"
    assert (quoted.group(4), quoted.group(5)) == (scored, scored)
    skill = " ".join(body(SKILLS / "verify-published-result/SKILL.md").split())
    assert (f"On release `{summary['kg_release_id']}` this returns AUROC {values['auroc']!r}, "
            f"average precision {values['average-precision']!r} and precision at 100 of "
            f"{values['precision-at-100']!r}, each on 8324 of 8324") in skill
    plugin = json.loads((ROOT / "docs/demo/plugin-run.json").read_text())["summary"]["metrics"]
    floats = " ".join(body(SKILLS / "run-benchmark/SKILL.md").split())
    assert (f"was {plugin['average_precision_sklearn']!r} from the plugin and "
            f"{summary['metrics']['average_precision_sklearn']!r} from the checkout") in floats


def test_skill_uses_only_tools_the_pinned_server_has():
    """The skill's graph steps each start with a tool name. Each must be in the pinned release."""
    text = (SKILL / "SKILL.md").read_text()
    steps = re.findall(r"^\d+\. `([a-z_]+)`", text, flags=re.MULTILINE)
    assert steps[:3] == ["dataset_info", "find_entity", "sparql_query"]
    assert set(steps) <= GRAPH_TOOLS, sorted(set(steps) - GRAPH_TOOLS)
    assert "`describe_resource`" in text and "describe_resource" in GRAPH_TOOLS


def test_skill_reference_states_which_server_release_its_queries_ran_on():
    text = (SKILL / "reference/kg-queries.md").read_text()
    assert f"plugin pins {pinned_tag()}" in " ".join(text.split())


def test_mcp_pin_is_the_newest_release():
    """A stale pin fails here. Offline, the check is skipped locally and fails in CI.

    The release workflow sets REWIRE_SKIP_PIN_FRESHNESS, so a release of this repository
    does not wait on another repository's tags.
    """
    if os.environ.get(SKIP_FRESHNESS):
        pytest.skip(f"{SKIP_FRESHNESS} is set")
    try:
        listing = subprocess.run(
            ["git", "ls-remote", "--tags", "--refs", GRAPH_SERVER],
            check=True, capture_output=True, text=True, timeout=60,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"Cannot list rdf-kg-mcp tags: {type(error).__name__}")
    newest = newest_release(listing)
    assert pinned_tag() == newest, (
        f"plugins/bio-model-benchmarks/.mcp.json pins rdf-kg-mcp {pinned_tag()}; "
        f"the newest release is {newest}. Raise the pin here and in README.md."
    )


def test_protocol_table_matches_the_sdk():
    table = (SKILL / "reference/protocols.md").read_text().split("## Summary", 1)[1]
    rows = [line for line in table.splitlines() if line.startswith("| `")]
    listed = [row.split("`")[1] for row in rows]
    assert listed == list(sdk.PROTOCOLS)
