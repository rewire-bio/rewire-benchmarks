"""The Claude Code plugin: its skills, commands, agent, MCP pins and copied recipes.

The plugin must ship the same skills that are reviewed in skills/ and the same recipes
that are validated in recipes/.
"""
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from rewirebench import mcp_server, sdk

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
SKILL = SKILLS / "bio-model-benchmarks"
PLUGIN = ROOT / "plugins/bio-model-benchmarks"
SKILL_NAMES = ["bio-model-benchmarks", "find-model-for-use-case", "prepare-benchmark-data",
               "run-benchmark", "verify-published-result"]
COMMANDS = ["data", "demo", "find", "plan", "run", "verify"]
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


def test_marketplace_lists_the_plugin():
    marketplace = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    plugin = json.loads((PLUGIN / ".claude-plugin/plugin.json").read_text())
    [entry] = marketplace["plugins"]
    assert entry["name"] == plugin["name"] == "bio-model-benchmarks"
    assert (ROOT / entry["source"]).resolve() == PLUGIN
    assert entry["version"] == plugin["version"]


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
    """No rewirebench release has `rewirebench mcp` yet, so the pin is a commit on main."""
    args = server_args("rewirebench")
    source = args[args.index("--from") + 1]
    match = re.fullmatch(
        r"rewirebench\[mcp\] @ git\+https://github\.com/rewire-bio/rewire-benchmarks@"
        r"([0-9a-f]{40})#subdirectory=packages/rewirebench", source)
    assert match, source
    assert args[args.index("--from") + 2:] == [
        "rewirebench", "mcp", "--recipes-dir", f"{PLUGIN_ROOT}/recipes"]
    assert match.group(1) in (ROOT / "README.md").read_text()


# Every backticked snake_case word in the new skills, the commands and the agent is either
# a tool of one of the three servers or one of these: fields, arguments and values.
NOT_TOOLS = {
    "average_precision_sklearn", "budget_bytes", "confirmed_amount",
    "expected_wall_seconds", "full_protocol", "kg_record_id", "kg_release_id",
    "limit_enforced", "plan_sha256", "pod_specs", "precision_at_capacity", "run_id",
    "started_by", "synthetic_smoke", "variant_type",
}


def recorded_tools() -> dict[str, set[str]]:
    """Tool names each server gave when it was started as the plugin declares it."""
    check = json.loads((ROOT / "docs/demo/plugin-servers.json").read_text())
    return {name: set(server["tools"]) for name, server in check["servers"].items()}


def test_recorded_tool_lists_match_the_sources_of_truth():
    tools = recorded_tools()
    assert tools["rewire-benchmark-graph"] == GRAPH_TOOLS
    assert tools["rewirebench"] == set(mcp_server.TOOLS)
    assert {"get_truth_set", "list_synthetic_protocols", "fetch_file"} <= tools["genomics-mcp"]


def test_skills_commands_and_agent_name_only_tools_that_exist():
    tools = set().union(*recorded_tools().values())
    pages = [SKILLS / name / "SKILL.md" for name in SKILL_NAMES if name != SKILL.name]
    pages += sorted((PLUGIN / "commands").glob("*.md")) + sorted((PLUGIN / "agents").glob("*.md"))
    for page in pages:
        words = set(re.findall(r"`([a-z0-9]+(?:_[a-z0-9]+)+)`", page.read_text()))
        unknown = sorted(words - tools - NOT_TOOLS)
        assert not unknown, f"{page.relative_to(ROOT)} names {unknown}: not a tool of any server"
    used = set().union(*(
        set(re.findall(r"`([a-z0-9]+(?:_[a-z0-9]+)+)`", page.read_text())) for page in pages))
    # The flows the plugin promises are all there.
    assert {"dataset_info", "run_query", "plan_run", "run_status", "get_receipt",
            "verify_receipt", "get_truth_set", "describe_protocol"} <= used
    assert NOT_TOOLS <= used, sorted(NOT_TOOLS - used)


def test_skills_never_tell_an_agent_to_confirm_a_paid_amount():
    for name in SKILL_NAMES:
        text = " ".join((SKILLS / name / "SKILL.md").read_text().split())
        if "confirmed_amount" in text:
            assert "Never copy an amount from the plan into `confirmed_amount`" in text
    agent = " ".join((PLUGIN / "agents/benchmarking-agent.md").read_text().split())
    assert "You never start a GPU run or a paid run" in agent
    assert "You never pass `confirmed_amount`" in agent


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
