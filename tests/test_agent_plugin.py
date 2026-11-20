"""The Claude Code plugin must ship the same skill that is reviewed in skills/."""
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
from rewirebench import sdk

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills/bio-model-benchmarks"
PLUGIN = ROOT / "plugins/bio-model-benchmarks"


def files(root):
    return {
        p.relative_to(root): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def test_plugin_skill_is_a_byte_identical_copy():
    source, copy = files(SKILL), files(PLUGIN / "skills/bio-model-benchmarks")
    assert sorted(source) == sorted(copy), "Copy skills/bio-model-benchmarks into the plugin"
    changed = [str(path) for path in source if source[path] != copy[path]]
    assert not changed, f"Plugin skill differs from skills/bio-model-benchmarks: {changed}"


def test_skill_frontmatter_and_length():
    text = (SKILL / "SKILL.md").read_text()
    header = text.split("---")[1]
    fields = dict(line.split(": ", 1) for line in header.strip().splitlines())
    assert fields["name"] == "bio-model-benchmarks"
    assert fields["description"]
    assert len(text.splitlines()) <= 200


def test_marketplace_lists_the_plugin():
    marketplace = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    plugin = json.loads((PLUGIN / ".claude-plugin/plugin.json").read_text())
    [entry] = marketplace["plugins"]
    assert entry["name"] == plugin["name"] == "bio-model-benchmarks"
    assert (ROOT / entry["source"]).resolve() == PLUGIN
    assert entry["version"] == plugin["version"]


GRAPH_SERVER = "https://github.com/rewire-bio/rdf-kg-mcp"
# The skill uses search, list_queries and run_query, which first shipped in v0.4.0.
MINIMUM_GRAPH_SERVER = (0, 4, 0)
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


def test_mcp_pin_is_the_newest_release():
    """A stale pin fails here. Offline, the check is skipped locally and fails in CI."""
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
