"""The Claude Code plugin must ship the same skill that is reviewed in skills/."""
import json
from pathlib import Path

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


def test_mcp_server_is_pinned():
    servers = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]
    server = servers["rewire-benchmark-graph"]
    assert server["command"] == "uvx"
    source = server["args"][server["args"].index("--from") + 1]
    assert source.startswith("git+https://github.com/rewire-bio/rdf-kg-mcp@v")
    assert server["args"][-2:] == ["serve", "https://benchmarks.rewirebio.io/agents/kg.tar.gz"]


def test_protocol_table_matches_the_sdk():
    table = (SKILL / "reference/protocols.md").read_text().split("## Summary", 1)[1]
    rows = [line for line in table.splitlines() if line.startswith("| `")]
    listed = [row.split("`")[1] for row in rows]
    assert listed == list(sdk.PROTOCOLS)
