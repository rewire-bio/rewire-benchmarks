"""Run one recipe through MCP tool calls only, and record every call.

This is the client side of the demo in docs/demo/. It starts the MCP servers the
``bio-model-benchmarks`` plugin declares, as processes speaking MCP on stdio, and makes
the calls the plugin's skills describe:

1. graph (optional): ``dataset_info``, ``find_entity``, ``sparql_query`` for the published
   values of the system the recipe reruns;
2. rewirebench: ``list_recipes``, ``describe_recipe``, ``plan_run``, ``run``, ``run_status``
   until finished, ``get_receipt``, ``verify_receipt``.

It calls no rewirebench function and starts no rewirebench command itself: the only way it
reaches the runner is ``session.call_tool``. It is a scripted client, not a model. The
transcript it writes is the list of tool calls with their arguments and results, with
local folder names replaced by labels.

Usage, from a checkout::

    uv run python scripts/demo/mcp_demo.py --recipe mfass-v2-kmer-baseline \\
        --output /absolute/new/folder --transcript transcript.json \\
        --graph plugins/bio-model-benchmarks/.mcp.json --strict

``--servers plugins/bio-model-benchmarks/.mcp.json`` instead starts each server exactly as
the plugin declares it and makes one read-only call to each.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

GRAPH = "rewire-benchmark-graph"
RUNNER = "rewirebench"
PLUGIN_ROOT = "${CLAUDE_PLUGIN_ROOT}"
POLL_SECONDS = 2.0
# Graph metric IRI suffix to the key the runner reports. The names differ: see
# skills/bio-model-benchmarks/reference/kg-queries.md.
METRIC_KEYS = {
    "precision-at-100": "precision_at_capacity",
    "average-precision": "average_precision_sklearn",
    "auroc": "auroc",
}
# A published value and a rerun of the same deterministic CPU baseline should agree far
# inside this. The recipe's own tolerances are checked by the runner, separately.
COMPARISON_TOLERANCE = 1e-9
# One read-only call per server, for --servers.
FIRST_CALLS: dict[str, tuple[str, dict[str, Any]]] = {
    "rewire-benchmark-graph": ("dataset_info", {}),
    "genomics-mcp": ("get_truth_set", {"sample": "HG002", "assembly": "GRCh38",
                                       "variant_type": "snv", "contig": "chr20"}),
    "rewirebench": ("list_recipes", {}),
}
LITERAL = re.compile(r'"(.*)"(?:\^\^\S+|@[A-Za-z-]+)?', re.DOTALL)
PUBLISHED_QUERY = """SELECT ?metric ?value ?scored ?eligible ?status WHERE {{
  ?result rb:resultFor {system} ; rb:resultOn {protocol} ; rb:metric ?metric ;
          mls:hasValue ?value ; rb:status ?status .
  OPTIONAL {{ ?result rb:coverage ?c . ?c rb:scoredCount ?scored ; rb:eligibleCount ?eligible }}
}} ORDER BY ?metric"""


class DemoError(RuntimeError):
    """A tool call failed, or a result was not what the demo requires."""


@dataclass
class Transcript:
    """Every tool call in order. ``labels`` maps local folder names to what replaces them."""

    labels: dict[str, str] = field(default_factory=dict)
    servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def label(self, path: str | Path, name: str) -> None:
        """Replace a folder, as given and as resolved, with a label in what is written."""
        for text in {str(path), str(Path(path).resolve())}:
            self.labels[text] = name

    def clean(self, value: Any) -> Any:
        """A copy of a JSON value with labelled folders replaced, longest name first."""
        text = json.dumps(value)
        for path in sorted(self.labels, key=len, reverse=True):
            text = text.replace(json.dumps(path)[1:-1], self.labels[path])
        return json.loads(text)

    def label_this_machine(self) -> None:
        """Label the folders a run can name: this Python, the temporary and home folders."""
        self.label(Path(sys.executable).parent, "<python>")
        self.label(Path(sys.prefix), "<environment>")
        self.label(tempfile.gettempdir(), "<temp>")
        self.label(Path.home(), "<home>")

    def document(self) -> dict[str, Any]:
        return self.clean({
            "what": "Tool calls made by scripts/demo/mcp_demo.py, a scripted MCP client. "
                    "No model chose the calls. Local folder names are replaced by labels.",
            "servers": self.servers,
            "calls": self.calls,
            "summary": self.summary,
        })

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.document(), indent=2) + "\n")


def _content(result: Any) -> Any:
    """The JSON a tool returned: its structured content, else its text read as JSON."""
    structured = getattr(result, "structuredContent", None)
    text = "\n".join(item.text for item in result.content if hasattr(item, "text"))
    try:
        return json.loads(text)
    except ValueError:
        return structured if structured is not None and not result.isError else text


async def call(session: ClientSession, transcript: Transcript, server: str, tool: str,
               arguments: dict[str, Any] | None = None, *, may_fail: bool = False) -> Any:
    """Call one tool, record it, and return its result as JSON."""
    arguments = arguments or {}
    result = await session.call_tool(tool, arguments)
    content = _content(result)
    transcript.calls.append({
        "number": len(transcript.calls) + 1,
        "at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "server": server, "tool": tool, "arguments": arguments,
        "is_error": bool(result.isError), "result": content,
    })
    if result.isError and not may_fail:
        raise DemoError(f"{server}.{tool} failed: {content}")
    return content


@asynccontextmanager
async def connect(transcript: Transcript, server: str, command: str, args: list[str],
                  *, env: dict[str, str] | None = None,
                  cwd: str | None = None) -> AsyncIterator[ClientSession]:
    """Start a server on stdio, initialise the session and record its name and tools."""
    parameters = StdioServerParameters(command=command, args=args, env=env, cwd=cwd)
    try:
        async with (stdio_client(parameters) as (read, write),
                    ClientSession(read, write) as session):
            started = await session.initialize()
            tools = await session.list_tools()
            transcript.servers[server] = {
                "command": [command, *args],
                "name": started.serverInfo.name,
                "version": started.serverInfo.version,
                "tools": sorted(tool.name for tool in tools.tools),
            }
            yield session
    except BaseExceptionGroup as group:
        # The transport's task group wraps what the caller raised. Give it back as it was.
        error: BaseException = group
        while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
            error = error.exceptions[0]
        raise error from None


def _rows(result: Any) -> list[dict[str, Any]]:
    """Rows of a ``sparql_query`` reply."""
    if isinstance(result, dict) and isinstance(result.get("rows"), list):
        return result["rows"]
    raise DemoError(f"sparql_query returned no rows: {result!r}")


def _plain(value: Any) -> Any:
    """A cell of a SPARQL row without its quotes, datatype or language tag.

    The graph server prints literals in Turtle form, such as ``"0.61"^^xsd:decimal``.
    """
    if isinstance(value, str):
        match = LITERAL.fullmatch(value)
        return match.group(1) if match else value
    return value


async def published_values(session: ClientSession, transcript: Transcript, *, text: str,
                           system: str, protocol: str) -> dict[str, Any]:
    """Ask the graph for its release and the published values of one system on a protocol."""
    info = await call(session, transcript, GRAPH, "dataset_info")
    await call(session, transcript, GRAPH, "find_entity", {"text": text, "type": "rb:Protocol"})
    reply = await call(session, transcript, GRAPH, "sparql_query", {
        "query": PUBLISHED_QUERY.format(system=system, protocol=protocol)})
    values: dict[str, Any] = {}
    for row in _rows(reply):
        metric = str(row["metric"]).removeprefix("metric:")
        values[metric] = {
            "value": float(_plain(row["value"])),
            "scored": int(_plain(row["scored"])) if row.get("scored") else None,
            "eligible": int(_plain(row["eligible"])) if row.get("eligible") else None,
            "status": str(row.get("status", "")).removeprefix("status:") or None,
        }
    if not values:
        raise DemoError(f"The graph holds no result for {system} on {protocol}")
    return {"kg_release_id": info["version"], "system": system, "protocol": protocol,
            "values": values}


async def run_recipe(session: ClientSession, transcript: Transcript, *, recipe: str,
                     output: Path, kg_release_id: str | None = None, strict: bool = False,
                     timeout_seconds: float = 900.0) -> dict[str, Any]:
    """List, describe, plan, run, poll, read the receipt and verify it. Tool calls only."""
    listed = await call(session, transcript, RUNNER, "list_recipes")
    if recipe not in [item["id"] for item in listed["recipes"]]:
        raise DemoError(f"list_recipes does not offer {recipe}")
    await call(session, transcript, RUNNER, "describe_recipe", {"recipe": recipe})
    arguments: dict[str, Any] = {"recipe": recipe, "backend": "local", "output": str(output)}
    if kg_release_id is not None:
        arguments["kg_release_id"] = kg_release_id
    plan = await call(session, transcript, RUNNER, "plan_run", arguments)
    if plan["quote"]["paid"] or not plan["runnable"]:
        raise DemoError(f"The plan is paid or not runnable: {plan}")
    status = await call(session, transcript, RUNNER, "run",
                        {**arguments, "plan_sha256": plan["plan_sha256"]})
    with anyio.fail_after(timeout_seconds):
        while status["state"] != "finished":
            await anyio.sleep(POLL_SECONDS)
            status = await call(session, transcript, RUNNER, "run_status",
                                {"run_id": status["run_id"]})
    if status["exit_code"] != 0 or not status["checks_passed"]:
        raise DemoError(f"The run failed: {status}")
    receipt = await call(session, transcript, RUNNER, "get_receipt", {"run_id": status["run_id"]})
    verified = await call(session, transcript, RUNNER, "verify_receipt",
                          {"run_id": status["run_id"], "strict": strict})
    if not verified["passed"]:
        raise DemoError(f"verify_receipt did not pass: {verified}")
    return {"plan_sha256": plan["plan_sha256"], "receipt": receipt, "verified": verified}


def compare(published: dict[str, Any], receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per published metric: the graph's value beside the receipt's."""
    rows = []
    for metric, entry in sorted(published["values"].items()):
        key = METRIC_KEYS.get(metric)
        observed = receipt["metrics"].get(key) if key else None
        rows.append({
            "metric": metric, "receipt_key": key, "published": entry["value"],
            "observed": observed,
            "difference": None if observed is None else abs(observed - entry["value"]),
            "agrees": observed is not None
                      and abs(observed - entry["value"]) <= COMPARISON_TOLERANCE,
        })
    return rows


def declared_server(config: str | Path, name: str) -> tuple[str, list[str]]:
    """Command line of a server in a plugin's ``.mcp.json``.

    ``${CLAUDE_PLUGIN_ROOT}`` becomes the folder that holds the file, as Claude Code
    sets it for an installed plugin.
    """
    path = Path(config).resolve()
    server = json.loads(path.read_text())["mcpServers"][name]
    root = str(path.parent)
    return server["command"], [arg.replace(PLUGIN_ROOT, root) for arg in server["args"]]


async def check_servers(config: str | Path, transcript: Transcript) -> None:
    """Start every server a plugin declares, list its tools and make one read-only call."""
    transcript.label(Path(config).resolve().parent, PLUGIN_ROOT)
    for name in json.loads(Path(config).read_text())["mcpServers"]:
        command, args = declared_server(config, name)
        async with connect(transcript, name, command, args) as session:
            if name in FIRST_CALLS:
                tool, arguments = FIRST_CALLS[name]
                await call(session, transcript, name, tool, arguments)
    transcript.summary = {
        name: {"version": server["version"], "tools": len(server["tools"])}
        for name, server in transcript.servers.items()
    }


def runner_command(recipes_dir: str | None) -> tuple[str, list[str]]:
    """This Python's rewirebench, so the demo runs the code of the checkout it is in."""
    args = ["-m", "rewirebench.cli", "mcp"]
    if recipes_dir:
        args += ["--recipes-dir", recipes_dir]
    return sys.executable, args


async def demo(options: argparse.Namespace, transcript: Transcript) -> None:
    transcript.label_this_machine()
    for path, name in options.label or []:
        transcript.label(path, name)
    if options.servers:
        await check_servers(options.servers, transcript)
        return
    output = Path(options.output)
    transcript.label(output, "<run folder>")
    published: dict[str, Any] | None = None
    if options.graph:
        command, args = declared_server(options.graph, GRAPH)
        async with connect(transcript, GRAPH, command, args) as session:
            published = await published_values(
                session, transcript, text=options.published_text,
                system=options.published_system, protocol=options.published_protocol)
    if options.runner:
        command, args = declared_server(options.runner, RUNNER)
        transcript.label(Path(options.runner).resolve().parent, PLUGIN_ROOT)
    else:
        command, args = runner_command(options.recipes_dir)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    async with connect(transcript, RUNNER, command, args, env=environment,
                       cwd=options.server_cwd) as session:
        result = await run_recipe(
            session, transcript, recipe=options.recipe, output=output, strict=options.strict,
            kg_release_id=published["kg_release_id"] if published else None)
    receipt = result["receipt"]
    transcript.summary = {
        "recipe": options.recipe,
        "plan_sha256": result["plan_sha256"],
        "kg_release_id": receipt["kg_release_id"],
        "code_commit": receipt["code"]["commit"],
        "recipe_sha256": receipt["recipe"]["sha256"],
        "checks_passed": receipt["checks_passed"],
        "metrics": receipt["metrics"],
        "verify": {"strict": result["verified"]["strict"], "passed": result["verified"]["passed"],
                   "not_checked": result["verified"]["not_checked"]},
        "wall_seconds": receipt["wall_seconds"],
        "cost": receipt["cost"],
    }
    if published:
        rows = compare(published, receipt)
        transcript.summary["published"] = published
        transcript.summary["comparison"] = rows
        if not all(row["agrees"] for row in rows):
            raise DemoError(f"The run does not agree with the graph: {rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recipe")
    parser.add_argument("--output", help="absolute path of a new run folder")
    parser.add_argument("--servers", metavar="MCP_JSON",
                        help="instead of a run: start each server this plugin .mcp.json "
                             "declares and make one read-only call to each")
    parser.add_argument("--transcript", required=True, help="where to write the transcript")
    parser.add_argument("--copy-receipt", help="folder to copy receipt.json and SHA256SUMS to")
    parser.add_argument("--recipes-dir", help="default: the server finds the checkout's recipes")
    parser.add_argument("--server-cwd", help="working directory of the rewirebench server")
    parser.add_argument("--strict", action="store_true", help="verify_receipt with strict")
    parser.add_argument("--graph", metavar="MCP_JSON",
                        help="ask the graph server this plugin .mcp.json declares first")
    parser.add_argument("--runner", metavar="MCP_JSON",
                        help="start rewirebench as this plugin .mcp.json declares it. "
                             "Default: this Python's rewirebench, from the checkout")
    parser.add_argument("--published-text", default="MFASS v2")
    parser.add_argument("--published-system", default="id:rewire-model-baseline-kmer-position-v2")
    parser.add_argument("--published-protocol", default="id:rewire-mfass-v2")
    parser.add_argument("--label", nargs=2, action="append", metavar=("PATH", "NAME"),
                        help="replace a local folder name with a label in the transcript")
    options = parser.parse_args()
    if not options.servers and not (options.recipe and options.output):
        parser.error("give --recipe and --output, or --servers")
    transcript = Transcript()
    try:
        anyio.run(demo, options, transcript)
    finally:
        # A failed demo still leaves the calls it made.
        transcript.write(Path(options.transcript))
    if options.copy_receipt and options.output:
        target = Path(options.copy_receipt)
        target.mkdir(parents=True, exist_ok=True)
        for name in ("receipt.json", "SHA256SUMS"):
            shutil.copyfile(Path(options.output) / name, target / name)
    print(json.dumps(transcript.document()["summary"], indent=2))


if __name__ == "__main__":
    main()
