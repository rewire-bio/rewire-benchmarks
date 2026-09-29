"""CLI for local campaigns. Start/resume are foreground unless --background is explicit."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .campaign import (
    create_campaign,
    export_campaign,
    read_manifests,
    run_campaign,
    scan_guarded,
    stop_campaign,
)
from .contracts import read_json
from .state import Store


def add_parser(subparsers):
    parser = subparsers.add_parser("research", help="Investigate discrepancies in pinned benchmark evidence")
    commands = parser.add_subparsers(dest="research_command", required=True)
    for name in ("scan", "start"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--manifests", required=True, help="One manifest or a JSON array of manifests")
        cmd.add_argument("--resolver", required=True, help="Private artifact-ID to absolute-path JSON")
        if name == "start":
            cmd.add_argument("--workspace", required=True, help="New ignored workbench campaign directory")
            cmd.add_argument("--limits", default="{}", help="JSON budget overrides; may only reduce defaults")
            cmd.add_argument("--replay-only", action="store_true", help="Numerical preflight without Codex calls")
            cmd.add_argument("--background", action="store_true")
            cmd.add_argument("--codex", default="codex", help="Installed Codex CLI executable")
    for name in ("status", "stop", "resume", "export"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--workspace", required=True)
        if name == "resume":
            cmd.add_argument("--background", action="store_true")
            cmd.add_argument("--codex", default="codex")
        elif name == "export":
            cmd.add_argument("--output", required=True, help="New directory for private review bundles")


def _launch(workspace, executable):
    workspace = Path(workspace).resolve()
    with (workspace/"supervisor.private.log").open("a") as stream:
        process = subprocess.Popen([sys.executable, "-m", "rewirebench.cli", "research", "resume",
                                    "--workspace", str(workspace), "--codex", executable],
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                   start_new_session=True)
    return {"workspace": str(workspace), "supervisor_pid": process.pid, "started": True}


def dispatch(args):
    command = args.research_command
    if command == "scan":
        return scan_guarded(read_manifests(args.manifests), read_json(args.resolver))
    if command == "start":
        create_campaign(args.workspace, read_manifests(args.manifests), read_json(args.resolver),
                        limits=json.loads(args.limits), mode="replay-only" if args.replay_only else "codex")
    if command in {"start", "resume"}:
        if args.background:
            return _launch(args.workspace, args.codex)
        return run_campaign(args.workspace, executable=args.codex)
    if command == "stop":
        return stop_campaign(args.workspace)
    if command == "export":
        return export_campaign(args.workspace, args.output)
    store = Store(args.workspace)
    try:
        return store.status()
    finally:
        store.close()
