"""Write a smoke-run receipt: recipe, pinned revision, runner resources, outputs with SHA-256.

Usage: receipt.py --recipe NAME --revision SHA --started EPOCH --out receipt.json
                  [--note TEXT ...] [--deviation TEXT ...] OUTPUT_PATH...

Every OUTPUT_PATH must exist and be non-empty, otherwise the receipt is not written and the
script exits non-zero: a smoke run passes only if it wrote the documented outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--started", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", action="append", default=[])
    ap.add_argument("--deviation", action="append", default=[])
    ap.add_argument("outputs", nargs="+")
    args = ap.parse_args()

    files = []
    missing = []
    for raw in args.outputs:
        path = Path(raw)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(raw)
            continue
        files.append({"path": raw, "bytes": path.stat().st_size, "sha256": sha256(path)})
    if missing:
        print(f"missing or empty documented outputs: {missing}", file=sys.stderr)
        return 1

    disk = shutil.disk_usage(".")
    receipt = {
        "recipe": args.recipe,
        "status": "cpu-smoke",
        "upstream_revision": args.revision,
        "wall_seconds": round(time.time() - args.started, 1),
        "runner": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
            "disk_free_gib": round(disk.free / 2**30, 1),
            "github_run": os.environ.get("GITHUB_SERVER_URL", "")
            + (f"/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
               if "GITHUB_RUN_ID" in os.environ else ""),
            "github_sha": os.environ.get("GITHUB_SHA"),
        },
        "deviations": args.deviation,
        "notes": args.note,
        "outputs": files,
    }
    Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
