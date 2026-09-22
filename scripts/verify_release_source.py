"""Require a clean, version-matched tag already merged into the release branch."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path


def verify(root: Path, tag: str, main_ref: str = "origin/main") -> dict:
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise ValueError("Release source must be a vMAJOR.MINOR.PATCH tag")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    version = tomllib.loads(
        (root / "packages/rewirebench/pyproject.toml").read_text()
    )["project"]["version"]
    if tag != f"v{version}":
        raise ValueError("Release tag does not match the SDK package version")
    commit = git("rev-parse", "HEAD")
    if git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}") != commit:
        raise ValueError("Checkout does not match the release tag")
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, main_ref],
        check=False, capture_output=True,
    )
    if ancestor.returncode:
        raise ValueError("Release source is not merged into the release branch")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Release source checkout contains uncommitted files")
    return {
        "schema_version": "1.0",
        "tag": tag,
        "sdk_version": version,
        "source_commit": commit,
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "release_branch_commit": git("rev-parse", main_ref),
        "source_date_epoch": int(git("show", "-s", "--format=%ct", "HEAD")),
        "publication_status": "not_published",
        "scope": "Release source identity only; platform evidence is recorded separately.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Checkout to validate; defaults to this script’s repository")
    parser.add_argument("--main-ref", default="origin/main")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.source_root.resolve(), args.tag, args.main_ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
