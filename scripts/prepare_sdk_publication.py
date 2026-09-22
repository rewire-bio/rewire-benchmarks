"""Verify CI release artifacts and stage them without duplicating large files.

Dry-run staging never contacts GitHub. Explicit --publish uploads a draft, verifies
server asset digests, and only then makes the release public. Never creates a tag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ENVIRONMENTS = ("core", "esm", "dnabert2", "sequence")
MAX_ASSET_BYTES = 2 * 1024**3  # GitHub requires each asset to be strictly smaller.


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_size(size):
    if not 0 < size < MAX_ASSET_BYTES:
        raise ValueError("Release assets must be nonempty and smaller than 2 GiB")


def verify_publisher(root, expected_sha, main_ref="origin/main"):
    """Record reviewed publishing code independently of immutable package source."""
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise ValueError("Publisher must use an exact commit SHA")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("rev-parse", "HEAD") != expected_sha:
        raise ValueError("Publisher checkout differs from the workflow commit")
    result = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor",
                             expected_sha, main_ref], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Publisher revision is not merged into main")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Publisher checkout contains uncommitted files")
    return {"commit": expected_sha, "tree": git("rev-parse", "HEAD^{tree}"),
            "main_commit": git("rev-parse", main_ref)}


def stage(artifacts, output, source, run, publisher):
    if (run.get("conclusion") != "success" or run.get("event") != "workflow_dispatch"
            or run.get("path") != ".github/workflows/sdk-release.yml"):
        raise ValueError("Artifacts must come from a successful SDK release preparation run")
    if output.exists():
        raise ValueError("Use a new publication staging directory")
    version = source["sdk_version"]
    common = {f"rewirebench-{version}-py3-none-any.whl", f"rewirebench-{version}.tar.gz"}
    selections, checks = {}, {}
    expected_dirs = {f"reviewed-sdk-release-candidate-{e}" for e in ENVIRONMENTS}
    if {p.name for p in artifacts.iterdir()} != expected_dirs:
        raise ValueError("Expected exactly the four release environment artifacts")
    for environment in ENVIRONMENTS:
        directory = artifacts / f"reviewed-sdk-release-candidate-{environment}"
        required = common | {
            f"rewirebench-{environment}.oci.tar", f"rewirebench-{environment}.sif",
            "release-source.json", "oci-image-inspect.json", "native-parity.json",
            "podman-parity.json", "apptainer-parity.json", "apptainer-version.txt",
            "podman-version.txt",
        }
        manifest = {}
        for line in (directory / "SHA256SUMS").read_text().splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
            if not match or match[2] in manifest:
                raise ValueError("Invalid or duplicate checksum entry")
            manifest[match[2]] = match[1]
        if set(manifest) != required:
            raise ValueError("Release artifact checksum inventory differs from required files")
        if {p.name for p in directory.iterdir()} != required | {"SHA256SUMS"}:
            raise ValueError("Unexpected release artifact files")
        for name, expected in manifest.items():
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Release files must be regular files")
            check_size(path.stat().st_size)
            if digest(path) != expected:
                raise ValueError(f"Checksum mismatch: {environment}/{name}")
        recorded = json.loads((directory / "release-source.json").read_text())
        for field in ("tag", "sdk_version", "source_commit", "source_tree"):
            if recorded[field] != source[field]:
                raise ValueError("Artifact source does not match reviewed release source")
        image = json.loads((directory / "oci-image-inspect.json").read_text())[0]
        if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
            raise ValueError("Only the tested Linux amd64 CPU images can be published")
        checks[environment] = {
            "source": recorded,
            "oci_image_id": image["Id"],
            "os": image["Os"], "architecture": image["Architecture"],
            "apptainer_version": (directory / "apptainer-version.txt").read_text().strip(),
            "podman_version": (directory / "podman-version.txt").read_text().strip(),
            "parity": {name: json.loads((directory / f"{name}-parity.json").read_text())
                       for name in ("native", "podman", "apptainer")},
            "checksums": manifest,
        }
        for name in common | {f"rewirebench-{environment}.oci.tar",
                              f"rewirebench-{environment}.sif"}:
            if name in selections and digest(selections[name]) != manifest[name]:
                raise ValueError("Matrix jobs produced different wheel or source archive bytes")
            selections[name] = directory / name
    output.mkdir(parents=True)
    for name, path in selections.items():
        os.link(path, output / name)  # Same GitHub runner filesystem; no second large copy.
    receipt = {
        "schema_version": "1.0", "source": source, "publisher": publisher,
        "preparation_run": {k: run[k] for k in ("id", "html_url", "conclusion", "event")},
        "tested_platform": "Linux amd64 CPU; archived/synthetic scoring and framework imports",
        "not_tested": ["GPU", "DGX", "Slurm", "Google Batch", "Windows", "Linux arm64",
                       "new model inference or model throughput"],
        "environments": checks,
    }
    (output / "validation-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assets = sorted(output.iterdir())
    (output / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in assets))
    for path in output.iterdir():
        check_size(path.stat().st_size)
    return receipt


def publish(output, source, repository):
    """Resume only an identical draft; never replace a published release or assets."""
    tag, commit = source["tag"], source["source_commit"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid repository")
    endpoint = f"repos/{repository}/releases"

    def api(*args):
        result = subprocess.run(["gh", "api", *args], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    # Refuse missing or moved remote tags; creating a release must never create a tag.
    tag_commit = api(f"repos/{repository}/commits/{tag}")["sha"]
    if tag_commit != commit:
        raise ValueError("Remote release tag differs from the reviewed source")
    # The tag endpoint omits draft releases, including one just created successfully.
    pages = api("--paginate", "--slurp", f"{endpoint}?per_page=100")
    matches = [release for page in pages for release in page if release["tag_name"] == tag]
    if len(matches) > 1:
        raise ValueError("Multiple releases use this tag; review before resuming")
    remote = matches[0] if matches else api(
        "--method", "POST", endpoint,
        "-f", f"tag_name={tag}", "-f", f"target_commitish={commit}", "-F", "draft=true",
        "-f", f"name=rewirebench {source['sdk_version']}",
        "-f", "body=Reviewed SDK and Linux amd64 CPU artifacts. See validation-receipt.json "
        "for exact source, executed platform checks and limitations. Installing these "
        "assets does not submit or publish scientific results.",
    )
    release_endpoint = f"{endpoint}/{remote['id']}"
    if not remote["draft"] or remote["target_commitish"] != commit:
        raise ValueError("Refusing to modify a published release or unrelated draft")
    expected = {p.name: {"size": p.stat().st_size, "digest": "sha256:" + digest(p)}
                for p in output.iterdir()}

    def verify_remote(assets):
        seen = set()
        for asset in assets:
            name = asset["name"]
            if name in seen or name not in expected:
                raise ValueError("Unexpected or duplicated asset in draft release")
            seen.add(name)
            if any(asset.get(k) != v for k, v in expected[name].items()):
                raise ValueError("Existing release asset differs; no overwriting allowed")
        return seen

    present = verify_remote(remote["assets"])
    for name in sorted(expected.keys() - present):
        subprocess.run(["gh", "release", "upload", tag, str(output / name),
                        "--repo", repository], check=True, capture_output=True, text=True)
    if verify_remote(api(release_endpoint)["assets"]) != expected.keys():
        raise ValueError("Incomplete release upload; draft retained")
    api("--method", "PATCH", release_endpoint, "-F", "draft=false")
    published = api(release_endpoint)
    if (published.get("draft") is not False or not published.get("published_at")
            or published.get("tag_name") != tag or published.get("target_commitish") != commit):
        raise ValueError("GitHub did not confirm public release; inspect the draft before retrying")
    if verify_remote(published["assets"]) != expected.keys():
        raise ValueError("Published release assets differ from the verified plan")
    return {"url": published["html_url"], "published_at": published["published_at"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--publisher-sha", required=True)
    args = parser.parse_args()
    source, run = (json.loads(path.read_text()) for path in (args.source, args.run))
    publisher = verify_publisher(Path(__file__).resolve().parents[1], args.publisher_sha)
    stage(args.artifacts, args.output, source, run, publisher)
    published = publish(args.output, source, args.repository) if args.publish else None
    print(json.dumps({"publish_requested": args.publish,
                      "publication_status": "published" if published else "dry_run",
                      "published_release": published, "tag": source["tag"],
                      "source_commit": source["source_commit"], "publisher": publisher,
                      "assets": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                                 for p in sorted(args.output.iterdir())]}, indent=2))


if __name__ == "__main__":
    main()
