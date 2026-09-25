"""Identity and preparation of the reviewed Pangolin per-gene masking patch.

Pangolin 5cf94b8 masks shared per-strand score arrays in place, so a gene's
masked scores depend on which overlapping genes on the same strand were
processed before it (upstream issue #29). The patch in
`benchmarks/mfass/patches/` gives each gene its own copy. It is applied to a
clean upstream tree and installed as a distinct version; nothing is patched at
import time. Runners hash the installed source so a result states which code
produced it.

Pangolin is GPL-3.0. The patch and any patched tree are GPL-3.0 derivatives.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

from mfass.specialist_provenance import file_sha256, pinned_file_check

UPSTREAM_REPOSITORY = "https://github.com/tkzeng/Pangolin"
UPSTREAM_REVISION = "5cf94b8db938c658391b4305cd7ce33297d44ff7"
UPSTREAM_VERSION = "1.0.2"
UPSTREAM_SOURCE_SHA256 = "f3f25c4febf64d01ef42f967dc5cf10f6856bf3aa92d26f09803ad408169da9b"

PATCH_ID = "pangolin-5cf94b8-mask-per-gene-1"
PATCH_FILE = Path(__file__).resolve().parents[2] / "patches" / f"{PATCH_ID}.patch"
PATCH_SHA256 = "d1c9fb69245f9d256e2547fd8bf61547686cfc3caf6c40c1351ab2d6997cb454"
PATCHED_VERSION = "1.0.2+rewire.maskpergene1"
PATCHED_SOURCE_SHA256 = "239c6ba9b1883d712d4856985c6eaa07e5b67f99790a718f93d6316ad076c083"

# The ensemble run_pangolin loads: final.{1,2,3}.{0,2,4,6}.3.v2.
ENSEMBLE_WEIGHTS = tuple(f"models/final.{j}.{i}.3.v2" for i in (0, 2, 4, 6) for j in (1, 2, 3))

# Upstream bytes at UPSTREAM_REVISION: (SHA-256, git blob ID). Blob IDs are from the
# GitHub tree API for the pinned commit and equal the local clone's objects.
UPSTREAM_FILES = {
    "__init__.py": ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
    "model.py": ("4a1c5c2570aafe1452bb43332255321677e6c6c817adf84b9dd438e3ca4be6f8",
                 "11dfb436b36f9b5783874b2db636a1788bfab619"),
    "models/final.1.0.3.v2": ("f0478fab173b75f7f7e9fe96688bad6c50fa4a46d70557f423b110caaf565501",
                              "fbe03218df2f781a9713b05bfa42001ae548782a"),
    "models/final.2.0.3.v2": ("c4c6bb4880fa6fb28b14182ae3ea0600edb07056158f55325b5e6e6e48fc9f26",
                              "ec4ee767bc83fa0397222f42c0da210a47f13fbb"),
    "models/final.3.0.3.v2": ("ec685a6e7105a4486c1f89a005458a13deb3fe7171f13d434f4877e386d10676",
                              "f31cbae821e38bc4361216c8e87c8db13fd5a9c0"),
    "models/final.1.2.3.v2": ("559c05de3e1ce65c2515ca3e92ef85edb0ec2e47686ca58060e25891ce06eb3a",
                              "39e8756faf1e427cd9a8c6c691532b3683d8916a"),
    "models/final.2.2.3.v2": ("48758ba8b95eee9aa9feea52672ef06ca1b34111299c27f8a710f734d8b9aae5",
                              "6fd0a31789c7de788227e103f33d697ad538def6"),
    "models/final.3.2.3.v2": ("7cb576c2b24db4fdd6970c4ca4fb7c20ae1b1d8ae80645ebbe689848b5743129",
                              "c7f2ae10caab8f5bae167309b2b7d3e6905871e5"),
    "models/final.1.4.3.v2": ("c50b12e0c0af776d5674ca5e346493f8265783494d4df383364de9c1136657f6",
                              "0af4fadb3f3dd54c49d41042befefa19fd4bd32a"),
    "models/final.2.4.3.v2": ("e03303bed4fd6f135ec0f6c1b192cce954ea42d0646f44d17b4a6fbb2b1f610e",
                              "8c003df629c9f1e8072000f8bbc3eeab94850b6c"),
    "models/final.3.4.3.v2": ("9476d2e25520d7ff15bece0cd5d3b657e3b1dd3cc5fcab1d9c3b62bea7a0c5b6",
                              "9ed4fad50ead41761073cdff3d9acafc6f78bdd4"),
    "models/final.1.6.3.v2": ("2aae563fa18a8a9b6699c6c96e0d32b8ec7543f8f805fb3bc9de77302cc9f66e",
                              "98e5539fe339385e9c8b19d2786c4f8c3f186a3b"),
    "models/final.2.6.3.v2": ("7d3c0b1b2a60067b940dec315567874fbc8bcd322f1b7c76bf969f51f0f53f7f",
                              "6a650d629fae6aee8171708241a0f580bb368cfd"),
    "models/final.3.6.3.v2": ("756e7721a382cace24e9bfea5b543af5623f2487d9a3efe7385e9c76367005fd",
                              "4a8e9e33d5832770e6563d0229bdb4bcae5b8398"),
}
UPSTREAM_SOURCE_BLOB = "194736ff64064dd4ee65e2a1a6260b059573c054"

KNOWN_SOURCES = {
    UPSTREAM_SOURCE_SHA256: f"upstream-{UPSTREAM_REVISION[:7]}-unpatched",
    PATCHED_SOURCE_SHA256: PATCH_ID,
}


def source_identity(source_path):
    """Name a pangolin.py by its bytes; unknown bytes are never assumed reviewed."""
    digest = file_sha256(source_path)
    return {"pangolin_source_sha256": digest,
            "pangolin_source_identity": KNOWN_SOURCES.get(digest, "unrecognised")}


def installed_identity():
    """Identity of the importable pangolin package, found without importing torch.

    `pangolin_upstream_files_verified` is true only when model.py, __init__.py and all
    12 ensemble weights equal the pinned upstream bytes (SHA-256 and git blob ID).
    """
    spec = importlib.util.find_spec("pangolin")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("pangolin is not installed")
    package = Path(next(iter(spec.submodule_search_locations)))
    identity = source_identity(package / "pangolin.py")
    try:
        version = importlib.metadata.version("pangolin")
    except importlib.metadata.PackageNotFoundError:
        version = "unreported"
    identity["pangolin_version"] = version
    identity["pangolin_patch"] = (
        {"id": PATCH_ID, "patch_sha256": PATCH_SHA256, "upstream_revision": UPSTREAM_REVISION,
         "upstream_source_sha256": UPSTREAM_SOURCE_SHA256}
        if identity["pangolin_source_identity"] == PATCH_ID else None)
    observed, mismatches = pinned_file_check(package, UPSTREAM_FILES)
    identity["pangolin_files"] = observed
    identity["pangolin_upstream_files_verified"] = not mismatches
    identity["pangolin_upstream_file_mismatches"] = mismatches
    return identity


def require_reviewed(identity, allow_unpatched=False):
    """Raise unless code and weights are the reviewed bytes."""
    problems = list(identity.get("pangolin_upstream_file_mismatches", ["unverified"]))
    if not identity.get("pangolin_upstream_files_verified"):
        problems.append("model code or weights differ from the pinned upstream bytes")
    if identity.get("pangolin_source_identity") != PATCH_ID and not allow_unpatched:
        problems.append(f"source is {identity.get('pangolin_source_identity')}, not {PATCH_ID}")
    if problems:
        raise ValueError("Pangolin identity check failed: " + "; ".join(map(str, problems)))


def _git(checkout, *args):
    return subprocess.run(["git", "-C", str(checkout), *args], check=True,
                          capture_output=True, text=True).stdout


def prepare_patched_tree(checkout, destination, patch_file=PATCH_FILE):
    """Export the pinned revision, apply the reviewed patch and verify every hash.

    The export comes from the commit object, not the working tree, so local
    edits in the checkout cannot leak in. Refuses to reuse a destination.
    """
    checkout, destination = Path(checkout), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to reuse {destination}")
    if file_sha256(patch_file) != PATCH_SHA256:
        raise ValueError("Patch file differs from the reviewed patch")
    if _git(checkout, "rev-parse", f"{UPSTREAM_REVISION}^{{commit}}").strip() != UPSTREAM_REVISION:
        raise ValueError("Checkout does not contain the pinned upstream revision")
    destination.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / "upstream.tar"
        _git(checkout, "archive", "--format=tar", "-o", str(archive), UPSTREAM_REVISION)
        with tarfile.open(archive) as tar:
            tar.extractall(destination, filter="data")
    if file_sha256(destination / "pangolin" / "pangolin.py") != UPSTREAM_SOURCE_SHA256:
        raise ValueError("Exported upstream source differs from the reviewed revision")
    subprocess.run(["git", "apply", "--verbose", str(Path(patch_file).resolve())],
                   cwd=destination, check=True, capture_output=True, text=True)
    patched = file_sha256(destination / "pangolin" / "pangolin.py")
    if patched != PATCHED_SOURCE_SHA256:
        raise ValueError("Patched source differs from the reviewed patched source")
    if pinned_file_check(destination / "pangolin", UPSTREAM_FILES)[1]:
        raise ValueError("Exported model code or weights differ from the pinned upstream bytes")
    if f'version="{PATCHED_VERSION}"' not in (destination / "setup.py").read_text():
        raise ValueError("Patched setup.py does not declare the patched version")
    receipt = {
        "patch_id": PATCH_ID, "patch_sha256": PATCH_SHA256,
        "upstream_repository": UPSTREAM_REPOSITORY, "upstream_revision": UPSTREAM_REVISION,
        "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
        "patched_source_sha256": patched, "patched_version": PATCHED_VERSION,
        "upstream_files": pinned_file_check(destination / "pangolin", UPSTREAM_FILES)[0],
        "licence": "GPL-3.0 (Pangolin); the patched tree is a GPL-3.0 derivative",
    }
    (destination / "REWIRE-PATCH-RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="export the pinned revision and apply the patch")
    prepare.add_argument("--checkout", required=True, help="git clone of upstream Pangolin")
    prepare.add_argument("--out", required=True, help="new directory for the patched tree")
    sub.add_parser("identity", help="print the installed package identity")
    args = ap.parse_args()
    result = (prepare_patched_tree(args.checkout, args.out) if args.command == "prepare"
              else installed_identity())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
