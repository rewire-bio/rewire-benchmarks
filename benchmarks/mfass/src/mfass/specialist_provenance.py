"""Local artifact identity for specialist runs, separate from source verification."""

import hashlib
from pathlib import Path


def file_sha256(path):
    """Hash file bytes in bounded memory, including compressed bytes if supplied."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path):
    """Git's object ID for a file's bytes, comparable with an upstream tree listing."""
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {Path(path).stat().st_size}\0".encode())
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_file_check(root, pinned):
    """Compare files under root with pinned {relative path: (sha256, git blob)}.

    Returns (observed {path: {"sha256", "git_blob"}}, list of mismatching paths).
    A missing file is a mismatch, never an error that could be mistaken for success.
    """
    observed, mismatches = {}, []
    for name, (sha256, blob) in pinned.items():
        path = Path(root) / name
        if not path.is_file():
            observed[name] = None
            mismatches.append(name)
            continue
        observed[name] = {"sha256": file_sha256(path), "git_blob": git_blob_sha1(path)}
        if observed[name] != {"sha256": sha256, "git_blob": blob}:
            mismatches.append(name)
    return observed, mismatches


def resolve_spliceai_annotation(annotation):
    """Match SpliceAI 1.3.1's exact bundled aliases; other strings are paths.

    Resolve before calling Annotator so the hashed file is also the file passed
    to its reader. The optional SpliceAI dependency is only needed for aliases.
    """
    if annotation in ("grch37", "grch38"):
        from pkg_resources import resource_filename

        return Path(resource_filename("spliceai", f"annotations/{annotation}.txt")).resolve()
    # An absolute custom path cannot accidentally become the reserved token
    # "grch38" when Path normalizes an input such as "./grch38".
    return Path(annotation).resolve()


def installed_package_files(package, relative_paths):
    """Version and SHA-256 of named files in an installed package, without importing it."""
    import importlib.metadata
    import importlib.util

    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"{package} is not installed")
    root = Path(next(iter(spec.submodule_search_locations)))
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "unreported"
    return version, {name: file_sha256(root / name) for name in relative_paths}


def specialist_artifacts(annotation, reference, annotation_release=None):
    """A digest identifies local bytes; a release label is only a declaration."""
    release = (annotation_release or "").strip()
    declared = bool(release) and release.casefold() != "unreported"
    return {
        "annotation_artifact": Path(annotation).name,
        "annotation_sha256": file_sha256(annotation),
        "reference_sha256": file_sha256(reference),
        "annotation_release": release if declared else "unreported",
        "annotation_release_status": "declared" if declared else "unreported",
        "artifact_identity_status": "local_file_hashes; upstream identity not verified",
    }
