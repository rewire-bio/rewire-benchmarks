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
