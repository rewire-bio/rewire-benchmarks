"""Remove old expanded references only when a matching gzip source survives."""

from __future__ import annotations

import gzip
import hashlib
import math
import stat
import time
import zlib
from pathlib import Path

REFERENCE_SUFFIXES = {".fa", ".fasta", ".fna", ".gtf", ".gff", ".gff3"}


def _digest(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.digest()


def _snapshot(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Not a regular file")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def clean(root: str | Path, *, older_than_days: float = 7, apply: bool = False) -> dict:
    """Preview by default. Never remove results, archives, releases or source datasets.

    Only expanded reference files beneath benchmarks/*/data are candidates. Each
    must be older than the cutoff and byte-identical to its adjacent .gz archive.
    Files must not be in use by a benchmark: stop local runs before applying.
    Byte totals are logical sizes; filesystem snapshots/hardlinks can retain space.
    """
    if not math.isfinite(older_than_days) or older_than_days < 0:
        raise ValueError("older_than_days must be a finite non-negative number")
    root = Path(root).resolve(strict=True)
    benchmarks = root / "benchmarks"
    if not benchmarks.is_dir() or benchmarks.is_symlink():
        raise ValueError("root must contain a real benchmarks directory")
    cutoff = time.time() - older_than_days * 86400
    report = {"root": str(root), "dry_run": not apply, "files": [], "skipped": [], "bytes": 0}
    for path in sorted(benchmarks.glob("*/data/**/*")):
        if path.suffix not in REFERENCE_SUFFIXES:
            continue
        # Never follow links out of the selected data directory (including parents).
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root):
            continue
        archive = Path(str(path) + ".gz")
        if not archive.is_file() or archive.is_symlink():
            continue
        try:
            before = _snapshot(path)
            source_before = _snapshot(archive)
            if path.stat().st_mtime > cutoff:
                continue
            with path.open("rb") as expanded, gzip.open(archive, "rb") as compressed:
                matches = _digest(expanded) == _digest(compressed)
            if not matches:
                report["skipped"].append({"path": str(path), "reason": "gzip differs"})
                continue
            if before != _snapshot(path) or source_before != _snapshot(archive):
                report["skipped"].append(
                    {"path": str(path), "reason": "file changed during verification"}
                )
                continue
            entry = {"path": str(path), "source": str(archive), "bytes": before[2]}
            if apply:
                path.unlink()
            report["files"].append(entry)
            report["bytes"] += before[2]
        except (OSError, EOFError, ValueError, zlib.error) as exc:
            report["skipped"].append({"path": str(path), "reason": type(exc).__name__})
    return report
