import gzip
import json
import os
import time

import pytest
from rewirebench.cleanup import clean


def reference(root, content=b">ref\nACGT\n", age=10):
    path = root / "benchmarks/mfass/data/ref/genome.fa"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    archive = path.with_suffix(".fa.gz")
    archive.write_bytes(gzip.compress(content))
    old = time.time() - age * 86400
    os.utime(path, (old, old))
    return path, archive


def test_preview_preserves_files_and_reports_verified_bytes(tmp_path):
    path, archive = reference(tmp_path)
    report = clean(tmp_path)
    assert report["dry_run"] is True
    assert report["bytes"] == path.stat().st_size
    assert report["files"][0]["source"] == str(archive)
    assert path.exists() and archive.exists()


def test_apply_removes_only_verified_old_expanded_reference(tmp_path):
    path, archive = reference(tmp_path)
    protected = [
        tmp_path / p
        for p in [
            "benchmarks/mfass/results/result.json",
            "workbench/report.json",
            "benchmarks/mfass/data/predictions.csv",
            "benchmarks/mfass/data/ref/annotation.db",
            "data/releases/catalogue.json",
        ]
    ]
    for item in protected:
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text("keep")
    report = clean(tmp_path, apply=True)
    assert report["files"][0]["path"] == str(path)
    assert not path.exists()
    assert gzip.decompress(archive.read_bytes()) == b">ref\nACGT\n"
    assert all(p.read_text() == "keep" for p in protected)
    assert clean(tmp_path, apply=True)["files"] == []


@pytest.mark.parametrize("kind", ["missing", "different", "corrupt", "truncated"])
def test_unverified_source_never_allows_removal(tmp_path, kind):
    path, archive = reference(tmp_path)
    if kind == "missing":
        archive.unlink()
    elif kind == "different":
        archive.write_bytes(gzip.compress(b"different"))
    elif kind == "corrupt":
        archive.write_bytes(b"bad gzip")
    else:
        archive.write_bytes(archive.read_bytes()[:12])
    assert clean(tmp_path, apply=True)["files"] == []
    assert path.exists()


def test_recent_reference_kept(tmp_path):
    path, _ = reference(tmp_path, age=1)
    assert clean(tmp_path, apply=True)["files"] == []
    assert path.exists()


@pytest.mark.parametrize("target", ["file", "archive", "directory"])
def test_symlinks_not_followed(tmp_path, target):
    outside = tmp_path / "outside"
    path, archive = reference(outside)
    root = tmp_path / "repo"
    (root / "benchmarks/mfass/data").mkdir(parents=True)
    local = root / "benchmarks/mfass/data/genome.fa"
    if target == "directory":
        (local.parent / "ref").symlink_to(path.parent, target_is_directory=True)
    elif target == "file":
        local.symlink_to(path)
        local.with_suffix(".fa.gz").write_bytes(archive.read_bytes())
    else:
        local.write_bytes(path.read_bytes())
        os.utime(local, (0, 0))
        local.with_suffix(".fa.gz").symlink_to(archive)
    assert clean(root, apply=True)["files"] == []
    assert path.exists() and archive.exists()


@pytest.mark.parametrize("days", [-1, float("nan"), float("inf")])
def test_invalid_age_rejected(tmp_path, days):
    with pytest.raises(ValueError):
        clean(tmp_path, older_than_days=days)


def test_missing_benchmarks_directory_rejected(tmp_path):
    with pytest.raises(ValueError):
        clean(tmp_path)


def test_changed_source_is_not_deleted(tmp_path, monkeypatch):
    from rewirebench import cleanup

    path, archive = reference(tmp_path)
    digest = cleanup._digest
    count = 0

    def change_during_hash(stream):
        nonlocal count
        result = digest(stream)
        count += 1
        if count == 2:
            archive.write_bytes(gzip.compress(b"changed"))
        return result

    monkeypatch.setattr(cleanup, "_digest", change_during_hash)
    report = clean(tmp_path, apply=True)
    assert path.exists()
    assert report["skipped"][0]["reason"] == "file changed during verification"


def test_cli_defaults_to_preview(tmp_path, capsys):
    from rewirebench.cli import main

    path, _ = reference(tmp_path)
    main(["clean", "--root", str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert path.exists()
