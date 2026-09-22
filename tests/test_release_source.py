"""A release must not be prepared from an unmerged or misidentified checkout."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "verify_release_source", Path(__file__).resolve().parents[1] / "scripts/verify_release_source.py"
)
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


@pytest.fixture
def repository(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Release test")
    package = tmp_path / "packages/rewirebench"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text('[project]\nversion = "0.5.0"\n')
    git("add", ".")
    git("commit", "-m", "Reviewed release")
    git("tag", "v0.5.0")
    return tmp_path, git


def test_matching_merged_tag_records_exact_source(repository):
    root, git = repository
    receipt = RELEASE.verify(root, "v0.5.0", "main")
    assert receipt["source_commit"] == git("rev-parse", "HEAD")
    assert receipt["sdk_version"] == "0.5.0"
    assert receipt["publication_status"] == "not_published"


@pytest.mark.parametrize("tag", ["main", "v0.4.0", "v0.5.0; echo wrong"])
def test_reject_wrong_release_identity(repository, tag):
    with pytest.raises(ValueError):
        RELEASE.verify(repository[0], tag, "main")


def test_reject_unmerged_tag(repository):
    root, git = repository
    git("switch", "-c", "feature")
    (root / "change").write_text("not reviewed")
    git("add", ".")
    git("commit", "-m", "Unmerged change")
    git("tag", "-f", "v0.5.0")
    with pytest.raises(ValueError, match="not merged"):
        RELEASE.verify(root, "v0.5.0", "main")


def test_reject_checkout_other_than_tag(repository):
    root, git = repository
    (root / "change").write_text("different checkout")
    git("add", ".")
    git("commit", "-m", "After tag")
    with pytest.raises(ValueError, match="does not match"):
        RELEASE.verify(root, "v0.5.0", "main")


def test_reject_dirty_release_source(repository):
    root, _ = repository
    (root / "untracked.py").write_text("print('not in the tag')")
    with pytest.raises(ValueError, match="uncommitted"):
        RELEASE.verify(root, "v0.5.0", "main")


def test_cli_can_validate_separate_immutable_source_checkout(repository, tmp_path):
    root, git = repository
    output = tmp_path.parent / f"{tmp_path.name}-source-receipt.json"
    result = subprocess.run([
        __import__("sys").executable, str(Path(RELEASE.__file__)),
        "--source-root", str(root), "--tag", "v0.5.0", "--main-ref", "main",
        "--output", str(output),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    import json
    assert json.loads(output.read_text())["source_commit"] == git("rev-parse", "HEAD")


def test_publisher_revision_must_be_clean_exact_and_merged(repository):
    root, git = repository
    spec = importlib.util.spec_from_file_location(
        "publication_for_source_test", Path(__file__).resolve().parents[1] / "scripts/prepare_sdk_publication.py"
    )
    publication = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publication)
    sha = git("rev-parse", "HEAD")
    assert publication.verify_publisher(root, sha, "main")["commit"] == sha
    with pytest.raises(ValueError, match="differs"):
        publication.verify_publisher(root, "0" * 40, "main")
    (root / "dirty").write_text("not in reviewed source")
    with pytest.raises(ValueError, match="uncommitted"):
        publication.verify_publisher(root, sha, "main")
    git("switch", "-c", "publisher-change")
    git("add", ".")
    git("commit", "-m", "Unreviewed publisher")
    with pytest.raises(ValueError, match="not merged"):
        publication.verify_publisher(root, git("rev-parse", "HEAD"), "main")
