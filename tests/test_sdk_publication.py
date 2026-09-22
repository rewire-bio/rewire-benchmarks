"""Release publication refuses source drift, corrupt assets and unsafe replacement."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_sdk_publication", Path(__file__).resolve().parents[1] / "scripts/prepare_sdk_publication.py"
)
PUBLICATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLICATION)


@pytest.fixture
def batch(tmp_path):
    source = {"tag": "v0.5.0", "sdk_version": "0.5.0", "source_commit": "a" * 40,
              "source_tree": "b" * 40}
    run = {"id": 42, "html_url": "https://github.com/example/repo/actions/runs/42",
           "event": "workflow_dispatch", "conclusion": "success",
           "path": ".github/workflows/sdk-release.yml"}
    artifacts = tmp_path / "artifacts"
    for environment in PUBLICATION.ENVIRONMENTS:
        directory = artifacts / f"reviewed-sdk-release-candidate-{environment}"
        directory.mkdir(parents=True)
        contents = {
            "rewirebench-0.5.0-py3-none-any.whl": "wheel",
            "rewirebench-0.5.0.tar.gz": "source",
            f"rewirebench-{environment}.oci.tar": "oci",
            f"rewirebench-{environment}.sif": "sif",
            "release-source.json": json.dumps(source),
            "oci-image-inspect.json": json.dumps([{"Os": "linux", "Architecture": "amd64",
                                                    "Id": "sha256:" + "c" * 64}]),
            "apptainer-version.txt": "1.5.3", "podman-version.txt": "podman test",
            **{f"{kind}-parity.json": '{"metric": 0.5}'
               for kind in ("native", "podman", "apptainer")},
        }
        for name, value in contents.items():
            (directory / name).write_text(value)
        rehash(directory)
    return artifacts, tmp_path / "staged", source, run


def rehash(directory):
    paths = sorted(p for p in directory.iterdir() if p.name != "SHA256SUMS")
    (directory / "SHA256SUMS").write_text(
        "".join(f"{PUBLICATION.digest(p)}  {p.name}\n" for p in paths)
    )


def test_verified_batch_stages_without_network_or_extra_large_copy(batch, monkeypatch):
    monkeypatch.setattr(PUBLICATION.subprocess, "run", lambda *a, **k: pytest.fail("network call"))
    artifacts, output, source, run = batch
    PUBLICATION.stage(*batch)
    assert len(list(output.iterdir())) == 12
    assert (output / "rewirebench-core.sif").stat().st_ino == (
        artifacts / "reviewed-sdk-release-candidate-core/rewirebench-core.sif"
    ).stat().st_ino
    receipt = json.loads((output / "validation-receipt.json").read_text())
    assert receipt["source"] == source and receipt["preparation_run"]["id"] == run["id"]


@pytest.mark.parametrize("change", ["corrupt", "source", "wheel", "platform", "unexpected"])
def test_reject_untrusted_artifact_batch(batch, change):
    artifacts, _, _, _ = batch
    directory = artifacts / "reviewed-sdk-release-candidate-core"
    if change == "corrupt":
        (directory / "rewirebench-core.sif").write_text("different bytes")
    else:
        if change == "source":
            source = json.loads((directory / "release-source.json").read_text())
            source["source_commit"] = "d" * 40
            (directory / "release-source.json").write_text(json.dumps(source))
        elif change == "wheel":
            (directory / "rewirebench-0.5.0-py3-none-any.whl").write_text("different wheel")
        elif change == "platform":
            (directory / "oci-image-inspect.json").write_text('[{"Os":"linux","Architecture":"arm64"}]')
        else:
            (directory / "unexpected").write_text("not approved")
        rehash(directory)
    with pytest.raises(ValueError):
        PUBLICATION.stage(*batch)


@pytest.mark.parametrize("field,value", [("conclusion", "failure"), ("event", "pull_request"),
                                         ("path", ".github/workflows/sdk.yml")])
def test_reject_wrong_preparation_run(batch, field, value):
    batch[3][field] = value
    with pytest.raises(ValueError, match="successful SDK release"):
        PUBLICATION.stage(*batch)


@pytest.mark.parametrize("size", [0, 2 * 1024**3, 2 * 1024**3 + 1])
def test_github_asset_limit_without_large_file_allocation(size):
    with pytest.raises(ValueError, match="2 GiB"):
        PUBLICATION.check_size(size)
    PUBLICATION.check_size(2 * 1024**3 - 1)


def test_published_release_is_never_modified(batch, monkeypatch):
    PUBLICATION.stage(*batch)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[:2] == ["gh", "api"]
        return SimpleNamespace(returncode=0, stdout=json.dumps({"draft": False}), stderr="")

    monkeypatch.setattr(PUBLICATION.subprocess, "run", run)
    with pytest.raises(ValueError, match="published release"):
        PUBLICATION.publish(batch[1], batch[2], "example/repo")
    assert len(calls) == 1


def test_identical_draft_resumes_then_publishes_after_digest_checks(batch, monkeypatch):
    PUBLICATION.stage(*batch)
    output, source = batch[1:3]
    remote = {"draft": True, "target_commitish": source["source_commit"], "assets": []}
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["gh", "api"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(remote), stderr="")
        if command[2] == "upload":
            path = Path(command[4])
            remote["assets"].append({"name": path.name, "size": path.stat().st_size,
                                     "digest": "sha256:" + PUBLICATION.digest(path)})
        elif command[2] == "edit":
            assert len(remote["assets"]) == len(list(output.iterdir()))
        else:
            pytest.fail("Unexpected GitHub write")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(PUBLICATION.subprocess, "run", run)
    PUBLICATION.publish(output, source, "example/repo")
    assert calls[-1][2] == "edit" and calls[-1][-1] == "--draft=false"


def test_draft_with_different_asset_is_not_overwritten(batch, monkeypatch):
    PUBLICATION.stage(*batch)
    remote = {"draft": True, "target_commitish": batch[2]["source_commit"], "assets": [
        {"name": "rewirebench-core.sif", "size": 3, "digest": "sha256:" + "0" * 64},
    ]}

    def run(command, **kwargs):
        assert command[:2] == ["gh", "api"]
        return SimpleNamespace(returncode=0, stdout=json.dumps(remote), stderr="")

    monkeypatch.setattr(PUBLICATION.subprocess, "run", run)
    with pytest.raises(ValueError, match="no overwriting"):
        PUBLICATION.publish(batch[1], batch[2], "example/repo")
