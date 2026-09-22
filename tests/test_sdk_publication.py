"""Release publication refuses source drift, corrupt assets and unsafe replacement."""
import importlib.util
import json
import os
import subprocess
import textwrap
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
    publisher = {"commit": "d" * 40, "tree": "e" * 40, "main_commit": "d" * 40}
    return artifacts, tmp_path / "staged", source, run, publisher


def rehash(directory):
    paths = sorted(p for p in directory.iterdir() if p.name != "SHA256SUMS")
    (directory / "SHA256SUMS").write_text(
        "".join(f"{PUBLICATION.digest(p)}  {p.name}\n" for p in paths)
    )


def test_verified_batch_stages_without_network_or_extra_large_copy(batch, monkeypatch):
    monkeypatch.setattr(PUBLICATION.subprocess, "run", lambda *a, **k: pytest.fail("network call"))
    artifacts, output, source, run, publisher = batch
    PUBLICATION.stage(*batch)
    assert len(list(output.iterdir())) == 12
    assert (output / "rewirebench-core.sif").stat().st_ino == (
        artifacts / "reviewed-sdk-release-candidate-core/rewirebench-core.sif"
    ).stat().st_ino
    receipt = json.loads((output / "validation-receipt.json").read_text())
    assert receipt["source"] == source and receipt["preparation_run"]["id"] == run["id"]
    assert receipt["publisher"] == publisher and publisher["commit"] != source["source_commit"]


@pytest.mark.parametrize("change", ["corrupt", "source", "wheel", "platform", "unexpected"])
def test_reject_untrusted_artifact_batch(batch, change):
    artifacts, _, _, _, _ = batch
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


def fake_github(monkeypatch, output, source, *, existing=True, published=False,
                upload_failure=False, confirmation_draft=False, corrupt_remote=False,
                final_asset_missing=False):
    remote = {"id": 123, "tag_name": source["tag"], "draft": not published,
              "target_commitish": source["source_commit"], "assets": [],
              "html_url": "https://github.com/example/repo/releases/tag/v0.5.0"}
    if corrupt_remote:
        remote["assets"] = [{"name": "rewirebench-core.sif", "size": 3,
                              "digest": "sha256:" + "0" * 64}]
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs.get("capture_output") is True
        assert kwargs.get("check") is True
        if command[:2] == ["gh", "api"]:
            if any("/releases/tags/" in arg for arg in command):
                # Real GitHub behavior: draft lookup by tag returns 404.
                raise subprocess.CalledProcessError(1, command, stderr="HTTP 404")
            if any("/commits/" in arg for arg in command):
                response = {"sha": source["source_commit"]}
            elif "--paginate" in command:
                assert "--slurp" in command
                response = [[{"tag_name": "v0.4.0"}], [remote] if existing else []]
            elif "POST" in command:
                response = remote  # Creation returns its ID, no tag lookup necessary.
            elif "PATCH" in command:
                assert len(remote["assets"]) == len(list(output.iterdir()))
                remote["draft"] = confirmation_draft
                remote["published_at"] = "2026-09-22T22:00:00Z"
                if final_asset_missing:
                    remote["assets"].pop()
                response = {"draft": False}  # Never trust PATCH alone.
            else:
                assert command[-1].endswith("/releases/123")
                response = remote
            return SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")
        assert command[:3] == ["gh", "release", "upload"]
        if upload_failure:
            raise subprocess.CalledProcessError(23, command, stderr="upload failed")
        path = Path(command[4])
        remote["assets"].append({"name": path.name, "size": path.stat().st_size,
                                 "digest": "sha256:" + PUBLICATION.digest(path)})
        return SimpleNamespace(returncode=0, stdout="CLI progress must not reach JSON stdout\n")

    monkeypatch.setattr(PUBLICATION.subprocess, "run", run)
    return calls, remote


@pytest.mark.parametrize("existing", [True, False])
def test_draft_lookup_and_creation_use_id_not_tag_404(batch, monkeypatch, capsys, existing):
    PUBLICATION.stage(*batch)
    output, source = batch[1:3]
    calls, remote = fake_github(monkeypatch, output, source, existing=existing)
    result = PUBLICATION.publish(output, source, "example/repo")
    assert result["published_at"] and result["url"] == remote["html_url"]
    assert calls[-1][-1].endswith("/releases/123")  # Fresh post-PATCH confirmation.
    assert len(remote["assets"]) == 12
    assert not capsys.readouterr().out
    assert any("POST" in call for call in calls) is (not existing)


def test_published_release_is_never_modified(batch, monkeypatch):
    PUBLICATION.stage(*batch)
    calls, _ = fake_github(monkeypatch, batch[1], batch[2], published=True)
    with pytest.raises(ValueError, match="published release"):
        PUBLICATION.publish(batch[1], batch[2], "example/repo")
    assert len(calls) == 2  # Tag and paginated list reads only.


def test_draft_with_different_asset_is_not_overwritten(batch, monkeypatch):
    PUBLICATION.stage(*batch)
    calls, _ = fake_github(monkeypatch, batch[1], batch[2], corrupt_remote=True)
    with pytest.raises(ValueError, match="no overwriting"):
        PUBLICATION.publish(batch[1], batch[2], "example/repo")
    assert len(calls) == 2


def test_upload_failure_propagates_and_draft_is_not_published(batch, monkeypatch):
    PUBLICATION.stage(*batch)
    calls, remote = fake_github(monkeypatch, batch[1], batch[2], upload_failure=True)
    with pytest.raises(subprocess.CalledProcessError) as error:
        PUBLICATION.publish(batch[1], batch[2], "example/repo")
    assert error.value.returncode == 23 and remote["draft"]
    assert not any("PATCH" in call for call in calls)


def test_patch_success_without_fresh_public_confirmation_is_failure(batch, monkeypatch):
    PUBLICATION.stage(*batch)
    fake_github(monkeypatch, batch[1], batch[2], confirmation_draft=True)
    with pytest.raises(ValueError, match="did not confirm"):
        PUBLICATION.publish(batch[1], batch[2], "example/repo")


def test_workflow_pipeline_propagates_nonzero_publisher(tmp_path):
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/sdk-publish.yml").read_text()
    step = workflow.split("      - name: Verify all assets and optionally publish\n")[1]
    command = textwrap.dedent(step.split("        run: |\n")[1].split("      - uses:")[0])
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python"
    python.write_text("#!/bin/sh\nprintf '%s\\n' 'publisher failure'\nexit 23\n")
    python.chmod(0o755)
    (tmp_path / "dist").mkdir()
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
               DRY_RUN="false", GITHUB_REPOSITORY="example/repo", GITHUB_SHA="d" * 40)
    result = subprocess.run(["bash", "-e", "-c", command], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 23  # tee must not turn this into a green check.


def test_fresh_published_asset_inventory_is_verified(batch, monkeypatch):
    PUBLICATION.stage(*batch)
    fake_github(monkeypatch, batch[1], batch[2], final_asset_missing=True)
    with pytest.raises(ValueError, match="Published release assets differ"):
        PUBLICATION.publish(batch[1], batch[2], "example/repo")
