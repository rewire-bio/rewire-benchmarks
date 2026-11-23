"""The recipe receipt workflow must produce a receipt a reviewer can rely on."""
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
HEAD = "${{ github.event.pull_request.head.sha || github.sha }}"


def workflow(name: str) -> dict[str, Any]:
    data = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
    # YAML 1.1 reads the key `on` as the boolean true.
    data["on"] = data.pop(True)
    return data


def test_recipe_receipt_cites_a_commit_on_a_branch():
    """A pull request checks out its head, not the temporary merge commit."""
    job = workflow("recipe-receipt.yml")["jobs"]["mfass-v2-kmer-baseline"]
    checkout = next(step for step in job["steps"] if "checkout" in step.get("uses", ""))
    assert checkout["with"]["ref"] == HEAD and job["env"]["RUN_COMMIT"] == HEAD
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "receipt verify" in commands and "--strict" in commands
    assert 'receipt["code"]["commit"] == wanted' in commands
    assert 'receipt["recipe"]["commit"] == wanted' in commands


def test_recipe_receipt_reruns_when_the_runner_changes_and_on_main():
    """The numbers come from the whole package, so any change to it reruns the recipe."""
    triggers = workflow("recipe-receipt.yml")["on"]
    for event in ("pull_request", "push"):
        assert {"recipes/**", "packages/rewirebench/**", "uv.lock"} <= set(triggers[event]["paths"])
    assert triggers["push"]["branches"] == ["main"]
    assert "schedule" in triggers and "workflow_dispatch" in triggers


def test_a_release_does_not_depend_on_another_repository_having_no_newer_tag():
    """The pin freshness check is for pull requests. A release of this repository must
    not be blocked by a new rdf-kg-mcp tag or by GitHub being unreachable."""
    steps = workflow("sdk-release.yml")["jobs"]["artifacts"]["steps"]
    [tests] = [step for step in steps if "pytest" in step.get("run", "")]
    assert tests["env"]["REWIRE_SKIP_PIN_FRESHNESS"] == "1"
    for name in ("sdk.yml",):
        for job in workflow(name)["jobs"].values():
            for step in job["steps"]:
                assert "REWIRE_SKIP_PIN_FRESHNESS" not in str(step)


def test_the_runner_image_is_pushed_only_from_main_and_never_by_a_pull_request():
    """A pull request builds and checks both platforms. Only the publish job can write
    packages, and it runs on main only."""
    image = workflow("image.yml")
    main_only = "github.event_name != 'pull_request' && github.ref == 'refs/heads/main'"
    assert image["permissions"] == {"contents": "read"}
    jobs = image["jobs"]
    assert jobs["publish"]["if"] == main_only and jobs["publish"]["needs"] == "build"
    assert jobs["verify"]["needs"] == "publish"
    writers = [name for name, job in jobs.items()
               if job.get("permissions", {}).get("packages") == "write"]
    assert writers == ["publish"]
    build = "\n".join(step.get("run", "") for step in jobs["build"]["steps"])
    assert "docker push" not in build and "docker login" not in build
    for step in jobs["build"]["steps"]:
        if "docker save" in step.get("run", "") or "upload-artifact" in step.get("uses", ""):
            assert step["if"] == main_only
    assert image["on"]["push"]["branches"] == ["main"]
    assert image["on"]["pull_request"]["paths"] == image["on"]["push"]["paths"]
    assert ".github/workflows/image.yml" in image["on"]["push"]["paths"]


def test_each_platform_of_the_runner_image_is_built_and_checked_on_its_own_architecture():
    """No emulation: linux/arm64 builds on an arm64 runner. Both workflows that build the
    image run the same check, and the pushed image is checked again by digest."""
    image = workflow("image.yml")
    matrix = image["jobs"]["build"]["strategy"]["matrix"]
    assert {(row["arch"], row["runner"]) for row in matrix["include"]} == {
        ("amd64", "ubuntu-24.04"), ("arm64", "ubuntu-24.04-arm")}
    assert matrix["arch"] == ["amd64", "arm64"] and matrix["environment"] == ["core", "sequence"]
    assert "qemu" not in (ROOT / ".github/workflows/image.yml").read_text().lower()
    build = "\n".join(step.get("run", "") for step in image["jobs"]["build"]["steps"])
    assert "--provenance=false" in build and "containers/check_image.sh docker" in build
    verify = "\n".join(step.get("run", "") for step in image["jobs"]["verify"]["steps"])
    assert 'docker pull --platform linux/amd64 "$IMAGE@$DIGEST"' in verify
    assert 'containers/check_image.sh docker "$IMAGE@$DIGEST"' in verify
    sdk = "\n".join(step.get("run", "") for step in workflow("sdk.yml")["jobs"]["python"]["steps"])
    assert "containers/check_image.sh podman" in sdk
