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
