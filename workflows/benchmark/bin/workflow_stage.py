#!/usr/bin/env python3
"""Whole-protocol orchestration. Scientific evaluation stays inside rewirebench."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SECRET_KEY = re.compile(r"(^|_)(token|password|secret|credential|api_key)(_|$)", re.IGNORECASE)


def validate_job(job):
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if SECRET_KEY.search(key):
                    raise ValueError("Credentials must not be stored in job manifests")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(job)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", job.get("id", "")):
        raise ValueError("Unsafe job ID")
    if not job.get("protocol") or not job.get("environment_id"):
        raise ValueError("Protocol and pinned environment identity are required")
    if job.get("kind") not in {"model", "baseline"}:
        raise ValueError("kind must be model or baseline")
    if job["kind"] == "model":
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", job.get("adapter", "")):
            raise ValueError("Model jobs require a Python module:class adapter")
        if not job.get("model", {}).get("name"):
            raise ValueError("Model jobs require a model identity")
    elif not job.get("baseline"):
        raise ValueError("Baseline jobs require an explicit baseline ID")
    return job


def resolve_assets(value, assets):
    if isinstance(value, dict):
        return {key: resolve_assets(item, assets) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_assets(item, assets) for item in value]
    if isinstance(value, str) and value.startswith("asset:"):
        relative = Path(value[6:])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Asset paths must remain inside the staged assets directory")
        result = Path(assets) / relative
        if not result.exists():
            raise ValueError("Missing staged model asset")
        return str(result.resolve())
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(stage, job, assets="assets"):
    from rewirebench import sdk
    validate_job(job)
    os.environ.pop("REWIRE_SUBMISSION_TOKEN", None)
    if job.get("container_image") not in (None, "native"):
        os.environ["REWIRE_CONTAINER_DIGEST"] = job["container_image"]
        os.environ["REWIRE_CONTAINER_RUNTIME"] = job.get("execution_profile", "unreported")
    if stage == "prepare":
        sources = list(Path("input").iterdir())
        if len(sources) != 1:
            raise ValueError("Exactly one file or source directory is required")
        sdk.prepare(job["protocol"], source=sources[0], output="prepared",
                    **job.get("prepare_options", {}))
    elif stage == "run":
        if job["kind"] == "baseline":
            subprocess.run([sys.executable, "-m", "rewirebench.cli", "run-baselines",
                            "--prepared", "prepared", "--output", "runs",
                            "--baseline", job["baseline"]], check=True)
            manifest = json.loads(Path("runs/baseline-manifest.json").read_text())
            selected = manifest.get("baselines", [])
            if (manifest.get("status") != "evaluated" or len(selected) != 1
                    or selected[0].get("baseline_id") != job["baseline"]
                    or selected[0].get("status") != "evaluated"):
                raise ValueError("Selected baseline did not evaluate successfully; inspect private run status")
            if len(list(Path("runs").rglob("report.json"))) != 1:
                raise ValueError("Evaluated baseline must produce exactly one report")
        else:
            module, name = job["adapter"].split(":")
            factory = getattr(importlib.import_module(module), name)
            options = resolve_assets(job.get("adapter_options", {}), assets)
            sdk.run("prepared", factory(**options), output="runs/model", model=job["model"],
                    batch_size=job.get("batch_size", 32),
                    prediction_type=job.get("prediction_type"),
                    allow_partial=job.get("allow_partial", False))
    elif stage == "evaluate":
        reports = sorted(Path("runs").rglob("report.json"))
        if len(reports) != 1:
            raise ValueError("Each matrix job must produce exactly one baseline or model report")
        original = json.loads(reports[0].read_text())
        model = dict(original["model"])
        model["configuration"] = original.get("model_configuration", {})
        model["input_information"] = original.get("input_information", "unreported")
        checked = sdk.evaluate("prepared", reports[0].parent / "predictions.json",
                               output="evaluated", model=model,
                               execution=original["execution"],
                               allow_partial=job.get("allow_partial", False))
        for field in ("metrics", "coverage", "predictions_sha256", "prepared_sha256"):
            if checked[field] != original[field]:
                raise ValueError(f"Independent rescoring changed {field}")
        # Keep original fit/inference environment, timing and baseline provenance intact.
        Path("evaluated/report.json").write_text(json.dumps(original, indent=2, allow_nan=False)+"\n")
        receipt = {"job_id": job["id"], "environment_id": job["environment_id"],
                   "configuration_sha256": hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
                   "runner_script_sha256": digest(__file__), "rescoring_matches": True,
                   "aggregation": "whole protocol; no sharding or cross-job averaging",
                   "submission_attempted": False}
        Path("evaluated/workflow-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    elif stage == "export":
        Path("bundles").mkdir()
        report = json.loads(Path("evaluated/report.json").read_text())
        if report["scope"] == "smoke":
            Path("bundles/NOT_SUBMITTABLE.txt").write_text("Smoke evaluation: not eligible for submission.\n")
        else:
            sdk.export("evaluated/report.json", output="bundles/contribution.json")
        receipt = json.loads(Path("evaluated/workflow-receipt.json").read_text())
        identity = {"configuration": receipt["configuration_sha256"],
                    "prepared": report["prepared_sha256"],
                    "predictions": report["predictions_sha256"],
                    "environment": report["environment"],
                    "runner": receipt["runner_script_sha256"]}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        destination = Path("delivery") / key
        shutil.copytree("evaluated", destination / "evaluated")
        shutil.copytree("bundles", destination / "bundles")
    else:
        raise ValueError("Unknown stage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "evaluate", "export"])
    parser.add_argument("--job-base64", required=True)
    parser.add_argument("--assets", default="assets")
    args = parser.parse_args()
    execute(args.stage, json.loads(base64.b64decode(args.job_base64)), args.assets)


if __name__ == "__main__":
    main()
