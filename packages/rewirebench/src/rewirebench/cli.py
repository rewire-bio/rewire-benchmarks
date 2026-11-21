"""Command-line counterpart to the local Python SDK."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from rewirebench import sdk
from rewirebench.submission import SubmissionError, submit


def _prepared_file(prepared: str) -> Path:
    path = Path(prepared)
    return path / "prepared.json" if path.is_dir() else path


def _baseline_metrics(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for report in sorted(Path(output).glob("*/report.json")):
        metrics[report.parent.name] = json.loads(report.read_text()).get("metrics")
    return metrics


def _recipe_command(args: argparse.Namespace) -> tuple[Any, int]:
    """Handle ``rewirebench recipe ...``. Returns the JSON result and the exit code."""
    from rewirebench import receipts, recipes

    if args.recipe_command == "schema":
        return recipes.json_schema(), 0
    if args.recipe_command == "validate" and args.paths:
        results = recipes.validate_files([Path(path) for path in args.paths])
        return results, 0 if all(entry["valid"] for entry in results) else 1
    directory = recipes.find_recipes_dir(args.recipes_dir)
    if directory is None:
        if args.recipe_command == "list":
            return {"recipes_dir": None, "recipes": [], "invalid": [],
                    "note": "No recipes directory found. Run from a rewire-benchmarks checkout, "
                            f"or set --recipes-dir or {recipes.RECIPES_DIR_VARIABLE}."}, 0
        raise ValueError(
            "No recipes directory found. Run from a rewire-benchmarks checkout, "
            f"or set --recipes-dir or {recipes.RECIPES_DIR_VARIABLE}."
        )
    if args.recipe_command == "validate":
        results = recipes.validate_files(recipes.recipe_files(directory))
        return results, 0 if results and all(entry["valid"] for entry in results) else 1
    if args.recipe_command == "list":
        # A file that is not a valid recipe is reported and does not hide the others.
        valid, invalid = recipes.list_recipes(directory)
        return {"recipes_dir": str(directory),
                "recipes": [recipes.summary(item, directory) for item in valid],
                "invalid": invalid}, 1 if invalid else 0
    loaded = recipes.get_recipe(args.recipe, directory)
    if args.recipe_command == "show":
        return {**recipes.summary(loaded, directory),
                "recipe": loaded.recipe.model_dump(mode="json", by_alias=True)}, 0
    from rewirebench import backends

    started_by = args.started_by or receipts.default_started_by()
    if args.recipe_command == "plan":
        backend = backends.get_backend(args.backend, spark=_spark_config(args, required=False))
        plan = backend.plan(
            loaded, recipes_dir=directory, output=args.output, inputs=args.inputs,
            started_by=started_by, kg_release_id=args.kg_release_id,
        )
        return plan.model_dump(mode="json"), 0
    if args.backend == "spark" and not args.in_container:
        # From this machine: SSH to the Spark and run the pinned image there. Without
        # --execute nothing is contacted and the steps are printed.
        import subprocess

        spark = backends.get_backend("spark", spark=_spark_config(args, required=args.execute))
        try:
            result = spark.run(
                loaded, recipes_dir=directory, output=args.output, inputs=args.inputs,
                started_by=started_by, kg_release_id=args.kg_release_id,
                dry_run=not args.execute,
            )
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            # An unreachable host or a failed step is reported like any other refusal.
            raise ValueError(f"The spark run did not complete: {error}") from error
        return result, 0 if result.get("dry_run") or result.get("checks_passed") else 1
    if args.in_container and args.backend != "spark":
        raise ValueError("--in-container is set by the spark backend only")
    receipt = receipts.run_recipe(
        loaded, output=args.output, recipes_dir=directory, inputs=args.inputs,
        started_by=args.started_by, kg_release_id=args.kg_release_id, backend=args.backend,
    )
    return {"recipe_id": loaded.recipe.id, "checks_passed": receipt.checks_passed,
            "metrics": receipt.metrics, "wall_seconds": receipt.wall_seconds,
            "peak_memory_bytes": receipt.peak_memory_bytes,
            "cost": receipt.cost.model_dump(mode="json"),
            "receipt": str(Path(args.output) / receipts.RECEIPT_FILENAME)}, \
        0 if receipt.checks_passed else 1


def _spark_config(args: argparse.Namespace, *, required: bool) -> Any:
    """The Spark to use: --spark-host and --spark-workdir, or the environment."""
    from rewirebench.backends import spark

    return spark.SparkConfig.from_environment(
        host=args.spark_host, workdir=args.spark_workdir, required=required
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rewirebench", description="Local biological benchmark evaluation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocols", help="List implemented protocols")
    listing = sub.add_parser("models", help="List the registered model adapters")
    listing.add_argument("--protocol", choices=sdk.PROTOCOLS,
                         help="Only the models this protocol accepts")
    server = sub.add_parser(
        "mcp", help="Serve the benchmarks MCP server on stdio (needs rewirebench[mcp])")
    server.add_argument("--recipes-dir", help="Default: the recipes/ of this checkout")
    cleanup = sub.add_parser("clean", help="Preview old expanded references with verified gzip copies")
    cleanup.add_argument("--root", required=True, help="Repository containing benchmarks/*/data")
    cleanup.add_argument("--older-than-days", type=float, default=7)
    cleanup.add_argument(
        "--apply", action="store_true", help="Remove verified copies; stop benchmark runs first"
    )
    inspect = sub.add_parser("inspect", help="Describe protocol inputs, datasets and execution rules")
    inspect.add_argument("protocol", choices=sdk.PROTOCOLS)
    baselines = sub.add_parser("baselines", help="Inspect reference baselines and selection gaps")
    baselines.add_argument("protocol", choices=sdk.PROTOCOLS)
    baseline_run = sub.add_parser("run-baselines", help="Run registered reference baselines locally")
    baseline_run.add_argument("--prepared", required=True)
    baseline_run.add_argument("--output", required=True)
    baseline_run.add_argument("--baseline", action="append")
    baseline_run.add_argument("--batch-size", type=int, default=32)
    baseline_run.add_argument("--baseline-options", default="{}",
                              help='JSON {baseline_id: {option: value}} for artifact-backed baselines')
    artifact = sub.add_parser("prepare-baseline-artifact",
                              help="Derive a hashed artifact for an artifact-backed baseline")
    artifact.add_argument("baseline", choices=["proteingym-evcouplings-independent-v1"])
    artifact.add_argument("--manifest", required=True, help="Reviewed model manifest JSON")
    artifact.add_argument("--output", required=True, help="New artifact JSON path")
    enqueue = sub.add_parser("enqueue", help="Freeze a reviewed SDK submission for later delivery")
    enqueue.add_argument("--bundle", required=True)
    enqueue.add_argument("--queue", required=True)
    for field in ("title", "summary", "source-url", "source-locator"):
        enqueue.add_argument("--" + field, required=True)
    enqueue.add_argument("--metric-path", required=True, help='JSON key list, e.g. ["auroc"]')
    drain = sub.add_parser("submit-queue", help="Inspect or explicitly deliver queued evaluations")
    drain.add_argument("--queue", required=True)
    drain.add_argument("--endpoint", default="https://benchmarks.rewire.it/api/trpc")
    drain.add_argument("--max-items", type=int, default=5)
    drain.add_argument("--send", action="store_true", help="Explicitly send; default is offline preview")
    prepare = sub.add_parser(
        "prepare", help="Validate local upstream resources and prepare a protocol"
    )
    prepare.add_argument("protocol", choices=sdk.PROTOCOLS)
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--options", default="{}", help="Protocol options as JSON")
    for name in ("run", "evaluate"):
        # No abbreviated options for `run`: a recipe step is checked by parsing it here,
        # and `--adap` must not be read as `--adapter`.
        cmd = sub.add_parser(name, allow_abbrev=name != "run")
        cmd.add_argument("--prepared", required=True)
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--model-name", help="Default: the registered model's title, "
                                              "or 'private model' with --adapter")
        cmd.add_argument("--training-overlap", default="unreported")
        cmd.add_argument("--allow-partial", action="store_true")
        if name == "run":
            which = cmd.add_mutually_exclusive_group(required=True)
            which.add_argument(
                "--model", help="A registered model; see `rewirebench models`"
            )
            which.add_argument(
                "--adapter", help="Explicit local Python module:class to execute"
            )
            cmd.add_argument("--adapter-options", default="{}")
            cmd.add_argument("--batch-size", type=int, default=32)
            cmd.add_argument("--prediction-type", choices=["scalar", "embedding"])
        else:
            inputs = cmd.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--predictions")
            inputs.add_argument("--embeddings", help="Keyed JSON vectors or safe NPZ {ids, embeddings}")
    export = sub.add_parser("export", help="Create a private-data-minimised contribution bundle")
    export.add_argument("--report", required=True)
    export.add_argument("--output", required=True)
    post = sub.add_parser("submit", help="Explicitly submit an exported bundle to review")
    post.add_argument("--bundle", required=True)
    for field in ("title", "summary", "source-url", "metric", "value", "source-locator"):
        post.add_argument("--" + field, required=True)
    post.add_argument("--endpoint", default="https://benchmarks.rewire.it/api/trpc")
    post.add_argument("--idempotency-key")
    post.add_argument("--dry-run", action="store_true")
    recipe = sub.add_parser("recipe", help="List, show, validate and run recipes from recipes/")
    recipe_sub = recipe.add_subparsers(dest="recipe_command", required=True)
    recipe_commands = {
        "list": "List the recipes with their SHA-256",
        "show": "Print one recipe as validated JSON",
        "validate": "Validate every recipe, or the named files",
        "schema": "Print the JSON Schema of a recipe",
        "plan": "Print what a run would need and cost on a backend. Starts nothing",
        "run": "Run a recipe and write receipt.json",
    }
    for name, text in recipe_commands.items():
        cmd = recipe_sub.add_parser(name, help=text)
        if name != "schema":
            cmd.add_argument("--recipes-dir", help="Default: the recipes/ of this checkout")
        if name in {"show", "plan", "run"}:
            cmd.add_argument("recipe", help="Recipe ID")
        if name == "validate":
            cmd.add_argument("paths", nargs="*", help="Recipe files; default is all of recipes/")
        if name in {"plan", "run"}:
            cmd.add_argument("--output", required=name == "run",
                             help="New folder for outputs and receipt")
            cmd.add_argument("--inputs", help="Folder holding the inputs; default is to fetch them")
            cmd.add_argument("--backend", default="local",
                             choices=["local", "spark", "runpod"] if name == "plan"
                             else ["local", "spark"],
                             help="runpod can be planned and priced, not run")
            cmd.add_argument("--started-by", choices=["human", "agent", "ci"],
                             help="Default: ci in CI, agent under a known agent harness, "
                                  "otherwise human. An agent should pass agent")
            cmd.add_argument("--kg-release-id")
            cmd.add_argument("--spark-host", help="SSH destination of the Spark; default "
                                                  "REWIREBENCH_SPARK_HOST")
            cmd.add_argument("--spark-workdir", help="Absolute folder on the Spark; default "
                                                     "REWIREBENCH_SPARK_WORKDIR")
        if name == "run":
            cmd.add_argument("--execute", action="store_true",
                             help="With --backend spark: connect and run. Without it the "
                                  "steps are printed and nothing is contacted")
            cmd.add_argument("--in-container", action="store_true",
                             help="Set by the spark backend inside the recipe's image")
    receipt = sub.add_parser("receipt", help="Check a run receipt")
    receipt_sub = receipt.add_subparsers(dest="receipt_command", required=True)
    check = receipt_sub.add_parser("verify", help="Recompute a receipt folder's digests and checks")
    check.add_argument("folder", help="Run folder holding receipt.json, or that file")
    check.add_argument("--recipes-dir", help="Default: the recipes/ of this checkout")
    check.add_argument("--inputs", help="Folder holding the inputs, to re-hash them")
    check.add_argument("--strict", action="store_true",
                       help="Also fail when a check could not be made. Passes only for a "
                            "recipe run from a clean commit")
    return parser


def parse_strict(argv: list[str]) -> argparse.Namespace:
    """Parse an argv list, raising ValueError where the command line would exit.

    Nothing is printed: ``--help`` in the list would otherwise write the help text to
    standard output, which the MCP server uses as its transport.
    """
    with (contextlib.redirect_stderr(io.StringIO()) as problem,
          contextlib.redirect_stdout(io.StringIO())):
        try:
            return build_parser().parse_args(argv)
        except SystemExit:
            lines = problem.getvalue().strip().splitlines()
            raise ValueError(
                lines[-1] if lines else "Not a command to run (it asks for help text)"
            ) from None


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "mcp":
        # Stdout belongs to the MCP transport from here on, so nothing below may print.
        try:
            from rewirebench import mcp_server
        except ImportError as error:
            parser.exit(2, f"The MCP server needs the mcp package ({error}). "
                           "Install rewirebench[mcp].\n")
        mcp_server.serve(recipes_dir=args.recipes_dir)
        return
    arguments = list(sys.argv[1:] if argv is None else argv)
    started_at = datetime.datetime.now(datetime.UTC)
    started = time.perf_counter()
    exit_code = 0
    try:
        if args.command == "recipe":
            result, exit_code = _recipe_command(args)
        elif args.command == "receipt":
            from rewirebench import receipts, recipes
            result = receipts.verify(args.folder, inputs=args.inputs,
                                     recipes_dir=recipes.find_recipes_dir(args.recipes_dir))
            exit_code = 0 if result["complete" if args.strict else "ok"] else 1
        elif args.command == "clean":
            from rewirebench.cleanup import clean
            result = clean(args.root, older_than_days=args.older_than_days, apply=args.apply)
        elif args.command == "protocols":
            result = list(sdk.PROTOCOLS)
        elif args.command == "models":
            from rewirebench import models
            result = models.list_models(args.protocol)
        elif args.command == "inspect":
            result = sdk.describe(args.protocol)
            from rewirebench.baselines import describe_baselines
            result["baselines"] = describe_baselines(args.protocol)
        elif args.command == "baselines":
            from rewirebench.baselines import describe_baselines
            result = describe_baselines(args.protocol)
        elif args.command == "run-baselines":
            from rewirebench.baselines import run_baselines
            result = run_baselines(args.prepared, output=args.output,
                baseline_ids=args.baseline, batch_size=args.batch_size,
                baseline_options=json.loads(args.baseline_options))
            from rewirebench import receipts
            receipts.write_adhoc_receipt(
                args.output, argv=arguments, input_files=[_prepared_file(args.prepared)],
                metrics=_baseline_metrics(args.output), started_at=started_at,
                wall_seconds=time.perf_counter() - started)
        elif args.command == "prepare-baseline-artifact":
            from rewirebench.adapters.evcouplings_independent import prepare_artifact
            result = prepare_artifact(args.manifest, args.output)
        elif args.command == "enqueue":
            from rewirebench.submission_queue import enqueue_submission
            result = enqueue_submission(args.bundle, queue=args.queue, title=args.title,
                summary=args.summary, source_url=args.source_url, source_locator=args.source_locator,
                metric_path=json.loads(args.metric_path))
        elif args.command == "submit-queue":
            from rewirebench.submission_queue import drain_submissions
            result = drain_submissions(args.queue, endpoint=args.endpoint, max_items=args.max_items,
                dry_run=not args.send, token=os.environ.get("REWIRE_SUBMISSION_TOKEN"))
        elif args.command == "prepare":
            result = sdk.prepare(
                args.protocol, source=args.source, output=args.output, **json.loads(args.options)
            )
            result = {
                "protocol_id": result["protocol_id"],
                "scope": result["scope"],
                "prepared": str(Path(args.output) / "prepared.json"),
            }
        elif args.command in {"run", "evaluate"}:
            model_name = args.model_name or "private model"
            if args.command == "run" and args.model:
                from rewirebench import models
                spec = models.get_model(args.model)
                model_name = args.model_name or spec.title
                adapter = models.load(spec, json.loads(args.adapter_options))
            elif args.command == "run":
                module, name = args.adapter.split(":", 1)
                factory = getattr(importlib.import_module(module), name)
                adapter = factory(**json.loads(args.adapter_options))
            kwargs = {
                "output": args.output,
                "model": {"name": model_name, "training_overlap": args.training_overlap},
                "allow_partial": args.allow_partial,
            }
            if args.command == "run":
                result = sdk.run(
                    args.prepared,
                    adapter,
                    batch_size=args.batch_size,
                    prediction_type=args.prediction_type,
                    **kwargs,
                )
            else:
                result = sdk.evaluate(
                    args.prepared, args.predictions, embeddings=args.embeddings, **kwargs,
                )
            from rewirebench import receipts
            used = [_prepared_file(args.prepared)]
            if args.command == "evaluate":
                used.append(Path(args.predictions or args.embeddings))
            written = receipts.write_adhoc_receipt(
                args.output, argv=arguments, input_files=used, metrics=result["metrics"],
                started_at=started_at, wall_seconds=time.perf_counter() - started)
            result = {
                "coverage": result["coverage"],
                "metrics": result["metrics"],
                "report": str(Path(args.output) / "report.json"),
            }
            if written is not None:
                result["receipt"] = str(Path(args.output) / receipts.RECEIPT_FILENAME)
        elif args.command == "export":
            sdk.export(args.report, output=args.output)
            result = {"bundle": args.output, "uploaded": False}
        else:
            result = submit(
                args.bundle,
                title=args.title,
                summary=args.summary,
                source_url=args.source_url,
                metric=args.metric,
                value=args.value,
                source_locator=args.source_locator,
                endpoint=args.endpoint,
                idempotency_key=args.idempotency_key,
                dry_run=args.dry_run,
                token=os.environ.get("REWIRE_SUBMISSION_TOKEN"),
            )
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, SubmissionError) as exc:
        parser.exit(2, f"{exc}\n")
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
