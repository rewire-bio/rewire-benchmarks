"""Command-line counterpart to the local Python SDK."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from rewirebench import sdk
from rewirebench.submission import SubmissionError, submit


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="rewirebench", description="Local biological benchmark evaluation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocols", help="List implemented protocols")
    inspect = sub.add_parser("inspect", help="Describe protocol inputs, datasets and execution rules")
    inspect.add_argument("protocol", choices=sdk.PROTOCOLS)
    prepare = sub.add_parser(
        "prepare", help="Validate local upstream resources and prepare a protocol"
    )
    prepare.add_argument("protocol", choices=sdk.PROTOCOLS)
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--options", default="{}", help="Protocol options as JSON")
    for name in ("run", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--prepared", required=True)
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--model-name", default="private model")
        cmd.add_argument("--training-overlap", default="unreported")
        cmd.add_argument("--allow-partial", action="store_true")
        if name == "run":
            cmd.add_argument(
                "--adapter", required=True, help="Explicit local Python module:class to execute"
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
    args = parser.parse_args(argv)
    try:
        if args.command == "protocols":
            result = list(sdk.PROTOCOLS)
        elif args.command == "inspect":
            result = sdk.describe(args.protocol)
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
            kwargs = {
                "output": args.output,
                "model": {"name": args.model_name, "training_overlap": args.training_overlap},
                "allow_partial": args.allow_partial,
            }
            if args.command == "run":
                module, name = args.adapter.split(":", 1)
                factory = getattr(importlib.import_module(module), name)
                result = sdk.run(
                    args.prepared,
                    factory(**json.loads(args.adapter_options)),
                    batch_size=args.batch_size,
                    prediction_type=args.prediction_type,
                    **kwargs,
                )
            else:
                result = sdk.evaluate(
                    args.prepared, args.predictions, embeddings=args.embeddings, **kwargs,
                )
            result = {
                "coverage": result["coverage"],
                "metrics": result["metrics"],
                "report": str(Path(args.output) / "report.json"),
            }
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


if __name__ == "__main__":
    main()
