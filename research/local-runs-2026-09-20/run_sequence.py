"""Run the two selected real-data controls. Raw inputs and predictions stay local."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import rewirebench as rb
from rewirebench.adapters.sequence import SequenceComposition, TrainMean
from rewirebench.resources.flip2.example_adapter import CompositionEmbeddings

WHEEL_SHA256 = "f5956f616c20f58299eef4d654b1297b0ce21e0033014cc5cb2c1a406f4b494a"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flip-source", type=Path, required=True)
    parser.add_argument("--mrna-source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    assert importlib.metadata.version("rewirebench") == "0.4.0"
    assert shutil.disk_usage(args.work.parent).free > 5 * 1024**3
    args.work.mkdir(parents=True, exist_ok=False)
    cases = [
        (
            "flip2",
            "flip2-fitness-v1",
            args.flip_source,
            {"dataset": "flip2-rhomax-by-wild-type"},
            CompositionEmbeddings,
            "Amino-acid composition + fixed ridge (Rewire control)",
            {
                "features": "40 amino-acid fractions; second component zero for Rhomax",
                "alpha": 10,
                "target_scaling": "training-only StandardScaler",
                "fit_intercept": True,
                "validation_used": False,
            },
        ),
        (
            "mrnabench",
            "mrnabench-sample-mrl-v1",
            args.mrna_source,
            {"dataset": "designed", "target": "target_mrl_designed", "require_official": True},
            SequenceComposition,
            "Sequence composition + RidgeCV (Rewire control)",
            {
                "features": "log1p length, A/C/G/T fractions, unknown fraction",
                "alphas": [0.001, 0.01, 0.1, 1, 10],
                "normalization": "none",
                "selection": "RidgeCV within training only",
                "split_seed": 2541,
                "validation_used": False,
            },
        ),
    ]
    for key, protocol, source, options, adapter_cls, model_name, config in cases:
        smoke = rb.prepare(
            protocol,
            source=source,
            output=args.work / (key + "-smoke-prepared"),
            **options,
            limit=32,
        )
        sr = rb.run(
            smoke,
            adapter_cls(),
            output=args.work / (key + "-smoke-result"),
            prediction_type="embedding",
        )
        assert sr["scope"] == "smoke"
        prepared = rb.prepare(
            protocol, source=source, output=args.work / (key + "-prepared"), **options
        )
        assert prepared["scope"] == "full"
        counts = dict(Counter(r["split"] for r in prepared["rows"]))
        for suffix, adapter, name, settings in [
            ("composition", adapter_cls(), model_name, config),
            (
                "train-mean",
                TrainMean(),
                "Training mean (Rewire control)",
                {
                    "fit_split": "train",
                    "constant": "arithmetic mean of training targets",
                    "validation_used": False,
                },
            ),
        ]:
            run_id = key + "-" + suffix
            local = args.work / run_id
            report = rb.run(
                prepared,
                adapter,
                output=local,
                model={
                    "name": name,
                    "training_overlap": "No pretraining; fitted only on this protocol training split. Split is not a claim of homology separation.",
                    "configuration": settings,
                    "input_information": "Complete source sequence only; no assay labels during prediction",
                },
            )
            assert report["completion"] == "complete" and report["coverage"]["unscored"] == 0
            public = args.evidence / run_id
            public.mkdir(parents=True, exist_ok=False)
            safe = copy.deepcopy(report)
            safe["protocol_configuration"].pop("split_membership", None)
            safe["protocol_configuration"]["split_counts"] = counts
            safe["release_wheel_sha256"] = WHEEL_SHA256
            safe["packages"] = {
                k: importlib.metadata.version(k)
                for k in ["rewirebench", "numpy", "scipy", "scikit-learn", "pandas", "pyarrow"]
            }
            safe["publication_note"] = (
                "Sanitized local execution report. Raw inputs, source membership, predictions and machine paths remain local. No published-score reproduction claim."
            )
            write(public / "report.json", safe)
            rb.export(report, output=public / "bundle.json")
            write(
                public / "source.json",
                {
                    "source_url": prepared["provenance"]["source_url"],
                    "retrieved_from": (
                        "https://flip.protein.properties/assets/splits/rhomax/by_wild_type.csv.gz"
                        if key == "flip2"
                        else prepared["provenance"]["source_url"]
                    ),
                    "source_bytes_note": (
                        "Official mirror gzip differs from Zenodo gzip; decompressed CSV matches pinned Zenodo v3 bytes."
                        if key == "flip2"
                        else "Processed parquet matches the pinned Hugging Face revision."
                    ),
                    "source_sha256": prepared["provenance"]["source_sha256"],
                    "verified_at": datetime.now(UTC).isoformat(),
                    "retrieval": "Reused local source bytes from prior recorded acquisition; checked against package pins before execution",
                    "split_counts": counts,
                    "provenance": prepared["provenance"],
                    "local_report_sha256": hashlib.sha256(
                        (local / "report.json").read_bytes()
                    ).hexdigest(),
                    "local_predictions_file_sha256": hashlib.sha256(
                        (local / "predictions.json").read_bytes()
                    ).hexdigest(),
                    "local_prepared_file_sha256": hashlib.sha256(
                        (args.work / (key + "-prepared") / "prepared.json").read_bytes()
                    ).hexdigest(),
                },
            )
            print(
                json.dumps(
                    {"run": run_id, "coverage": report["coverage"], "metrics": report["metrics"]}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
