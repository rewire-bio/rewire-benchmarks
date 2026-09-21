"""Local CPU execution; pass prepared data, verified checkpoint and new output paths.

No network calls. Public exports contain aggregate reports, not labels, sequences,
embeddings or local paths. The caller supplies a frozen PYTHONPATH package copy.
"""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
from rewirebench import sdk
from rewirebench.adapters.esm_embeddings import ESM2Embeddings
from rewirebench.baselines import run_baselines
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--model-name", default="esm2_t6_8M_UR50D", choices=["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D"])
    parser.add_argument("--verify-existing", action="store_true", help="Audit immutable prior outputs without rerunning models")
    args = parser.parse_args()
    if not args.verify_existing:
        args.output.mkdir(parents=True, exist_ok=False)
    args.evidence.mkdir(parents=True, exist_ok=False)
    layer, dimension, display = (6, 320, "8M") if args.model_name == "esm2_t6_8M_UR50D" else (12, 480, "35M")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    prepared = sdk._load(args.prepared, "prepared.json")
    sdk._validate_prepared(prepared)
    if prepared["dataset_id"] != "flip2-rhomax-by-wild-type" or prepared["scope"] != "full":
        raise ValueError("This receipt is specific to the complete archived Rhomax split")
    code_start = sdk._code_digest()
    if args.verify_existing:
        report = json.loads((args.output / "esm2/report.json").read_text())
        assert report["environment"]["sdk_code_sha256"] == code_start
        assert report["prepared_sha256"] == prepared["prepared_sha256"]
        controls = json.loads((args.output / "controls/baseline-manifest.json").read_text())
        elapsed = None
    else:
        start = time.perf_counter()
        # Default construction retains compatibility with the exact archived 8M code.
        encoder = ESM2Embeddings(args.checkpoint, device="cpu", **({"model_name": args.model_name} if layer == 12 else {}))
        report = sdk.run(prepared, encoder, output=args.output / "esm2", batch_size=4,
            prediction_type="embedding", model={
                "name": f"ESM-2 {display} frozen residue-mean embeddings + fixed ridge (Rewire)",
                "training_overlap": "UniRef50 pretraining overlap is unreported; the linear head fits only archived training rows",
                "input_information": "Complete amino-acid sequences; no labels enter encoder; no MSA/templates",
                "configuration": {"checkpoint": args.model_name, "representation_layer": layer,
                    "pooling": "mean over residues excluding BOS/EOS and padding", "encoder_frozen": True,
                    "dimension": dimension, "head": prepared["metadata"]["embedding_probe"],
                    "device": "cpu", "torch_threads": 4, "torch_seed": 0,
                    "validation_used": False, "scope": "one complete Rhomax split, not FLIP2 suite"},
            })
        elapsed = time.perf_counter() - start
        controls = run_baselines(prepared, output=args.output / "controls")
    if controls["status"] != "evaluated":
        raise RuntimeError("A reference control failed")
    runs = [("esm2", args.output / "esm2", report)]
    for item in controls["baselines"]:
        directory = args.output / "controls" / item["baseline_id"]
        runs.append((item["baseline_id"], directory, json.loads((directory / "report.json").read_text())))
    test = [row for row in prepared["rows"] if row["split"] == "test"]
    target = np.asarray([row["target"] for row in test])
    checks = []
    for ident, directory, item in runs:
        predictions = json.loads((directory / "predictions.json").read_text())
        assert set(predictions) == {row["id"] for row in test}
        values = np.asarray([predictions[row["id"]] for row in test])
        # FLIP2 upstream shifts all scored targets by their minimum, even if positive.
        relevance = target - float(target.min())
        correlation = float(spearmanr(target, values).statistic) if len(set(values)) > 1 else None
        ndcg = float(ndcg_score(relevance[None, :], values[None, :]))
        assert item["metrics"]["spearman"] is None if correlation is None else abs(correlation - item["metrics"]["spearman"]) < 1e-12
        assert abs(ndcg - item["metrics"]["ndcg"]) < 1e-12
        assert item["coverage"] == {"denominator": 184, "scored": 184, "unscored": 0}
        sdk.export(item, output=args.evidence / f"{ident}.bundle.json")
        # Reports contain no prepared rows, predictions, embeddings or paths.
        write(args.evidence / f"{ident}.report.json", item)
        checks.append({"run": ident, "scored": len(values), "spearman": correlation, "ndcg": ndcg,
                       "independent_metric_recomputation": "passed", "tolerance": 1e-12,
                       "predictions_sha256": item["predictions_sha256"],
                       "evidence_origin": "Rewire local model execution, not paper reproduction"})
    code_end = sdk._code_digest()
    assert code_start == code_end
    package_root = Path(sdk.__file__).parent
    write(args.evidence / "package-code-manifest.json", {
        "sdk_code_sha256": code_start,
        "files": {str(p.relative_to(package_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(package_root.rglob("*.py"))},
    })
    write(args.evidence / "verification.json", {
        "schema_version": "1.0", "review_method": "automated independent metric recomputation from local predictions",
        "source_verification": prepared["provenance"].get("data_verification"),
        "prepared_sha256": prepared["prepared_sha256"], "dataset_id": prepared["dataset_id"],
        "scope": "one complete Rhomax split; no suite aggregate", "split_counts": {"train": 584, "validation": 116, "test": 184},
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "code_hash_start": code_start, "code_hash_end": code_end,
        "esm2_load_inference_fit_evaluate_seconds": elapsed,
        "timing_scope": "elapsed is null when auditing existing outputs; execution.inference_and_fit_seconds in the original report covers encoding plus fitting, excludes checkpoint load/acquisition",
        "environment": {p: importlib.metadata.version(p) for p in ("torch", "fair-esm", "numpy", "scipy", "scikit-learn")},
        "runs": checks, "limitations": ["No confidence intervals", "Single task and split", "Unreported pretraining overlap", "No container validation", "No upload/publication performed by this script"],
    })
    print(json.dumps({"verified": checks, "elapsed_seconds": elapsed, "code_sha256": code_start}))


if __name__ == "__main__":
    main()
