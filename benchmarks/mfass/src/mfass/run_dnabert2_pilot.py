"""Zero-cost, unscored local DNABERT-2 feasibility run on MFASS sequence pairs.

This deliberately does not read assay labels or produce a leaderboard result. It
measures whether a pinned checkpoint can load and embed representative 170bp
reference/mutant pairs on the local machine before anyone plans a full run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import pathlib
import platform
import resource
import shutil
import time
from datetime import UTC, datetime

MODEL_ID = "zhihan1996/DNABERT-2-117M"
MODEL_REVISION = "b5ae377faa374ee160eec1c27b8494436cc94451"
SEED = "rewire-mfass-dnabert2-pilot-v1"
MIN_FREE_BYTES = 1024**3
ESTIMATED_DOWNLOAD_BYTES = 750 * 1024**2


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_pairs(cohort: pathlib.Path, split: pathlib.Path, per_orientation: int = 16):
    """Select reproducible held-out pairs from both legacy orientation strata."""
    with split.open(newline="") as fh:
        heldout = {r["id"] for r in csv.DictReader(fh, delimiter="\t") if r["split"] == "test"}
    strata = {"assay": [], "reverse_complement": []}
    with cohort.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["id"] not in heldout:
                continue
            orientation = r["legacy_sequence_orientation"]
            if orientation not in strata:
                raise ValueError(f"{r['id']}: unknown legacy orientation")
            p = int(r["rel_position"]) - 1
            reference, mutant = r["reference_sequence"], r["mutant_sequence"]
            if not (len(reference) == len(mutant) == 170 and 0 <= p < 170):
                raise ValueError(f"{r['id']}: invalid assay pair")
            if [i for i, (a, b) in enumerate(zip(reference, mutant)) if a != b] != [p]:
                raise ValueError(f"{r['id']}: invalid mutant position")
            strata[orientation].append((r["id"], reference, mutant))
    selected = []
    for orientation, pairs in strata.items():
        if len(pairs) < per_orientation:
            raise ValueError(f"{orientation}: too few test pairs")
        pairs.sort(key=lambda pair: hashlib.sha256(f"{SEED}:{pair[0]}".encode()).hexdigest())
        selected.extend((orientation, *pair) for pair in pairs[:per_orientation])
    selected.sort(key=lambda pair: pair[1])
    return selected


def peak_rss_bytes() -> int:
    # macOS reports ru_maxrss in bytes, unlike Linux's KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=pathlib.Path,
                    default=pathlib.Path("benchmarks/mfass/data/cohort.tsv"))
    ap.add_argument("--split", type=pathlib.Path,
                    default=pathlib.Path("benchmarks/mfass/splits/split-v2.tsv"))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("benchmarks/mfass/feasibility/dnabert2-local.json"))
    ap.add_argument("--per-orientation", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = ap.parse_args()
    if args.per_orientation < 1 or args.batch_size < 1:
        ap.error("sample size and batch size must be positive")

    pairs = sample_pairs(args.cohort, args.split, args.per_orientation)
    free_before = shutil.disk_usage(args.out.parent.resolve() if args.out.parent.exists()
                                    else args.out.parent.parent.resolve()).free
    if free_before < MIN_FREE_BYTES + ESTIMATED_DOWNLOAD_BYTES:
        raise SystemExit("Insufficient free disk to cache checkpoint and preserve a 1 GiB reserve")

    report = {
        "kind": "unscored_local_feasibility",
        "status": "started",
        "created_utc": datetime.now(UTC).isoformat(),
        "model": MODEL_ID,
        "checkpoint_and_code_revision": MODEL_REVISION,
        "source": {
            "cohort_sha256": sha256(args.cohort),
            "split_sha256": sha256(args.split),
            "sample_seed": SEED,
            "sample_ids": [pair[1] for pair in pairs],
            "sample_orientation_counts": {"assay": args.per_orientation,
                                          "reverse_complement": args.per_orientation},
            "pairs": len(pairs),
            "sequences": len(pairs) * 2,
            "length_bases": 170,
        },
        "environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "device": args.device,
            "batch_size": args.batch_size, "dtype": "float32",
            "disk_free_bytes_before": free_before,
        },
        "accuracy": None,
        "leaderboard_eligible": False,
    }

    t_all = time.perf_counter()
    failures = []
    try:
        import torch
        import transformers
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        report["environment"]["torch"] = torch.__version__
        report["environment"]["transformers"] = transformers.__version__
        report["environment"]["mps_available"] = bool(torch.backends.mps.is_available())
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")

        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = AutoModel.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
            use_safetensors=True, torch_dtype=torch.float32,
        )
        model.eval().to(args.device)
        load_seconds = time.perf_counter() - t0
        t_hash = time.perf_counter()
        loaded_code = pathlib.Path(inspect.getfile(type(model)))
        pinned_code = pathlib.Path(hf_hub_download(MODEL_ID, "bert_layers.py",
                                                   revision=MODEL_REVISION))
        if sha256(loaded_code) != sha256(pinned_code):
            raise RuntimeError("Loaded remote code differs from the pinned repository revision")
        report["environment"]["loaded_remote_code_module"] = type(model).__module__
        report["environment"]["loaded_remote_code_sha256"] = sha256(loaded_code)
        report["environment"]["checkpoint_safetensors_sha256"] = sha256(
            pathlib.Path(hf_hub_download(MODEL_ID, "model.safetensors",
                                          revision=MODEL_REVISION)))
        report["timing_seconds"] = {"load_tokenizer_and_checkpoint": load_seconds,
                                    "verify_pinned_artifacts": time.perf_counter() - t_hash}
        report["memory"] = {"peak_process_rss_bytes_after_load": peak_rss_bytes()}
        free_after_load = shutil.disk_usage(args.out.parent.parent.resolve()).free
        if free_after_load < MIN_FREE_BYTES:
            raise RuntimeError("Checkpoint load left less than 1 GiB disk free")

        lengths = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(pairs), args.batch_size):
                batch_pairs = pairs[start:start + args.batch_size]
                sequences = [s for _, _, ref, mut in batch_pairs for s in (ref, mut)]
                encoded = tokenizer(sequences, return_tensors="pt", padding=True,
                                    truncation=False)
                lengths.extend(int(n) for n in encoded["attention_mask"].sum(dim=1))
                encoded = {k: v.to(args.device) for k, v in encoded.items()}
                try:
                    output = model(**encoded)
                    if output is None:
                        raise RuntimeError("model returned no output")
                except Exception as exc:  # noqa: BLE001
                    failures.append({"sample_ids": [p[1] for p in batch_pairs],
                                     "error_type": type(exc).__name__,
                                     "message": str(exc)[:300]})
        inference_time = time.perf_counter() - t0
        report["timing_seconds"]["embed_pairs"] = inference_time
        report["timing_seconds"]["end_to_end"] = time.perf_counter() - t_all
        report["token_lengths"] = {"min": min(lengths), "max": max(lengths),
                                   "mean": sum(lengths) / len(lengths)}
        report["memory"]["peak_process_rss_bytes"] = peak_rss_bytes()
        report["environment"]["disk_free_bytes_after"] = shutil.disk_usage(
            args.out.parent.parent.resolve()).free
        report["failures"] = failures
        if failures:
            report["status"] = "inference_failures"
        else:
            report["status"] = "feasible"
            report["projected_full_test_hours"] = (
                inference_time / len(pairs) * 8324 / 3600
            )
            report["full_run_within_12h"] = report["projected_full_test_hours"] <= 12
            report["full_run_disk_reserve_ok"] = (
                report["environment"]["disk_free_bytes_after"] >= MIN_FREE_BYTES
            )
    except Exception as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["failure"] = {"error_type": type(exc).__name__, "message": str(exc)[:500]}
        report["timing_seconds"] = {**report.get("timing_seconds", {}),
                                    "end_to_end": time.perf_counter() - t_all}
        report["memory"] = {**report.get("memory", {}),
                            "peak_process_rss_bytes": peak_rss_bytes()}
        report["environment"]["disk_free_bytes_after"] = shutil.disk_usage(
            args.out.parent.parent.resolve()).free

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "feasible":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
