"""Scoring loop and output writing shared by the SpliceAI and Pangolin runners.

Failure policy: a runner's per-variant function returns a score or one of the
tool's own expected skips (no coordinate, no gene, reference mismatch, a
nonfinite model score). Anything else raises. The exception stops the run
before a checkpoint row is written, so a resumed run retries that variant and
no result claims to be complete.

Outputs are written predictions first, then unscored reasons, then the result
JSON, each through a temporary file renamed into place. The JSON is written
last and records the SHA-256 of the other two, so a JSON's presence means its
tables are complete and identifies their exact bytes.
"""
import csv
import io
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

PREDICTION_FIELDS = ("id", "group", "label", "score")
EXPECTED_SKIP_PREFIXES = (
    "no hg38 coordinate",
    "no annotated gene overlapping the variant",
    "skipped by SpliceAI:",
    "skipped by Pangolin",
    "nonfinite model score",
)


def output_paths(out):
    out = Path(out)
    return {"json": out, "predictions": out.with_suffix(".predictions.tsv"),
            "unscored": out.with_suffix(".unscored.tsv")}


def refuse_finished(out):
    """Fail fast before scoring if this result already exists."""
    if output_paths(out)["json"].exists():
        raise FileExistsError(f"Refusing to overwrite {output_paths(out)['json']}")


def refuse_existing(out):
    """Refuse a finished result; move aside tables from an interrupted final write.

    Tables without their JSON were never bound to a result, so they are renamed
    with a timestamp, not deleted and not reused.
    """
    paths = output_paths(out)
    if paths["json"].exists():
        raise FileExistsError(f"Refusing to overwrite {paths['json']}")
    moved = []
    for key in ("predictions", "unscored"):
        if paths[key].exists():
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            aside = paths[key].with_name(f"{paths[key].name}.orphan-{stamp}")
            os.replace(paths[key], aside)
            moved.append(aside.name)
    return moved


def score_all(test, score_one, log=None, progress_every=250):
    """Score every selected row, reusing checkpoint rows. Returns scores, unscored, seconds."""
    scores, unscored = [], []
    t0 = time.perf_counter()
    for i, row in enumerate(test):
        prior = log.get(row["id"]) if log else None
        if prior is None:
            started = time.perf_counter()
            score, reason, raw = score_one(i, row)  # unexpected errors propagate
            if score is None and not reason.startswith(EXPECTED_SKIP_PREFIXES):
                raise RuntimeError(f"{row['id']}: unexpected unscored reason {reason!r}")
            if log:
                log.append(row["id"], score, time.perf_counter() - started, reason, raw)
        else:
            score = float(prior["score"]) if prior["status"] == "scored" else None
            reason = prior["reason"]
        scores.append(np.nan if score is None else score)
        if score is None:
            unscored.append((row["id"], reason))
        if (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{len(test)}  {(time.perf_counter() - t0) / (i + 1):.3f}s/variant",
                  flush=True)
    seconds = time.perf_counter() - t0
    if log:
        log.check_complete([row["id"] for row in test])
        # Sum of per-variant scoring time across every checkpoint segment.
        seconds = sum(float(row["seconds"]) for row in log.rows.values())
    return np.asarray(scores, dtype=float), unscored, seconds


def timing_scope(log):
    scope = ("code/weight and artifact hashing, model/reference load and prediction; "
             "cohort preparation excluded")
    if log:
        scope += ("; score_test sums per-variant time over every checkpoint segment, while "
                  "hashing and load times are from the final segment only")
    return scope


def _atomic_write(path, text):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _table(header, rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def write_outputs(result, out, test, groups, scores, unscored):
    """Write predictions, unscored reasons, then the result JSON binding both."""
    import hashlib

    paths = output_paths(out)
    orphans = refuse_existing(out)
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    predictions = _table(PREDICTION_FIELDS, (
        [row["id"], groups[row["id"]], row["sdv"], "" if np.isnan(s) else f"{s:.4f}"]
        for row, s in zip(test, scores)))
    _atomic_write(paths["predictions"], predictions)
    tables = {"predictions": {"file": paths["predictions"].name, "rows": len(test),
                              "sha256": hashlib.sha256(predictions.encode()).hexdigest()}}
    if unscored:
        text = _table(("id", "reason"), unscored)
        _atomic_write(paths["unscored"], text)
        tables["unscored"] = {"file": paths["unscored"].name, "rows": len(unscored),
                              "sha256": hashlib.sha256(text.encode()).hexdigest()}
    result.config["output_tables"] = tables
    result.config["orphaned_tables_moved_aside"] = orphans
    result.finalise()
    _atomic_write(paths["json"], json.dumps(asdict(result), indent=2, allow_nan=False))
    return paths["json"]
