"""Compare two scored methods on the variants both could score.

Methods disagree about which inputs they can handle, so a table of per-method
metrics is not a comparison: each row is computed on a different denominator. This
restricts to the intersection, recomputes both, and gives a paired interval on the
difference by resampling whole groups.
"""
import argparse
import csv
import json
import pathlib

import numpy as np
from rewirebench import metrics as M


def load_predictions(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = {}
    seen = set()
    for r in rows:
        if not r["id"] or r["id"] in seen:
            raise ValueError("Prediction IDs must be nonempty and unique")
        seen.add(r["id"])
        if int(r["label"]) not in (0, 1) or not r["group"]:
            raise ValueError("Invalid prediction label or group")
        if r["score"] in ("", "NA", None):
            continue
        value = float(r["score"])
        if not np.isfinite(value):
            raise ValueError("Nonfinite prediction score")
        out[r["id"]] = (r["group"], int(r["label"]), value)
    return out


def check_protocol_metadata(baseline_path, candidate_path):
    """Refuse conflicting pinned manifests; legacy tables retain unknown status."""
    records = []
    for path in (baseline_path, candidate_path):
        path = pathlib.Path(path)
        suffix = ".predictions.tsv"
        sidecar = path.with_name(path.name[:-len(suffix)] + ".json") if path.name.endswith(suffix) else None
        records.append(json.loads(sidecar.read_text()) if sidecar and sidecar.exists() else {})
    a, b = (record.get("config", {}) for record in records)
    checked = []
    for key in ("split_sha256", "cohort_sha256"):
        if a.get(key) and b.get(key):
            if a[key] != b[key]:
                raise ValueError(f"Incompatible paired evaluation {key}")
            checked.append(key)
    if any(record.get("config", {}).get("scope") == "smoke" or
           record.get("benchmark", "").endswith("-smoke") for record in records):
        raise ValueError("Smoke runs cannot enter a benchmark comparison")
    return {"checked_hashes": checked,
            "status": "shared_dataset_and_split" if len(checked) == 2 else "protocol_compatibility_unverified",
            "note": "Shared IDs, labels and groups are checked. Matching rows alone do not establish matching inputs, training or evaluation protocols."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="predictions.tsv for the reference method")
    ap.add_argument("--candidate", required=True, help="predictions.tsv for the method under test")
    ap.add_argument("--capacity", type=int, default=100)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.capacity < 1 or args.draws < 1:
        ap.error("capacity and draws must be positive")
    if args.out and pathlib.Path(args.out).exists():
        raise FileExistsError("Refusing to overwrite an existing comparison")
    compatibility = check_protocol_metadata(args.baseline, args.candidate)
    b = load_predictions(args.baseline)
    c = load_predictions(args.candidate)
    common = sorted(set(b) & set(c))
    if not common:
        raise SystemExit("no variants scored by both methods")

    if any(b[i][:2] != c[i][:2] for i in common):
        raise ValueError("Paired predictions disagree on label or independent group")
    labels = np.array([b[i][1] for i in common])
    if len(set(labels)) != 2:
        raise ValueError("Comparison requires both outcome classes")
    groups = np.array([b[i][0] for i in common])
    sb = np.array([b[i][2] for i in common])
    sc = np.array([c[i][2] for i in common])

    report = {
        "compatibility": compatibility,
        "baseline_file": args.baseline,
        "candidate_file": args.candidate,
        "capacity": args.capacity,
        "seed": args.seed,
        "denominators": {
            "baseline_scored": len(b),
            "candidate_scored": len(c),
            "common": len(common),
            "baseline_only": len(set(b) - set(c)),
            "candidate_only": len(set(c) - set(b)),
        },
        "on_common_subset": {
            "baseline": M.point_metrics(labels, sb, args.capacity),
            "candidate": M.point_metrics(labels, sc, args.capacity),
        },
        "independent_groups": len(set(groups)),
    }

    for metric in ("precision_at_capacity", "average_precision_sklearn", "auroc"):
        try:
            report.setdefault("paired", {})[metric] = M.paired_group_bootstrap(
                labels, sb, sc, groups, capacity=args.capacity,
                metric=metric, n=args.draws, seed=args.seed,
            )
        except ValueError as exc:
            report.setdefault("paired", {})[metric] = {"refused": str(exc)}

    print(json.dumps(report, indent=2))
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("x") as fh:
            fh.write(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
