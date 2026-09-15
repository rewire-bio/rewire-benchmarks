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
    for r in rows:
        if r["score"] in ("", "NA", None):
            continue
        out[r["id"]] = (r["group"], int(r["label"]), float(r["score"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="predictions.tsv for the reference method")
    ap.add_argument("--candidate", required=True, help="predictions.tsv for the method under test")
    ap.add_argument("--capacity", type=int, default=100)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    b = load_predictions(args.baseline)
    c = load_predictions(args.candidate)
    common = sorted(set(b) & set(c))
    if not common:
        raise SystemExit("no variants scored by both methods")

    labels = np.array([b[i][1] for i in common])
    groups = np.array([b[i][0] for i in common])
    sb = np.array([b[i][2] for i in common])
    sc = np.array([c[i][2] for i in common])

    report = {
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
        "independent_groups": int(len(set(groups))),
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
        p.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
