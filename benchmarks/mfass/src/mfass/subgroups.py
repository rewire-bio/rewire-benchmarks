"""Stratify a comparison by distance to the nearest exon boundary.

Canonical splice sites are largely a solved problem, and a method that only wins
there has not helped with the variants a laboratory actually struggles to
interpret. Across the whole MFASS cohort the canonical band holds 16.6% of the
splice-disrupting variants, matching the source paper's ~17%; within a given test
split the share differs (13.3% on the reported three-method common subset).

**Confound, stated because it changes how these bands read.** The band variable is
the minimum absolute distance to either exon boundary, and that same quantity is a
fitted feature of the trivial baseline in run_baseline.py. Stratifying on one of a
model's own inputs is not a neutral cut for that model, so a band-wise reversal
involving the baseline is confounded with its supervision, not merely imprecise.

Bands are defined on the minimum absolute distance from the variant to either exon
boundary, in window coordinates:

    canonical   <= 2 bases   the essential acceptor/donor dinucleotides
    near        3 to 10
    mid         11 to 30
    distal      > 30
"""
import argparse
import csv
import json
import pathlib

import numpy as np

from rewirebench import metrics as M

BANDS = [("canonical <=2", 0, 2), ("near 3-10", 3, 10), ("mid 11-30", 11, 30), ("distal >30", 31, 10**9)]


def band_of(row):
    pos = float(row["rel_position"])
    acceptor = float(row["intron1_len"])
    donor = acceptor + float(row["exon_len"])
    d = min(abs(pos - acceptor), abs(pos - donor))
    for name, lo, hi in BANDS:
        if lo <= d <= hi:
            return name
    return BANDS[-1][0]


def load_predictions(path):
    with open(path, newline="") as fh:
        out = {}
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["score"] in ("", "NA", None):
                continue
            out[r["id"]] = (r["group"], int(r["label"]), float(r["score"]))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="benchmarks/mfass/data/cohort.tsv")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="name=path/to/predictions.tsv, repeatable")
    ap.add_argument("--capacity", type=int, default=100,
                    help="review capacity over the whole test set; scaled per band by band size")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.cohort, newline="") as fh:
        cohort = {r["id"]: r for r in csv.DictReader(fh, delimiter="\t")}

    preds = {}
    for spec in args.methods:
        name, path = spec.split("=", 1)
        preds[name] = load_predictions(path)

    common = sorted(set.intersection(*(set(p) for p in preds.values())))
    bands = {i: band_of(cohort[i]) for i in common}

    report = {"capacity_total": args.capacity, "common_variants": len(common), "bands": {}}
    for name, _, _ in BANDS:
        ids = [i for i in common if bands[i] == name]
        if not ids:
            continue
        labels = np.array([preds[list(preds)[0]][i][1] for i in ids])
        n_pos = int(labels.sum())
        cap = max(1, round(args.capacity * len(ids) / len(common)))
        entry = {
            "variants": len(ids),
            "positives": n_pos,
            "prevalence": round(float(labels.mean()), 5),
            "share_of_all_positives": None,
            "capacity_scaled": cap,
            "methods": {},
        }
        for mname, p in preds.items():
            s = np.array([p[i][2] for i in ids])
            if n_pos == 0 or n_pos == len(ids):
                entry["methods"][mname] = {"note": "band has a single class; metrics undefined"}
                continue
            entry["methods"][mname] = {
                "precision_at_scaled_capacity": M.precision_at_n(labels, s, cap),
                "average_precision_sklearn": float(M.point_metrics(labels, s, cap)["average_precision_sklearn"]),
                "auroc": float(M.point_metrics(labels, s, cap)["auroc"]),
            }
        report["bands"][name] = entry

    total_pos = sum(b["positives"] for b in report["bands"].values())
    for b in report["bands"].values():
        b["share_of_all_positives"] = round(b["positives"] / total_pos, 4) if total_pos else None

    print(json.dumps(report, indent=2))
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
