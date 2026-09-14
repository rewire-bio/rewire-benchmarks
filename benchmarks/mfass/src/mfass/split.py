"""Group the MFASS cohort into independent units and emit a split manifest.

The grouping unit is the connected component of the relations that must not cross
the train/test boundary, not a concatenated key. Concatenating keys reports more
independent units than exist whenever one value of one key spans two values of
another.

For MFASS the mandatory relation is the exon: many variants share an exon and were
assayed in the same minigene context. Extra relations can be supplied with --key
once a mapping exists, for example an exon-to-gene table.
"""
import argparse
import csv
import json
import pathlib

from rewirebench.splits import assign_groups, connected_components, naive_key_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="benchmarks/mfass/data/cohort.tsv")
    ap.add_argument("--out", default="benchmarks/mfass/splits/split-v1.tsv")
    ap.add_argument("--key", action="append", default=None,
                    help="equivalence key column; repeatable. Default: ensembl_id")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    keys = args.key or ["ensembl_id"]
    with open(args.cohort, newline="") as fh:
        records = list(csv.DictReader(fh, delimiter="\t"))

    groups = connected_components(records, keys)
    split, by_group, manifest = assign_groups(
        records, groups, label_field="sdv",
        test_frac=args.test_frac, seed=args.seed,
    )
    n_naive = naive_key_count(records, keys)
    manifest = {
        "benchmark": "mfass-v1",
        "keys": keys,
        "naive_concatenated_key_count": n_naive,
        "inflation_factor": round(n_naive / manifest["independent_groups"], 4),
        **manifest,
    }
    print(json.dumps(manifest, indent=2))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "group", "split"])
        for r in records:
            w.writerow([r["id"], str(groups[r["id"]]), split[r["id"]]])
    with open(out.with_suffix(".manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
