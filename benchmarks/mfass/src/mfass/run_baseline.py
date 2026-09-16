"""Trivial baseline for the MFASS benchmark.

Deliberately simple and deliberately allowed to win: position relative to the
exon boundaries, allele identity, and k-mer composition of the window around the
variant, fed to gradient-boosted trees. No pretrained model, no genome retrieval.

Everything is fitted on the train split only. The test split is scored once.
"""
import argparse
import csv
import hashlib
import itertools
import json
import pathlib
import time

import numpy as np
from rewirebench import metrics as M
from rewirebench.results import BenchmarkResult, write_result
from sklearn.ensemble import HistGradientBoostingClassifier

BASES = "ACGT"
REGIONS = ["exon", "upstr_intron", "downstr_intron"]


def kmers(k):
    return ["".join(p) for p in itertools.product(BASES, repeat=k)]


def featurise(rows, k=3, window=21):
    """Positional and compositional features. No label information is used."""
    vocab = {km: i for i, km in enumerate(kmers(k))}
    n_kmer = len(vocab)
    X = np.zeros((len(rows), 8 + 2 + len(REGIONS) + 8 + n_kmer), dtype=np.float32)

    for i, r in enumerate(rows):
        pos = float(r["rel_position"])
        i1, ex = float(r["intron1_len"]), float(r["exon_len"])
        acceptor, donor = i1, i1 + ex          # exon boundaries within the window
        c = 0
        X[i, c] = pos; c += 1
        X[i, c] = float(r["rel_position_scaled"]); c += 1
        X[i, c] = pos - acceptor; c += 1                       # signed dist to acceptor
        X[i, c] = pos - donor; c += 1                          # signed dist to donor
        X[i, c] = min(abs(pos - acceptor), abs(pos - donor)); c += 1
        X[i, c] = ex; c += 1
        X[i, c] = i1; c += 1
        X[i, c] = float(r["intron2_len"]); c += 1
        # Raw conservation. phyloP and phastCons are alignment statistics, not
        # trained predictors, so they belong in a trivial baseline. CADD is in the
        # cohort but is a trained model and gets its own comparator row instead.
        for col in ("phylop_score", "mean_phastCons_score"):
            v = r.get(col, "NA")
            X[i, c] = float(v) if v not in ("NA", "", None) else np.nan
            c += 1
        for reg in REGIONS:
            X[i, c] = 1.0 if r["label"] == reg else 0.0; c += 1
        ref, alt = r["ref_allele"], r["alt_allele"]
        for b in BASES:
            X[i, c] = 1.0 if ref == b else 0.0; c += 1
        for b in BASES:
            X[i, c] = 1.0 if alt == b else 0.0; c += 1

        # `sequence` is reverse-complemented for 7,770 cohort rows while
        # rel_position remains in assay orientation. The validated mutant pair
        # from build_dataset is the only safe source for a centred window.
        seq = r["mutant_sequence"].upper()
        p = int(pos) - 1
        if not 0 <= p < len(seq):
            raise ValueError(f"{r['id']}: invalid assay-oriented variant position")
        lo, hi = max(0, p - window // 2), min(len(seq), p + window // 2 + 1)
        sub = seq[lo:hi]
        counts = np.zeros(n_kmer, dtype=np.float32)
        for j in range(len(sub) - k + 1):
            idx = vocab.get(sub[j:j + k])
            if idx is not None:
                counts[idx] += 1
        total = counts.sum()
        if total:
            counts /= total
        X[i, c:c + n_kmer] = counts
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="benchmarks/mfass/data/cohort.tsv")
    ap.add_argument("--split", default="benchmarks/mfass/splits/split-v2.tsv")
    ap.add_argument("--out", default="benchmarks/mfass/results/baseline-kmer-position-v2.json")
    ap.add_argument("--capacity", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    t_all = time.perf_counter()
    with open(args.cohort, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if rows and not all("mutant_sequence" in r and "reference_sequence" in r for r in rows):
        raise SystemExit("Cohort predates assay-pair validation; rebuild with mfass-build first")
    with open(args.split, newline="") as fh:
        sp = {r["id"]: (r["group"], r["split"]) for r in csv.DictReader(fh, delimiter="\t")}
    if len(rows) != 27733 or len(sp) != len(rows) or {r["id"] for r in rows} != set(sp):
        raise SystemExit("MFASS-v2 cohort/split IDs do not reconcile")
    complement = str.maketrans("ACGT", "TGCA")
    for r in rows:
        pos = int(r["rel_position"]) - 1
        reference, mutant, legacy = (r[k].upper() for k in
                                     ("reference_sequence", "mutant_sequence", "sequence"))
        if not (len(reference) == len(mutant) == len(legacy) == 170 and 0 <= pos < 170):
            raise SystemExit(f"{r['id']}: invalid assay pair length or position")
        if [j for j, (a, b) in enumerate(zip(reference, mutant)) if a != b] != [pos]:
            raise SystemExit(f"{r['id']}: assay mutation is not at rel_position - 1")
        strand = r["strand"]
        if strand not in ("+", "-"):
            raise SystemExit(f"{r['id']}: unsupported strand")
        ref, alt = r["ref_allele"].upper(), r["alt_allele"].upper()
        if strand == "-":
            ref, alt = ref.translate(complement), alt.translate(complement)
        if reference[pos] != ref or mutant[pos] != alt:
            raise SystemExit(f"{r['id']}: assay pair alleles do not match source variant")
        orientation = ("assay" if legacy == mutant else "reverse_complement" if
                       legacy == mutant.translate(complement)[::-1] else None)
        if orientation is None or orientation != r["legacy_sequence_orientation"]:
            raise SystemExit(f"{r['id']}: legacy orientation does not reconcile")

    train = [r for r in rows if sp[r["id"]][1] == "train"]
    test = [r for r in rows if sp[r["id"]][1] == "test"]
    if (len(train) != 19409 or len(test) != 8324 or
            sum(int(r["sdv"]) for r in train) != 735 or
            sum(int(r["sdv"]) for r in test) != 315):
        raise SystemExit("Canonical MFASS-v2 split counts changed")
    cohort_sha = hashlib.sha256(pathlib.Path(args.cohort).read_bytes()).hexdigest()
    split_sha = hashlib.sha256(pathlib.Path(args.split).read_bytes()).hexdigest()
    t_validate = time.perf_counter() - t_all

    t0 = time.perf_counter()
    Xtr, Xte = featurise(train), featurise(test)
    ytr = np.array([int(r["sdv"]) for r in train])
    yte = np.array([int(r["sdv"]) for r in test])
    t_feat = time.perf_counter() - t0

    t0 = time.perf_counter()
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=args.seed,
    )
    clf.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = clf.predict_proba(Xte)[:, 1]
    t_pred = time.perf_counter() - t0

    groups_test = [sp[r["id"]][0] for r in test]
    result = BenchmarkResult(
        benchmark="mfass-v2",
        method="baseline-kmer-position",
        family="trivial baseline",
        description=("exon-boundary distances, allele identity, phyloP and phastCons "
                     "conservation, and 3-mer composition of a 21bp assay-oriented window, "
                     "HistGradientBoosting"),
        split=args.split,
        metrics=M.point_metrics(yte, scores, args.capacity),
        coverage={"scored": len(test), "unscored": 0, "denominator": len(test)},
        timing_seconds={
            "load_and_validate_cohort_and_split": round(t_validate, 3),
            "featurise_train_and_test": round(t_feat, 3),
            "fit": round(t_fit, 3),
            "predict_test": round(t_pred, 3),
            "end_to_end": round(time.perf_counter() - t_all, 3),
            "per_variant_total": round((t_validate + t_feat + t_fit + t_pred) / len(rows), 6),
        },
        independent_groups=len(set(groups_test)),
        pretrained=False,
        config={"trained_on_variants": len(train), "trained_on_positives": int(ytr.sum()),
                "seed": args.seed, "kmer_k": 3, "window": 21,
                "features": "exon-boundary distances, allele identity, phyloP, phastCons, 3-mers",
                "sequence_orientation": "validated assay-oriented original_seq at rel_position - 1",
                "cohort_sha256": cohort_sha, "split_sha256": split_sha},
        notes=("Corrected from mfass-v1: the old baseline centred windows using raw sequence, "
               "which was reverse-complemented for 7,770 variants; v1 artifacts are preserved."),
    )
    out = write_result(result, args.out)
    np.save(out.with_suffix(".scores.npy"), scores)
    with open(out.with_suffix(".predictions.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "group", "label", "score"])
        for r, s in zip(test, scores):
            w.writerow([r["id"], sp[r["id"]][0], r["sdv"], f"{s:.10f}"])
    print(json.dumps(json.loads(out.read_text()), indent=2))


if __name__ == "__main__":
    main()
