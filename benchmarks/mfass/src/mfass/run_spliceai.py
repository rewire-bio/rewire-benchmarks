"""Score the MFASS test split with SpliceAI 1.3.1.

Uses the official five-model ensemble and the bundled GRCh38 annotation via
`spliceai.utils`, which is the same path the distributed CLI takes. Every setting
that moves the number is pinned explicitly, including the ones already sitting at
their defaults.

Scope note worth keeping in view: MFASS measured exon recognition in a minigene
construct, while SpliceAI scores the variant in its genomic context. That is how a
laboratory would actually use the tool, and it is the comparison the source paper
made, but the two are not measuring the same molecule.
"""
import argparse
import csv
import json
import os
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

from rewirebench import metrics as M  # noqa: E402
from rewirebench.results import BenchmarkResult, write_result  # noqa: E402

# SpliceAI's documented defaults, stated rather than inherited.
MASK_DEFAULT = 0        # -M in the CLI: unmasked. Pangolin's equivalent defaults to True.
DISTANCE_DEFAULT = 50   # -D in the CLI: how far to look for a gained or lost site.
                        # Not the model's 10kb input context.


def _patch_numpy_fromstring():
    """Restore one_hot_encode under modern NumPy.

    SpliceAI 1.3.1 calls np.fromstring(seq, np.int8), whose binary mode NumPy has
    removed. This is a minimal transcription of the package's own function with that
    single call swapped for np.frombuffer on the same bytes: identical map, identical
    character translation, identical dtype and length. Nothing about the model or its
    numerics changes, and the patch is recorded in the result.

    The translation to \x01-\x04 above the lookup is load-bearing. Dropping it makes
    every base index the all-zero row and every delta score come out as 0.0.
    """
    import numpy as _np
    import spliceai.utils as _u

    _map = _np.asarray([[0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    def one_hot_encode(seq):
        seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
        seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')
        return _map[_np.frombuffer(seq.encode('latin-1'), _np.int8) % 5]

    _u.one_hot_encode = one_hot_encode


class _Record:
    """Minimal stand-in for the pysam VCF record get_delta_scores expects."""

    def __init__(self, chrom, pos, ref, alts):
        self.chrom = chrom
        self.pos = pos
        self.ref = ref
        self.alts = alts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="benchmarks/mfass/data/cohort.tsv")
    ap.add_argument("--split", default="benchmarks/mfass/splits/split-v2.tsv")
    ap.add_argument("--ref", default="benchmarks/mfass/data/ref/GRCh38.primary_assembly.genome.fa")
    ap.add_argument("--annotation", default="grch38", help="'grch38' for the bundled table, or a path")
    ap.add_argument("--distance", type=int, default=DISTANCE_DEFAULT)
    ap.add_argument("--mask", type=int, default=MASK_DEFAULT)
    ap.add_argument("--out", default="benchmarks/mfass/results/spliceai-1.3.1.json")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N, for smoke tests")
    args = ap.parse_args()

    from spliceai.utils import Annotator, get_delta_scores
    _patch_numpy_fromstring()

    with open(args.cohort, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    with open(args.split, newline="") as fh:
        sp = {r["id"]: (r["group"], r["split"]) for r in csv.DictReader(fh, delimiter="\t")}

    test = [r for r in rows if sp[r["id"]][1] == "test"]
    if args.limit:
        test = test[: args.limit]

    t0 = time.perf_counter()
    ann = Annotator(args.ref, args.annotation)
    t_load = time.perf_counter() - t0

    scores, unscored = [], []
    t0 = time.perf_counter()
    for i, r in enumerate(test):
        pos = r["snp_position_hg38_1based"]
        if pos in ("NA", "", None):
            scores.append(np.nan)
            unscored.append((r["id"], "no hg38 coordinate"))
            continue
        rec = _Record(r["chr"], int(pos), r["ref_allele"], [r["alt_allele"]])
        try:
            ds = get_delta_scores(rec, ann, args.distance, args.mask)
        except Exception as exc:  # noqa: BLE001
            scores.append(np.nan)
            unscored.append((r["id"], f"{type(exc).__name__}: {exc}"))
            continue
        if not ds:
            scores.append(np.nan)
            unscored.append((r["id"], "no annotated gene overlapping the variant"))
            continue
        # Delta score fields 2..5 are DS_AG, DS_AL, DS_DG, DS_DL. The maximum is the
        # tool's own headline number and the one a laboratory ranks on.
        best = 0.0
        for entry in ds:
            parts = entry.split("|")
            best = max(best, max(float(x) for x in parts[2:6]))
        scores.append(best)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(test)}  {(time.perf_counter()-t0)/(i+1):.3f}s/variant", flush=True)
    t_score = time.perf_counter() - t0

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray([int(r["sdv"]) for r in test])
    ok = ~np.isnan(scores)

    # A variant SpliceAI cannot score is a coverage gap, not a negative prediction.
    # Metrics are computed on the scored subset; coverage reports against the whole.
    groups = np.asarray([sp[r["id"]][0] for r in test])
    m = M.point_metrics(labels[ok], scores[ok], capacity=100)
    m["scored_subset_note"] = "metrics computed on scored variants only; see coverage"

    result = BenchmarkResult(
        benchmark="mfass-v1",
        method="spliceai-1.3.1",
        family="specialist",
        description="SpliceAI 1.3.1 official five-model ensemble, max delta score over AG/AL/DG/DL",
        split=args.split,
        metrics=m,
        coverage={"scored": int(ok.sum()), "unscored": int((~ok).sum()), "denominator": len(test)},
        timing_seconds={
            "load_models_and_reference": round(t_load, 3),
            "score_test": round(t_score, 3),
            "per_variant_total": round((t_load + t_score) / max(len(test), 1), 6),
        },
        independent_groups=int(len(set(groups[ok]))),
        pretrained=True,
        contamination=(
            "SpliceAI was trained on GENCODE transcripts on the reference genome, not on MFASS "
            "assay outcomes, so the labels are independent of its training signal. Whether any "
            "assayed exon appeared in its training transcripts is unchecked."
        ),
        config={
            "version": "1.3.1",
            "models": "bundled 5-model ensemble (spliceai1-5.h5)",
            "annotation": args.annotation,
            "reference": pathlib.Path(args.ref).name,
            "distance_D": args.distance,
            "mask_M": args.mask,
            "score": "max(DS_AG, DS_AL, DS_DG, DS_DL)",
            "context_bases": 10000,
            "patches": ["one_hot_encode: np.fromstring -> np.frombuffer for NumPy 2 "
                        "compatibility; identical dtype, values and length"],
        },
        notes=(
            "MFASS measures exon recognition in a minigene construct; SpliceAI scores the variant "
            "in genomic context. Both predict splice disruption for the same variant, but they are "
            "not measuring the same molecule."
        ),
    )
    out = write_result(result, args.out)
    with open(out.with_suffix(".predictions.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "group", "label", "score"])
        for r, s in zip(test, scores):
            w.writerow([r["id"], sp[r["id"]][0], r["sdv"], "" if np.isnan(s) else f"{s:.4f}"])
    if unscored:
        with open(out.with_suffix(".unscored.tsv"), "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["id", "reason"])
            w.writerows(unscored)
    print(json.dumps(json.loads(out.read_text())["metrics"], indent=2))
    print(f"coverage: {int(ok.sum())}/{len(test)} scored", flush=True)


if __name__ == "__main__":
    main()
