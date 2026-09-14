"""Score the MFASS test split with Pangolin.

Uses the official 12-model ensemble via `pangolin.pangolin.process_variant`, the
same path the distributed CLI takes.

The masking default is the point of interest. Pangolin's `-m/--mask` defaults to
"True", zeroing splice gains at annotated sites and splice losses at unannotated
sites. SpliceAI's `-M` defaults to 0, meaning unmasked. Two tools whose defaults
disagree about what a score means will disagree about a variant for reasons that
have nothing to do with their networks, so this runner takes an explicit --mask and
the benchmark reports both settings.
"""
import argparse
import csv
import json
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from rewirebench import metrics as M  # noqa: E402
from rewirebench.results import BenchmarkResult, write_result  # noqa: E402

MASK_DEFAULT = "True"     # Pangolin's own default. SpliceAI's equivalent is unmasked.
DISTANCE_DEFAULT = 50     # Matches SpliceAI's -D default. Not the model's input context.


class _Args:
    """Stand-in for the argparse namespace process_variant expects."""

    def __init__(self, reference_file, distance, mask):
        self.reference_file = reference_file
        self.distance = distance
        self.mask = mask
        self.score_cutoff = None
        self.score_exons = "False"


def _load_models():
    import torch
    from pkg_resources import resource_filename
    from pangolin.model import Pangolin

    L, W, AR = 32, np.array([11] * 5), np.array([1] * 5)
    # Widths and dilations follow the package's own main(); see pangolin.pangolin.
    from pangolin.pangolin import L as _L, W as _W, AR as _AR  # noqa: F401
    L, W, AR = _L, _W, _AR

    models = []
    for i in [0, 2, 4, 6]:
        for j in range(1, 4):
            m = Pangolin(L, W, AR)
            w = torch.load(
                resource_filename("pangolin", "models/final.%s.%s.3.v2" % (j, i)),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            m.load_state_dict(w)
            m.eval()
            models.append(m)
    return models


def _max_abs_score(raw):
    """Pangolin returns 'gene|pos:gain|pos:loss|Warnings:...' per gene, comma joined."""
    best = 0.0
    if not raw or raw == -1:
        return None
    for chunk in raw.split(","):
        for field in chunk.split("|"):
            if ":" not in field or field.startswith("Warnings"):
                continue
            try:
                best = max(best, abs(float(field.split(":", 1)[1])))
            except ValueError:
                continue
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="benchmarks/mfass/data/cohort.tsv")
    ap.add_argument("--split", default="benchmarks/mfass/splits/split-v2.tsv")
    ap.add_argument("--ref", default="benchmarks/mfass/data/ref/GRCh38.primary_assembly.genome.fa")
    ap.add_argument("--db", default="benchmarks/mfass/data/ref/gencode.v44.annotation.db")
    ap.add_argument("--distance", type=int, default=DISTANCE_DEFAULT)
    ap.add_argument("--mask", default=MASK_DEFAULT, choices=["True", "False"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_path = args.out or f"benchmarks/mfass/results/pangolin-mask{args.mask}.json"

    import os

    import gffutils
    import torch
    import pangolin.pangolin as pp
    from pangolin.pangolin import process_variant

    # Performance only, no effect on numerics: process_variant reopens the FASTA on
    # every call, and torch defaults to a subset of cores.
    torch.set_num_threads(os.cpu_count() or 4)
    _fasta_cache = {}
    _orig_fasta = pp.pyfastx.Fasta

    def _cached_fasta(path, *a, **k):
        if path not in _fasta_cache:
            _fasta_cache[path] = _orig_fasta(path, *a, **k)
        return _fasta_cache[path]

    pp.pyfastx.Fasta = _cached_fasta

    with open(args.cohort, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    with open(args.split, newline="") as fh:
        sp = {r["id"]: (r["group"], r["split"]) for r in csv.DictReader(fh, delimiter="\t")}

    test = [r for r in rows if sp[r["id"]][1] == "test"]
    if args.limit:
        test = test[: args.limit]

    t0 = time.perf_counter()
    gtf = gffutils.FeatureDB(args.db)
    models = _load_models()
    pargs = _Args(args.ref, args.distance, args.mask)
    t_load = time.perf_counter() - t0

    scores, unscored = [], []
    t0 = time.perf_counter()
    for i, r in enumerate(test):
        pos = r["snp_position_hg38_1based"]
        if pos in ("NA", "", None):
            scores.append(np.nan)
            unscored.append((r["id"], "no hg38 coordinate"))
            continue
        try:
            raw = process_variant(i, r["chr"], int(pos), r["ref_allele"], r["alt_allele"],
                                  gtf, models, pargs)
        except Exception as exc:  # noqa: BLE001
            scores.append(np.nan)
            unscored.append((r["id"], f"{type(exc).__name__}: {exc}"))
            continue
        s = _max_abs_score(raw)
        if s is None:
            scores.append(np.nan)
            unscored.append((r["id"], "skipped by Pangolin (no gene, ref mismatch or unsupported)"))
        else:
            scores.append(s)
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(test)}  {(time.perf_counter()-t0)/(i+1):.3f}s/variant", flush=True)
    t_score = time.perf_counter() - t0

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray([int(r["sdv"]) for r in test])
    groups = np.asarray([sp[r["id"]][0] for r in test])
    ok = ~np.isnan(scores)

    m = M.point_metrics(labels[ok], scores[ok], capacity=100)
    m["scored_subset_note"] = "metrics computed on scored variants only; see coverage"

    result = BenchmarkResult(
        benchmark="mfass-v1",
        method=f"pangolin-mask{args.mask}",
        family="specialist",
        description=("Pangolin official 12-model ensemble, max absolute predicted change in "
                     f"splice site usage, mask={args.mask}"),
        split=args.split,
        metrics=m,
        coverage={"scored": int(ok.sum()), "unscored": int((~ok).sum()), "denominator": len(test)},
        timing_seconds={
            "load_models_and_annotation": round(t_load, 3),
            "score_test": round(t_score, 3),
            "per_variant_total": round((t_load + t_score) / max(len(test), 1), 6),
        },
        independent_groups=int(len(set(groups[ok]))),
        pretrained=True,
        contamination=(
            "Pangolin was trained on splice site usage across GTEx tissues and four species, not on "
            "MFASS assay outcomes, so the labels are independent of its training signal. Whether any "
            "assayed exon appeared in its training annotation is unchecked."
        ),
        config={
            "models": "official 12-model ensemble (final.{1,2,3}.{0,2,4,6}.3.v2)",
            "annotation": pathlib.Path(args.db).name,
            "annotation_release": "GENCODE v44",
            "reference": pathlib.Path(args.ref).name,
            "distance_d": args.distance,
            "mask_m": args.mask,
            "score": "max absolute predicted change in splice site usage over reported sites",
            "context_bases": 10000,
            "torch_threads": None,  # filled below
            "patches": ["cached the per-call pyfastx.Fasta handle and raised torch thread "
                        "count; performance only, no effect on scores"],
        },
        notes=(
            "mask=True is Pangolin's own default and zeroes splice gains at annotated sites and "
            "losses at unannotated sites. mask=False matches SpliceAI's unmasked default. Both are "
            "run so the effect of the differing defaults is measured rather than assumed."
        ),
    )
    import os as _os
    result.config["torch_threads"] = _os.cpu_count()
    out = write_result(result, out_path)
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
