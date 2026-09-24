"""Score the MFASS test split with Pangolin.

Uses the official 12-model ensemble via `pangolin.pangolin.process_variant`, the
same path the distributed CLI takes.

The masking default is the point of interest. Pangolin's `-m/--mask` defaults to
"True", zeroing splice gains at annotated sites and splice losses at unannotated
sites. SpliceAI's `-M` defaults to 0, meaning unmasked. Two tools whose defaults
disagree about what a score means will disagree about a variant for reasons that
have nothing to do with their networks, so this runner takes an explicit --mask and
records only the selected setting. The archived evaluation used mask=False;
the mask=True comparison remains a proposed sensitivity analysis.
"""
import argparse
import csv
import json
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from rewirebench.results import BenchmarkResult, write_result

from mfass.specialist_provenance import specialist_artifacts

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
    from pangolin.model import Pangolin
    from pkg_resources import resource_filename

    L, W, AR = 32, np.array([11] * 5), np.array([1] * 5)
    # Widths and dilations follow the package's own main(); see pangolin.pangolin.
    from pangolin.pangolin import AR as _AR
    from pangolin.pangolin import L as _L
    from pangolin.pangolin import W as _W
    L, W, AR = _L, _W, _AR

    models = []
    for i in [0, 2, 4, 6]:
        for j in range(1, 4):
            m = Pangolin(L, W, AR)
            w = torch.load(
                resource_filename("pangolin", f"models/final.{j}.{i}.3.v2"),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            m.load_state_dict(w)
            m.eval()
            models.append(m)
    return models


def _max_abs_score(raw):
    """Pangolin returns 'gene|pos:gain|pos:loss|Warnings:...' per gene, comma joined."""
    best = None
    if not raw or raw == -1:
        return None
    for chunk in raw.split(","):
        for field in chunk.split("|"):
            if ":" not in field or field.startswith("Warnings"):
                continue
            try:
                value = abs(float(field.split(":", 1)[1]))
                if not np.isfinite(value):
                    return None
                best = value if best is None else max(best, value)
            except ValueError:
                continue
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="benchmarks/mfass/data/cohort.tsv")
    ap.add_argument("--split", default="benchmarks/mfass/splits/split-v2.tsv")
    ap.add_argument("--ref", default="benchmarks/mfass/data/ref/GRCh38.primary_assembly.genome.fa")
    ap.add_argument("--db", default="benchmarks/mfass/data/ref/gencode.v44.annotation.db")
    ap.add_argument("--annotation-release", help="Declared release label; not source verification")
    ap.add_argument("--distance", type=int, default=DISTANCE_DEFAULT)
    ap.add_argument("--mask", default=MASK_DEFAULT, choices=["True", "False"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.limit < 0 or args.distance < 0:
        ap.error("limit and distance must be nonnegative")
    from rewirebench.protocols.mfass import prepare as prepare_protocol
    from rewirebench.protocols.mfass import score as score_protocol
    dataset = prepare_protocol(pathlib.Path(args.cohort), split=args.split,
                               **({"limit": args.limit} if args.limit else {}))
    destination = pathlib.Path(args.out or f"benchmarks/mfass/results/pangolin-mask{args.mask}.json")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    for suffix in (".json", ".predictions.tsv", ".unscored.tsv"):
        if destination.with_suffix(suffix).exists():
            raise FileExistsError(f"Refusing to overwrite {destination.with_suffix(suffix)}")

    out_path = args.out or f"benchmarks/mfass/results/pangolin-mask{args.mask}.json"

    t0 = time.perf_counter()
    artifacts = specialist_artifacts(args.db, args.ref, args.annotation_release)
    t_hash = time.perf_counter() - t0

    import os

    import gffutils
    import pangolin.pangolin as pp
    import torch
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
    groups = np.asarray([sp[r["id"]][0] for r in test])
    ok = np.isfinite(scores)

    scored = score_protocol(dataset, {r["id"]: float(s) for r, s in zip(test, scores)
                                      if np.isfinite(s)})
    m = scored["metrics"]
    m["scored_subset_note"] = "metrics computed on scored variants only; see coverage"

    result = BenchmarkResult(
        benchmark="mfass-v2-smoke" if args.limit else "mfass-v2",
        method=f"pangolin-mask{args.mask}",
        family="specialist",
        description=("Pangolin official 12-model ensemble, max absolute predicted change in "
                     f"splice site usage, mask={args.mask}"),
        split=args.split,
        metrics=m,
        coverage={key: scored["coverage"][key] for key in ("scored", "unscored", "denominator")},
        timing_seconds={
            "hash_reference_and_annotation": round(t_hash, 3),
            "load_models_and_annotation": round(t_load, 3),
            "score_test": round(t_score, 3),
            "per_variant_total": round((t_hash + t_load + t_score) / max(len(test), 1), 6),
        },
        independent_groups=len(set(groups[ok])),
        pretrained=True,
        contamination=(
            "Pangolin was trained on splice site usage across GTEx tissues and four species, not on "
            "MFASS assay outcomes, so the labels are independent of its training signal. Whether any "
            "assayed exon appeared in its training annotation is unchecked."
        ),
        config={
            "scope": dataset["scope"],
            "timing_scope": ("artifact hashing, model/reference load and prediction; "
                             "cohort preparation excluded"),
            "selected_test_rows": len(test),
            "canonical_test_rows": 8324,
            "cohort_sha256": dataset["provenance"]["cohort_sha256"],
            "split_sha256": dataset["provenance"]["split_sha256"],
            "not_selected_smoke_rows": 8324 - len(test),
            "input_context": "genomic GRCh38; differs from assay-pair encoder inputs",
            "models": "official 12-model ensemble (final.{1,2,3}.{0,2,4,6}.3.v2)",
            "annotation": pathlib.Path(args.db).name,
            **artifacts,
            "reference": pathlib.Path(args.ref).name,
            "distance_d": args.distance,
            "mask_m": args.mask,
            "score": "max absolute predicted change in splice site usage over reported sites",
            "context_bases": 10000,
            "torch_threads": None,  # filled below
            "patches": [("cached the per-call pyfastx.Fasta handle and raised torch thread "
                        "count; performance only, no effect on scores")],
        },
        notes=(
            f"This run used mask={args.mask}. Pangolin's mask=True setting zeroes splice gains "
            "at annotated sites and losses at unannotated sites; mask=False disables that "
            "masking. This result does not measure the effect of the other setting."
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
