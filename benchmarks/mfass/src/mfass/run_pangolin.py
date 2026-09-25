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

Upstream Pangolin 5cf94b8 applies each gene's mask to arrays shared by every
gene on the same strand, so masked scores depend on gene order. mask=True is
refused unless the installed source is the reviewed per-gene patch and model.py
and all 12 ensemble weights equal the pinned upstream bytes
(`mfass.pangolin_patch`). Their hashes are recorded in every result.

Expected skips (no coordinate, Pangolin's own -1 skip, nonfinite scores) become
unscored rows. Any other error stops the run; see `mfass.specialist_run`.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from rewirebench.results import BenchmarkResult

from mfass.checkpoint import Checkpoint
from mfass.pangolin_patch import PATCH_ID, installed_identity, require_reviewed
from mfass.specialist_provenance import file_sha256, specialist_artifacts
from mfass.specialist_run import refuse_finished, score_all, timing_scope, write_outputs

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


def _reported_values(raw):
    """Scores in 'gene|pos:gain|pos:loss|Warnings:...' chunks, comma joined."""
    values = []
    if not raw or raw == -1:
        return values
    for chunk in raw.split(","):
        for field in chunk.split("|"):
            if ":" not in field or field.startswith("Warnings"):
                continue
            try:
                values.append(float(field.split(":", 1)[1]))
            except ValueError:
                continue
    return values


def _max_abs_score(raw):
    """Maximum absolute reported change, or None if absent or any value is nonfinite."""
    values = _reported_values(raw)
    if not values or not np.isfinite(values).all():
        return None
    return max(abs(v) for v in values)


SKIPPED = "skipped by Pangolin (no gene, ref mismatch or unsupported)"


def _score_variant(process_variant, i, row, gtf, models, pargs):
    """Return (score or None, unscored reason, upstream output) for one variant."""
    pos = row["snp_position_hg38_1based"]
    if pos in ("NA", "", None):
        return None, "no hg38 coordinate", ""
    printed = io.StringIO()
    # process_variant reports why it skipped a variant only by printing. Unexpected
    # exceptions propagate and stop the run without a checkpoint row.
    with contextlib.redirect_stdout(printed):
        raw = process_variant(i, row["chr"], int(pos), row["ref_allele"], row["alt_allele"],
                              gtf, models, pargs)
    if raw == -1:
        detail = " ".join(printed.getvalue().split())
        return None, f"{SKIPPED}: {detail}" if detail else SKIPPED, ""
    values = _reported_values(raw)
    if not values:
        raise RuntimeError(f"{row['id']}: Pangolin output has no scores: {raw!r}")
    if not np.isfinite(values).all():
        return None, "nonfinite model score", str(raw)
    return _max_abs_score(raw), "", raw


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
    ap.add_argument("--threads", type=int, default=None,
                    help="torch intra-op threads; default is every core")
    ap.add_argument("--interop-threads", type=int, default=None,
                    help="torch inter-op threads; default is torch's")
    ap.add_argument("--checkpoint", default=None,
                    help="resumable per-variant log; resumes only with identical settings")
    ap.add_argument("--require-verified-code", action="store_true",
                    help="refuse unless installed code and weights are the reviewed bytes, "
                         "including the masking patch, whatever the mask setting")
    args = ap.parse_args()
    if args.limit < 0 or args.distance < 0:
        ap.error("limit and distance must be nonnegative")
    if any(n is not None and n < 1 for n in (args.threads, args.interop_threads)):
        ap.error("thread counts must be positive")
    from rewirebench.protocols.mfass import prepare as prepare_protocol
    from rewirebench.protocols.mfass import score as score_protocol
    dataset = prepare_protocol(pathlib.Path(args.cohort), split=args.split,
                               **({"limit": args.limit} if args.limit else {}))
    out_path = args.out or f"benchmarks/mfass/results/pangolin-mask{args.mask}.json"
    refuse_finished(out_path)

    t0 = time.perf_counter()
    code = installed_identity()
    try:
        if args.mask == "True" or args.require_verified_code:
            # Upstream masks shared arrays in place, so masked scores depend on gene order.
            require_reviewed(code)
    except ValueError as error:
        raise SystemExit(f"Refusing to run: {error}. Install the reviewed patch {PATCH_ID} "
                         "(see mfass.pangolin_patch).") from None
    t_code = time.perf_counter() - t0

    t0 = time.perf_counter()
    artifacts = specialist_artifacts(args.db, args.ref, args.annotation_release)
    t_hash = time.perf_counter() - t0

    import os

    import gffutils
    import pangolin.pangolin as pp
    import torch
    from pangolin.pangolin import process_variant

    # Performance only: process_variant reopens the FASTA on every call, and torch
    # defaults to a subset of cores. Thread counts can change floating-point
    # reduction order, so they are recorded and must match within a comparison.
    # The inter-op pool must be sized before any parallel work.
    if args.interop_threads:
        torch.set_num_interop_threads(args.interop_threads)
    torch.set_num_threads(args.threads or os.cpu_count() or 4)
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

    threads = {"intra_op": torch.get_num_threads(),
               "inter_op": torch.get_num_interop_threads(),
               "scope": ("torch intra-op and inter-op pools as set. Other runtime threads "
                         "(I/O, Python) are not counted.")}
    runner_sha256 = file_sha256(__file__)
    settings = {
        "runner": "mfass-pangolin", "runner_sha256": runner_sha256,
        "cohort_sha256": dataset["provenance"]["cohort_sha256"],
        "split_sha256": dataset["provenance"]["split_sha256"],
        "selected_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in test).encode()).hexdigest(),
        "mask": args.mask, "distance": args.distance, "torch_threads": threads,
        "torch_version": getattr(torch, "__version__", "unreported"),
        "annotation_sha256": artifacts["annotation_sha256"],
        "reference_sha256": artifacts["reference_sha256"], **code,
    }
    log = Checkpoint(args.checkpoint, settings) if args.checkpoint else None
    try:
        scores, unscored, t_score = score_all(
            test, lambda i, r: _score_variant(process_variant, i, r, gtf, models, pargs), log)
    finally:
        if log:
            log.close()
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
            "verify_code_and_weights": round(t_code, 3),
            "hash_reference_and_annotation": round(t_hash, 3),
            "load_models_and_annotation": round(t_load, 3),
            "score_test": round(t_score, 3),
            "per_variant_total": round((t_code + t_hash + t_load + t_score) / max(len(test), 1), 6),
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
            "timing_scope": timing_scope(log),
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
            **code,
            "runner_sha256": runner_sha256,
            "torch_version": settings["torch_version"],
            "torch_threads": threads,
            "selected_ids_sha256": settings["selected_ids_sha256"],
            "patches": [("cached the per-call pyfastx.Fasta handle and set the torch thread "
                         "count; performance only")] +
                       ([(f"{PATCH_ID}: per-gene copy of strand score arrays before masking "
                          "(installed source; no change when mask=False)")]
                        if code["pangolin_source_identity"] == PATCH_ID else []),
            "checkpoint": ({"file": log.path.name, "fingerprint": log.fingerprint,
                            "resumed": log.resumed,
                            "discarded_partial_line": log.discarded_partial_line,
                            "sha256": file_sha256(log.path)} if log else None),
        },
        notes=(
            f"This run used mask={args.mask}. Pangolin's mask=True setting zeroes splice gains "
            "at annotated sites and losses at unannotated sites; mask=False disables that "
            "masking. This result does not measure the effect of the other setting."
        ),
    )
    out = write_outputs(result, out_path, test, {r["id"]: sp[r["id"]][0] for r in test},
                        scores, unscored)
    print(json.dumps(json.loads(out.read_text())["metrics"], indent=2))
    print(f"coverage: {int(ok.sum())}/{len(test)} scored", flush=True)


if __name__ == "__main__":
    main()
