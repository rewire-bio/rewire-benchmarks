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
import hashlib
import json
import logging
import os
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from rewirebench.results import BenchmarkResult

from mfass.checkpoint import Checkpoint
from mfass.specialist_provenance import (
    file_sha256,
    pinned_file_check,
    resolve_spliceai_annotation,
    specialist_artifacts,
)
from mfass.specialist_run import refuse_finished, score_all, timing_scope, write_outputs

# SpliceAI's documented defaults, stated rather than inherited.
MASK_DEFAULT = 0        # -M in the CLI: unmasked. Pangolin's equivalent defaults to True.
DISTANCE_DEFAULT = 50   # -D in the CLI: how far to look for a gained or lost site.
                        # Not the model's 10kb input context.

UPSTREAM_REPOSITORY = "https://github.com/Illumina/SpliceAI"
UPSTREAM_REVISION = "b3c7f17b4137cb32b30c56c064f8b90b9b8f38d0"
# spliceai/utils.py at the reviewed v1.3.1 revision (audit S03-S05).
UPSTREAM_UTILS_SHA256 = "207e147c73bdc1970fdb176dcbf088883f9a58756bfeb0d93494e75018ae477c"
# Installed bytes that must equal upstream at UPSTREAM_REVISION: (SHA-256, git blob ID).
# Blob IDs are from the GitHub tree API for that commit.
UPSTREAM_FILES = {
    "__init__.py": ("5e0ade84087ee01040dbc3344359cbfdb26e2fcf98c214e15110ac1d05562037",
                    "b5d843e672cfff52c6d050205dead13d7c31533e"),
    "utils.py": (UPSTREAM_UTILS_SHA256, "66f2c3c14198d9020a7913f0412fc91ade008b91"),
    "__main__.py": ("1090e97a5f7d8f937048b7d1f08d98fb593bd86953e9d64178aa3ca096961a77",
                    "2c59acd61ce528dd0ed6d6c14244579354fed146"),
    "models/spliceai1.h5": ("e1fd5adcef7489d604b10e79c40078ef790d51ef048c4ce3869c9119ac5de42b",
                            "e5e6bf5d2f61f9f9f8f1c93386881992a96a82c0"),
    "models/spliceai2.h5": ("6ab042b82ab966b6d3582cb31b96f0859ea08a864f168d69e83aa14450a3b66e",
                            "b6c197b52fc39e81f42a2dfb25eeef11db3c94af"),
    "models/spliceai3.h5": ("e2e790bde53dfdf410c6dc434a86122a7d12f3f38dc2ef45d85986e9ecf22fad",
                            "5009694c892a782faa70f66e23f0f74cf69c6809"),
    "models/spliceai4.h5": ("ca88ac9e58e69ba6fdeed319b72f063f164c9abf7392eaccef903e94c1d99dd6",
                            "81ce1c887855f34abf9af5f5c65d85fd04d3132d"),
    "models/spliceai5.h5": ("791cd22c62a80a08d2ca674615a93ce8159d7b55bd157cfef2983b1bd6b41391",
                            "752679198e4f52f69b93945b2c49e4efc8534860"),
}
# Present in the PyPI 1.3.1 wheel but not in the upstream tree; neither is imported by
# utils.py. Pinned by SHA-256 so any change is still detected.
WHEEL_ONLY_FILES = {
    "__clean_main__.py": "d91e273a60b1b1c079cd29974f1c9901d997bc6f57226ff094ad15d0c1e49b03",
    "normalise_chrom.py": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
}


def installed_identity():
    """Installed SpliceAI bytes compared with the pinned upstream files, without importing it."""
    import importlib.metadata
    import importlib.util

    spec = importlib.util.find_spec("spliceai")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("spliceai is not installed")
    root = pathlib.Path(next(iter(spec.submodule_search_locations)))
    observed, mismatches = pinned_file_check(root, UPSTREAM_FILES)
    wheel_only = {name: file_sha256(root / name) if (root / name).is_file() else None
                  for name in WHEEL_ONLY_FILES}
    mismatches += [name for name, digest in wheel_only.items()
                   if digest != WHEEL_ONLY_FILES[name]]
    try:
        version = importlib.metadata.version("spliceai")
    except importlib.metadata.PackageNotFoundError:
        version = "unreported"
    return {"spliceai_version": version, "spliceai_files": observed,
            "spliceai_wheel_only_files_sha256": wheel_only,
            "spliceai_source_identity": ("upstream-b3c7f17" if observed["utils.py"] and
                                         observed["utils.py"]["sha256"] == UPSTREAM_UTILS_SHA256
                                         else "unrecognised"),
            "spliceai_upstream_files_verified": not mismatches,
            "spliceai_upstream_file_mismatches": mismatches}


class _Warnings(logging.Handler):
    """Collect the warnings get_delta_scores logs when it skips a record."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _score_variant(get_delta_scores, row, ann, distance, mask):
    """Return (score or None, unscored reason, upstream output) for one variant."""
    pos = row["snp_position_hg38_1based"]
    if pos in ("NA", "", None):
        return None, "no hg38 coordinate", ""
    rec = _Record(row["chr"], int(pos), row["ref_allele"], [row["alt_allele"]])
    captured = _Warnings()
    logging.getLogger().addHandler(captured)
    try:
        # Unexpected exceptions propagate: they stop the run without a checkpoint row.
        ds = get_delta_scores(rec, ann, distance, mask)
    finally:
        logging.getLogger().removeHandler(captured)
    if not ds:
        # With no warning, SpliceAI found no annotated gene; otherwise it names the skip.
        reason = captured.messages[0] if captured.messages else ""
        return None, (f"skipped by SpliceAI: {reason}" if reason else
                      "no annotated gene overlapping the variant"), ""
    # Delta score fields 2..5 are DS_AG, DS_AL, DS_DG, DS_DL. The maximum is the
    # tool's own headline number and the one a laboratory ranks on.
    best = 0.0
    for entry in ds:
        values = [float(x) for x in entry.split("|")[2:6]]
        if len(values) != 4 or not np.isfinite(values).all():
            return None, "nonfinite model score", ",".join(ds)
        best = max(best, max(values))
    return best, "", ",".join(ds)


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
    ap.add_argument("--annotation-release", help="Declared release label; not source verification")
    ap.add_argument("--distance", type=int, default=DISTANCE_DEFAULT)
    ap.add_argument("--mask", type=int, choices=(0, 1), default=MASK_DEFAULT)
    ap.add_argument("--out", default="benchmarks/mfass/results/spliceai-1.3.1.json")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N, for smoke tests")
    ap.add_argument("--threads", type=int, default=None,
                    help="TensorFlow intra-op threads; default is TensorFlow's")
    ap.add_argument("--inter-threads", type=int, default=None,
                    help="TensorFlow inter-op threads; default is TensorFlow's")
    ap.add_argument("--checkpoint", default=None,
                    help="resumable per-variant log; resumes only with identical settings")
    ap.add_argument("--require-verified-code", action="store_true",
                    help="refuse unless installed code and weights equal the pinned upstream bytes")
    args = ap.parse_args()
    if args.limit < 0 or args.distance < 0:
        ap.error("limit and distance must be nonnegative")
    if any(n is not None and n < 1 for n in (args.threads, args.inter_threads)):
        ap.error("thread counts must be positive")
    from rewirebench.protocols.mfass import prepare as prepare_protocol
    from rewirebench.protocols.mfass import score as score_protocol
    dataset = prepare_protocol(pathlib.Path(args.cohort), split=args.split,
                               **({"limit": args.limit} if args.limit else {}))
    refuse_finished(args.out)

    t0 = time.perf_counter()
    code = installed_identity()
    if args.require_verified_code and not code["spliceai_upstream_files_verified"]:
        raise SystemExit("Installed SpliceAI differs from the pinned upstream bytes: "
                         f"{code['spliceai_upstream_file_mismatches']}")
    t_code = time.perf_counter() - t0

    t0 = time.perf_counter()
    annotation_path = resolve_spliceai_annotation(args.annotation)
    artifacts = specialist_artifacts(annotation_path, args.ref, args.annotation_release)
    t_hash = time.perf_counter() - t0

    import tensorflow as tf
    # Set before any TensorFlow operation. Thread counts can change floating-point
    # reduction order, so they are recorded and fingerprinted.
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.inter_threads:
        tf.config.threading.set_inter_op_parallelism_threads(args.inter_threads)
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
    ann = Annotator(args.ref, str(annotation_path))
    t_load = time.perf_counter() - t0

    threads = {"intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
               "inter_op": tf.config.threading.get_inter_op_parallelism_threads(),
               "scope": ("TensorFlow intra-op and inter-op pools as set; 0 means TensorFlow's "
                         "default. Other runtime threads (I/O, Python) are not counted.")}
    runner_sha256 = file_sha256(__file__)
    settings = {
        "runner": "mfass-spliceai", "runner_sha256": runner_sha256,
        "cohort_sha256": dataset["provenance"]["cohort_sha256"],
        "split_sha256": dataset["provenance"]["split_sha256"],
        "selected_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in test).encode()).hexdigest(),
        "mask": args.mask, "distance": args.distance, "tensorflow_threads": threads,
        "tensorflow_version": tf.__version__,
        "annotation_sha256": artifacts["annotation_sha256"],
        "reference_sha256": artifacts["reference_sha256"], **code,
    }
    log = Checkpoint(args.checkpoint, settings) if args.checkpoint else None
    try:
        scores, unscored, t_score = score_all(
            test, lambda i, r: _score_variant(get_delta_scores, r, ann, args.distance, args.mask),
            log, progress_every=500)
    finally:
        if log:
            log.close()
    ok = np.isfinite(scores)

    # A variant SpliceAI cannot score is a coverage gap, not a negative prediction.
    # Metrics are computed on the scored subset; coverage reports against the whole.
    groups = np.asarray([sp[r["id"]][0] for r in test])
    scored = score_protocol(dataset, {r["id"]: float(s) for r, s in zip(test, scores)
                                      if np.isfinite(s)})
    m = scored["metrics"]
    m["scored_subset_note"] = "metrics computed on scored variants only; see coverage"

    result = BenchmarkResult(
        benchmark="mfass-v2-smoke" if args.limit else "mfass-v2",
        method="spliceai-1.3.1",
        family="specialist",
        description="SpliceAI 1.3.1 official five-model ensemble, max delta score over AG/AL/DG/DL",
        split=args.split,
        metrics=m,
        coverage={key: scored["coverage"][key] for key in ("scored", "unscored", "denominator")},
        timing_seconds={
            "verify_code_and_weights": round(t_code, 3),
            "hash_reference_and_annotation": round(t_hash, 3),
            "load_models_and_reference": round(t_load, 3),
            "score_test": round(t_score, 3),
            "per_variant_total": round((t_code + t_hash + t_load + t_score) / max(len(test), 1), 6),
        },
        independent_groups=len(set(groups[ok])),
        pretrained=True,
        contamination=(
            "SpliceAI was trained on GENCODE transcripts on the reference genome, not on MFASS "
            "assay outcomes, so the labels are independent of its training signal. Whether any "
            "assayed exon appeared in its training transcripts is unchecked."
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
            "version": "1.3.1",
            "models": "bundled 5-model ensemble (spliceai1-5.h5)",
            # A custom path is recorded by file name, not local directory.
            "annotation": (args.annotation if args.annotation in ("grch37", "grch38")
                           else annotation_path.name),
            **artifacts,
            **code,
            "runner_sha256": runner_sha256,
            "tensorflow_version": settings["tensorflow_version"],
            "tensorflow_threads": threads,
            "selected_ids_sha256": settings["selected_ids_sha256"],
            "checkpoint": ({"file": log.path.name, "fingerprint": log.fingerprint,
                            "resumed": log.resumed,
                            "discarded_partial_line": log.discarded_partial_line,
                            "sha256": file_sha256(log.path)} if log else None),
            "reference": pathlib.Path(args.ref).name,
            "distance_D": args.distance,
            "mask_M": args.mask,
            "score": "max(DS_AG, DS_AL, DS_DG, DS_DL)",
            "context_bases": 10000,
            "patches": [("one_hot_encode: np.fromstring replaced by np.frombuffer on the same "
                         "bytes, always applied; identical dtype, values and length")],
        },
        notes=(
            "MFASS measures exon recognition in a minigene construct; SpliceAI scores the variant "
            "in genomic context. Both predict splice disruption for the same variant, but they are "
            "not measuring the same molecule."
        ),
    )
    out = write_outputs(result, args.out, test, {r["id"]: sp[r["id"]][0] for r in test},
                        scores, unscored)
    print(json.dumps(json.loads(out.read_text())["metrics"], indent=2))
    print(f"coverage: {int(ok.sum())}/{len(test)} scored", flush=True)


if __name__ == "__main__":
    main()
