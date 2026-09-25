"""Freeze, run, verify and report the annotation-matched S0/S1/P0/P1 specialist study.

This adds no scoring logic. `freeze` records every resource, code, weight and
environment hash, the fixed thread configuration and the exact runner and
comparison commands in one timestamped manifest. `run` re-verifies what a step
depends on, then executes it with the existing CLIs, logging to a per-step file.
`verify` checks that each finished result used the frozen artifacts, that its
prediction and unscored tables are the exact bytes it recorded, that every row
matches the canonical IDs, labels and groups, that its metrics and coverage
recompute from those rows, and that each masking pair differs only in its mask.
`all` runs the four conditions, verification, three contrasts and the report in
order, resuming finished or interrupted steps and stopping at the first failure.
Protocol: docs/mfass-specialist-comparison.md and its registration.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from rewirebench.protocols.mfass import COHORT_SHA256, SPLIT_SHA256

from mfass.specialist_provenance import file_sha256

CANONICAL_TEST_ROWS = 8324
CONDITIONS = {
    "S0": {"tool": "spliceai", "mask": "0", "stem": "spliceai-1.3.1-gencode44-canonical-mask0"},
    "S1": {"tool": "spliceai", "mask": "1", "stem": "spliceai-1.3.1-gencode44-canonical-mask1"},
    "P0": {"tool": "pangolin", "mask": "False", "stem": "pangolin-gencode44-canonical-maskFalse"},
    "P1": {"tool": "pangolin", "mask": "True", "stem": "pangolin-gencode44-canonical-maskTrue"},
}
# (name, candidate, baseline): compare reports candidate minus baseline.
CONTRASTS = (("S1-S0", "S1", "S0"), ("P1-P0", "P1", "P0"), ("P0-S0", "P0", "S0"))
SEQUENCE = (*CONDITIONS, "verify", *(c[0] for c in CONTRASTS), "report")
DISTANCE = 50
MIN_COMMON = 100
DRAWS, SEED, CAPACITY = 2000, 20260914, 100
METRIC_KEYS = ("n", "positives", "precision_at_capacity", "recall_at_capacity",
               "average_precision_sklearn", "auroc")
# Config keys expected to differ within a masking pair. Everything else must match.
PAIR_VARIABLE_KEYS = {"mask_M", "mask_m", "checkpoint", "output_tables",
                      "orphaned_tables_moved_aside"}
PATCHED_SPLICEAI_IDENTITY = "upstream-b3c7f17"

IDENTITY_SNIPPET = {
    "spliceai": ("import json; from mfass.run_spliceai import installed_identity; "
                 "print(json.dumps(installed_identity()))"),
    "pangolin": ("import json; from mfass.pangolin_patch import installed_identity; "
                 "print(json.dumps(installed_identity()))"),
}
DISTRIBUTIONS_SNIPPET = ("import json, importlib.metadata as m; print(json.dumps(sorted("
                         "f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions())))")


def _utc():
    return datetime.now(UTC).isoformat(timespec="seconds")


def _python_json(python, snippet):
    out = subprocess.run([str(python), "-c", snippet], check=True, capture_output=True,
                         text=True, env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"})
    return json.loads(out.stdout.strip().splitlines()[-1])


def _git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def _thread_env(count):
    return {key: str(count) for key in THREAD_ENV_KEYS}


def thread_plan(tf_intra, tf_inter, torch_intra, torch_inter, cpu_threads):
    """Fixed per-step thread settings; every step's compute pools fit the CPU budget."""
    plan = {
        "spliceai": {"intra_op": tf_intra, "inter_op": tf_inter, "env": _thread_env(tf_intra)},
        "pangolin": {"intra_op": torch_intra, "inter_op": torch_inter,
                     "env": _thread_env(torch_intra)},
        # Contrasts (numpy/scikit-learn bootstrap) and in-process verification and report.
        "other_steps": {"env": _thread_env(cpu_threads)},
        "cpu_threads": cpu_threads,
        "scope": ("One step runs at a time. TensorFlow intra-op plus inter-op pools are within "
                  "the CPU budget. Torch's intra-op pool is within the budget; its inter-op pool "
                  "is 1 and idle during sequential forward passes. Runtime I/O and Python threads "
                  "are not counted. Manifest hashing in the parent process is outside all "
                  "recorded timings."),
    }
    if min(tf_intra, tf_inter, torch_intra, torch_inter) < 1:
        raise SystemExit("thread counts must be positive")
    if tf_intra + tf_inter > cpu_threads or torch_intra > cpu_threads:
        raise SystemExit(f"thread plan exceeds the {cpu_threads}-thread budget")
    return plan


def _runner_argv(python, condition, paths, threads, out_dir):
    spec = CONDITIONS[condition]
    stem = out_dir / condition / spec["stem"]
    tool_threads = threads[spec["tool"]]
    common = ["--cohort", paths["cohort"], "--split", paths["split"], "--ref", paths["fasta"],
              "--annotation-release", paths["annotation_release"],
              "--distance", str(DISTANCE), "--mask", spec["mask"],
              "--threads", str(tool_threads["intra_op"]),
              "--out", f"{stem}.json", "--checkpoint", str(out_dir / condition / "checkpoint.tsv"),
              "--require-verified-code"]
    if spec["tool"] == "spliceai":
        return [str(python), "-m", "mfass.run_spliceai",
                "--annotation", paths["spliceai_annotation"],
                "--inter-threads", str(tool_threads["inter_op"]), *common]
    return [str(python), "-m", "mfass.run_pangolin", "--db", paths["pangolin_db"],
            "--interop-threads", str(tool_threads["inter_op"]), *common]


RUNNER_FILES = ["benchmarks/mfass/src/mfass/" + name for name in (
    "run_spliceai.py", "run_pangolin.py", "checkpoint.py", "compare.py", "specialist_run.py",
    "specialist_provenance.py", "pangolin_patch.py", "matched_annotation.py",
    "matched_study.py")] + ["packages/rewirebench/src/rewirebench/metrics.py",
                            "packages/rewirebench/src/rewirebench/results.py",
                            "packages/rewirebench/src/rewirebench/protocols/mfass.py",
                            "benchmarks/mfass/patches/pangolin-5cf94b8-mask-per-gene-1.patch",
                            "benchmarks/mfass/scripts/run_matched_study.sh",
                            "uv.lock"]


def check_identities(code):
    """Refuse unless both tools are the reviewed, pinned bytes."""
    from mfass.pangolin_patch import PATCH_ID, PATCHED_SOURCE_SHA256
    problems = []
    spliceai, pangolin = code["spliceai"], code["pangolin"]
    if spliceai.get("spliceai_source_identity") != PATCHED_SPLICEAI_IDENTITY:
        problems.append("SpliceAI source is not the reviewed v1.3.1 bytes")
    if not spliceai.get("spliceai_upstream_files_verified"):
        problems.append(f"SpliceAI files differ from upstream: "
                        f"{spliceai.get('spliceai_upstream_file_mismatches')}")
    if (pangolin.get("pangolin_source_identity") != PATCH_ID or
            pangolin.get("pangolin_source_sha256") != PATCHED_SOURCE_SHA256):
        problems.append(f"Pangolin source is not the reviewed patch {PATCH_ID}")
    if not pangolin.get("pangolin_upstream_files_verified"):
        problems.append(f"Pangolin model code or weights differ from upstream: "
                        f"{pangolin.get('pangolin_upstream_file_mismatches')}")
    if problems:
        raise SystemExit("Refusing to freeze: " + "; ".join(problems))


def check_eligibility(eligibility, annotation_summary_sha256, resources):
    """Bind the label-free eligibility check to this annotation, source and inputs."""
    from mfass.pangolin_patch import PATCHED_SOURCE_SHA256
    problems = []
    if eligibility.get("test_variants") != CANONICAL_TEST_ROWS:
        problems.append("does not cover the canonical test arm")
    if eligibility.get("labels_read") is not False:
        problems.append("does not state that labels were not read")
    if eligibility.get("annotation_dir_summary_sha256") != annotation_summary_sha256:
        problems.append("was produced from a different annotation build")
    if eligibility.get("pangolin_source_sha256") != PATCHED_SOURCE_SHA256:
        problems.append("used a different Pangolin get_genes source")
    for key in ("cohort", "split", "fasta"):
        if eligibility.get(f"{key}_sha256") != resources[key]:
            problems.append(f"used a different {key}")
    if problems:
        raise SystemExit("Refusing to freeze: eligibility summary " + "; ".join(problems))


def freeze(args):
    out_dir = Path(args.out_dir).resolve()
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; choose a new output directory")
    threads = thread_plan(args.tf_intra, args.tf_inter, args.torch_intra, args.torch_inter,
                          args.cpu_threads)
    annotation = Path(args.annotation_dir).resolve()
    summary = json.loads((annotation / "summary.json").read_text())
    cross = json.loads((annotation / "cross-check.json").read_text())
    if not (summary["cross_check_passed"] and cross["passed"]):
        raise SystemExit("Annotation cross-check has not passed; refusing to freeze")
    fasta = Path(args.fasta).resolve()
    paths = {
        "cohort": str(Path(args.cohort).resolve()), "split": str(Path(args.split).resolve()),
        "fasta": str(fasta), "fai": f"{fasta}.fai",
        "spliceai_annotation": str(annotation / "spliceai.tsv"),
        "pangolin_db": str(annotation / "pangolin.db"),
        "annotation_release": summary["gtf"]["annotation_release"],
    }
    resources = {name: file_sha256(paths[name]) for name in
                 ("cohort", "split", "fasta", "fai", "spliceai_annotation", "pangolin_db")}
    if (resources["cohort"], resources["split"]) != (COHORT_SHA256, SPLIT_SHA256):
        raise SystemExit("Cohort or split is not the canonical MFASS v2 file")
    for name, key in (("spliceai_annotation", "spliceai.tsv"), ("pangolin_db", "pangolin.db")):
        if resources[name] != summary["outputs"][key]:
            raise SystemExit(f"{key} differs from the annotation build summary")
    annotation_summary_sha256 = file_sha256(annotation / "summary.json")
    eligibility = json.loads(Path(args.eligibility_summary).read_text())
    check_eligibility(eligibility, annotation_summary_sha256, resources)
    pythons = {"spliceai": Path(args.spliceai_python), "pangolin": Path(args.pangolin_python)}
    code = {tool: _python_json(python, IDENTITY_SNIPPET[tool]) for tool, python in pythons.items()}
    check_identities(code)
    environments = {}
    for tool, python in pythons.items():
        distributions = _python_json(python, DISTRIBUTIONS_SNIPPET)
        environments[tool] = {
            "python": str(python), "distributions": distributions,
            "distributions_sha256": hashlib.sha256("\n".join(distributions).encode()).hexdigest()}
    repo = Path(_git("rev-parse", "--show-toplevel"))
    steps = {c: _runner_argv(pythons[CONDITIONS[c]["tool"]], c, paths, threads, out_dir)
             for c in CONDITIONS}
    for name, candidate, baseline in CONTRASTS:
        def predictions(c):
            return str(out_dir / c / f"{CONDITIONS[c]['stem']}.predictions.tsv")
        steps[name] = [str(pythons["pangolin"]), "-m", "mfass.compare",
                       "--baseline", predictions(baseline), "--candidate", predictions(candidate),
                       "--capacity", str(CAPACITY), "--draws", str(DRAWS), "--seed", str(SEED),
                       "--min-common", str(MIN_COMMON),
                       "--out", str(out_dir / "contrasts" / f"{name}.json")]
    manifest = {
        "study": "mfass-specialist-annotation-matched-sensitivity",
        "protocol": "docs/mfass-specialist-comparison.md",
        "registration": args.registration,
        "registration_sha256": file_sha256(repo / args.registration),
        "frozen_at_utc": _utc(),
        "draft": args.draft,
        "status": ("DRAFT for review only; not a registration and refused by run and all"
                   if args.draft else
                   "frozen; the output directory was empty when this manifest was written"),
        "git": {"head": _git("rev-parse", "HEAD"),
                "dirty_paths": _git("status", "--porcelain").splitlines()},
        "runner_files_sha256": {f: file_sha256(repo / f) for f in RUNNER_FILES},
        "paths": paths, "resources_sha256": resources,
        "annotation": {"summary_sha256": annotation_summary_sha256,
                       "cross_check_sha256": file_sha256(annotation / "cross-check.json"),
                       "outputs_sha256": summary["outputs"]},
        "eligibility": {"summary_sha256": file_sha256(args.eligibility_summary), **eligibility},
        "code": code, "environments": environments, "threads": threads,
        "distance": DISTANCE, "conditions": CONDITIONS,
        "contrasts": [{"name": n, "candidate": c, "baseline": b} for n, c, b in CONTRASTS],
        "analysis": {"capacity": CAPACITY, "draws": DRAWS, "seed": SEED,
                     "min_common": MIN_COMMON, "interval": "95% percentile, whole-group",
                     "single_class_refusal": "more than 5% of draws"},
        "sequence": list(SEQUENCE), "out_dir": str(out_dir), "steps": steps,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    print(f"{manifest_path}  sha256 {file_sha256(manifest_path)}")


def _check_frozen(manifest, step):
    """Re-hash what a step depends on; any drift stops execution."""
    repo = Path(_git("rev-parse", "--show-toplevel"))
    drift = [f for f, h in manifest["runner_files_sha256"].items() if file_sha256(repo / f) != h]
    if file_sha256(repo / manifest["registration"]) != manifest["registration_sha256"]:
        drift.append(manifest["registration"])
    if step in CONDITIONS:
        tool = CONDITIONS[step]["tool"]
        needed = ["cohort", "split", "fasta", "fai",
                  "spliceai_annotation" if tool == "spliceai" else "pangolin_db"]
        drift += [n for n in needed
                  if file_sha256(manifest["paths"][n]) != manifest["resources_sha256"][n]]
        python = manifest["environments"][tool]["python"]
        if _python_json(python, IDENTITY_SNIPPET[tool]) != manifest["code"][tool]:
            drift.append(f"{tool} code or weights")
        distributions = _python_json(python, DISTRIBUTIONS_SNIPPET)
        if distributions != manifest["environments"][tool]["distributions"]:
            drift.append(f"{tool} environment")
    if drift:
        raise SystemExit(f"Frozen inputs changed since the manifest was written: {drift}")


def _after(created, frozen_at):
    try:
        return datetime.fromisoformat(created) > datetime.fromisoformat(frozen_at)
    except (TypeError, ValueError):
        return False


def _result_path(manifest, condition):
    return Path(manifest["out_dir"]) / condition / f"{CONDITIONS[condition]['stem']}.json"


def _result(manifest, condition):
    return json.loads(_result_path(manifest, condition).read_text())


def canonical_test_rows(manifest):
    """Canonical test IDs in cohort order, with labels and groups."""
    with open(manifest["paths"]["split"], newline="") as handle:
        split = {r["id"]: r for r in csv.DictReader(handle, delimiter="\t")}
    with open(manifest["paths"]["cohort"], newline="") as handle:
        return [{"id": r["id"], "label": r["sdv"], "group": split[r["id"]]["group"]}
                for r in csv.DictReader(handle, delimiter="\t")
                if split[r["id"]]["split"] == "test"]


def _read_table(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def check_tables(manifest, condition, result, canonical):
    """Bind a result's tables and checkpoint to its JSON; return problems and table facts."""
    from rewirebench.metrics import point_metrics

    from mfass.specialist_run import EXPECTED_SKIP_PREFIXES, output_paths
    problems = []
    paths = output_paths(_result_path(manifest, condition))
    recorded = result["config"].get("output_tables") or {}
    if "predictions" not in recorded:
        return [f"{condition}: result records no prediction table"], {}
    for key in ("predictions", "unscored"):
        present = paths[key].exists()
        if (key in recorded) != present:
            problems.append(f"{condition}: {key} table presence differs from the result")
        elif present and file_sha256(paths[key]) != recorded[key]["sha256"]:
            problems.append(f"{condition}: {key} table bytes differ from the result")
    if problems:
        return problems, {}
    rows = _read_table(paths["predictions"])
    if [r["id"] for r in rows] != [r["id"] for r in canonical]:
        problems.append(f"{condition}: prediction IDs are not the canonical test IDs in order")
        return problems, {}
    if any((r["label"], r["group"]) != (c["label"], c["group"]) for r, c in zip(rows, canonical)):
        problems.append(f"{condition}: prediction labels or groups differ from the canonical files")
    scored = [r for r in rows if r["score"] != ""]
    values = [float(r["score"]) for r in scored]
    if not all(math.isfinite(v) for v in values):
        problems.append(f"{condition}: nonfinite prediction score")
        return problems, {}
    blank = [r["id"] for r in rows if r["score"] == ""]
    unscored = _read_table(paths["unscored"]) if paths["unscored"].exists() else []
    if [r["id"] for r in unscored] != blank:
        problems.append(f"{condition}: unscored table does not list exactly the blank rows")
    unexpected = sorted({r["reason"] for r in unscored
                         if not r["reason"].startswith(EXPECTED_SKIP_PREFIXES)})
    if unexpected:
        problems.append(f"{condition}: unexpected unscored reasons {unexpected[:5]}")
    coverage = result["coverage"]
    if (coverage["scored"], coverage["unscored"]) != (len(scored), len(blank)):
        problems.append(f"{condition}: coverage differs from the prediction rows")
    if result["independent_groups"] != len({r["group"] for r in scored}):
        problems.append(f"{condition}: independent group count differs from the prediction rows")
    labels = [int(r["label"]) for r in scored]
    if len(set(labels)) == 2:
        recomputed = point_metrics(labels, values, CAPACITY)
        for key in METRIC_KEYS:
            if not math.isclose(recomputed[key], result["metrics"].get(key, math.nan),
                                rel_tol=0, abs_tol=1e-12):
                problems.append(f"{condition}: metric {key} does not recompute from the rows")
    else:
        problems.append(f"{condition}: scored rows lack both outcome classes")
    checkpoint = Path(manifest["out_dir"]) / condition / "checkpoint.tsv"
    record = result["config"].get("checkpoint") or {}
    if not checkpoint.exists() or file_sha256(checkpoint) != record.get("sha256"):
        problems.append(f"{condition}: checkpoint bytes differ from the result")
    else:
        logged = {r["id"]: r for r in _read_table_skip_fingerprint(checkpoint)}
        for r in rows:
            entry = logged.get(r["id"])
            want = "" if entry is None or entry["status"] != "scored" else (
                f"{float(entry['score']):.4f}")
            if entry is None or want != r["score"]:
                problems.append(f"{condition}: checkpoint and predictions disagree at {r['id']}")
                break
    return problems, {"predictions_sha256": recorded["predictions"]["sha256"],
                      "unscored_reasons": dict(Counter(r["reason"] for r in unscored))}


def _read_table_skip_fingerprint(path):
    with open(path, newline="") as handle:
        next(handle)
        return list(csv.DictReader(handle, delimiter="\t"))


def verify_results(manifest, results, canonical=None):
    """Check provenance and bound tables of finished condition results.

    Metrics are recomputed only to confirm they match the rows; they are not reported.
    """
    problems, coverage, tables = [], {}, {}
    frozen_at = manifest.get("frozen_at_utc")
    for condition, result in results.items():
        spec, config = CONDITIONS[condition], result["config"]
        tool = spec["tool"]
        annotation_key = "spliceai_annotation" if tool == "spliceai" else "pangolin_db"
        threads = manifest["threads"][tool]
        observed_threads = config.get("tensorflow_threads" if tool == "spliceai"
                                      else "torch_threads") or {}
        expected = {
            "scope": "full", "cohort_sha256": COHORT_SHA256, "split_sha256": SPLIT_SHA256,
            "annotation_sha256": manifest["resources_sha256"][annotation_key],
            "reference_sha256": manifest["resources_sha256"]["fasta"],
            ("mask_M" if tool == "spliceai" else "mask_m"):
                int(spec["mask"]) if tool == "spliceai" else spec["mask"],
            ("distance_D" if tool == "spliceai" else "distance_d"): manifest["distance"],
            **manifest["code"][tool],
        }
        for key, value in expected.items():
            if config.get(key) != value:
                problems.append(f"{condition}: {key} differs from manifest")
        if (observed_threads.get("intra_op"), observed_threads.get("inter_op")) != (
                threads["intra_op"], threads["inter_op"]):
            problems.append(f"{condition}: thread configuration differs from manifest")
        if result["benchmark"] != "mfass-v2" or config.get("selected_test_rows") != CANONICAL_TEST_ROWS:
            problems.append(f"{condition}: not a full canonical run")
        if result["coverage"]["denominator"] != CANONICAL_TEST_ROWS:
            problems.append(f"{condition}: denominator is not {CANONICAL_TEST_ROWS}")
        if frozen_at and not _after(result.get("created_utc"), frozen_at):
            problems.append(f"{condition}: result predates the frozen manifest")
        if canonical is not None:
            found, facts = check_tables(manifest, condition, result, canonical)
            problems += found
            tables[condition] = facts
        coverage[condition] = result["coverage"]
    for first, second in (("S0", "S1"), ("P0", "P1")):
        if first in results and second in results:
            a, b = results[first]["config"], results[second]["config"]
            differing = sorted(k for k in set(a) | set(b)
                               if k not in PAIR_VARIABLE_KEYS and a.get(k) != b.get(k))
            if differing:
                problems.append(f"{first}/{second} differ beyond mask: {differing}")
    return {"passed": not problems, "problems": problems, "coverage": coverage,
            "tables": tables, "metrics_reported": False}


def _verify_conditions(manifest, conditions):
    missing = [c for c in conditions if not _result_path(manifest, c).exists()]
    if missing:
        raise SystemExit(f"Finished results are missing for {missing}")
    report = verify_results(manifest, {c: _result(manifest, c) for c in conditions},
                            canonical_test_rows(manifest))
    if not report["passed"]:
        raise SystemExit(f"Condition results fail verification: {report['problems']}")
    return report


def _log_event(manifest, manifest_sha256, **event):
    logs = Path(manifest["out_dir"]) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "events.jsonl").open("a") as handle:
        handle.write(json.dumps({"utc": _utc(), "manifest_sha256": manifest_sha256, **event}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _execute(manifest, manifest_sha256, step):
    """Run one recorded command, appending its output to logs/<step>.log."""
    argv = manifest["steps"][step]
    tool = CONDITIONS[step]["tool"] if step in CONDITIONS else None
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "3",
           **manifest["threads"][tool or "other_steps"]["env"]}
    Path(argv[argv.index("--out") + 1]).parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(manifest["out_dir"]) / "logs" / f"{step}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_event(manifest, manifest_sha256, step=step, event="start", argv=argv)
    started = time.monotonic()
    with log_path.open("a") as log:
        log.write(f"\n=== {_utc()} start {step} manifest {manifest_sha256}\n")
        log.flush()
        code = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                              check=False).returncode
        log.write(f"=== {_utc()} exit {code}\n")
    _log_event(manifest, manifest_sha256, step=step, event="exit", returncode=code,
               seconds=round(time.monotonic() - started, 1))
    if code != 0:
        raise SystemExit(f"{step} failed with exit code {code}; see {log_path}")


def _contrast_complete(manifest, contrast, verified):
    out = Path(manifest["steps"][contrast["name"]][-1])
    if not out.exists():
        return False
    report = json.loads(out.read_text())
    tables = verified["tables"]
    expected = (tables[contrast["baseline"]]["predictions_sha256"],
                tables[contrast["candidate"]]["predictions_sha256"])
    if (report.get("baseline_sha256"), report.get("candidate_sha256")) != expected:
        raise SystemExit(f"{out} was computed from other prediction bytes; move it aside")
    return True


def run_step(manifest, manifest_sha256, step, check_only=False):
    """Run or confirm one step. Returns 'done', 'ran' or 'checked'."""
    if step not in SEQUENCE:
        raise SystemExit(f"Unknown step {step}; choose from {list(SEQUENCE)}")
    if manifest.get("draft", True) and not check_only:
        raise SystemExit("This manifest is a draft (or predates draft marking); freeze a "
                         "final manifest after the reviewed code is committed")
    if step in CONDITIONS:
        if _result_path(manifest, step).exists():
            _verify_conditions(manifest, [step])
            return "done"
        _check_frozen(manifest, step)
        if check_only:
            return "checked"
        _execute(manifest, manifest_sha256, step)
        _verify_conditions(manifest, [step])
        return "ran"
    _check_frozen(manifest, step)
    if step == "verify":
        verified = _verify_conditions(manifest, list(CONDITIONS))
        out = Path(manifest["out_dir"]) / "verification.json"
        if not check_only:
            out.write_text(json.dumps(verified, indent=2) + "\n")
        return "ran"
    contrast = next((c for c in manifest["contrasts"] if c["name"] == step), None)
    if contrast:
        verified = _verify_conditions(manifest, [contrast["candidate"], contrast["baseline"]])
        if _contrast_complete(manifest, contrast, verified):
            return "done"
        if check_only:
            return "checked"
        _execute(manifest, manifest_sha256, step)
        if not _contrast_complete(manifest, contrast, verified):
            raise SystemExit(f"{step} did not produce a report bound to the verified predictions")
        return "ran"
    report_path = Path(manifest["out_dir"]) / "report.json"
    if report_path.exists():
        return "done"
    if check_only:
        return "checked"
    write_report(manifest, manifest_sha256)
    return "ran"


def tie_summary(values, capacity=CAPACITY):
    """How the review cutoff falls among tied scores; labels are not used."""
    ordered = sorted(values, reverse=True)
    if len(ordered) < capacity:
        return {"scored": len(ordered), "capacity_exceeds_list": True}
    cutoff = ordered[capacity - 1]
    above = sum(v > cutoff for v in ordered)
    tied = sum(v == cutoff for v in ordered)
    return {"scored": len(ordered), "score_at_capacity": cutoff, "above_cutoff": above,
            "tied_at_cutoff": tied, "slots_filled_from_ties": capacity - above,
            "tie_break": "label-independent permutation, rewirebench.metrics.precision_at_n"}


def _scores(manifest, condition):
    from mfass.specialist_run import output_paths
    rows = _read_table(output_paths(_result_path(manifest, condition))["predictions"])
    return {r["id"]: float(r["score"]) for r in rows if r["score"] != ""}


def write_report(manifest, manifest_sha256):
    verified = _verify_conditions(manifest, list(CONDITIONS))
    report = {"manifest_sha256": manifest_sha256, "generated_utc": _utc(),
              "denominator": CANONICAL_TEST_ROWS, "conditions": {}, "contrasts": {},
              "notes": ("Exploratory, unadjusted intervals; not a confirmatory comparison. "
                        "Missing scores are coverage gaps, not negative predictions.")}
    scores = {c: _scores(manifest, c) for c in CONDITIONS}
    for condition in CONDITIONS:
        result = _result(manifest, condition)
        report["conditions"][condition] = {
            "coverage": result["coverage"], "metrics": {k: result["metrics"][k] for k in METRIC_KEYS},
            "unscored_reasons": verified["tables"][condition]["unscored_reasons"],
            "ties": tie_summary(list(scores[condition].values())),
            "predictions_sha256": verified["tables"][condition]["predictions_sha256"],
            "timing_seconds": result["timing_seconds"]}
    for contrast in manifest["contrasts"]:
        out = Path(manifest["steps"][contrast["name"]][-1])
        compared = json.loads(out.read_text())
        common = sorted(set(scores[contrast["baseline"]]) & set(scores[contrast["candidate"]]))
        report["contrasts"][contrast["name"]] = {
            "candidate": contrast["candidate"], "baseline": contrast["baseline"],
            "denominators": compared["denominators"],
            "independent_groups": compared["independent_groups"],
            "compatibility": compared["compatibility"]["status"],
            "on_common_subset": compared["on_common_subset"], "paired": compared["paired"],
            "ties_on_common": {side: tie_summary([scores[contrast[side]][i] for i in common])
                               for side in ("baseline", "candidate")},
            "report_sha256": file_sha256(out)}
    out_dir = Path(manifest["out_dir"])
    text = json.dumps(report, indent=2) + "\n"
    temporary = out_dir / ".report.json.partial"
    temporary.write_text(text)
    os.replace(temporary, out_dir / "report.json")
    (out_dir / "report.md").write_text(_markdown(report))
    print(f"report: {out_dir / 'report.json'}")


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def _markdown(report):
    lines = ["# MFASS annotation-matched specialist study: results",
             "", f"Manifest SHA-256 `{report['manifest_sha256']}`. Generated {report['generated_utc']}.",
             report["notes"], "", "## Conditions", "",
             "| Condition | Scored / 8,324 | P@100 | AP | AUROC | Tied at the 100th score |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, c in report["conditions"].items():
        m = c["metrics"]
        lines.append(f"| {name} | {c['coverage']['scored']:,} | {_fmt(m['precision_at_capacity'])} "
                     f"| {_fmt(m['average_precision_sklearn'])} | {_fmt(m['auroc'])} "
                     f"| {c['ties'].get('tied_at_cutoff', 'n/a')} |")
    lines += ["", "## Contrasts (candidate minus baseline, common scored variants)", "",
              "| Contrast | Common | Positives | Groups | Metric | Observed delta | 95% interval |",
              "|---|---:|---:|---:|---|---:|---|"]
    for name, c in report["contrasts"].items():
        d = c["denominators"]
        for metric, value in c["paired"].items():
            interval = (f"refused: {value['refused']}" if "refused" in value else
                        f"[{value['ci95_low']:.3f}, {value['ci95_high']:.3f}]")
            delta = "n/a" if "refused" in value else f"{value['observed_delta']:+.3f}"
            lines.append(f"| {name} | {d['common']:,} | {d['common_positives']} "
                         f"| {c['independent_groups']} | {metric} | {delta} | {interval} |")
    return "\n".join(lines) + "\n"


def _load_manifest(path):
    return json.loads(Path(path).read_text()), file_sha256(path)


def run(args):
    manifest, digest = _load_manifest(args.manifest)
    outcome = run_step(manifest, digest, args.step, args.check_only)
    print(f"{args.step}: {outcome}")


def run_all(args):
    manifest, digest = _load_manifest(args.manifest)
    _log_event(manifest, digest, step="all", event="start")
    for step in SEQUENCE:
        try:
            outcome = run_step(manifest, digest, step)
        except BaseException as stop:
            _log_event(manifest, digest, step=step, event="failed",
                       detail=f"{type(stop).__name__}: {stop}")
            raise
        _log_event(manifest, digest, step=step, event=outcome)
        print(f"{_utc()} {step}: {outcome}", flush=True)
    _log_event(manifest, digest, step="all", event="complete")


def verify(args):
    manifest, _ = _load_manifest(args.manifest)
    results = {c: _result(manifest, c) for c in CONDITIONS if _result_path(manifest, c).exists()}
    report = verify_results(manifest, results, canonical_test_rows(manifest))
    report["conditions_present"] = sorted(results)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    for name in ("cohort", "split", "fasta", "annotation-dir", "eligibility-summary",
                 "spliceai-python", "pangolin-python", "out-dir", "manifest", "registration"):
        f.add_argument(f"--{name}", required=True)
    f.add_argument("--cpu-threads", type=int, default=6)
    f.add_argument("--tf-intra", type=int, default=5)
    f.add_argument("--tf-inter", type=int, default=1)
    f.add_argument("--torch-intra", type=int, default=6)
    f.add_argument("--torch-inter", type=int, default=1)
    f.add_argument("--draft", action="store_true",
                   help="write a review-only manifest that run and all refuse to execute")
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--step", required=True, help=", ".join(SEQUENCE))
    r.add_argument("--check-only", action="store_true", help="verify frozen inputs; do not run")
    a = sub.add_parser("all", help="run every step in order; resumes and stops on failure")
    a.add_argument("--manifest", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    args = ap.parse_args()
    {"freeze": freeze, "run": run, "all": run_all, "verify": verify}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
