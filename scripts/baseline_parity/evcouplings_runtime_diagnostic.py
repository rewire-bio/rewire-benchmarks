"""Numerical runtime diagnostic for the EVCouplings independent-model fixtures.

Run with the rewirebench environment. It prepares the synthetic fixture
artifact in this runtime and compares it, over every assay, site, alphabet entry
and receipt mutant, with:

- the frozen macOS receipt (``upstream-receipt.json``), always;
- optionally, a receipt produced by the unmodified pinned upstream in this same
  runtime (``--same-runtime-receipt``, from evcouplings_upstream_receipt.py).

It records the runtime identity (platform, CPU, NumPy/SciPy, BLAS and NumPy CPU
dispatch) and writes JSON; a short summary goes to stdout. It asserts nothing;
the tests do. No biological data is used.

    python scripts/baseline_parity/evcouplings_runtime_diagnostic.py \
        [--same-runtime-receipt same.json] --output diagnostic.json
"""
import argparse
import hashlib
import io
import json
import platform
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evcouplings_compare as compare

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "packages/rewirebench/tests/fixtures/proteingym_independent"


def runtime():
    config = io.StringIO()
    with redirect_stdout(config):
        np.show_config()
    try:
        from numpy.core._multiarray_umath import __cpu_features__ as features
        enabled = sorted(k for k, v in features.items() if v)
    except ImportError:
        enabled = "unavailable"
    cpu = "unreported"
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), "unreported")
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_model": cpu, "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "numpy_cpu_features_enabled": enabled,
            "numpy_show_config": config.getvalue()}


def prepare(tmp):
    from rewirebench.adapters import evcouplings_independent as evc

    spec = json.loads((FIXTURE / "fixture.json").read_text())
    provenance = {k: "synthetic fixture; not a biological model" for k in evc.PROVENANCE_FIELDS}
    provenance["alignment_sha256"] = "0" * 64
    manifest = {"schema": evc.MANIFEST_SCHEMA, "dms_labels_used": False, "assays": {
        assay: {"model_path": str(FIXTURE / s["model"]), "format": "plmc_v2", "precision": "float32",
                "model_sha256": hashlib.sha256((FIXTURE / s["model"]).read_bytes()).hexdigest(),
                "model_id": s["model_id"], "msa_start": s["msa_start"], "provenance": provenance}
        for assay, s in spec.items()}}
    (tmp / "manifest.json").write_text(json.dumps(manifest))
    result = evc.prepare_artifact(tmp / "manifest.json", tmp / "artifact.json")
    adapter = evc.EVCouplingsIndependent(result["artifact"], result["artifact_sha256"])
    predictions = {}
    for assay, s in spec.items():
        wild_type = json.loads(Path(result["artifact"]).read_text())["assays"][assay]["wild_type_sequence"]
        rows = [{"id": m, "assay_id": assay, "wild_type_sequence": wild_type, "mutant": m} for m in s["scored"]]
        predictions[assay] = adapter.predict(rows)
    return json.loads(Path(result["artifact"]).read_text())["assays"], spec, predictions


def summarise(label, report, failures):
    print(f"== {label}: {len(failures)} checks outside 1e-6")
    for assay, entry in report.items():
        for position, site in entry["sites"].items():
            print(f"  {assay[:5]} {position:>4}: raw {site['raw_max_abs_difference']:.2e} "
                  f"mean {site['mean_difference']:+.2e} after-mean {site['max_abs_difference_after_mean']:.2e} "
                  f"contrast {site['contrast_max_abs_difference']:.2e} "
                  f"flags adapter {site['adapter_warnflag']} upstream {site['upstream_warnflag']}")
        worst = max(v["abs_difference"] for v in entry["scores"].values())
        print(f"  {assay[:5]} scores: max abs difference {worst:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--same-runtime-receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory() as tmp:
        assays, spec, predictions = prepare(Path(tmp))
    frozen = json.loads((FIXTURE / "upstream-receipt.json").read_text())
    result = {"schema": "rewire-evcouplings-runtime-diagnostic-v1", "runtime": runtime(),
              "frozen_receipt_environment": frozen["environment"],
              "adapter_warnflags": {a: {p: s["warnflag"] for p, s in assays[a]["optimizer"]["sites"].items()}
                                    for a in assays}}
    report, failures = compare.compare(assays, frozen["assays"], spec, predictions)
    result["versus_frozen_receipt"] = {"failures": failures, "report": report}
    summarise("adapter vs frozen macOS receipt", report, failures)
    live = None
    if args.same_runtime_receipt:
        live = json.loads(args.same_runtime_receipt.read_text())
        report, failures = compare.compare(assays, live["assays"], spec, predictions)
        failures += compare.warnflag_mismatches(report)
        result["same_runtime_upstream_environment"] = live["environment"]
        result["versus_same_runtime_upstream"] = {"failures": failures, "report": report}
        summarise("adapter vs same-runtime upstream", report, failures)
        upstream_vs_frozen = {
            assay: {"max_raw_field_difference": max(
                float(np.abs(np.asarray(row) - np.asarray(frozen["assays"][assay]["independent_fields"][i])).max())
                for i, row in live["assays"][assay]["independent_fields"].items()),
                    "warnflags_same_runtime": live["assays"][assay]["bfgs_warnflags"],
                    "warnflags_frozen": frozen["assays"][assay]["bfgs_warnflags"]}
            for assay in spec}
        result["same_runtime_upstream_vs_frozen"] = upstream_vs_frozen
        print("== same-runtime upstream vs frozen receipt:", json.dumps(upstream_vs_frozen))
    result["common_component"] = common_component(assays, spec, frozen, live)
    print("== site means minus the exact stationary mean -N_eff(1-sum f)/(2 lambda_h q):")
    for assay, sites in result["common_component"].items():
        for position, site in sites.items():
            print(f"  {assay[:5]} {position:>4}: optimum {site['stationary_mean']:+.4e} "
                  + " ".join(f"{k} {v:+.2e}" for k, v in site.items() if k.endswith("_minus_optimum")))
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")


def common_component(assays, spec, frozen, live):
    """Distance of each execution's site mean from the exact optimum's mean.

    Summing the gradient over the alphabet gives N_eff (1 - sum f) + 2 lambda_h sum h,
    so at the exact optimum mean(h) = -N_eff (1 - sum f) / (2 lambda_h q).
    """
    from rewirebench.adapters.evcouplings_independent import read_plmc_v2

    result = {}
    for assay, s in spec.items():
        model = read_plmc_v2(FIXTURE / s["model"])
        n_eff, lam, q = float(model["N_eff"]), float(model["lambda_h"]), model["num_symbols"]
        result[assay] = {}
        for i, index in enumerate(model["index_list"]):
            position = str(index + s["msa_start"] - 1)
            optimum = -n_eff * (1 - model["f_i"][i].astype(float).sum()) / (2 * lam * q)
            site = {"stationary_mean": optimum,
                    "adapter_minus_optimum": float(np.mean(assays[assay]["fields"][position])) - optimum,
                    "frozen_minus_optimum": float(np.mean(frozen["assays"][assay]["independent_fields"][str(index)])) - optimum}
            if live:
                site["same_runtime_upstream_minus_optimum"] = (
                    float(np.mean(live["assays"][assay]["independent_fields"][str(index)])) - optimum)
            result[assay][position] = site
    return result


if __name__ == "__main__":
    main()
