"""Run the unmodified pinned ProteinGym/EVCouplings scorer on the synthetic fixture.

Execute with a Python environment where the pinned EVCouplings checkout is
installed (it must not import rewirebench). The script:

1. checks the upstream source hashes against the reviewed pins;
2. lays out each fixture model as ProteinGym expects and runs the unmodified
   ``score_mutants.py`` command for the assay's DMS_index in the pinned reference;
3. calls the pinned ``calculations.predict_mutation_table`` on each invalid
   mutant alone to record upstream's own error, since one invalid mutant aborts
   the whole upstream assay run;
4. records the BFGS warning flag of each site in upstream's own
   ``to_independent_model`` call, through a pass-through wrapper around
   ``scipy.optimize.fmin_bfgs`` that returns the identical iterate;
5. writes a receipt with predictions, independent fields, versions and hashes.

    python scripts/baseline_parity/evcouplings_upstream_receipt.py \
        --upstream /path/to/upstream --output receipt.json
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "packages/rewirebench/tests/fixtures/proteingym_independent"
REFERENCE = ROOT / "packages/rewirebench/src/rewirebench/resources/proteingym/DMS_substitutions.csv"
PINS = {
    "evcouplings/evcouplings/couplings/model.py": "6422cfc7892cb5d076194354b05fe91fe40a49cee28548e62c4e211f32cfa050",
    "proteingym/proteingym/baselines/EVmutation/score_mutants.py": "9802fcd2e50ac0396a445b3c0bf9e4cfb865f1e57ae0c6382a12214ee0336050",
    "proteingym/proteingym/baselines/EVmutation/calculations.py": "07e2059186bdf2f673c46817c0a2c630658269008474b3c05f10585b8d246c6e",
    "proteingym/config.json": "7cd239d1e4c8b474f6ce2bd6574423b53ab48204f852f219e09a36f25d513608",
}
REVISIONS = {"evcouplings": "e1362407a0b65d63ca07df55f44cb17b0a3722b7",
             "proteingym": "144fe22b07dfaeec2b366f2346203a9838a55b4c"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for relative, digest in PINS.items():
        if sha(args.upstream / relative) != digest:
            raise SystemExit(f"Upstream source hash mismatch: {relative}")
    for name, revision in REVISIONS.items():
        head = subprocess.run(["git", "-C", str(args.upstream / name), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
        if head != revision:
            raise SystemExit(f"{name} checkout is {head}, expected {revision}")
    import evcouplings.couplings.model as installed
    if sha(installed.__file__) != PINS["evcouplings/evcouplings/couplings/model.py"]:
        raise SystemExit("Installed EVCouplings model.py differs from the pinned source")
    if "rewirebench" in sys.modules:
        raise SystemExit("Upstream receipt must not import rewirebench")

    scorer_dir = args.upstream / "proteingym/proteingym/baselines/EVmutation"
    sys.path.insert(0, str(scorer_dir))
    import calculations  # pinned ProteinGym module
    import numpy as np
    import pandas as pd
    import scipy.optimize
    from evcouplings.couplings import CouplingsModel

    with REFERENCE.open(newline="") as stream:
        reference = list(csv.DictReader(stream))
    fixture = json.loads((FIXTURE / "fixture.json").read_text())
    receipt = {"schema": "rewire-evcouplings-upstream-receipt-v1",
               "created_at": datetime.now(UTC).isoformat(),
               "command": "score_mutants.py --DMS_reference_file_path --DMS_data_folder "
                          "--model_folder --output_scores_folder --DMS_index",
               "upstream_revisions": REVISIONS, "upstream_source_sha256": PINS,
               "installed_model_py_sha256": sha(installed.__file__),
               "reference_sha256": sha(REFERENCE),
               "fixture_json_sha256": sha(FIXTURE / "fixture.json"),
               "environment": {"python": platform.python_version(), "platform": platform.platform(),
                               **{p: importlib.metadata.version(p)
                                  for p in ("numpy", "scipy", "pandas", "numba")}},
               "assays": {}}
    for assay_id, spec in sorted(fixture.items()):
        index = next(i for i, r in enumerate(reference) if r["DMS_id"] == assay_id)
        ref = reference[index]
        model_path = FIXTURE / spec["model"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prefix = spec["model_id"]
            couplings = tmp / "models" / prefix / prefix / "couplings"
            couplings.mkdir(parents=True)
            shutil.copy(model_path, couplings / f"{prefix}.model")
            (tmp / "dms").mkdir()
            (tmp / "out").mkdir()
            wt = ref["target_seq"]
            with (tmp / "dms" / ref["DMS_filename"]).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"])
                for mutant in spec["scored"]:
                    sequence = list(wt)
                    for part in mutant.split(":"):
                        sequence[int(part[1:-1]) - 1] = part[-1]
                    # Placeholder values: the scorer copies but never reads them.
                    writer.writerow([mutant, "".join(sequence), 0.0, 0])
            run = subprocess.run(
                [sys.executable, "score_mutants.py", "--DMS_reference_file_path", str(REFERENCE),
                 "--DMS_data_folder", str(tmp / "dms"), "--model_folder", str(tmp / "models"),
                 "--output_scores_folder", str(tmp / "out"), "--DMS_index", str(index)],
                cwd=scorer_dir, capture_output=True, text=True, check=True)
            table = pd.read_csv(tmp / "out" / f"{assay_id}.csv")
        model = CouplingsModel(str(model_path))
        real_bfgs, warnflags = scipy.optimize.fmin_bfgs, []

        def observed_bfgs(*a, _real=real_bfgs, _flags=warnflags, **kw):
            # Same call as upstream plus full_output, which does not alter the iterate.
            result = _real(*a, **kw, full_output=True)
            _flags.append(int(result[6]))
            return result[0]

        scipy.optimize.fmin_bfgs = observed_bfgs
        try:
            independent = model.to_independent_model()
        finally:
            scipy.optimize.fmin_bfgs = real_bfgs
        offset = -(int(ref["MSA_start"]) - 1)
        errors = {}
        for mutant in spec["upstream_errors"]:
            try:
                calculations.predict_mutation_table(
                    independent, pd.DataFrame({"mutant": [mutant]}), "prediction_independent",
                    sep=":", offset=offset)
                errors[mutant] = None
            except Exception as exc:  # noqa: BLE001 - record upstream's own failure
                errors[mutant] = {"type": type(exc).__name__, "message": str(exc).splitlines()[0]}
        receipt["assays"][assay_id] = {
            "DMS_index": index, "MSA_start": int(ref["MSA_start"]), "model": spec["model"],
            "model_sha256": sha(model_path),
            "stdout_offset": [line for line in run.stdout.splitlines() if line.startswith("Offset")],
            "prediction_independent": dict(zip(table["mutant"], table["prediction_independent"])),
            "prediction_epistatic": dict(zip(table["mutant"], table["prediction_epistatic"])),
            "independent_fields": {str(int(p)): independent.h_i[i].tolist()
                                   for i, p in enumerate(model.index_list)},
            "alphabet": "".join(model.alphabet), "lambda_h": float(model.lambda_h),
            "N_eff": float(model.N_eff), "invalid_mutants": errors,
            "J_ij_cleared": bool(np.all(independent.J_ij == 0)),
            "bfgs_warnflags": {str(int(p)): flag for p, flag in zip(model.index_list, warnflags)},
        }
    args.output.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
