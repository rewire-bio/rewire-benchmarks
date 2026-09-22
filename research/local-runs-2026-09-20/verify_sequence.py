"""Independently audit saved sequence runs; publish aggregates, never source rows.

This checker uses sklearn/scipy directly, not rewirebench fitting or scoring.
It checks installed SDK input allowlisting separately. It does not establish
reproduction of a published model or absence of biological train/test overlap.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TOLERANCE = 1e-9
RUNNER_REVISION = "fcccbbcdbe3d5cd64a1a312d536615273320f7b4"
RUNNER_URL = f"https://github.com/rewire-bio/rewire-benchmarks/blob/{RUNNER_REVISION}/packages/rewirebench/src/rewirebench/"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_digest(root):
    state = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        state.update(str(path.relative_to(root)).encode())
        state.update(path.read_bytes())
    return state.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def check(checks, name, passed, detail, source, locator):
    checks.append(
        {
            "check": name,
            "status": "passed" if passed else "failed",
            "method": "automated independent calculation",
            "detail": detail,
            "source_url": source,
            "evidence_location": locator,
        }
    )
    if not passed:
        raise AssertionError(name + ": " + detail)


def features(sequence, kind):
    if kind == "flip2":
        # Rhomax is a single sequence; the paired-input component is zero.
        return [sequence.count(aa) / len(sequence) for aa in "ACDEFGHIKLMNPQRSTVWY"] + [0.0] * 20
    sequence = sequence.upper()
    counts = [sequence.count(base) for base in "ACGT"]
    return (
        [math.log1p(len(sequence))]
        + [c / len(sequence) for c in counts]
        + [(len(sequence) - sum(counts)) / len(sequence)]
    )


def source_checks(args, kind, prepared):
    checks = []
    root = files("rewirebench").joinpath("resources", kind)
    pins = json.loads(root.joinpath("sources.json").read_text())
    rows = prepared["rows"]
    counts = dict(Counter(r["split"] for r in rows))
    if kind == "flip2":
        pin = next(p for p in pins["datasets"] if p["dataset_id"] == "flip2-rhomax-by-wild-type")
        source = args.flip_source
        source_url = pin["mirror_url"] if digest(source) == pin["mirror_sha256"] else pin["url"]
        check(
            checks,
            "source_bytes",
            digest(source) in (pin["sha256"], pin["mirror_sha256"]),
            "Archived/mirror compressed source hash matches its respective pin: " + digest(source),
            source_url,
            "complete gzip file",
        )
        raw = gzip.decompress(source.read_bytes())
        check(
            checks,
            "source_csv_bytes",
            hashlib.sha256(raw).hexdigest() == pin["csv_sha256"],
            "Decompressed source CSV matches canonical v3 hash " + pin["csv_sha256"],
            pin["url"],
            "complete decompressed CSV",
        )
        original = list(csv.DictReader(io.StringIO(raw.decode())))
        source_rows = Counter(
            (
                "validation" if r["validation"] == "True" else r["set"],
                r["sequence"],
                float(r["target"]),
            )
            for r in original
        )
        actual_rows = Counter((r["split"], r["inputs"]["sequence"], r["target"]) for r in rows)
        check(
            checks,
            "source_rows_and_splits",
            source_rows == actual_rows and counts == pin["counts"],
            "Every source sequence, label and archived assignment retained, including duplicate multiplicity; counts "
            + json.dumps(counts, sort_keys=True),
            pin["url"],
            "set/validation/sequence/target columns; all 884 source rows",
        )
        evaluator = next(s for s in pins["sources"] if s["role"] == "baselines_aggregate.py")
        upstream = args.flip_upstream / "baselines_aggregate.py"
        refs = [(upstream, evaluator)]
        refs.append(
            (
                args.flip_upstream / "baselines_linear_models.py",
                next(s for s in pins["sources"] if s["role"] == "baselines_linear_models.py"),
            )
        )
        locator = "baselines/aggregate.py:131-135; Spearman and shifted-target NDCG"
    else:
        pin = pins["datasets"]["designed"]
        source = args.mrna_source
        source_url = pin["url"]
        check(
            checks,
            "source_bytes",
            digest(source) == pin["sha256"],
            "Official pinned parquet hash " + pin["sha256"],
            source_url,
            "complete parquet file",
        )
        frame = pd.read_parquet(source, columns=["sequence", "target_mrl_designed"]).reset_index(
            drop=True
        )
        eligible = frame[frame.target_mrl_designed.notna()]
        train, heldout = train_test_split(
            eligible.index.to_numpy(), test_size=0.3, random_state=2541
        )
        validation, test = train_test_split(heldout, test_size=0.5, random_state=2541)
        membership = {
            name: [int(i) for i in values]
            for name, values in [("train", train), ("validation", validation), ("test", test)]
        }
        assigned = {i: name for name, values in membership.items() for i in values}
        identical = len({r["source_index"] for r in rows}) == len(eligible)
        for r in rows:
            i = r["source_index"]
            identical &= (
                r["split"] == assigned[i]
                and r["inputs"]["sequence"] == frame.at[i, "sequence"]
                and r["target"] == float(frame.at[i, "target_mrl_designed"])
            )
        split_hash = hashlib.sha256(
            json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        check(
            checks,
            "source_rows_and_splits",
            identical
            and membership == prepared["metadata"]["split_membership"]
            and split_hash == prepared["provenance"]["split_sha256"]
            and counts == {"train": 70011, "validation": 15003, "test": 15003},
            f"All {len(frame)} source rows reconciled; {len(frame) - len(eligible)} missing targets excluded; independently generated seed-2541 assignments match; split hash {split_hash}; counts "
            + json.dumps(counts, sort_keys=True),
            source_url,
            "sequence/target_mrl_designed columns; 70/15/15 split after missing-target removal",
        )
        evaluator = pins["code"]["mrna_bench/linear_probe/evaluator.py"]
        refs = [
            (args.mrna_upstream / key.replace("/", "."), value)
            for key, value in pins["code"].items()
            if (args.mrna_upstream / key.replace("/", ".")).exists()
        ]
        locator = "mrna_bench/linear_probe/evaluator.py:9-36; eval_regression"
    references = []
    for path, reference in refs:
        check(
            checks,
            "upstream_source_" + path.name,
            digest(path) == reference["sha256"],
            "Pinned source file SHA-256 " + reference["sha256"],
            reference["url"],
            "complete source file",
        )
        references.append({"url": reference["url"], "sha256": reference["sha256"]})
    check(
        checks,
        "prepared_scope",
        prepared["scope"] == "full" and prepared["metadata"]["smoke_limit_per_split"] is None,
        "Complete selected split/target; not a suite aggregate and not a smoke run",
        source_url,
        "prepared scope and limit metadata",
    )
    check(
        checks,
        "opaque_unique_ids",
        len({r["id"] for r in rows}) == len(rows)
        and all(set(r["inputs"]) == {"sequence"} for r in rows),
        "Every prepared ID is unique; allowlisted biological inputs contain only sequence",
        RUNNER_URL + "sdk.py#L162-L163",
        "_inputs",
    )
    return checks, references, evaluator["url"], locator, counts


def main():
    parser = argparse.ArgumentParser()
    for arg in ("work", "evidence", "flip-source", "mrna-source", "flip-upstream", "mrna-upstream"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    from rewirebench import sdk

    for kind in ("flip2", "mrnabench"):
        prepared_path = args.work / (kind + "-prepared") / "prepared.json"
        prepared = load(prepared_path)
        shared, refs, evaluator_url, locator, counts = source_checks(args, kind, prepared)
        rows = prepared["rows"]
        train = [r for r in rows if r["split"] == "train"]
        test = [r for r in rows if r["split"] == "test"]
        # Check the released allowlist function, never use its fit/scoring code.
        inputs = sdk._inputs(rows)
        check(
            shared,
            "adapter_input_allowlist",
            all(
                set(r) == {"id", "sequence"}
                and r["id"] == original["id"]
                and r["sequence"] == original["inputs"]["sequence"]
                for r, original in zip(inputs, rows)
            ),
            "All adapter inputs inspected: only opaque ID and sequence; no target, split or source index",
            RUNNER_URL + "sdk.py#L162-L163",
            "_inputs applied to every prepared row",
        )
        del inputs
        x_train = np.asarray([features(r["inputs"]["sequence"], kind) for r in train])
        x_test = np.asarray([features(r["inputs"]["sequence"], kind) for r in test])
        y_train = np.asarray([r["target"] for r in train])
        y_test = np.asarray([r["target"] for r in test])
        if kind == "flip2":
            scaler = StandardScaler().fit(y_train[:, None])
            head = Ridge(alpha=10, solver="auto", tol=1e-5, max_iter=1000000, fit_intercept=True)
            head.fit(x_train, scaler.transform(y_train[:, None]).ravel())
            composed = scaler.inverse_transform(head.predict(x_test)[:, None]).ravel()
            fitting_detail = "40 composition features independently constructed; ridge alpha 10 and target scaling fitted on 584 training rows only; validation and test labels unused in fitting. This is not the published one-hot baseline."
        else:
            head = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10]).fit(x_train, y_train)
            composed = head.predict(x_test)
            fitting_detail = f"6 composition features independently constructed; unnormalized RidgeCV fitted on 70011 training rows only; selected alpha {float(head.alpha_)}; validation and test labels unused in fitting. Evaluation uses held-out test, not upstream default validation."
        for suffix in ("composition", "train-mean"):
            run_id = kind + "-" + suffix
            local = args.work / run_id
            public = args.evidence / run_id
            report = load(public / "report.json")
            source = load(public / "source.json")
            checks = list(shared)
            predictions = json.loads(
                (local / "predictions.json").read_text(), object_pairs_hook=unique_object
            )
            check(
                checks,
                "prediction_coverage",
                set(predictions) == {r["id"] for r in test}
                and all(np.isfinite(v) for v in predictions.values())
                and report["coverage"]
                == {"scored": len(test), "denominator": len(test), "unscored": 0},
                f"All {len(test)} test IDs scored exactly once; no unknown IDs, missing predictions or nonfinite scores",
                evaluator_url,
                "prediction file and reported coverage",
            )
            p = np.asarray([predictions[r["id"]] for r in test])
            independent = (
                composed
                if suffix == "composition"
                else np.repeat(sum(float(v) for v in y_train) / len(y_train), len(test))
            )
            prediction_delta = float(np.max(np.abs(p - independent)))
            check(
                checks,
                "independent_train_only_refit",
                prediction_delta <= TOLERANCE,
                (
                    fitting_detail
                    if suffix == "composition"
                    else "Arithmetic mean of training labels only; no validation or test labels used in fit."
                )
                + f" Maximum absolute prediction difference {prediction_delta:.17g}; tolerance {TOLERANCE}.",
                RUNNER_URL
                + (
                    "protocols/" + kind + ".py"
                    if suffix == "composition"
                    else "adapters/sequence.py"
                ),
                "independent feature extraction and fitting in verify_sequence.py",
            )
            constant = np.ptp(p) == 0 or np.ptp(y_test) == 0
            metrics = {"spearman": None if constant else float(spearmanr(p, y_test).statistic)}
            if kind == "flip2":
                metrics.update(
                    ndcg=float(ndcg_score((y_test - y_test.min())[None, :], p[None, :])),
                    n=len(test),
                )
            else:
                metrics.update(
                    mse=float(np.mean((p - y_test) ** 2)),
                    pearson=None if constant else float(pearsonr(p, y_test).statistic),
                )
            delta = 0.0
            for key, value in metrics.items():
                observed = report["metrics"][key]
                if value is None:
                    assert observed is None, "Undefined correlation must be null"
                else:
                    delta = max(delta, abs(value - observed))
            check(
                checks,
                "independent_metrics",
                metrics.keys() == report["metrics"].keys() and delta <= TOLERANCE,
                "Pinned upstream metric expressions recomputed from saved predictions; undefined constant correlations retained as null; max absolute difference "
                + str(delta),
                evaluator_url,
                locator,
            )
            check(
                checks,
                "artifact_integrity",
                digest(local / "predictions.json")
                == report["predictions_sha256"]
                == source["local_predictions_file_sha256"]
                and digest(prepared_path) == source["local_prepared_file_sha256"]
                and digest(local / "report.json") == source["local_report_sha256"]
                and package_digest(Path(sdk.__file__).parent)
                == report["environment"]["sdk_code_sha256"],
                "Saved predictions, prepared input, original report and installed released SDK match recorded hashes",
                RUNNER_URL + "sdk.py",
                "SHA-256 receipt fields",
            )
            check(
                checks,
                "scope_and_claims",
                report["completion"] == "complete"
                and report["scope"] == "full"
                and report["protocol_results"]["suite_complete"] is False
                and report["independently_reproduced"] is False,
                "Complete selected evaluation, never whole-suite completion or published-score reproduction",
                RUNNER_URL + "protocols/" + kind + ".py",
                "report completion and scientific scope",
            )
            document = {
                "schema_version": "1.0",
                "run_id": run_id,
                "status": "passed",
                "review_method": "automated independent source reconciliation, train-only refit and metric recalculation",
                "reviewed_at": datetime.now(UTC).isoformat(),
                "reviewer": "Codex independent audit worker",
                "tolerance_absolute": TOLERANCE,
                "counts": counts,
                "prediction_max_absolute_error": prediction_delta,
                "metric_max_absolute_error": delta,
                "recomputed_metrics": metrics,
                "source_references": refs,
                "checks": checks,
                "audit_script_sha256": digest(__file__),
                "limitations": [
                    "Independent calculation uses the same installed numerical libraries as execution; it is not an independent software-stack reproduction.",
                    "Checks establish agreement for these saved inputs and controls; they do not establish absence of homology or biological overlap between source splits.",
                    "These Rewire controls are not published-model reproductions. No uncertainty interval or repeated-seed estimate was measured.",
                    "Raw data and predictions remain local; public aggregate receipts alone cannot reconstruct individual predictions.",
                ],
            }
            (public / "audit.json").write_text(
                json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            with (public / "audit.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
                writer.writeheader()
                writer.writerows(checks)
            print(
                json.dumps(
                    {
                        "run": run_id,
                        "checks": len(checks),
                        "max_prediction_error": prediction_delta,
                        "max_metric_error": delta,
                        "metrics": metrics,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
