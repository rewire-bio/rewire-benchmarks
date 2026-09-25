"""Compare prepared EVCouplings independent fields and scores with an upstream receipt.

Shared by the tests and ``evcouplings_runtime_diagnostic.py``. It only compares
two executions: expected scores always come from the upstream receipt, never
from a formula here. For every site it reports:

- raw field differences, and their split into the site's mean difference (a
  common offset) and the largest remaining difference after removing it;
- every wild-type-relative contrast, h(a) - h(wt) for all alphabet symbols a,
  which determines every possible substitution score at that site;
- both executions' BFGS warning flags when the receipt records them.

Scores are compared for every mutant in the receipt. Tolerances are
math.isclose(rel_tol=1e-6, abs_tol=1e-6), matching the parity tests.

Coverage is checked before any number is compared, so a receipt or artifact
cannot silently shrink the comparison. Assays must equal the fixture's; field
sites must equal the model index list, mapped artifact positions and optimizer
sites exactly; every row must have one value per alphabet symbol; receipt scores
must cover exactly the fixture mutants; and warning flags must be recorded for
exactly the model sites. Any gap or extra key raises ``CoverageError``.
"""
import math

import numpy as np

TOLERANCE = 1e-6


class CoverageError(ValueError):
    """A receipt or artifact does not cover exactly the sites being compared."""


def close(a, b):
    return math.isclose(a, b, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def check_coverage(artifact_assays, receipt_assays, fixture, predictions):
    """Raise CoverageError unless both sides cover exactly the same sites and keys."""
    problems = []
    if set(receipt_assays) != set(fixture) or not set(fixture) <= set(artifact_assays) \
            or set(predictions) != set(fixture):
        problems.append(f"assays: fixture {sorted(fixture)}, receipt {sorted(receipt_assays)}, "
                        f"artifact {sorted(artifact_assays)}, predictions {sorted(predictions)}")
    for assay in sorted(set(fixture) & set(receipt_assays) & set(artifact_assays)):
        mine, upstream, spec = artifact_assays[assay], receipt_assays[assay], fixture[assay]
        indices = {str(m) for m in mine["model_metadata"]["index_list"]}
        positions = {str(m + spec["msa_start"] - 1) for m in mine["model_metadata"]["index_list"]}
        q = len(mine["alphabet"])
        checks = {
            "receipt field sites": (set(upstream.get("independent_fields", {})), indices),
            "receipt warning flags": (set(upstream.get("bfgs_warnflags", {})), indices),
            "artifact field positions": (set(mine["fields"]), positions),
            "artifact optimizer sites": (set(mine["optimizer"]["sites"]), positions),
            "receipt scores": (set(upstream.get("prediction_independent", {})), set(spec["scored"])),
            "adapter scores": (set(predictions.get(assay, {})), set(spec["scored"])),
        }
        for label, (found, expected) in checks.items():
            if found != expected:
                problems.append(f"{assay} {label}: missing {sorted(expected - found)}, extra {sorted(found - expected)}")
        rows = [*upstream.get("independent_fields", {}).values(), *mine["fields"].values()]
        if any(not isinstance(row, list) or len(row) != q for row in rows):
            problems.append(f"{assay}: a field row does not have one value per alphabet symbol ({q})")
        flags = upstream.get("bfgs_warnflags", {}).values()
        if any(type(flag) is not int for flag in flags):
            problems.append(f"{assay}: receipt warning flags must be integers")
    if problems:
        raise CoverageError("; ".join(problems))


def compare(artifact_assays, receipt_assays, fixture, predictions):
    """Return a JSON-serialisable report and a list of failed checks.

    Raises CoverageError first if either side does not cover exactly the
    expected assays, sites, alphabet entries, warning flags and mutants.

    ``artifact_assays`` is a prepared artifact's ``assays``; ``receipt_assays``
    an upstream receipt's ``assays``; ``fixture`` the fixture spec;
    ``predictions`` maps assay to {mutant: adapter score}.
    """
    check_coverage(artifact_assays, receipt_assays, fixture, predictions)
    report, failures = {}, []
    for assay, spec in fixture.items():
        mine, upstream = artifact_assays[assay], receipt_assays[assay]
        alphabet = mine["alphabet"]
        if alphabet != upstream["alphabet"]:
            failures.append(f"{assay}: alphabet differs")
            continue
        meta = mine["model_metadata"]
        wild_type = dict(zip(meta["index_list"], meta["target_seq"]))
        sites = {}
        for index, row in upstream["independent_fields"].items():
            position = str(int(index) + spec["msa_start"] - 1)
            a, b = np.asarray(mine["fields"][position]), np.asarray(row)
            difference = a - b
            wt = alphabet.index(wild_type[int(index)])
            contrast_a, contrast_b = a - a[wt], b - b[wt]
            raw_fail = [alphabet[k] for k in range(len(a)) if not close(a[k], b[k])]
            contrast_fail = [alphabet[k] for k in range(len(a)) if not close(contrast_a[k], contrast_b[k])]
            site = {
                "raw_max_abs_difference": float(np.abs(difference).max()),
                "mean_difference": float(difference.mean()),
                "max_abs_difference_after_mean": float(np.abs(difference - difference.mean()).max()),
                "contrast_max_abs_difference": float(np.abs(contrast_a - contrast_b).max()),
                "raw_symbols_outside_tolerance": raw_fail,
                "contrast_symbols_outside_tolerance": contrast_fail,
                "adapter_warnflag": mine["optimizer"]["sites"][position]["warnflag"],
                "adapter_stationarity_residual": mine["optimizer"]["sites"][position]["stationarity_residual"],
                "upstream_warnflag": upstream["bfgs_warnflags"][index],
            }
            sites[position] = site
            if raw_fail:
                failures.append(f"{assay} position {position}: raw fields outside tolerance for {''.join(raw_fail)}")
            if contrast_fail:
                failures.append(f"{assay} position {position}: WT contrasts outside tolerance for {''.join(contrast_fail)}")
        scores = {}
        for mutant, expected in upstream["prediction_independent"].items():
            observed = predictions[assay][mutant]
            scores[mutant] = {"adapter": observed, "upstream": expected,
                              "abs_difference": abs(observed - expected), "within_tolerance": close(observed, expected)}
            if not close(observed, expected):
                failures.append(f"{assay} {mutant}: score {observed!r} vs upstream {expected!r}")
        report[assay] = {"sites": sites, "scores": scores}
    return report, failures


def warnflag_mismatches(report):
    return [f"{assay} position {position}: adapter warnflag {site['adapter_warnflag']}, "
            f"upstream {site['upstream_warnflag']}"
            for assay, entry in report.items() for position, site in entry["sites"].items()
            if site["adapter_warnflag"] != site["upstream_warnflag"]]
