"""Unit tests and result integrity checks for ClinGen PP3/BP4 local posterior calibration."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np

from clingen_pp3_bp4 import calibrate, extract_cohorts

ROOT = Path(__file__).resolve().parents[1]


def test_posterior_cutoffs_match_tavtigian_pejaver():
    post_p, post_b = calibrate.posterior_cutoffs()
    assert math.isclose(post_p["VeryStrong_8"], 1124 * 0.0441 / (1123 * 0.0441 + 1), rel_tol=1e-12)
    for key, pts, approx in (("Supporting_1", 1, 0.100), ("Moderate_2", 2, 0.211), ("Strong_4", 4, 0.607)):
        lr = 1124.0 ** (pts / 8.0)
        expected = (lr * 0.0441) / ((lr - 1.0) * 0.0441 + 1.0)
        assert math.isclose(post_p[key], expected, rel_tol=1e-12)
        assert math.isclose(post_p[key], approx, abs_tol=1e-3)
    assert post_b["Strong_4"] > post_b["Moderate_2"] > post_b["Supporting_1"] > 0.9


def test_fast_binary_search_matches_pejaver_reference_loop():
    rng = np.random.default_rng(42)
    pos = np.clip(rng.normal(0.75, 0.2, 400), 0.0, 1.0)
    neg = np.clip(rng.normal(0.25, 0.2, 500), 0.0, 1.0)
    g = np.clip(rng.uniform(0.0, 1.0, 600), 0.0, 1.0)
    thrs = np.round(np.linspace(1.0, 0.0, 101), 2)

    ref = calibrate.compute_local_posteriors_reference(pos, neg, g, thrs, increment=0.01)
    fast = calibrate.compute_local_posteriors_fast(pos, neg, g, thrs, increment=0.01)
    assert np.allclose(fast, ref, atol=1e-12)


def test_discounted_thresholds_direction_and_invalid_cap():
    vals_p = np.linspace(0.5, 0.99, 100)
    # Conservative pathogenic bound is near 95th percentile (top ~5th largest value)
    d_p = calibrate.discount_bootstrap_thresholds(vals_p, benign=False)
    assert math.isclose(d_p, float(np.sort(vals_p)[::-1][5]), abs_tol=1e-12)
    # Conservative benign bound is near 5th percentile (5th smallest value)
    d_b = calibrate.discount_bootstrap_thresholds(vals_p, benign=True)
    assert math.isclose(d_b, float(np.sort(vals_p)[5]), abs_tol=1e-12)
    # >5% NaNs yields NaN
    too_many_nans = vals_p.copy()
    too_many_nans[:6] = math.nan
    assert math.isnan(calibrate.discount_bootstrap_thresholds(too_many_nans, benign=False))


def test_calibration_monotonicity_on_separable_synthetic_data():
    rng = np.random.default_rng(7)
    pos = np.clip(rng.beta(5, 1.5, 500), 0.0, 1.0)
    neg = np.clip(rng.beta(1.5, 5, 500), 0.0, 1.0)
    g = rng.uniform(0.0, 1.0, 800)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        res = calibrate.calibrate_predictor(pos, neg, g, increment=0.01, decimals=2, n_boot=100, seed=7)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert res["auroc"] > 0.90
    pp3 = res["discounted_thresholds"]["PP3"]
    bp4 = res["discounted_thresholds"]["BP4"]
    assert pp3["Supporting_1"] is not None
    assert bp4["Supporting_1"] is not None
    assert pp3["Supporting_1"] > bp4["Supporting_1"]
    if pp3["Moderate_2"] is not None:
        assert pp3["Moderate_2"] >= pp3["Supporting_1"]
    if pp3["Strong_4"] is not None:
        assert pp3["Strong_4"] >= pp3["Moderate_2"]
    encoded = json.dumps(res, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_sparse_calibration_emits_strict_json_nulls_and_no_warnings():
    rng = np.random.default_rng(11)
    pos = np.clip(rng.beta(2.0, 2.0, 160), 0.0, 1.0)
    neg = np.clip(rng.beta(1.6, 2.4, 160), 0.0, 1.0)
    g = np.array([], dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        res = calibrate.calibrate_predictor(pos, neg, g, increment=0.05, decimals=2, n_boot=50, seed=11)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert res["discounted_thresholds"]["PP3"]["VeryStrong_8"] is None
    encoded = json.dumps(res, allow_nan=False)
    decoded = json.loads(
        encoded,
        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Non-RFC JSON constant: {c}")),
    )
    assert decoded["point_thresholds"]["PP3"]["VeryStrong_8"] is None


def test_missense_filter_excludes_nonsense_and_splice():
    assert extract_cohorts.is_clean_missense("SO:0001583|missense_variant")
    assert not extract_cohorts.is_clean_missense("SO:0001583|missense_variant,SO:0001575|splice_donor_variant")
    assert not extract_cohorts.is_clean_missense("SO:0001587|nonsense")
    assert extract_cohorts.has_prior_pb_classification("Pathogenic/Likely_pathogenic")
    assert not extract_cohorts.has_prior_pb_classification("Uncertain_significance")


def test_cohorts_disjoint_if_already_extracted():
    p23 = ROOT / "cohorts/variants_post_2023_12.tsv"
    p_ref = ROOT / "cohorts/gnomad_ref_missense.tsv"
    assert p23.exists() and p_ref.exists()
    with p23.open() as h:
        k23 = {(r["chrom"], r["pos"], r["ref"], r["alt"]) for r in csv.DictReader(h, delimiter="\t")}
    with p_ref.open() as h:
        kref = {(r["chrom"], r["pos"], r["ref"], r["alt"]) for r in csv.DictReader(h, delimiter="\t")}
    assert k23.isdisjoint(kref)


def test_committed_results_strict_json_and_checksums():
    results_dir = ROOT / "results"
    for name in ("calibration_summary.json", "receipt.json"):
        raw = (results_dir / name).read_text()
        json.loads(
            raw,
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"{name} contains non-RFC constant {c}")),
        )
    sums_lines = [line for line in (results_dir / "SHA256SUMS").read_text().splitlines() if line.strip()]
    assert len(sums_lines) == 4
    for line in sums_lines:
        expected_sha, fname = line.split("  ", 1)
        actual_sha = hashlib.sha256((results_dir / fname).read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"Checksum mismatch for {fname}"
