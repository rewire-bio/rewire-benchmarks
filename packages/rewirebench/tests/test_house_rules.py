"""The house rules must be enforced, not documented."""
import numpy as np
import pytest

from rewirebench import (
    BenchmarkResult, connected_components, naive_key_count,
    paired_group_bootstrap, precision_at_n,
)


def _ok(**over):
    base = dict(
        benchmark="x-v1", method="m", family="trivial baseline", description="d",
        split="s", metrics={"auroc": 0.5},
        coverage={"scored": 10, "unscored": 0, "denominator": 10},
        timing_seconds={"total": 1.0}, independent_groups=50,
    )
    base.update(over)
    return BenchmarkResult(**base)


def test_coverage_must_reconcile():
    r = _ok(coverage={"scored": 9, "unscored": 0, "denominator": 10})
    with pytest.raises(ValueError, match="does not reconcile"):
        r.validate()


def test_timing_required():
    with pytest.raises(ValueError, match="timing_seconds"):
        _ok(timing_seconds={}).validate()


def test_pretrained_requires_contamination_statement():
    with pytest.raises(ValueError, match="contamination"):
        _ok(pretrained=True, config={"revision": "abc"}).validate()


def test_pretrained_requires_config():
    with pytest.raises(ValueError, match="config is required"):
        _ok(pretrained=True, contamination="unknown").validate()


def test_pretrained_passes_when_declared():
    _ok(pretrained=True, contamination="unknown", config={"revision": "abc"}).validate()


def test_too_few_groups_refused():
    with pytest.raises(ValueError, match="independent_groups"):
        _ok(independent_groups=1).validate()


def test_connected_components_merge_transitively():
    recs = [
        {"id": "v1", "patient": "P1", "gene": "BRCA1"},
        {"id": "v2", "patient": "P2", "gene": "BRCA1"},
        {"id": "v3", "patient": "P2", "gene": "TP53"},
        {"id": "v4", "patient": "P3", "gene": "TP53"},
        {"id": "v5", "patient": "P4", "gene": "MLH1"},
    ]
    groups = connected_components(recs, ["patient", "gene"])
    assert len(set(groups.values())) == 2
    # the concatenated key claims five independent units; there are two
    assert naive_key_count(recs, ["patient", "gene"]) == 5


def test_bootstrap_refuses_when_groups_are_class_pure():
    labels = np.array([1] * 5 + [0] * 5)
    groups = np.array([0] * 5 + [1] * 5)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="single-class"):
        paired_group_bootstrap(
            labels, rng.normal(size=10), rng.normal(size=10), groups,
            capacity=3, n=500,
        )


def test_precision_at_n_is_tie_robust():
    labels = np.array([1] * 5 + [0] * 95)
    flat = np.zeros(100)          # every score identical
    vals = {precision_at_n(labels, flat, 10, np.random.default_rng(s)) for s in range(20)}
    assert len(vals) > 1, "identical scores must not yield a deterministic top-n"
    assert all(0.0 <= v <= 1.0 for v in vals)
