"""Integrity of the published issue #19 matched-annotation study bundle.

No inference. Checks the checksums, that no private path remains, and that each
condition's coverage and metrics and each contrast's paired population recompute
exactly from the published prediction tables.
"""
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import pytest
from rewirebench.metrics import point_metrics

BUNDLE = Path(__file__).resolve().parents[1] / "results" / "matched-annotation-v1"
CONDITIONS = ("S0", "S1", "P0", "P1")
CONTRASTS = {"S1-S0": ("S1", "S0"), "P1-P0": ("P1", "P0"), "P0-S0": ("P0", "S0")}
METRICS = ("n", "positives", "precision_at_capacity", "recall_at_capacity",
           "average_precision_sklearn", "auroc")


def _predictions(condition):
    [path] = (BUNDLE / condition).glob("*.predictions.tsv")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _result(condition):
    [path] = (BUNDLE / condition).glob("*.json")
    return json.loads(path.read_text())


def test_checksums_and_no_private_paths():
    lines = (BUNDLE / "SHA256SUMS").read_text().splitlines()
    listed = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        listed[name] = digest
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == digest, name
    files = {str(p.relative_to(BUNDLE)) for p in BUNDLE.rglob("*") if p.is_file()}
    assert files - {"SHA256SUMS"} == set(listed)
    for name in listed:
        text = (BUNDLE / name).read_text(errors="replace")
        assert not any(m in text for m in ("/Volumes/", "/Users/", "/private/", "/home/")), name
        # Cohort alleles are withheld; raw checkpoints (which carry them) are not published.
        assert not re.search(r"variant file \(ref base: [ACGTNacgtn]", text), name
        assert not re.search(r"[ACGTN]+\|ENSG\d", text), name  # SpliceAI raw ALLELE|gene
    assert not any(name.endswith("checkpoint.tsv") for name in listed)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_condition_rows_coverage_and_metrics_recompute(condition):
    rows, result = _predictions(condition), _result(condition)
    assert len(rows) == 8324 and len({r["id"] for r in rows}) == 8324
    scored = [r for r in rows if r["score"] != ""]
    assert result["coverage"] == {"scored": len(scored), "unscored": 8324 - len(scored),
                                  "denominator": 8324}
    recomputed = point_metrics([int(r["label"]) for r in scored],
                               [float(r["score"]) for r in scored], 100)
    for key in METRICS:
        assert math.isclose(recomputed[key], result["metrics"][key], rel_tol=0, abs_tol=1e-12)
    table = result["config"]["output_tables"]["predictions"]["sha256"]
    [path] = (BUNDLE / condition).glob("*.predictions.tsv")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == table


@pytest.mark.parametrize("name", CONTRASTS)
def test_contrast_population_matches_published_predictions(name):
    candidate, baseline = CONTRASTS[name]
    report = json.loads((BUNDLE / "contrasts" / f"{name}.json").read_text())
    scored = {c: {r["id"]: r for r in _predictions(c) if r["score"] != ""}
              for c in (candidate, baseline)}
    common = sorted(set(scored[candidate]) & set(scored[baseline]))
    d = report["denominators"]
    assert d["common"] == len(common)
    assert d["common_id_sha256"] == hashlib.sha256("\n".join(common).encode()).hexdigest()
    assert d["common_positives"] == sum(int(scored[baseline][i]["label"]) for i in common)
    for side, condition in (("baseline", baseline), ("candidate", candidate)):
        [path] = (BUNDLE / condition).glob("*.predictions.tsv")
        assert report[f"{side}_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["seed"] == 20260914
    assert all(v.get("draws") == 2000 for v in report["paired"].values())
