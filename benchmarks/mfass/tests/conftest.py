"""Shared synthetic fixtures for specialist runner tests; no real data or models."""
import csv
import os
from types import SimpleNamespace

import pytest

# Modules whose tests are scientific gates for the matched specialist study. With
# MFASS_REQUIRE_GATES=1 a skip in these modules (missing gffutils or pinned Pangolin
# checkout) is reported as a failure, so a gate cannot pass by not running.
GATE_MODULES = {"test_pangolin_patch.py", "test_matched_annotation.py",
                "test_matched_study.py", "test_specialist_provenance.py"}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (os.environ.get("MFASS_REQUIRE_GATES") == "1" and report.skipped
            and item.path.name in GATE_MODULES):
        report.outcome = "failed"
        report.longrepr = f"gate test skipped with MFASS_REQUIRE_GATES=1: {report.longrepr}"


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    """Replace canonical preparation/scoring; no real biological data or model."""
    import rewirebench.protocols.mfass as protocol

    cohort, split = tmp_path / "cohort.tsv", tmp_path / "split.tsv"
    rows = [{"id": str(i), "chr": "chr1", "snp_position_hg38_1based": str(i + 1),
             "ref_allele": "A", "alt_allele": "G", "sdv": str(i % 2)} for i in range(3)]
    for path, contents in (
        (cohort, rows),
        (split, [{"id": r["id"], "group": r["id"], "split": "test"} for r in rows]),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=contents[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(contents)
    monkeypatch.setattr(protocol, "prepare", lambda *a, **kw: {
        "scope": "smoke", "provenance": {"cohort_sha256": "fixture", "split_sha256": "fixture"},
    })
    monkeypatch.setattr(protocol, "score", lambda dataset, scores: {
        "metrics": {}, "coverage": {"scored": len(scores), "unscored": 8324 - len(scores),
                                      "denominator": 8324},
    })
    reference, annotation = tmp_path / "ref.fa", tmp_path / "annotation.db"
    reference.write_text(">chr1\nAAA\n")
    annotation.write_text("synthetic annotation")
    output = tmp_path / "result.json"
    args = ["--cohort", str(cohort), "--split", str(split), "--ref", str(reference),
            "--limit", "3", "--out", str(output)]
    return SimpleNamespace(args=args, reference=reference, annotation=annotation, output=output)
