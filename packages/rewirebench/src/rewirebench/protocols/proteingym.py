"""Pinned ProteinGym v1.3 DMS substitutions, with explicit incomplete coverage.

Metrics and category-balanced aggregation follow upstream performance_DMS_benchmarks.py
at 144fe22b07dfaeec2b366f2346203a9838a55b4c, lines 10–75, 199–210, 261, 283–295.
Copyright (c) 2023 OATML-Markslab et al.; upstream MIT notice in packaged resources.
"""
from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import defaultdict
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, roc_auc_score

PROTOCOL_ID = "proteingym-v1.3-dms-substitutions"
UPSTREAM_REVISION = "144fe22b07dfaeec2b366f2346203a9838a55b4c"
REFERENCE_SHA256 = "a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308"
METRICS = ("Spearman", "AUC", "MCC", "NDCG", "Top_recall")
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def resource_path(name: str) -> Path:
    packaged = Path(str(files("rewirebench").joinpath("resources", "proteingym", name)))
    return packaged


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _substitutions(mutant: str, sequence: str) -> list[tuple[int, str, str]]:
    result = []
    seen = set()
    for part in mutant.split(":"):
        match = re.fullmatch(r"([ACDEFGHIKLMNPQRSTVWY])([1-9]\d*)([ACDEFGHIKLMNPQRSTVWY])", part)
        if not match:
            raise ValueError(f"Not a substitution: {part!r}")
        wt, position, alt = match.groups()
        index = int(position) - 1
        if index < 0 or index >= len(sequence) or sequence[index] != wt or index in seen:
            raise ValueError(f"Mutation does not match target sequence or repeats a position: {part}")
        if wt == alt:
            raise ValueError(f"No-op mutation is not a substitution: {part}")
        seen.add(index)
        result.append((index, wt, alt))
    return result


def prepare(source: Path, **options) -> dict:
    """Read local extracted official assays, without network access.

    reference must match the shipped, pinned upstream reference. assay_ids selects
    complete assays; limit deliberately creates a smoke subset. expected_hashes can
    verify pre-existing user pins, but is not evidence of official data provenance.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError("source must be an extracted DMS substitutions directory")
    reference = Path(options.get("reference") or resource_path("DMS_substitutions.csv"))
    if _sha(reference) != REFERENCE_SHA256:
        raise ValueError("Reference metadata must match the pinned ProteinGym revision")
    with reference.open(newline="") as handle:
        refs = list(csv.DictReader(handle))
    by_id = {r["DMS_id"]: r for r in refs}
    selected = options.get("assay_ids")
    if isinstance(selected, str):
        selected = selected.split(",")
    selected = list(selected) if selected is not None else list(by_id)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(by_id):
        raise ValueError("Choose unique assay IDs from the pinned reference")
    limit = options.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    expected_hashes = options.get("expected_hashes") or {}
    if set(expected_hashes) - set(selected):
        raise ValueError("Expected hashes contain unselected or unknown assays")
    rows, assay_metadata, hashes = [], {}, {}
    for assay_id in selected:
        ref = by_id[assay_id]
        path = (source / ref["DMS_filename"]).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Assay path escapes source directory")
        if not path.is_file():
            raise ValueError(f"Missing assay {assay_id}; select assay_ids for an explicit subset")
        actual_hash = _sha(path)
        if assay_id in expected_hashes and expected_hashes[assay_id] != actual_hash:
            raise ValueError(f"Assay hash mismatch: {assay_id}")
        hashes[assay_id] = actual_hash
        with path.open(newline="") as handle:
            assay_rows = list(csv.DictReader(handle))
        required = {"mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"}
        if not assay_rows or not required.issubset(assay_rows[0]):
            raise ValueError(f"Missing required columns or empty assay: {assay_id}")
        expected_count = int(ref["DMS_total_number_mutants"])
        if len(assay_rows) > expected_count:
            raise ValueError(f"Too many rows for {assay_id}: expected {expected_count}")
        if len(assay_rows) != expected_count and limit is None:
            raise ValueError(f"Assay {assay_id} has {len(assay_rows)} rows; expected {expected_count}; use limit for smoke input")
        sequence = ref["target_seq"]
        seen = set()
        for row in assay_rows:
            mutant = row["mutant"]
            if mutant in seen:
                raise ValueError(f"Duplicate mutant in {assay_id}: {mutant}")
            seen.add(mutant)
            substitutions = _substitutions(mutant, sequence)
            mutated = list(sequence)
            for index, _, alt in substitutions:
                mutated[index] = alt
            if row["mutated_sequence"] != "".join(mutated):
                raise ValueError(f"Mutated sequence disagrees with substitution: {assay_id}/{mutant}")
            target = float(row["DMS_score"])
            binary = float(row["DMS_score_bin"])
            if not math.isfinite(target) or binary not in (0, 1):
                raise ValueError("DMS labels must be finite, with binary labels 0 or 1")
        selected_rows = assay_rows if limit is None else assay_rows[:limit]
        for row in selected_rows:
            rows.append({
                "id": f"{assay_id}::{row['mutant']}", "split": "test",
                "inputs": {"assay_id": assay_id, "wild_type_sequence": sequence,
                           "mutant": row["mutant"], "mutated_sequence": row["mutated_sequence"]},
                "target": float(row["DMS_score"]), "target_binary": int(float(row["DMS_score_bin"])),
                "group": ref["UniProt_ID"], "assay_id": assay_id,
                "selection_type": ref["coarse_selection_type"],
            })
        assay_metadata[assay_id] = {
            "UniProt_ID": ref["UniProt_ID"], "selection_type": ref["coarse_selection_type"],
            "expected_count": expected_count, "prepared_count": len(selected_rows),
            "taxon": ref["taxon"], "MSA_Neff_L_category": ref["MSA_Neff_L_category"],
        }
    scope = "smoke" if limit is not None else ("full" if set(selected) == set(by_id) else "subset")
    return {
        "protocol_id": PROTOCOL_ID, "protocol_version": "1.3", "dataset_id": "proteingym-dms-substitutions-v1.3",
        "scope": scope, "rows": rows,
        "provenance": {"upstream_revision": UPSTREAM_REVISION,
                       "reference_sha256": REFERENCE_SHA256, "assay_sha256": hashes,
                       "data_verification": "local_bytes_hashed_not_independently_source_verified",
                       "expected_hashes_checked": sorted(expected_hashes.keys() & hashes.keys()),
                       "source_url": f"https://github.com/OATML-Markslab/ProteinGym/tree/{UPSTREAM_REVISION}"},
        "metadata": {"assays": assay_metadata, "official_assay_count": len(refs),
                     "official_variant_count": sum(int(r["DMS_total_number_mutants"]) for r in refs),
                     "score_direction": "higher_is_better", "binary_labels": "official_DMS_score_bin",
                     "aggregation": "round_assay_3dp_then_mean_by_UniProt_and_selection_then_equal_category_mean"},
    }


def _ndcg(truth: np.ndarray, prediction: np.ndarray) -> float:
    # Preserve upstream tie handling and top floor(n * .1), including small-sample 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        gains = (truth - np.min(truth)) / (np.max(truth) - np.min(truth))
    k = int(np.floor(len(truth) * .1))
    ranks = np.argsort(np.argsort(-prediction)) + 1
    keep = (ranks <= k) & (gains != 0)
    if not np.any(keep):
        return 0.0
    dcg = np.sum([g / np.log2(r + 1) for r, g in zip(ranks[keep], gains[keep])])
    ideal = np.argsort(np.argsort(-gains)) + 1
    keep = (ideal <= k) & (gains != 0)
    idcg = np.sum([g / np.log2(r + 1) for r, g in zip(ideal[keep], gains[keep])])
    return float(dcg / idcg) if idcg else float("nan")


def _metrics(truth: np.ndarray, prediction: np.ndarray, binary: np.ndarray) -> dict:
    true_top = truth >= np.percentile(truth, 90)
    pred_top = prediction >= np.percentile(prediction, 90)
    values = {
        "Spearman": float(spearmanr(truth, prediction).statistic),
        "AUC": float(roc_auc_score(binary, prediction)) if len(set(binary)) == 2 else float("nan"),
        "MCC": float(matthews_corrcoef(binary, prediction >= np.median(prediction))),
        "NDCG": _ndcg(truth, prediction),
        "Top_recall": float(np.sum(true_top & pred_top) / np.sum(true_top)),
    }
    return {name: (float(np.round(value, 3)) if math.isfinite(value) else None)
            for name, value in values.items()}


def _mean(values) -> float | None:
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def score(dataset: dict, predictions: dict[str, float]) -> dict:
    """Score keyed fitness predictions. No complete-track score for partial inputs."""
    if dataset.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Wrong protocol for ProteinGym scorer")
    rows = dataset["rows"]
    known = {r["id"] for r in rows}
    if len(known) != len(rows) or set(predictions) - known:
        raise ValueError("Duplicate dataset IDs or unknown prediction IDs")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in predictions.values()):
        raise ValueError("Predictions must be finite scalar numbers")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["assay_id"]].append(row)
    per_assay, grouped_metrics = {}, defaultdict(list)
    for assay_id, assay in grouped.items():
        scored = [r for r in assay if r["id"] in predictions]
        meta = dataset["metadata"]["assays"][assay_id]
        expected = meta["expected_count"]
        metrics = {name: None for name in METRICS}
        if scored:
            metrics = _metrics(np.array([r["target"] for r in scored]),
                               np.array([predictions[r["id"]] for r in scored]),
                               np.array([r["target_binary"] for r in scored]))
        complete = len(scored) == expected and len(assay) == expected
        per_assay[assay_id] = {"metrics": metrics, "scored": len(scored),
                              "eligible": expected, "prepared": len(assay),
                              "status": "complete" if complete else "partial",
                              "UniProt_ID": meta["UniProt_ID"], "selection_type": meta["selection_type"]}
        if complete:
            grouped_metrics[(meta["UniProt_ID"], meta["selection_type"])].append(metrics)
    by_protein_category = {
        f"{protein}::{category}": {name: _mean(v[name] for v in values) for name in METRICS}
        for (protein, category), values in grouped_metrics.items()
    }
    by_category = defaultdict(list)
    for (protein, category) in grouped_metrics:
        by_category[category].append(by_protein_category[f"{protein}::{category}"])
    categories = {cat: {name: _mean(v[name] for v in values) for name in METRICS}
                  for cat, values in by_category.items()}
    complete_track = (dataset.get("scope") == "full"
                      and len(per_assay) == dataset["metadata"]["official_assay_count"]
                      and all(v["status"] == "complete" for v in per_assay.values()))
    balanced_raw = {name: _mean(v[name] for v in categories.values()) for name in METRICS}
    balanced = {name: float(np.round(value, 3)) if value is not None else None
                for name, value in balanced_raw.items()}
    return {
        "metrics": balanced if complete_track else {}, "per_assay": per_assay,
        "by_protein_selection_type": by_protein_category, "by_selection_type": categories,
        "status": "complete_track" if complete_track else "partial_track",
        "aggregation": dataset["metadata"]["aggregation"],
        "complete_assays": sum(v["status"] == "complete" for v in per_assay.values()),
        "total_assays": dataset["metadata"]["official_assay_count"],
        "data_verification": dataset.get("provenance", {}).get("data_verification", "unreported"),
        "uncertainty": "not_estimated; upstream cross-model bootstrap differences are not an absolute score CI",
        "coverage": {"denominator": sum(v["expected_count"] for v in dataset["metadata"]["assays"].values()),
                     "scored": len(predictions), "prepared": len(rows),
                     "eligible": sum(v["expected_count"] for v in dataset["metadata"]["assays"].values())},
    }
