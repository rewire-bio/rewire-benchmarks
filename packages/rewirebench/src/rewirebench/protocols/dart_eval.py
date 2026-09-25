"""DART-Eval task 1 paired regulatory-element zero-shot evaluation.

The evaluator owns pair identities. Adapters see independent opaque sequence IDs.
A hash supplied by a caller verifies local integrity, not Synapse authenticity.
"""
from __future__ import annotations

import hashlib
import json
import numbers
import secrets
from collections import Counter
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

PROTOCOL_ID = "dart-eval-task1-zero-shot-v1"
PROTOCOL_VERSION = "1"
UPSTREAM_REVISION = "af2a86d666c35304257c2fa7e15180e1fbcabb01"
DATASET_ID = "dart-eval-task1-encode-ccre"
INPUT_CONTRACT = "opaque-independent-sequences-v1"
CHROMOSOMES = ("chr5", "chr10", "chr14", "chr18", "chr20", "chr22")
CAPABILITIES = {"prediction_types": ["scalar"], "allow_fit": False,
                "allow_validation": False, "embedding_kind": None}
METRICS = ("acc", "pval", "signed_rank_sum", "mean_diff", "q05_diff", "q25_diff",
           "median_diff", "q75_diff", "q95_diff")


def _resource(name):
    return files("rewirebench").joinpath("resources", "dart_eval", name)


def describe():
    return {
        "protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION,
        "capabilities": CAPABILITIES, "datasets": [DATASET_ID],
        "splits": ["test"], "targets": ["regulatory element above paired control"],
        "metrics": list(METRICS), "inputs": ["sequence"],
        "primary_metric": "acc",
        "metric_directions": {key: "higher" if key == "acc" else "not_applicable" for key in METRICS},
        "statistic_interpretation": "pval is inferential; ranks and score differences are descriptive, not cross-model quality rankings",
        "requirements": ["numpy", "scipy", "h5py for upstream data.h5"],
        "statistical_procedure": "Frozen scipy1.12 Wilcoxon auto/greater/wilcox semantics",
        "source_revision": UPSTREAM_REVISION,
        "data_access": "Authenticated Synapse syn64314109 version 1 download. Local file SHA256 required.",
        "limitations": "Local HDF5 content has no independently verified source hash; runs remain subset scope. Demo is synthetic.",
    }


def submission_contract(dataset_id):
    if dataset_id != DATASET_ID:
        raise ValueError(f"Unsupported DART-Eval dataset {dataset_id!r}")
    return {"metrics": list(METRICS), "protocol_version": PROTOCOL_VERSION}


def _hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sequence(onehot):
    onehot = np.asarray(onehot)
    if onehot.shape != (350, 4) or not np.isin(onehot, (0, 1)).all():
        raise ValueError("DART input must be a binary 350 by 4 one-hot array")
    total = onehot.sum(axis=1)
    if (total > 1).any():
        raise ValueError("DART sequence has ambiguous multi-hot bases")
    chars = np.asarray(list("ACGT"))[onehot.argmax(axis=1)]
    chars[total == 0] = "N"
    return "".join(chars)


def _read_h5(path, limit):
    try:
        import h5py
    except ImportError as exc:
        raise ValueError("Install rewirebench[sequence] to read upstream data.h5") from exc
    with h5py.File(path, "r") as handle:
        if "test" not in handle or not isinstance(handle.get("test", getlink=True), h5py.HardLink):
            raise ValueError("Expected an internal test HDF5 group")
        if not isinstance(handle["test"], h5py.Group) or any(k not in handle["test"] for k in ("seqs", "ctrls", "idxs")):
            raise ValueError("Expected upstream data.h5 test/{seqs,ctrls,idxs}")
        group = handle["test"]
        # Do not follow arbitrary HDF5 external links while loading local data.
        for name in ("seqs", "ctrls", "idxs"):
            if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                raise ValueError("DART HDF5 external or symbolic links are not allowed")  # noqa: TRY004 - public validation API
            if not isinstance(group[name], h5py.Dataset) or group[name].is_virtual or group[name].external:
                raise ValueError("DART HDF5 virtual or external datasets are not allowed")
        count = len(group["idxs"])
        if count < 1 or group["seqs"].shape != (count, 350, 4) or group["ctrls"].shape != (count, 350, 4):
            raise ValueError("DART HDF5 sequence and pair dimensions do not match")
        ids = group["idxs"][:]
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(set(ids.tolist())) != count:
            raise ValueError("DART HDF5 pair indices must be unique integers")
        pairs = [{"source_index": int(ids[i]), "sequence": _sequence(group["seqs"][i]),
                  "control": _sequence(group["ctrls"][i])} for i in range(min(limit or count, count))]
        return pairs, count


def prepare(source: Path, **options):
    dataset = options.get("dataset", DATASET_ID)
    submission_contract(dataset)
    limit = options.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer number of pairs")
    demo = str(source) == "demo"
    if demo:
        raw = _resource("demo.json").read_bytes()
        index = json.loads(_resource("sources.json").read_text())
        expected = next(item["sha256"] for item in index["files"] if item["path"] == "demo.json")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Packaged DART demo checksum mismatch")
        fixture = json.loads(raw)
        pairs, count = fixture["pairs"], len(fixture["pairs"])
        pairs = pairs[:limit] if limit else pairs
        source_sha = hashlib.sha256(raw).hexdigest()
        scope, verification = "smoke", "synthetic_demo_not_biological_evidence"
    else:
        path = Path(source)
        if path.is_dir():
            path = path / "data.h5"
        source_sha = _hash_file(path)
        expected = options.get("source_sha256")
        if not isinstance(expected, str) or len(expected) != 64 or source_sha != expected.lower():
            raise ValueError("source_sha256 must match the exact local data.h5 bytes; no upstream hash is claimed")
        pairs, count = _read_h5(path, limit)
        scope = "smoke" if limit else "subset"
        verification = "local_bytes_hashed_not_independently_source_verified"
    rows = []
    for pair in pairs:
        pair_id = secrets.token_hex(32)
        for role, key, target in (("element", "sequence", 1), ("control", "control", 0)):
            rows.append({"id": secrets.token_hex(32), "split": "test",
                         "inputs": {"sequence": pair[key]}, "target": target,
                         "pair_id": pair_id, "pair_role": role,
                         "source_index": pair["source_index"]})
    rows.sort(key=lambda row: row["id"])
    data = {
        "protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION,
        "dataset_id": DATASET_ID, "scope": scope, "rows": rows,
        "metadata": {"adapter_input_contract": INPUT_CONTRACT,
                     "canonical_test_count": count * 2, "canonical_pair_count": count,
                     "canonical_count_origin": "synthetic fixture" if demo else "local HDF5 test group; not independently checked",
                     "chromosomes": list(CHROMOSOMES), "shuffle_seed": 0,
                     "score_semantics": "higher scalar sequence likelihood; element minus control; ties count as incorrect",
                     "metric": "acc", "metric_direction": "higher", "task": "zero_shot_paired_ranking",
                     "source_reuse_terms": "Unreported; consult source Synapse and ENCODE access and reuse terms",
                     "smoke_limit_pairs": limit},
        "provenance": {"upstream_revision": UPSTREAM_REVISION, "source_sha256": source_sha,
                       "source_url": "https://www.synapse.org/Synapse:syn64314109.1",
                       "data_verification": verification,
                       "split_origin": "Upstream test group; chr5,chr10,chr14,chr18,chr20,chr22 and seed0 expected, not inferred from arrays",
                       "reproduction_claim": "local_evaluation_not_paper_reproduction"},
    }
    validate_prepared(data)
    return data


def validate_prepared(data):
    if data.get("protocol_id") != PROTOCOL_ID or data.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported DART-Eval protocol version")
    submission_contract(data.get("dataset_id"))
    metadata = data.get("metadata", {})
    if metadata.get("adapter_input_contract") != INPUT_CONTRACT:
        raise ValueError("DART adapters require opaque independent sequences")
    if metadata.get("chromosomes") != list(CHROMOSOMES) or metadata.get("shuffle_seed") != 0:
        raise ValueError("DART zero-shot chromosome/seed contract mismatch")
    if data.get("scope") not in {"smoke", "subset"}:
        raise ValueError("Unverified local DART data cannot claim full official scope")
    rows = data.get("rows", [])
    ids = [row.get("id", "") for row in rows]
    if not ids or len(set(ids)) != len(ids) or ids != sorted(ids) or any(
        len(i) != 64 or any(c not in "0123456789abcdef" for c in i) for i in ids
    ):
        raise ValueError("DART input IDs must be unique opaque hex IDs in independent order")
    pairs = {}
    for row in rows:
        sequence = row.get("inputs", {}).get("sequence", "")
        if row.get("split") != "test" or set(row.get("inputs", {})) != {"sequence"}:
            raise ValueError("DART adapters receive only test sequences")
        if len(sequence) != 350 or set(sequence) - set("ACGTN"):
            raise ValueError("DART sequences must contain 350 DNA bases")
        role = row.get("pair_role")
        if role not in {"element", "control"} or row.get("target") != int(role == "element"):
            raise ValueError("Invalid DART pair role or target")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or len(pair_id) != 64 or any(c not in "0123456789abcdef" for c in pair_id):
            raise ValueError("Invalid opaque DART pair identity")
        if not isinstance(row.get("source_index"), int) or isinstance(row["source_index"], bool) or row["source_index"] < 0:
            raise ValueError("Invalid DART source pair index")
        pair = pairs.setdefault(pair_id, {})
        if role in pair:
            raise ValueError("Duplicate DART pair member")
        pair[role] = row
    if any(set(pair) != {"element", "control"} for pair in pairs.values()):
        raise ValueError("Prepared DART pairs must contain both members")
    indices = [pair["element"]["source_index"] for pair in pairs.values()]
    if len(set(indices)) != len(indices) or any(pair["element"]["source_index"] != pair["control"]["source_index"] for pair in pairs.values()):
        raise ValueError("DART pair source indices must be unique and matched")
    count = metadata.get("canonical_pair_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < len(pairs) or metadata.get("canonical_test_count") != 2 * count:
        raise ValueError("DART canonical sequence/pair denominators mismatch")


def _wilcoxon_112(diffs):
    """Frozen scipy1.12 auto/greater/wilcox semantics, including integer truncation.

    Modern SciPy changes auto selection and tied exact ranks; using its defaults
    would change published p-values. Upstream warns that tied "exact" p-values
    are not an exact tied null distribution. We preserve, rather than repair it.
    """
    use_exact = len(diffs) <= 50 and not np.any(diffs == 0)
    nonzero = diffs if use_exact else diffs[diffs != 0]
    ranks = rankdata(np.abs(nonzero), method="average")
    statistic = float(ranks[nonzero > 0].sum())
    count = len(nonzero)
    if use_exact:
        # Distribution of sum of randomly selected ranks 1..N, scipy1.12's
        # _get_wilcoxon_distr recurrence. N<=50 bounds memory/time.
        pmf = np.array([1.0])
        for rank in range(1, count + 1):
            nxt = np.zeros(len(pmf) + rank)
            nxt[:len(pmf)] += pmf * 0.5
            nxt[rank:] += pmf * 0.5
            pmf = nxt
        probability = float(np.clip(pmf[int(statistic):].sum(), 0, 1))
    else:
        mean = count * (count + 1.0) * 0.25
        variance = count * (count + 1.0) * (2.0 * count + 1)
        _, repeats = np.unique(ranks, return_counts=True)
        variance -= 0.5 * (repeats * (repeats * repeats - 1)).sum()
        probability = float(norm.sf((statistic - mean) / np.sqrt(variance / 24)))
    return statistic, probability


def score(data, predictions):
    validate_prepared(data)
    rows = data["rows"]
    if not isinstance(predictions, Mapping):
        raise ValueError("DART predictions must be keyed by sequence ID")  # noqa: TRY004 - public validation API
    if set(predictions) - {r["id"] for r in rows}:
        raise ValueError("Unknown DART prediction IDs")
    if any(isinstance(v, bool) or not isinstance(v, numbers.Real) or not np.isfinite(v) for v in predictions.values()):
        raise ValueError("DART predictions must be finite scalar scores")
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], {})[row["pair_role"]] = row["id"]
    diffs, missing = [], Counter()
    for pair in pairs.values():
        absent = [role for role, ident in pair.items() if ident not in predictions]
        if absent:
            missing["missing_both" if len(absent) == 2 else "missing_" + absent[0]] += 1
        else:
            diffs.append(float(predictions[pair["element"]]) - float(predictions[pair["control"]]))
    diffs = np.asarray(diffs, dtype=np.float64)
    if not np.isfinite(diffs).all():
        raise ValueError("DART score differences overflowed")
    metrics = dict.fromkeys(METRICS)
    if len(diffs):
        metrics.update(acc=float((diffs > 0).mean()), mean_diff=float(diffs.mean()),
                       median_diff=float(np.median(diffs)))
        for q in (5, 25, 75, 95):
            metrics[f"q{q:02d}_diff"] = float(np.percentile(diffs, q))
        if np.any(diffs != 0):
            statistic, pvalue = _wilcoxon_112(diffs)
            metrics.update(pval=pvalue, signed_rank_sum=statistic)
        else:
            metrics["unavailable_reason"] = "All differences are zero; upstream scipy1.12 Wilcoxon is undefined"
    else:
        metrics["unavailable_reason"] = "No complete scored pairs"
    metrics["n"] = len(predictions)
    metrics["n_pairs"] = len(diffs)
    metrics["pairs_denominator"] = data["metadata"]["canonical_pair_count"]
    denominator = data["metadata"]["canonical_test_count"]
    pair_denominator = data["metadata"]["canonical_pair_count"]
    return {"protocol_id": PROTOCOL_ID, "scope": data["scope"], "complete": False,
            "metrics": metrics, "primary_metric": "acc", "score_direction": "higher",
            "metric_directions": {key: "higher" if key == "acc" else "not_applicable" for key in METRICS},
            "coverage": {"scored": len(predictions), "unscored": denominator - len(predictions),
                         "denominator": denominator, "selected": len(rows)},
            "pair_coverage": {"scored": len(diffs), "unscored": pair_denominator - len(diffs),
                              "denominator": pair_denominator, "selected": len(pairs)},
            "unscored_reasons": {"not_selected": denominator - len(rows), "missing_prediction": len(rows) - len(predictions)},
            "unscored_pair_reasons": {"not_selected": pair_denominator - len(pairs), **dict(missing)},
            # Ties count as incorrect in acc; integer scores such as motif counts tie often.
            "pair_diagnostics": {"tied_pairs": int((diffs == 0).sum())},
            "metrics_scope": "Complete scored pairs only; incomplete pairs excluded with explicit coverage. Local inputs are not source-verified.",
            "evaluation_claim": "local_evaluation_not_paper_reproduction"}
