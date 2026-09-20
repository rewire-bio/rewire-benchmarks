"""Portable, fail-closed sequence-protocol contribution rules.

The JSON contract is also tested by the database's TypeScript intake validator.
It carries only public dataset identities, source pins and metric definitions.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib.resources import files

from rewirebench.provenance import is_digest


@lru_cache(maxsize=1)
def contract():
    return json.loads(files("rewirebench").joinpath("resources/sequence-submission-contract.json").read_text())


SEQUENCE_PROTOCOLS = frozenset(contract()["protocols"])


def validate_sequence_bundle(bundle):
    rules = contract()
    protocol = rules["protocols"].get(bundle["protocol_id"])
    if protocol is None:
        raise ValueError("Unsupported sequence protocol")
    if bundle.get("evaluation_claim") != rules["evaluation_claim"]:
        raise ValueError("Sequence contributions must disclaim published-result reproduction")
    if bundle.get("data_verification") not in rules["data_verification"]:
        raise ValueError("Sequence contributions must declare source verification")
    if bundle["protocol_version"] != protocol["protocol_version"]:
        raise ValueError("Unsupported sequence protocol version")
    dataset = protocol["datasets"].get(bundle["dataset_id"])
    if dataset is None:
        raise ValueError("Unknown sequence dataset, split or target")
    methods = protocol["methods_by_execution"].get(bundle["execution_status"], [])
    if bundle.get("evaluation_method") not in methods:
        raise ValueError("Evaluation method is incompatible with protocol or execution status")
    provenance = bundle["provenance"]
    if not isinstance(provenance, dict) or provenance.get("upstream_revision") != protocol["upstream_revision"]:
        raise ValueError("Sequence contributions require the exact upstream evaluator revision")
    if any(not is_digest(provenance.get(key), 64) for key in protocol["required_hashes"]):
        raise ValueError("Missing sequence dataset or split hash")
    verified = bundle["data_verification"] == "pinned_source_bytes"
    if verified:
        if dataset["denominator"] is None or not dataset["verified_provenance"]:
            raise ValueError("This dataset has no independently verified source pin")
        if any(provenance.get(key) != value for key, value in dataset["verified_provenance"].items()):
            raise ValueError("Source or split hashes differ from the pinned sequence dataset")
        if bundle["coverage"]["denominator"] != dataset["denominator"]:
            raise ValueError("Coverage must retain the pinned target/split denominator")
    if bundle["scope"] == "full" and not verified:
        raise ValueError("Local unverified sequence copies cannot claim full benchmark scope")
    metrics = bundle["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != set(protocol["metrics"]):
        raise ValueError("Include all and only prescribed sequence metrics and counts")
    for key, constraints in protocol["metrics"].items():
        value = metrics[key]
        if value is None and not constraints["integer"]:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Sequence metrics must be finite numbers or null performance metrics")
        if constraints["integer"] and (type(value) is not int or value > 9007199254740991):
            raise ValueError("Sequence counts must be safe integers")
        if ((constraints["minimum"] is not None and value < constraints["minimum"])
                or (constraints["maximum"] is not None and value > constraints["maximum"])):
            raise ValueError(f"Sequence metric {key} is outside its valid range")
    if not any(metrics[key] is not None for key in protocol["performance_metrics"]):
        raise ValueError("No numerical performance metric to submit")
    if "n" in metrics and metrics["n"] != bundle["coverage"]["scored"]:
        raise ValueError("Sequence metric n must reconcile with scored sequences")
    if bundle["protocol_id"] == "dart-eval-task1-zero-shot-v1":
        pairs, denominator, scored = metrics["n_pairs"], metrics["pairs_denominator"], metrics["n"]
        if (denominator * 2 != bundle["coverage"]["denominator"]
                or pairs > scored // 2 or pairs < max(0, scored - denominator)):
            raise ValueError("DART pair coverage does not reconcile with sequence coverage")
        rank = metrics["signed_rank_sum"]
        if rank is not None and rank > pairs * (pairs + 1) / 2:
            raise ValueError("DART signed-rank sum exceeds the possible pair ranks")
        if metrics["acc"] is None:
            raise ValueError("A scored DART pair set requires accuracy")
