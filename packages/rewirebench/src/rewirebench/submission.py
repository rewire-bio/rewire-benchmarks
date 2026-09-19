"""Explicit, verified-email submission to the existing private review queue."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from rewirebench.provenance import PUBLIC_FIELDS, is_digest
from rewirebench.sdk import LOCAL_COPY_PROTOCOLS, LOCAL_EVALUATION_CLAIM

DEFAULT_ENDPOINT = "https://benchmarks.rewire.it/api/trpc"


class SubmissionError(RuntimeError):
    """A failed submission, with no secret-bearing server body in its message."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the bearer token to an unexpected destination.
        return None


MFASS_METRICS = frozenset({
    "n", "positives", "prevalence", "capacity", "precision_at_capacity",
    "recall_at_capacity", "average_precision_sklearn", "auroc",
})
MFASS_PERFORMANCE_METRICS = MFASS_METRICS - {"n", "positives", "prevalence", "capacity"}
PROTEINGYM_METRICS = frozenset({"Spearman", "AUC", "MCC", "NDCG", "Top_recall"})
SUPPORTED_PROTOCOLS = frozenset({
    "mfass-v2", "mfass-v2-frozen-encoder", "proteingym-v1.3-dms-substitutions",
    *LOCAL_COPY_PROTOCOLS,
})


@lru_cache(maxsize=1)
def _proteingym_assays() -> dict[str, int]:
    """Only public assay identifiers from the exact pinned reference are permitted."""
    data = files("rewirebench").joinpath(
        "resources", "proteingym", "DMS_substitutions.csv",
    ).read_bytes()
    if hashlib.sha256(data).hexdigest() != (
        "a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308"
    ):
        raise ValueError("Packaged ProteinGym reference checksum mismatch")
    return {row["DMS_id"]: int(row["DMS_total_number_mutants"])
            for row in csv.DictReader(io.StringIO(data.decode("utf-8")))}


def _scalar_metrics(value, allowed: frozenset[str], performance: frozenset[str]) -> bool:
    if not isinstance(value, dict) or not value or set(value) - allowed:
        raise ValueError("Only known scalar metrics for the selected protocol may be submitted")
    for item in value.values():
        if item is not None and (type(item) not in (int, float) or not math.isfinite(item)):
            raise ValueError("Metric values must be finite numbers or null")
    return any(key in performance and item is not None for key, item in value.items())


def _validate_metrics(bundle):
    """Allow known scientific summaries, never arbitrary nested model output.

    This excludes raw prediction dictionaries and private paths as metric keys.
    The explicit model name and training-overlap description remain user-provided
    text; contributors must inspect those fields before sending a bundle.
    """
    protocol = bundle["protocol_id"]
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError("Unsupported protocol for SDK submission")
    if protocol in LOCAL_COPY_PROTOCOLS:
        _validate_local_copy_metrics(bundle)
        return
    value = bundle["metrics"]
    if protocol in {"mfass-v2", "mfass-v2-frozen-encoder"}:
        if not _scalar_metrics(value, MFASS_METRICS, MFASS_PERFORMANCE_METRICS):
            raise ValueError("No numerical performance metrics to submit")
        return
    if not isinstance(value, dict) or not value:
        raise ValueError("No numerical performance metrics to submit")
    if "per_assay" not in value:
        if not _scalar_metrics(value, PROTEINGYM_METRICS, PROTEINGYM_METRICS):
            raise ValueError("No numerical performance metrics to submit")
        if (bundle["scope"] != "full" or bundle["completion"] != "complete"
                or bundle["coverage"]["denominator"] != sum(_proteingym_assays().values())):
            raise ValueError("ProteinGym suite metrics require complete full-track coverage")
        return
    if set(value) != {"per_assay"}:
        raise ValueError("Per-assay summaries cannot contain additional fields")
    assays = value["per_assay"]
    official = _proteingym_assays()
    if not isinstance(assays, dict) or not assays or set(assays) - official.keys():
        raise ValueError("Per-assay metrics require pinned official ProteinGym assay IDs")
    any_performance = False
    total_scored = total_eligible = 0
    for assay_id, summary in assays.items():
        if not isinstance(summary, dict) or set(summary) != {"metrics", "scored", "denominator"}:
            raise ValueError("Per-assay summaries allow only metrics, scored and denominator")
        any_performance |= _scalar_metrics(
            summary["metrics"], PROTEINGYM_METRICS, PROTEINGYM_METRICS,
        )
        scored, denominator = summary["scored"], summary["denominator"]
        if (type(scored) is not int or type(denominator) is not int
                or scored < 0 or denominator <= 0 or scored > denominator
                or denominator != official[assay_id]):
            raise ValueError("Per-assay coverage must retain the official denominator")
        total_scored += scored
        total_eligible += denominator
    if not any_performance:
        raise ValueError("No numerical performance metrics to submit")
    if (total_scored != bundle["coverage"]["scored"]
            or total_eligible != bundle["coverage"]["denominator"]):
        raise ValueError("Per-assay coverage does not reconcile with total coverage")
    if bundle["scope"] == "full" and set(assays) != official.keys():
        raise ValueError("Full-track summaries must include all official assays")


def _validate_local_copy_metrics(bundle):
    """One named local dataset, prescribed scalar metrics, no reproduction claim."""
    from rewirebench.protocols import genomic_benchmarks as gb
    from rewirebench.protocols import tdc_admet as tdc

    if (
        bundle.get("evaluation_claim") != LOCAL_EVALUATION_CLAIM
        or bundle.get("data_verification")
        != "local_bytes_hashed_not_independently_source_verified"
    ):
        raise ValueError("Local-copy results must disclaim paper reproduction and source verification")
    if bundle["protocol_id"] == tdc.PROTOCOL_ID:
        datasets = {f"tdc-admet-{name}": metric for name, metric in tdc.ADMET_METRICS.items()}
        if bundle["dataset_id"] not in datasets:
            raise ValueError("Expected a known TDC ADMET dataset")
        performance = frozenset({datasets[bundle["dataset_id"]]})
        version, revision = tdc.UPSTREAM_REVISION, tdc.UPSTREAM_REVISION
        required_hashes = {"test_sha256"}
    else:
        datasets = {f"genomic-benchmarks-{name}" for name in gb.DATASETS}
        if bundle["dataset_id"] not in datasets:
            raise ValueError("Expected a known Genomic Benchmarks dataset")
        performance = frozenset(
            {"accuracy", "f1_macro", "f1_weighted"}
            if bundle["dataset_id"] == "genomic-benchmarks-human_ensembl_regulatory"
            else {"accuracy", "f1"}
        )
        version, revision = gb.PROTOCOL_VERSION, gb.UPSTREAM_REVISION
        required_hashes = {"train_sha256", "test_sha256"}
    if bundle["protocol_version"] != version:
        raise ValueError("Unsupported version for this local-copy protocol")
    provenance = bundle["provenance"]
    if (
        not isinstance(provenance, dict)
        or provenance.get("upstream_revision") != revision
        or any(not is_digest(provenance.get(key), 64) for key in required_hashes)
    ):
        raise ValueError("Local-copy results require the evaluator revision and local source hashes")
    metrics = bundle["metrics"]
    allowed = performance | {"n"}
    if not _scalar_metrics(metrics, allowed, performance):
        raise ValueError("No numerical performance metrics to submit")
    if set(metrics) != allowed:
        raise ValueError("Include all prescribed metrics for the selected dataset")
    if type(metrics["n"]) is not int or metrics["n"] != bundle["coverage"]["scored"]:
        raise ValueError("Metric n must reconcile with scored coverage")
    for key in performance:
        value = metrics[key]
        if value is None:
            continue
        lower, upper = (-1, 1) if key == "spearman" else (0, math.inf if key == "mae" else 1)
        if not lower <= value <= upper:
            raise ValueError(f"Metric {key} is outside its valid range")


def _validate_bundle(bundle):
    required = {
        "schema_version",
        "kind",
        "protocol_id",
        "protocol_version",
        "dataset_id",
        "scope",
        "completion",
        "model",
        "metrics",
        "coverage",
        "provenance",
        "execution_status",
        "review_status",
        "independently_reproduced",
        "prepared_sha256",
        "predictions_sha256",
    }
    if (
        not isinstance(bundle, dict)
        or not required.issubset(bundle)
        or set(bundle) - required - {"data_verification", "evaluation_claim"}
    ):
        raise ValueError("Submit only the allowlisted bundle produced by rewirebench.export")
    if bundle.get("data_verification", "unreported") not in {
        "pinned_source_bytes",
        "local_bytes_hashed_not_independently_source_verified",
        "unreported",
    }:
        raise ValueError("Invalid data verification declaration")
    if "evaluation_claim" in bundle and bundle["evaluation_claim"] != LOCAL_EVALUATION_CLAIM:
        raise ValueError("Invalid evaluation claim")
    if (
        bundle["schema_version"] != "1.0"
        or bundle["kind"] != "rewire_benchmark_submission"
        or bundle["scope"] not in {"full", "subset"}
        or bundle["completion"] not in {"complete", "partial"}
        or bundle["review_status"] != "unreviewed_contribution"
        or bundle["independently_reproduced"] is not False
        or bundle["execution_status"] not in {"imported_predictions", "local_adapter"}
    ):
        raise ValueError("Invalid bundle status; smoke tests cannot be submitted")
    for key in ("protocol_id", "protocol_version", "dataset_id"):
        if not isinstance(bundle[key], str) or not bundle[key].strip():
            raise ValueError(f"Missing {key}")
    model = bundle["model"]
    if (
        not isinstance(model, dict)
        or set(model) != {"name", "training_overlap"}
        or any(not isinstance(v, str) or not v.strip() for v in model.values())
    ):
        raise ValueError("Model export may contain only name and training_overlap")
    coverage = bundle["coverage"]
    if not isinstance(coverage, dict) or set(coverage) != {"denominator", "scored", "unscored"}:
        raise ValueError("Invalid coverage")
    if (
        any(type(v) is not int or v < 0 for v in coverage.values())
        or (coverage["scored"] + coverage["unscored"] != coverage["denominator"])
        or coverage["scored"] == 0
    ):
        raise ValueError("Coverage does not reconcile")
    if bundle["completion"] == "complete" and (bundle["scope"] != "full" or coverage["unscored"]):
        raise ValueError("Partial coverage cannot be declared complete")

    _validate_metrics(bundle)
    provenance = bundle["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("Invalid provenance")  # noqa: TRY004 - public validation API
    for key, value in {
        **provenance,
        "prepared_sha256": bundle["prepared_sha256"],
        "predictions_sha256": bundle["predictions_sha256"],
    }.items():
        length = (
            64 if key in {"prepared_sha256", "predictions_sha256"} else PUBLIC_FIELDS.get(key)
        )
        if length is None or not is_digest(value, length):
            raise ValueError("Export provenance may contain only allowlisted revision and checksum fields")


def submit(
    bundle: dict | str | Path,
    *,
    title: str,
    summary: str,
    source_url: str,
    metric: str,
    value: str,
    source_locator: str,
    token: str | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    idempotency_key: str | None = None,
    dry_run: bool = False,
    timeout: float = 30,
) -> dict:
    """Submit one run to review, never directly to a leaderboard.

    Obtain a Firebase ID token from the verified-email contribution session.
    Tokens are used only for the request, never saved in the bundle or report.
    Production deliberately refuses requests while submissions are disabled.
    Repeating an identical payload uses the same deterministic idempotency key.
    """
    data = json.loads(Path(bundle).read_text()) if isinstance(bundle, (str, Path)) else bundle
    _validate_bundle(data)
    if (
        not title.strip()
        or len(summary.strip()) < 10
        or not all(isinstance(v, str) and v.strip() for v in (metric, value, source_locator))
    ):
        raise ValueError(
            "Supply a title, summary (10+ characters), metric, value and evidence location"
        )
    source = urllib.parse.urlsplit(source_url)
    if source.scheme != "https" or not source.hostname or source.username or source.password:
        raise ValueError("A public HTTPS source URL without credentials is required")
    contribution = {
        "type": "result",
        "title": title,
        "summary": summary,
        "source_urls": [source_url],
        "public_credit": False,
        "details": {
            "model": data["model"]["name"],
            "benchmark": data["protocol_id"],
            "protocol": data["protocol_version"],
            "metric": metric,
            "value": value,
            "source_locator": source_locator,
            "rewire_bundle": data,
        },
    }
    encoded_contribution = json.dumps(contribution, sort_keys=True, allow_nan=False)
    key = idempotency_key or hashlib.sha256(encoded_contribution.encode()).hexdigest()
    if not 16 <= len(key) <= 128:
        raise ValueError("idempotency_key must be 16..128 characters")
    if len(json.dumps(contribution["details"]).encode()) > 24_000:
        raise ValueError("Contribution details exceed 24,000 bytes; submit aggregate metrics only")
    payload = {"contribution": contribution, "idempotencyKey": key}
    if dry_run:
        return payload
    if not token or not token.strip():
        raise ValueError("A verified-email Firebase ID token is required; use dry_run to preview")
    url = urllib.parse.urlsplit(endpoint)
    local = url.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        (url.scheme != "https" and not (url.scheme == "http" and local))
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Submission endpoint must use HTTPS (HTTP allowed only for a local emulator)"
        )
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/submission.create",
        data=json.dumps(payload, allow_nan=False).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
    )
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            raise SubmissionError(
                "Rewire submissions are currently disabled. Keep your export and submit later."
            ) from None
        if exc.code in (401, 403):
            raise SubmissionError(
                "Verify your email and refresh your contribution access token."
            ) from None
        if exc.code == 429:
            raise SubmissionError(
                "Submission rate limit reached; retry later with the same idempotency key."
            ) from None
        raise SubmissionError(
            f"Submission rejected (HTTP {exc.code}); keep the bundle and idempotency key."
        ) from None
    except (urllib.error.URLError, TimeoutError, ValueError):
        raise SubmissionError(
            "Submission outcome is uncertain; retry the same payload and idempotency key."
        ) from None
    if (
        not isinstance(result, dict)
        or "error" in result
        or not isinstance(result.get("result"), dict)
        or not isinstance(result["result"].get("data"), dict)
    ):
        raise SubmissionError(
            "Submission was not acknowledged; retain the bundle and retry with the same key."
        )
    return {
        "submission": result["result"]["data"],
        "idempotency_key": key,
        "publication_status": "pending_review",
    }
