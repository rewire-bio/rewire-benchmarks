"""Explicit, verified-email submission to the existing private review queue."""

from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "https://benchmarks.rewire.it/api/trpc"


class SubmissionError(RuntimeError):
    """A failed submission, with no secret-bearing server body in its message."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the bearer token to an unexpected destination.
        return None


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
    if not isinstance(bundle, dict) or set(bundle) != required:
        raise ValueError("Submit only the allowlisted bundle produced by rewirebench.export")
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

    numeric_values = []

    def metrics(value, depth=0):
        if depth > 6:
            raise ValueError("Metrics nesting exceeds six levels")
        if isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(k, str) or not k:
                    raise ValueError("Invalid metric name")
                metrics(v, depth + 1)
        elif value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("Metric values must be finite numbers or null")

        elif type(value) in (int, float):
            numeric_values.append(value)

    if not isinstance(bundle["metrics"], dict) or not bundle["metrics"]:
        raise ValueError("No aggregate metrics to submit")
    metrics(bundle["metrics"])
    if not numeric_values:
        raise ValueError("No numerical metrics to submit")
    provenance = bundle["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("Invalid provenance")
    for key, value in {
        **provenance,
        "prepared_sha256": bundle["prepared_sha256"],
        "predictions_sha256": bundle["predictions_sha256"],
    }.items():
        length = 64 if key.endswith("_sha256") else 40 if key.endswith("_revision") else 0
        if (
            not length
            or not isinstance(value, str)
            or len(value) != length
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Export provenance may contain only revision and checksum fields")


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
    if len(json.dumps(contribution["details"]).encode()) > 24 * 1024:
        raise ValueError("Contribution details exceed 24 KiB; submit aggregate metrics only")
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
