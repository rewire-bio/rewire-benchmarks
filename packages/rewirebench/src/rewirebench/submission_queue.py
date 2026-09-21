"""Explicit, resumable submission outside model execution environments.

Only allowlisted SDK bundles and public evidence metadata are persisted. This is a
local private queue, not a catalogue release, credential store or background daemon.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from rewirebench.submission import (
    DEFAULT_ENDPOINT,
    SubmissionError,
    _acknowledgement,
    _NoRedirect,
    submit,
)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _retry_identity(token, *, allow_unsigned=False):
    """Hash unverified Firebase issuer/project/subject claims solely to bind retries.

    This does NOT authenticate the token. The server remains responsible for
    signature, expiry, revocation, verified email and ownership validation. Only
    this fingerprint is persisted, never the token, UID, email or raw claims.
    Emulator unsigned JWTs are accepted only for an explicitly local endpoint.
    """
    try:
        if not isinstance(token, str) or len(token) > 20_000:
            raise ValueError
        parts = token.split(".")
        if len(parts) != 3 or any(
            re.fullmatch(r"[A-Za-z0-9_-]+", part) is None for part in parts[:2]
        ) or (parts[2] and re.fullmatch(r"[A-Za-z0-9_-]+", parts[2]) is None):
            raise ValueError
        def decode(part):
            return json.loads(base64.b64decode(part + "=" * (-len(part) % 4),
                                              altchars=b"-_", validate=True))
        header, claims = decode(parts[0]), decode(parts[1])
        if not isinstance(header, dict) or not isinstance(claims, dict):
            raise TypeError
        if not (header.get("alg") == "RS256" and parts[2]) and not (
            allow_unsigned and header.get("alg") == "none" and not parts[2]
        ):
            raise ValueError
        issuer, project, subject = (claims.get(key) for key in ("iss", "aud", "sub"))
        if (not isinstance(project, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", project) is None
                or issuer != f"https://securetoken.google.com/{project}"
                or not isinstance(subject, str) or not 1 <= len(subject) <= 128
                or any(ord(character) < 32 for character in subject)):
            raise ValueError
        return hashlib.sha256(_encoded({"issuer": issuer, "project": project,
                                        "subject": subject})).hexdigest()
    except (ValueError, TypeError, UnicodeError, binascii.Error):
        raise ValueError("A structurally valid Firebase ID token is required for retry identity; refresh sign-in") from None


def _write(path, value):
    fd, temporary = tempfile.mkstemp(prefix=".queue-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_encoded(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _locked(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    # flock is released on process exit, including termination during a request.
    import fcntl
    fd = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process owns this submission queue") from None
        yield directory
    finally:
        os.close(fd)


def _metric(bundle, path):
    value = bundle["metrics"]
    if not path or not all(isinstance(key, str) and key for key in path):
        raise ValueError("metric_path must name a numeric metric in the exported bundle")
    try:
        for key in path:
            value = value[key]
    except (KeyError, TypeError):
        raise ValueError("metric_path does not resolve in the exported bundle") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Selected metric is unavailable or nonfinite")
    return str(value)


def enqueue_submission(bundle, *, queue, title, summary, source_url, source_locator,
                       metric_path, metric=None):
    """Freeze a validated payload. Derive its value from the bundle, never retype it.

    metric_path is a list of keys, supporting nested per-assay metrics. All metrics
    remain in the bundle. Call once per evaluation, not once per displayed metric.
    """
    data = json.loads(Path(bundle).read_text()) if isinstance(bundle, (str, Path)) else bundle
    value = _metric(data, metric_path)
    arguments = {"title": title, "summary": summary, "source_url": source_url,
                 "source_locator": source_locator, "metric": metric or metric_path[-1],
                 "value": value}
    payload = submit(data, **arguments, dry_run=True)
    key = payload["idempotencyKey"]
    with _locked(queue) as directory:
        # One identity for an evaluation regardless of headline metric or evidence wording.
        evaluation_key = hashlib.sha256(_encoded({
            k: data.get(k) for k in ("protocol_id", "protocol_version", "dataset_id",
                                    "prepared_sha256", "predictions_sha256", "model")
        })).hexdigest()
        for path in directory.glob("*.json"):
            old = json.loads(path.read_text())
            if old.get("evaluation_key") == evaluation_key:
                if old["idempotency_key"] != key:
                    raise ValueError("Evaluation already queued; changed metadata requires a reviewed revision")
                return {"idempotency_key": key, "status": old["status"], "created": False}
        _write(directory / f"{key}.json", {
            "schema_version": "1.0", "evaluation_key": evaluation_key,
            "idempotency_key": key, "bundle": data, "arguments": arguments,
            "status": "queued", "attempts": 0,
        })
    return {"idempotency_key": key, "status": "queued", "created": True}


def _endpoint(endpoint):
    value = urllib.parse.urlsplit(endpoint)
    local = value.hostname in {"localhost", "127.0.0.1", "::1"}
    if (not value.hostname or value.username or value.password or value.query or value.fragment
            or (value.scheme != "https" and not (local and value.scheme == "http"))):
        raise ValueError("Use an HTTPS endpoint, or HTTP on a local emulator")
    return endpoint.rstrip("/")


def submission_status(identifier, *, token, endpoint=DEFAULT_ENDPOINT, timeout=30):
    """Retrieve only the private ID and review status; never persist private bodies."""
    endpoint = _endpoint(endpoint)
    if not token or not token.strip():
        raise ValueError("A verified-email access token is required")
    _acknowledgement({"id": identifier, "status": "submitted"})
    query = urllib.parse.urlencode({"input": json.dumps({"id": identifier})})
    request = urllib.request.Request(f"{endpoint}/submission.get?{query}",
                                     headers={"Authorization": "Bearer " + token})
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout) as response:
            result = json.load(response)["result"]["data"]
        result = _acknowledgement(result)
        if result["id"] != identifier:
            raise ValueError("Invalid tracking response")
        return {"id": identifier, "status": result["status"]}
    except urllib.error.HTTPError as error:
        raise SubmissionError("Private submission tracking is unavailable; retry later",
                              http_status=error.code) from None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise SubmissionError("Private submission tracking is unconfirmed; retry later") from None


def drain_submissions(queue, *, token=None, endpoint=DEFAULT_ENDPOINT, max_items=5,
                      dry_run=True, timeout=30):
    """Explicitly deliver a bounded batch using the normal SDK submission API.

    Dry runs validate without networking or changing queue state. Real delivery
    stops on the first problem, preserving subsequent items and stable retry keys.
    Unverified issuer/project/subject claims bind retries to one account; only
    their fingerprint is saved. Server authentication remains authoritative.
    """
    endpoint = _endpoint(endpoint)
    if type(max_items) is not int or not 1 <= max_items <= 100:
        raise ValueError("max_items must be between 1 and 100")
    if not dry_run and (not token or not token.strip()):
        raise ValueError("A verified-email access token is required")
    retry_identity = None if dry_run else _retry_identity(token, allow_unsigned=(
        urllib.parse.urlsplit(endpoint).hostname in {"localhost", "127.0.0.1", "::1"}
    ))
    outcomes = []
    with _locked(queue) as directory:
        for path in sorted(directory.glob("*.json")):
            if len(outcomes) >= max_items:
                break
            item = json.loads(path.read_text())
            if item["status"] == "failed":
                continue
            receipt = item.get("receipt")
            if item["status"] == "submitted" and item.get("tracking"):
                continue
            payload = submit(item["bundle"], **item["arguments"], dry_run=True)
            key = payload["idempotencyKey"]
            if key != item["idempotency_key"] or path.stem != key:
                raise ValueError("Queue payload changed; refusing to reuse its idempotency key")
            if dry_run:
                outcomes.append({"idempotency_key": key, "status": item["status"], "dry_run": True})
                continue
            if item.get("endpoint") and item["endpoint"] != endpoint:
                raise ValueError("Queue item is bound to a different submission endpoint")
            if item.get("retry_identity_sha256") not in (None, retry_identity):
                raise ValueError("Queue item is bound to a different Firebase account or project")
            if not item.get("retry_identity_sha256") and (item.get("attempts", 0) or receipt):
                raise ValueError("Attempted queue item has no retry identity; reconcile its original owner before retrying")
            item["retry_identity_sha256"] = retry_identity
            item["endpoint"] = endpoint
            item["attempts"] += 1
            item["status"] = "sending"
            _write(path, item)
            try:
                if not receipt:
                    receipt = submit(item["bundle"], **item["arguments"], token=token,
                                     endpoint=endpoint, idempotency_key=key, timeout=timeout)
                    # Only accept the acknowledged private record identity/status.
                    acknowledgement = _acknowledgement(receipt["submission"])
                    identifier = acknowledgement["id"]
                    item["receipt"] = {"submission": {"id": identifier,
                        "status": acknowledgement["status"]}, "idempotency_key": key,
                        "publication_status": "pending_review"}
                    _write(path, item)
                item["tracking"] = _acknowledgement(submission_status(
                    item["receipt"]["submission"]["id"],
                    token=token, endpoint=endpoint, timeout=timeout))
                item["status"] = "submitted"
                item.pop("last_error", None)
            except SubmissionError as error:
                code = error.http_status
                item["status"] = ("blocked" if code in {401, 403, 429, 503} else
                                  "failed" if code in {400, 409, 413, 422} else "uncertain")
                item["last_error"] = {"http_status": code,
                                      "reason": "Submission or tracking requires attention"}
                _write(path, item)
                outcomes.append({"idempotency_key": key, "status": item["status"],
                                 "acknowledged": "receipt" in item})
                break
            _write(path, item)
            outcomes.append({"idempotency_key": key, "status": item["status"],
                             "submission_id": item["receipt"]["submission"]["id"]})
    return {"dry_run": dry_run, "items": outcomes, "published": False}
