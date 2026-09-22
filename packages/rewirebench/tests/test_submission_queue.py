import base64
import io
import json
from unittest.mock import patch

import pytest
from rewirebench.submission import SubmissionError
from rewirebench.submission_queue import _locked, drain_submissions, enqueue_submission
from test_submission import arguments, bundle


def identity_token(subject="account-a", project="rewire-it", *, issued_at=1, **overrides):
    """Deliberately unsigned test fixture, not an authentic Firebase credential."""
    encode = lambda data: base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")
    claims = {"iss": f"https://securetoken.google.com/{project}", "aud": project,
              "sub": subject, "iat": issued_at, "email": "private@example.org"} | overrides
    return ".".join([encode({"alg": "RS256", "typ": "JWT"}), encode(claims), "test-signature"])


def enqueue(tmp_path, data=None, **overrides):
    args = arguments()
    args.pop("value")
    return enqueue_submission(data or bundle(), queue=tmp_path,
                              metric_path=["auroc"], **(args | overrides))


def entry(tmp_path):
    path = next(tmp_path.glob("*.json"))
    return path, json.loads(path.read_text())


def test_enqueue_derives_metric_and_is_idempotent(tmp_path):
    assert enqueue(tmp_path)["created"]
    assert not enqueue(tmp_path)["created"]
    path, item = entry(tmp_path)
    assert item["arguments"]["value"] == "0.7"
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="reviewed revision"):
        enqueue(tmp_path, title="Changed title")


def test_rejects_smoke_and_unknown_metric(tmp_path):
    data = bundle() | {"scope": "smoke"}
    with pytest.raises(ValueError):
        enqueue(tmp_path, data)
    with pytest.raises(ValueError, match="unavailable"):
        enqueue(tmp_path, bundle() | {"metrics": {"auroc": None}})


def test_dry_run_never_networks_or_changes_entry(tmp_path):
    enqueue(tmp_path)
    path, _ = entry(tmp_path)
    before = path.read_bytes()
    with patch("urllib.request.build_opener", side_effect=AssertionError("network forbidden")):
        assert drain_submissions(tmp_path)["dry_run"]
    assert path.read_bytes() == before


def test_changed_queue_payload_rejected(tmp_path):
    enqueue(tmp_path)
    path, item = entry(tmp_path)
    item["arguments"]["title"] = "altered"
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="payload changed"):
        drain_submissions(tmp_path)


def test_single_coordinator_lock(tmp_path):
    with _locked(tmp_path), pytest.raises(ValueError, match="Another process"):
        drain_submissions(tmp_path)


@pytest.mark.parametrize("code,status", [(401, "blocked"), (403, "blocked"),
    (429, "blocked"), (503, "blocked"), (409, "failed"), (500, "uncertain"), (None, "uncertain")])
def test_failures_preserve_retry_keys_and_redact(tmp_path, code, status):
    queued = enqueue(tmp_path)
    from rewirebench.submission import submit as real_submit
    def send(*args, **kwargs):
        if kwargs.get("dry_run"):
            return real_submit(*args, **kwargs)
        raise SubmissionError("secret-provider-body", http_status=code)
    with patch("rewirebench.submission_queue.submit", side_effect=send):
        result = drain_submissions(tmp_path, token=identity_token(), dry_run=False)
    path, item = entry(tmp_path)
    assert item["status"] == status
    assert item["idempotency_key"] == queued["idempotency_key"]
    assert "secret" not in path.read_text() + json.dumps(result)


def test_acknowledged_submission_is_not_reposted_when_tracking_fails(tmp_path):
    enqueue(tmp_path)
    from rewirebench.submission import submit as real_submit
    calls = []
    def send(*args, **kwargs):
        if kwargs.get("dry_run"):
            return real_submit(*args, **kwargs)
        calls.append(kwargs["idempotency_key"])
        return {"submission": {"id": "private-id", "status": "submitted"}}
    with patch("rewirebench.submission_queue.submit", side_effect=send), patch(
        "rewirebench.submission_queue.submission_status",
        side_effect=SubmissionError("expired", http_status=401),
    ):
        drain_submissions(tmp_path, token=identity_token(), dry_run=False)
    assert entry(tmp_path)[1]["status"] == "blocked"
    with patch("rewirebench.submission_queue.submit", side_effect=send), patch(
        "rewirebench.submission_queue.submission_status",
        return_value={"id": "private-id", "status": "submitted"},
    ):
        result = drain_submissions(tmp_path, token=identity_token(issued_at=2), dry_run=False)
        assert not drain_submissions(tmp_path, token=identity_token(issued_at=2), dry_run=False)["items"]
    assert len(calls) == 1
    assert result["items"][0]["status"] == "submitted"


def test_uncertain_submission_retries_identical_payload_and_key(tmp_path):
    enqueue(tmp_path)
    from rewirebench.submission import submit as real_submit
    calls = []
    def send(*args, **kwargs):
        if kwargs.get("dry_run"):
            return real_submit(*args, **kwargs)
        calls.append((args[0], kwargs["idempotency_key"]))
        if len(calls) == 1:
            raise SubmissionError("uncertain")
        return {"submission": {"id": "private-id", "status": "submitted"}}
    with patch("rewirebench.submission_queue.submit", side_effect=send), patch(
        "rewirebench.submission_queue.submission_status",
        return_value={"id": "private-id", "status": "submitted"},
    ):
        drain_submissions(tmp_path, token=identity_token(), dry_run=False)
        drain_submissions(tmp_path, token=identity_token(), dry_run=False)
    assert len(calls) == 2 and calls[0] == calls[1]


def test_empty_metric_path_has_clear_error(tmp_path):
    args = arguments()
    args.pop("value")
    with pytest.raises(ValueError, match="metric_path"):
        enqueue_submission(bundle(), queue=tmp_path, metric_path=[], **args)


@pytest.mark.parametrize("changed", [
    {"subject": "account-b"}, {"project": "another-project"},
])
def test_uncertain_retry_cannot_switch_account_or_project(tmp_path, changed):
    enqueue(tmp_path)
    from rewirebench.submission import submit as real_submit
    attempts = []
    def send(*args, **kwargs):
        if kwargs.get("dry_run"):
            return real_submit(*args, **kwargs)
        attempts.append(kwargs["token"])
        raise SubmissionError("Server may have accepted; response was lost")
    with patch("rewirebench.submission_queue.submit", side_effect=send):
        drain_submissions(tmp_path, token=identity_token(), dry_run=False)
        path, item = entry(tmp_path)
        persisted = path.read_bytes()
        assert item["status"] == "uncertain"
        assert len(item["retry_identity_sha256"]) == 64
        with pytest.raises(ValueError, match="different Firebase account or project"):
            drain_submissions(tmp_path, token=identity_token(**changed), dry_run=False)
        assert path.read_bytes() == persisted
    assert len(attempts) == 1
    assert identity_token() not in path.read_text()
    assert "account-a" not in path.read_text() and "private@example.org" not in path.read_text()


def test_refreshed_same_account_retries_after_uncertain_response(tmp_path):
    enqueue(tmp_path)
    from rewirebench.submission import submit as real_submit
    attempts = []
    def send(*args, **kwargs):
        if kwargs.get("dry_run"):
            return real_submit(*args, **kwargs)
        attempts.append(kwargs["idempotency_key"])
        if len(attempts) == 1:
            raise SubmissionError("lost response")
        return {"submission": {"id": "private-id", "status": "submitted"}}
    with patch("rewirebench.submission_queue.submit", side_effect=send), patch(
        "rewirebench.submission_queue.submission_status",
        return_value={"id": "private-id", "status": "in_review"},
    ):
        drain_submissions(tmp_path, token=identity_token(issued_at=1), dry_run=False)
        fingerprint = entry(tmp_path)[1]["retry_identity_sha256"]
        drain_submissions(tmp_path, token=identity_token(issued_at=2), dry_run=False)
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert entry(tmp_path)[1]["retry_identity_sha256"] == fingerprint
    assert entry(tmp_path)[1]["tracking"]["status"] == "in_review"


@pytest.mark.parametrize("token", [
    "not-a-token", "e30.e30.", "%%%.$$$.signature", "a.b.c", None,
    identity_token(sub=""), identity_token(aud=[]), identity_token(iss="https://other.example"),
    identity_token(sub="x" * 129), identity_token(aud="other-project"),
])
def test_malformed_retry_identity_rejected_before_network_or_queue_changes(tmp_path, token):
    enqueue(tmp_path)
    path, _ = entry(tmp_path)
    before = path.read_bytes()
    with (
        patch("urllib.request.build_opener", side_effect=AssertionError("network forbidden")),
        pytest.raises(ValueError),
    ):
        drain_submissions(tmp_path, token=token, dry_run=False)
    assert path.read_bytes() == before


def test_unsigned_emulator_token_requires_local_endpoint():
    from rewirebench.submission_queue import _retry_identity
    encode = lambda value: base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
    token = encode({"alg": "none"}) + "." + encode({
        "sub": "local-user", "aud": "demo-rewire",
        "iss": "https://securetoken.google.com/demo-rewire",
    }) + "."
    with pytest.raises(ValueError):
        _retry_identity(token)
    assert len(_retry_identity(token, allow_unsigned=True)) == 64


def test_attempted_legacy_item_without_identity_is_not_reassigned(tmp_path):
    enqueue(tmp_path)
    path, item = entry(tmp_path)
    item.update(attempts=1, status="uncertain")
    path.write_text(json.dumps(item))
    with (
        patch("urllib.request.build_opener", side_effect=AssertionError("network forbidden")),
        pytest.raises(ValueError, match="reconcile its original owner"),
    ):
        drain_submissions(tmp_path, token=identity_token(), dry_run=False)


def test_tracking_status_and_private_response_fields_are_allowlisted():
    from rewirebench.submission_queue import submission_status
    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(json.dumps({"result": {"data": {
                "id": "private-id", "status": "in_review", "email": "private@example.org",
                "token": "do-not-persist", "review_notes": ["private text"],
            }}}).encode())
    with patch("urllib.request.build_opener", return_value=Opener()):
        result = submission_status("private-id", token=identity_token())
    assert result == {"id": "private-id", "status": "in_review"}


@pytest.mark.parametrize("status", ["private-secret", "", "pending", ["submitted"], None])
def test_invalid_tracking_success_is_redacted(status):
    from rewirebench.submission_queue import submission_status
    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(json.dumps({"result": {"data": {
                "id": "private-id", "status": status,
            }}}).encode())
    with (
        patch("urllib.request.build_opener", return_value=Opener()),
        pytest.raises(SubmissionError) as caught,
    ):
        submission_status("private-id", token=identity_token())
    assert "private-secret" not in str(caught.value)


def test_reflected_token_in_success_response_never_enters_queue(tmp_path):
    enqueue(tmp_path)
    token = identity_token()
    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(json.dumps({"result": {"data": {
                "id": "private-id", "status": token,
            }}}).encode())
    with patch("urllib.request.build_opener", return_value=Opener()):
        result = drain_submissions(tmp_path, token=token, dry_run=False)
    path, item = entry(tmp_path)
    assert item["status"] == "uncertain" and "receipt" not in item
    assert token not in path.read_text() + json.dumps(result)


def test_multiple_items_receive_their_own_acknowledgements_and_tracking(tmp_path):
    enqueue(tmp_path)
    enqueue(tmp_path, bundle() | {"predictions_sha256": "e" * 64})
    class Opener:
        created = 0
        def open(self, request, timeout):
            if request.data:
                self.created += 1
                result = {"id": f"private-{self.created}", "status": "submitted"}
            else:
                import urllib.parse
                raw = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)["input"][0]
                result = {"id": json.loads(raw)["id"], "status": "in_review"}
            return io.BytesIO(json.dumps({"result": {"data": result}}).encode())
    opener = Opener()
    with patch("urllib.request.build_opener", return_value=opener):
        result = drain_submissions(tmp_path, token=identity_token(), dry_run=False)
    assert opener.created == 2
    assert {row["submission_id"] for row in result["items"]} == {"private-1", "private-2"}
    for path in tmp_path.glob("*.json"):
        item = json.loads(path.read_text())
        assert item["receipt"]["submission"]["id"] == item["tracking"]["id"]


def test_uncertain_retry_cannot_switch_endpoint(tmp_path):
    enqueue(tmp_path)
    from rewirebench.submission import submit as real_submit
    def send(*args, **kwargs):
        if kwargs.get("dry_run"):
            return real_submit(*args, **kwargs)
        raise SubmissionError("lost response")
    with patch("rewirebench.submission_queue.submit", side_effect=send):
        drain_submissions(tmp_path, token=identity_token(), dry_run=False)
        path, _ = entry(tmp_path)
        before = path.read_bytes()
        with pytest.raises(ValueError, match="different submission endpoint"):
            drain_submissions(tmp_path, token=identity_token(), dry_run=False,
                              endpoint="https://other.example/api/trpc")
        assert path.read_bytes() == before


def test_tracking_rejects_acknowledgement_for_another_submission():
    from rewirebench.submission_queue import submission_status
    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(b'{"result":{"data":{"id":"another-private-id","status":"submitted"}}}')
    with (
        patch("urllib.request.build_opener", return_value=Opener()),
        pytest.raises(SubmissionError, match="unconfirmed"),
    ):
        submission_status("private-id", token=identity_token())
