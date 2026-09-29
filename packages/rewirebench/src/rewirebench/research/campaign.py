"""Durable, bounded research campaigns over pinned evidence manifests."""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import psutil

from rewirebench import sdk

from .contracts import (
    DEFAULT_LIMITS,
    LIMITS_SCHEMA,
    OPERATIONS,
    REGISTRY_HELP,
    assert_public,
    canonical,
    critic_schema,
    digest,
    hypothesis_schema,
    operation_signature,
    planner_schema,
    read_json,
    redact_public,
    selected_question,
    validate,
    validate_bundle,
    validate_critic,
    validate_followup_test,
    validate_manifest,
    validate_plan,
    validate_proposed_plan,
    validate_question_design,
    validate_spec,
)
from .operations import load_table, replay, validate_operation
from .reasoner import CodexReasoner, ReasoningFailure
from .state import Store, now
from .supervisor import ResourceStop, local_environment, run_guarded, workspace_size


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def read_manifests(path):
    value = read_json(path)
    values = value if isinstance(value, list) else [value]
    seen = set()
    for manifest in values:
        validate_manifest(manifest)
        if manifest["id"] in seen:
            raise ValueError("Duplicate manifest ID")
        seen.add(manifest["id"])
    if not values:
        raise ValueError("Supply at least one manifest")
    return values


def scan(manifests, resolver):
    """Read-only triage; download/reproduction occurs inside guarded campaigns."""
    results = []
    for manifest in manifests:
        validate_manifest(manifest)
        result = {"manifest_id": manifest["id"], "catalogue_release_id": manifest["catalogue_release_id"],
                  "question": manifest["question"], "priority": 0, "signals": [], "blockers": [],
                  "eligible": False}
        try:
            table = load_table(manifest, resolver)
            result["summary"] = replay(table, manifest, list(manifest["expected_metrics"]))
            checks = result["summary"]["checks"]
            failed = sum(c["status"] == "failed" for c in checks)
            if failed:
                result["signals"].append(f"{failed} reported metrics did not replay within tolerance")
                result["priority"] += 100
            for method, metrics in result["summary"]["metrics"].items():
                cov = result["summary"]["coverage"][method]
                if cov["missing"]:
                    result["signals"].append(f"{method}: {cov['missing']} missing predictions")
                    result["priority"] += 10
                if any(metrics.get(m) is not None and metrics[m] < 0
                       for m in ("Spearman", "spearman", "pearson")):
                    result["signals"].append(f"{method}: negative rank or linear correlation")
                    result["priority"] += 20
                scored = [r["scores"][method] for r in table["rows"] if r["scores"][method] is not None]
                if scored and len(set(scored)) == 1:
                    result["signals"].append(f"{method}: constant predictions; correlation is undefined")
                    result["priority"] += 15
            if len(manifest["expected_metrics"]) > 1:
                result["signals"].append("Multiple methods available for paired population reconciliation")
                result["priority"] += 5
            result["eligible"] = True
        except (ValueError, OSError) as exc:
            result["blockers"].append(str(exc))
            table_ref = next(a for a in manifest["artifacts"] if a["id"] == manifest["table_artifact_id"])
            result["download_available"] = bool(table_ref["uri"])
        if not manifest["semantics"]["independent_unit"]:
            result["blockers"].append("No documented independence unit for uncertainty estimation")
        result["independent_validation"] = False
        results.append(result)
    return sorted(results, key=lambda r: (-r["priority"], r["manifest_id"]))


def scan_guarded(manifests, resolver):
    """CLI triage gets the same numerical-process limits as campaign experiments."""
    with tempfile.TemporaryDirectory(prefix="rewire-research-scan-") as directory:
        root = Path(directory)
        request, result = root/"request.json", root/"result.json"
        write_new(request, {"manifests": manifests, "resolver": resolver})
        execution = run_guarded([sys.executable, "-m", "rewirebench.research.campaign",
                                 str(request), str(result)], cwd=root, workspace=root,
                                stdout=root/"stdout.log", stderr=root/"stderr.log",
                                seconds=DEFAULT_LIMITS["experiment_seconds"],
                                memory_bytes=DEFAULT_LIMITS["memory_bytes"],
                                workspace_bytes=DEFAULT_LIMITS["workspace_bytes"],
                                deadline=time.time()+DEFAULT_LIMITS["experiment_seconds"],
                                env=local_environment())
        if execution["returncode"] or not result.exists():
            raise ValueError("Research scan failed within the supervised process")
        return read_json(result)


def create_campaign(workspace, manifests, resolver, *, limits=None, mode="codex"):
    limits = dict(DEFAULT_LIMITS, **(limits or {}))
    validate(limits, LIMITS_SCHEMA)
    if mode not in {"codex", "replay-only"}:
        raise ValueError("Unknown research mode")
    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise ValueError("Campaign workspace must be new; use resume for an existing campaign")
    if not isinstance(resolver, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                           or not Path(v).is_absolute() for k, v in resolver.items()):
        raise ValueError("Private resolver must map artifact IDs to absolute local file paths")
    manifests = [validate_manifest(m) for m in manifests]
    if not manifests or len({m["id"] for m in manifests}) != len(manifests):
        raise ValueError("Manifest IDs must be unique and nonempty")
    workspace.mkdir(parents=True, mode=0o700)
    store = Store(workspace, create=True)
    try:
        created, identifier = now(), "campaign-"+uuid.uuid4().hex[:16]
        for key, value in {"id": identifier, "created_at": created, "state": "created",
                           "limits": limits, "deadline": time.time()+limits["campaign_seconds"],
                           "mode": mode, "codex_calls_used": 0, "stop_requested": False,
                           "code_sha256": sdk._code_digest()}.items():
            store.put(key, value)
        write_new(workspace/"resolver.private.json", resolver)
        os.chmod(workspace/"resolver.private.json", 0o600)
        for manifest in manifests:
            with store.db:
                store.db.execute("INSERT INTO cases(id,manifest) VALUES (?,?)",
                                 (manifest["id"], canonical(manifest)))
        return store.status()
    finally:
        store.close()


def operation(identifier, kind, *, methods=None, metric=None, field=None, recipe=None,
              expected="The pinned data and declared method support this diagnostic."):
    return {"id": identifier, "kind": kind, "methods": methods or [], "metric": metric,
            "field": field, "recipe": recipe, "expected_observation": expected}


def _core_hypothesis(manifest):
    tests = [operation("verify-evidence", "verify", expected="All available evidence matches its declared byte hash."),
             operation("replay-metrics", "replay", expected="Recorded metrics replay within the stated tolerance.")]
    if "sdk:train-mean-v1" in manifest["local_recipes"]:
        tests.append(operation("local-train-mean", "local_recipe", recipe="sdk:train-mean-v1",
                               expected="The SDK trains only on training labels and reproduces the constant control."))
    return {"id": "evidence-and-method", "explanation": "The discrepancy may arise from stale evidence, metric calculation or an inadequate trivial control.",
            "tests": tests}


def _replay_plan(manifest):
    return {"hypotheses": [_core_hypothesis(manifest)], "multiple_testing": "descriptive_only",
            "stopping_rule": "Stop after registered verification and replay. No autonomous hypothesis claims.",
            "requested_tools": []}


def _supervised_hypotheses(manifest, *, initial, question_design=None, followup_test=None, question=None):
    core = _core_hypothesis(manifest) if initial else None
    hypotheses = [core] if core else []
    if initial and question_design:
        selected = selected_question(validate_question_design(question_design, manifest))
        if selected:
            test = copy.deepcopy(selected["decisive_test"])
            test["id"] = "question-decisive-test"
            hypotheses.append({"id": "selected-question", "explanation": selected["hypothesis"] +
                               " Alternative explanation: " + selected["alternative_explanation"],
                               "tests": [test]})
    if followup_test is not None:
        if initial:
            raise ValueError("A follow-up test cannot be supplied to the initial round")
        test = copy.deepcopy(validate_followup_test(followup_test, manifest))
        test["id"] = "followup-decisive-test"
        hypotheses.append({"id": "followup-question", "explanation": question or
                           "The critic's follow-up question requires this registered decisive test.",
                           "tests": [test]})
    return hypotheses


def merge_plan(manifest, proposed, *, initial, question_design=None, followup_test=None, question=None):
    """Preserve supervised tests and keep explicitly blocked questions out of execution.

    Analysis parameters are never repaired or guessed. Original model output is
    retained by the reasoning checkpoint. Expected observations from a duplicate
    supervised operation are retained on its single frozen test.
    """
    plan = copy.deepcopy(proposed)
    hypotheses = _supervised_hypotheses(manifest, initial=initial, question_design=question_design,
                                        followup_test=followup_test, question=question)
    supervised = list(hypotheses)

    for hi, hypothesis in enumerate(plan["hypotheses"]):
        blocker = hypothesis.pop("blocker", None)
        if blocker is not None:
            if hypothesis["tests"]:
                raise ValueError("A blocked hypothesis cannot contain executable tests")
            plan["requested_tools"].append(
                f"{hypothesis['id']}: {hypothesis['explanation']} Blocker: {blocker}")
            continue
        if not hypothesis["tests"]:
            raise ValueError("An executable hypothesis must contain at least one test")
        kept = []
        duplicate_hosts = []
        for ti, test in enumerate(hypothesis["tests"]):
            duplicate_pair = next(((h, t) for h in supervised for t in h["tests"]
                                   if operation_signature(t) == operation_signature(test)), None)
            if duplicate_pair:
                host, duplicate = duplicate_pair
                if host not in duplicate_hosts:
                    duplicate_hosts.append(host)
                if duplicate["expected_observation"] != test["expected_observation"]:
                    duplicate["expected_observation"] += " Additional proposed expectation: " + test["expected_observation"]
                continue
            test["id"] = f"agent-{hi+1}-{ti+1}-{test['id'][:170]}"
            kept.append(test)
        if kept:
            hypothesis["id"] = f"agent-{hi+1}-{hypothesis['id'][:175]}"
            hypothesis["tests"] = kept
            hypotheses.append(hypothesis)
        else:
            for host in duplicate_hosts:
                host["explanation"] += " Additional proposed explanation: " + hypothesis["explanation"]
    plan["hypotheses"] = hypotheses
    return validate_plan(plan)


def _reason(store, reasoner, manifest, role, round_number, payload, schema):
    row = store.db.execute("SELECT * FROM reasoning WHERE case_id=? AND role=? AND round=?",
                           (manifest["id"], role, round_number)).fetchone()
    if row:
        if row["state"] == "completed":
            value = json.loads(row["payload"])
            if role == "critic":
                # Keep old checkpoints immutable. They lack an executable
                # follow-up contract, so read them without guessing a test.
                value = validate_critic(value, manifest,
                                        followups_remaining=payload["followups_remaining"],
                                        attempts=payload["attempts"], allow_legacy=True)
            return value
        raise ReasoningFailure("An interrupted or failed reasoning call is retained; it is not automatically retried")
    limits = store.get("limits")
    used = store.get("codex_calls_used", 0)
    if used >= limits["codex_calls"]:
        raise ResourceStop("codex_call_budget")
    with store.db:
        store.db.execute("INSERT INTO reasoning(case_id,role,round,state) VALUES (?,?,?,'running')",
                         (manifest["id"], role, round_number))
        store.db.execute("UPDATE meta SET value=? WHERE key='codex_calls_used'", (canonical(used+1),))
    directory = store.workspace/"private"/manifest["id"]/f"{round_number}-{role}"

    def on_start(pid):
        with store.db:
            store.db.execute("UPDATE reasoning SET pid=?,pid_created=? WHERE case_id=? AND role=? AND round=?",
                             (pid, psutil.Process(pid).create_time(), manifest["id"], role, round_number))
    try:
        value, metadata = reasoner.run(role=role, payload=payload, schema=schema,
                                      directory=directory, workspace=store.workspace, limits=limits,
                                      deadline=store.get("deadline"),
                                      stopped=lambda: store.get("stop_requested", False), on_start=on_start)
        with store.db:
            store.db.execute("UPDATE reasoning SET state='completed',payload=?,metadata=? WHERE case_id=? AND role=? AND round=?",
                             (canonical(value), canonical(metadata), manifest["id"], role, round_number))
        return value
    except Exception as exc:
        with store.db:
            store.db.execute("UPDATE reasoning SET state='failed',error=? WHERE case_id=? AND role=? AND round=?",
                             (str(exc), manifest["id"], role, round_number))
        raise


def _freeze(store, manifest, round_number, plan, question, question_design=None, followup_test=None):
    validate_plan(plan)
    for hypothesis in plan["hypotheses"]:
        for test in hypothesis["tests"]:
            # Unknown methods/metrics/recipes cannot become executable plans.
            # Valid scientific requests with insufficient evidence (for example
            # undocumented dependence) are blocked by the registered operation.
            validate_operation(test, manifest)
    if question_design is not None:
        validate_question_design(question_design, manifest)
    if round_number > 0 and followup_test is None:
        raise ValueError("A new follow-up specification requires a structured decisive test")
    if followup_test is not None:
        validate_followup_test(followup_test, manifest)
    spec = validate_spec({"schema_version": "1.0", "id": f"{manifest['id']}-round-{round_number}",
                          "manifest_id": manifest["id"], "manifest_sha256": digest(manifest),
                          "catalogue_release_id": manifest["catalogue_release_id"], "question": question,
                          "created_at": now(), "round": round_number, "evidence": manifest["artifacts"],
                          "plan": plan, "permitted_actions": list(OPERATIONS),
                          "exposure": {"previously_exposed": manifest["semantics"]["exposed"],
                                       "usage": "exploration", "independent_validation": False},
                          "budget": store.get("limits"),
                          **({"question_design": question_design} if question_design is not None else {}),
                          **({"followup_test": followup_test} if followup_test is not None else {})})
    sha = digest(spec)
    # Commit the specification before any numerical job is started.
    with store.db:
        store.db.execute("INSERT INTO plans VALUES (?,?,?,?,?)",
                         (spec["id"], manifest["id"], round_number, canonical(spec), sha))
        store.db.execute("UPDATE cases SET round=?,state='running' WHERE id=?",
                         (round_number, manifest["id"]))
    write_new(store.workspace/"specs"/f"{spec['id']}.json", spec)
    return spec


def _attempt(store, manifest, resolver, spec, test):
    # Different wording/IDs do not rerun the same test over the same evidence.
    identity = {k: v for k, v in test.items() if k not in {"id", "expected_observation"}}
    fingerprint = digest({"manifest": digest(manifest), "operation": identity})
    previous = store.db.execute("SELECT record FROM attempts WHERE fingerprint=?", (fingerprint,)).fetchone()
    if previous:
        return json.loads(previous[0])
    identifier = f"attempt-{fingerprint[:24]}"
    record = {"id": identifier, "plan_sha256": digest(spec), "operation": test,
              "status": "interrupted", "started_at": now(), "finished_at": now(),
              "receipt_sha256": None, "receipt": None, "error": "Job has not completed"}
    with store.db:
        store.db.execute("INSERT INTO attempts(id,case_id,fingerprint,record,state) VALUES (?,?,?,?,'running')",
                         (identifier, manifest["id"], fingerprint, canonical(record)))
    job = store.workspace/"jobs"/identifier
    job.mkdir(parents=True, exist_ok=False)
    request, response = job/"request.private.json", job/"result.private.json"
    limits = store.get("limits")
    write_new(request, {"manifest": manifest, "resolver": resolver, "operation": test,
                        "expected_code_sha256": store.get("code_sha256"),
                        "cache": str(store.workspace/"cache"),
                        "max_bytes": max(0, limits["workspace_bytes"]-workspace_size(store.workspace))})

    def on_start(pid):
        with store.db:
            store.db.execute("UPDATE attempts SET pid=?,pid_created=? WHERE id=?",
                             (pid, psutil.Process(pid).create_time(), identifier))
    resource_error = None
    try:
        result = run_guarded([sys.executable, "-m", "rewirebench.research.operations", str(request), str(response)],
                             cwd=job, workspace=store.workspace, stdout=job/"stdout.private.log",
                             stderr=job/"stderr.private.log", seconds=limits["experiment_seconds"],
                             memory_bytes=limits["memory_bytes"], workspace_bytes=limits["workspace_bytes"],
                             deadline=store.get("deadline"), stopped=lambda: store.get("stop_requested", False),
                             env=local_environment(), on_start=on_start)
        if result["returncode"] or not response.exists():
            record.update(status="failed", error="Registered operation failed; inspect private process log")
        else:
            value = read_json(response)
            if value["receipt"] and value["receipt"]["code_sha256"] != store.get("code_sha256"):
                raise ValueError("Operation receipt came from changed runner code")
            record.update(status=value["status"], receipt=value["receipt"], error=value["error"])
            if value["receipt"]:
                record["receipt_sha256"] = digest(value["receipt"])
    except ResourceStop as exc:
        record.update(status="interrupted", error=str(exc))
        if str(exc) in {"campaign_deadline", "workspace_limit", "stop_requested"}:
            resource_error = exc
    except (ValueError, OSError) as exc:
        record.update(status="failed", error=str(exc))
    record["finished_at"] = now()
    with store.db:
        store.db.execute("UPDATE attempts SET state=?,record=? WHERE id=?",
                         (record["status"], canonical(record), identifier))
    if resource_error:
        raise resource_error
    return record


def _records(store, case_id):
    return [json.loads(r[0]) for r in store.db.execute(
        "SELECT record FROM attempts WHERE case_id=? ORDER BY rowid", (case_id,))]


def _records_through_round(store, case_id, round_number):
    """Resume must not expose later attempts to an earlier saved critic decision."""
    hashes = {r[0] for r in store.db.execute(
        "SELECT sha256 FROM plans WHERE case_id=? AND round<=?", (case_id, round_number))}
    return [record for record in _records(store, case_id) if record["plan_sha256"] in hashes]


def _bundle(store, manifest, status, critic=None, error=None):
    specs = [json.loads(r[0]) for r in store.db.execute(
        "SELECT spec FROM plans WHERE case_id=? ORDER BY round", (manifest["id"],))]
    attempts = _records(store, manifest["id"])
    reasoning = [json.loads(r[0]) for r in store.db.execute(
        "SELECT metadata FROM reasoning WHERE case_id=? AND state='completed' ORDER BY id", (manifest["id"],))]
    limitations = list(manifest["verification"]["limitations"])
    limitations.extend(["Exploratory analysis of exposed outcomes; no independent validation or novelty claim.",
                        "AI-assisted interpretation requires human scientific review before publication.",
                        "All intervals are descriptive and unadjusted; unsuccessful tests remain in the report."])
    design_artifact = None
    design_row = store.db.execute(
        "SELECT payload FROM reasoning WHERE case_id=? AND role='hypothesizer' AND round=0 AND state='completed'",
        (manifest["id"],)).fetchone()
    if design_row:
        try:
            design = validate_question_design(json.loads(design_row[0]), manifest)
            design_artifact = {"design": design, "sha256": digest(design)}
        except ValueError:
            limitations.append("The hypothesis response failed validation; its private checkpoint is retained.")
    if error:
        limitations.append(error)
    if critic is None:
        # A budget stop or later planning failure must not discard an earlier
        # completed interpretation. Validate checkpoints, retaining their bytes.
        for row in store.db.execute(
                "SELECT round,payload FROM reasoning WHERE case_id=? AND role='critic' "
                "AND state='completed' ORDER BY round DESC", (manifest["id"],)):
            try:
                critic = validate_critic(
                    json.loads(row["payload"]), manifest,
                    followups_remaining=store.get("limits")["followup_rounds"]-row["round"],
                    attempts=_records_through_round(store, manifest["id"], row["round"]),
                    allow_legacy=True)
            except ValueError:
                limitations.append("A saved critic response failed validation; its interpretation was not imported.")
                continue
            if critic["next_test"] is not None:
                attempt = next((a for a in attempts if operation_signature(a["operation"]) ==
                                operation_signature(critic["next_test"])), None)
                if attempt is None:
                    limitations.append("The latest completed critic requested a follow-up that was not executed before the campaign stopped.")
                elif attempt["status"] != "completed":
                    limitations.append("The requested follow-up operation was attempted but did not complete; no finding is established by it.")
                else:
                    limitations.append("The requested follow-up operation completed without a validated follow-up critique; retained findings concern earlier evidence.")
                limitations.extend(["The pending follow-up question was:", critic["next_question"]])
            break
    if critic:
        limitations.extend(critic["limitations"])
    findings = critic["findings"] if critic else []
    if not critic:
        for attempt in attempts:
            if attempt["receipt"] and attempt["operation"]["kind"] == "replay":
                checks = attempt["receipt"]["numerical"]["checks"]
                findings.append(f"Metric replay: {sum(c['status']=='passed' for c in checks)}/{len(checks)} checks passed.")
    value = {"schema_version": "1.0", "id": f"{manifest['id']}-{store.get('id')}",
             "catalogue_release_id": manifest["catalogue_release_id"], "manifest_id": manifest["id"],
             "title": manifest["title"], "question": manifest["question"], "status": status,
             "claim_level": "exploratory", "outcome": critic["outcome"] if critic else
                 ("blocked" if status in {"blocked", "failed", "stopped"} else "inconclusive"),
             "created_at": store.get("created_at"), "plan_sha256": digest(specs),
             "attempts": attempts, "findings": findings, "limitations": list(dict.fromkeys(limitations)),
             "review": {"status": "pending", "method": "ai_assisted"},
             "artifacts": manifest["artifacts"], "specs": specs,
             "execution": {"campaign_id": store.get("id"), "code_sha256": store.get("code_sha256"),
                           "codex_calls": len(list(store.db.execute("SELECT id FROM reasoning WHERE case_id=?", (manifest["id"],)))),
                           "reasoning": reasoning}}
    if design_artifact:
        value["question_design_artifact"] = design_artifact
    return validate_bundle(value)


def investigate(store, manifest, resolver, reasoner):
    limits, critic = store.get("limits"), None
    mode = store.get("mode")
    question_design = None
    if mode != "replay-only":
        question_design = _reason(
            store, reasoner, manifest, "hypothesizer", 0,
            {"manifest": manifest, "registered_operations": REGISTRY_HELP,
             "instructions": (
                 "Before planning experiments, propose one to three precise candidate questions within "
                 "the supplied research scope and rank them by explanatory value and present testability. "
                 "For each define population, comparison and measured outcome, a falsifiable hypothesis, "
                 "a competing explanation, and distinguishable supporting and contradicting results. "
                 "Explain what we would learn and name confounders, missing evidence and future independent "
                 "validation. Use the manifest's actual methods and annotations; their availability does "
                 "not prove subgroup variation, independence, a biological effect or novelty. Choose a "
                 "decisive operation permitted by the response schema. Only give a null test with an "
                 "explicit blocker if this question cannot be distinguished using available operations "
                 "and evidence. Select the strongest testable question; choose null only when none are "
                 "testable. Do not present missing literature review as completed or claim novelty. "
                 "The supervisor will include the selected decisive test in the initial plan, after "
                 "evidence verification. Additional experiment planning happens in a separate session."
             )}, hypothesis_schema(manifest))
        validate_question_design(question_design, manifest)
        selected = selected_question(question_design)
        if selected is None:
            saved = store.db.execute("SELECT spec FROM plans WHERE case_id=? AND round=0",
                                     (manifest["id"],)).fetchone()
            if not saved:
                _freeze(store, manifest, 0, _replay_plan(manifest), manifest["question"], question_design)
            error = "No candidate research question has a decisive test with the available evidence and operations"
            store.set_case(manifest["id"], "blocked", error=error,
                           bundle=_bundle(store, manifest, "blocked", error=error))
            return
    for round_number in range(1 if mode == "replay-only" else limits["followup_rounds"]+1):
        saved = store.db.execute("SELECT spec FROM plans WHERE case_id=? AND round=?",
                                 (manifest["id"], round_number)).fetchone()
        if saved:
            spec = json.loads(saved[0])
        else:
            question = (critic or {}).get("next_question") or (
                selected_question(question_design)["question"] if question_design else manifest["question"])
            followup_test = critic["next_test"] if round_number > 0 else None
            if mode == "replay-only":
                plan = _replay_plan(manifest)
            else:
                schema = planner_schema(manifest, initial=round_number == 0,
                                        question_design=question_design, followup_test=followup_test)
                supervised = _supervised_hypotheses(
                    manifest, initial=round_number == 0, question_design=question_design,
                    followup_test=followup_test, question=question)
                payload = {"manifest": manifest, "question": question,
                           "question_design": question_design if round_number == 0 else None,
                           "initial_question_design": question_design,
                           "followup_test": followup_test,
                           "registered_operations": REGISTRY_HELP, "round": round_number,
                           "planner_instructions": "Address the current question. The supervisor supplies exactly the operations in supervisor_tests; initial_question_design is historical context on follow-up rounds. Propose only additional hypotheses that distinguish alternatives or confounders. An executable hypothesis has blocker=null and one or two registered tests. A blocked hypothesis has an explicit blocker and tests=[]; it becomes a requested_tools review item and must never use a dummy operation. You may propose no additional hypotheses. Every scheduled test executes unconditionally unless evidence prerequisites or resource limits stop it; prose stopping rules cannot skip tests. Do not repeat supervisor tests or move available operations into requested_tools prose. Choose exactly the signatures and manifest values allowed by the response schema. Use subgroups for per-category coverage; coverage itself has no field argument. Null argument slots must stay null. Availability of an operation is not evidence for a biological explanation.",
                           "previous_attempts": _records(store, manifest["id"]),
                           "supervisor_tests": [t for h in supervised for t in h["tests"]],
                           "critic": critic, "core_tests_added_by_supervisor":
                               _core_hypothesis(manifest) if round_number == 0 else None}
                plan = _reason(store, reasoner, manifest, "planner", round_number, payload, schema)
                validate_proposed_plan(plan, manifest, initial=round_number == 0,
                                       question_design=question_design, followup_test=followup_test)
                plan = merge_plan(manifest, plan, initial=round_number == 0,
                                  question_design=question_design, followup_test=followup_test, question=question)
            spec = _freeze(store, manifest, round_number, plan, question,
                           question_design if round_number == 0 else None, followup_test=followup_test)
        for hypothesis in spec["plan"]["hypotheses"]:
            for test in hypothesis["tests"]:
                if time.time() >= store.get("deadline"):
                    raise ResourceStop("campaign_deadline")
                attempt = _attempt(store, manifest, resolver, spec, test)
                if test["kind"] in {"verify", "replay"} and attempt["status"] != "completed":
                    bundle = _bundle(store, manifest, "blocked", error="Evidence prerequisites failed; dependent analyses were not run")
                    store.set_case(manifest["id"], "blocked", error="Evidence prerequisites failed", bundle=bundle)
                    return
        if mode == "replay-only":
            break
        remaining = limits["followup_rounds"]-round_number
        attempts = _records_through_round(store, manifest["id"], round_number)
        critic = _reason(store, reasoner, manifest, "critic", round_number,
                         {"manifest": manifest, "specification": spec,
                          "question_design": spec.get("question_design"),
                          "initial_question_design": question_design,
                          "critic_instructions": "Evaluate the current specification.question and its selected decisive test or followup_test against the predeclared expectations. initial_question_design records the original candidates and is historical context, not a requirement that later questions stay identical. Separate evidence for the hypothesis from alternatives and confounders; identify missing validation. A completed test does not by itself answer the question. Request a follow-up only if budget remains and a new registered operation materially distinguishes a remaining uncertainty. Supply both next_question and next_test for that request; otherwise follow_up=false and both are null. Missing tools belong in limitations, never dummy operations. Do not claim novelty from hypothesis generation.",
                          "attempts": attempts,
                          "followups_remaining": remaining}, critic_schema(manifest, followups_remaining=remaining))
        critic = validate_critic(critic, manifest, followups_remaining=remaining, attempts=attempts)
        if not critic["follow_up"]:
            break
    records = _records(store, manifest["id"])
    completed = any(r["status"] == "completed" for r in records)
    status = "completed" if completed else "blocked"
    bundle = _bundle(store, manifest, status, critic)
    store.set_case(manifest["id"], status, bundle=bundle)


def run_campaign(workspace, *, reasoner=None, executable="codex"):
    store = Store(workspace)
    try:
        with store.lease():
            if store.get("code_sha256") != sdk._code_digest():
                raise ValueError("Runner code changed since campaign creation; retain this campaign and start a new one")
            if store.get("state") == "completed":
                return store.status()
            store.put("stop_requested", False)
            store.put("state", "running")
            store.put("error", None)
            if time.time() >= store.get("deadline"):
                store.put("state", "stopped")
                store.put("error", "campaign_deadline")
                return store.status()
            resolver = read_json(store.workspace/"resolver.private.json")
            if store.get("mode") == "codex" and reasoner is None:
                try:
                    reasoner = CodexReasoner(executable)
                except ReasoningFailure as exc:
                    store.put("state", "blocked")
                    store.put("error", str(exc))
                    return store.status()
            manifests = [json.loads(r[0]) for r in store.db.execute("SELECT manifest FROM cases")]
            # Priority uses manifest metadata only. Loading tables and computing
            # metrics always happens in supervised subprocesses, not this process.
            ordered = sorted(({"manifest_id": m["id"], "priority":
                               sum(any(v is not None and v < 0 for v in ms.values())
                                   for ms in m["expected_metrics"].values())}
                              for m in manifests), key=lambda x: (-x["priority"], x["manifest_id"]))
            lookup = {m["id"]: m for m in manifests}
            for candidate in ordered:
                identifier = candidate["manifest_id"]
                case = store.case(identifier)
                if case["state"] in {"completed", "blocked", "failed"}:
                    continue
                manifest = lookup[identifier]
                try:
                    investigate(store, manifest, resolver, reasoner)
                except ResourceStop as exc:
                    bundle = _bundle(store, manifest, "stopped", error=str(exc))
                    store.set_case(identifier, "stopped", error=str(exc), bundle=bundle)
                    store.put("state", "stopped")
                    store.put("error", str(exc))
                    return store.status()
                except ReasoningFailure as exc:
                    bundle = _bundle(store, manifest, "blocked", error=str(exc))
                    store.set_case(identifier, "blocked", error=str(exc), bundle=bundle)
                    store.put("state", "blocked")
                    store.put("error", str(exc))
                    return store.status()
                except (ValueError, OSError) as exc:
                    bundle = _bundle(store, manifest, "failed", error=str(exc))
                    store.set_case(identifier, "failed", error=str(exc), bundle=bundle)
            store.put("state", "completed")
            return store.status()
    finally:
        store.close()


def stop_campaign(workspace):
    store = Store(workspace)
    try:
        store.put("stop_requested", True)
        return store.status()
    finally:
        store.close()


def export_campaign(workspace, output):
    store, destination = Store(workspace), Path(output)
    try:
        if destination.exists():
            raise ValueError("Export destination must be new; earlier exports remain immutable")
        if store.get("state") == "running":
            raise ValueError("Stop the campaign before exporting a consistent snapshot")
        bundles = [json.loads(r[0]) for r in store.db.execute("SELECT bundle FROM cases WHERE bundle IS NOT NULL ORDER BY id")]
        destination.mkdir(parents=True)
        inventory = []
        for bundle in bundles:
            validate_bundle(bundle)
            # Only prose/error fields can be redacted. Frozen specs and receipts
            # stay byte-semantically intact so their checksums remain verifiable.
            public = copy.deepcopy(bundle)
            for key in ("findings", "limitations"):
                public[key] = redact_public(public[key])
            for attempt in public["attempts"]:
                attempt["error"] = redact_public(attempt["error"])
            validate_bundle(public)
            assert_public(public)
            name = bundle["id"]+".json"
            write_new(destination/name, public)
            inventory.append({"id": bundle["id"], "file": name, "semantic_sha256": digest(public)})
        write_new(destination/"index.json", {"schema_version": "1.0", "campaign_id": store.get("id"),
                                            "review_status": "pending", "uploaded": False, "bundles": inventory})
        return {"output": str(destination), "bundles": len(bundles), "uploaded": False,
                "review_status": "pending"}
    finally:
        store.close()


if __name__ == "__main__":
    scan_request = read_json(sys.argv[1])
    write_new(sys.argv[2], scan(scan_request["manifests"], scan_request["resolver"]))
