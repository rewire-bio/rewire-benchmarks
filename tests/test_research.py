"""Scientific lineage, resource bounds and recovery invariants for research campaigns."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import psutil
import pytest
from rewirebench import sdk
from rewirebench.research.campaign import (
    create_campaign,
    export_campaign,
    merge_plan,
    operation,
    run_campaign,
    scan,
)
from rewirebench.research.contracts import (
    BUNDLE_SCHEMA,
    MANIFEST_SCHEMA,
    SPEC_SCHEMA,
    critic_schema,
    digest,
    file_sha,
    hypothesis_schema,
    planner_schema,
    read_json,
    validate,
    validate_bundle,
    validate_critic,
    validate_manifest,
    validate_proposed_plan,
    validate_question_design,
    validate_spec,
    validate_table,
)
from rewirebench.research.operations import (
    BlockedOperation,
    bootstrap,
    execute_operation,
    metric_values,
    paired,
    replay,
    subgroup_bins,
    validate_operation,
)
from rewirebench.research.reasoner import ReasoningFailure
from rewirebench.research.state import Store
from rewirebench.research.supervisor import ResourceStop, run_guarded


@pytest.fixture
def evidence(tmp_path):
    table = {"schema_version": "1.0", "metric_kind": "mfass", "rows": [
        {"id": f"r{i}", "y": i % 2, "group": f"g{i//4}", "split": "test",
         "scores": {"baseline": float(i%2), "candidate": float(i%2)+i/100},
         "features": {"region": "exon" if i < 12 else "intron", "distance": i}}
        for i in range(24)]}
    path = tmp_path/"table.json"
    path.write_text(json.dumps(table))
    manifest = {"schema_version": "1.0", "id": "test-investigation", "title": "Test investigation",
                "question": "Does the result replicate under the registered metric?",
                "catalogue_release_id": "2026-09-23-0123456789ab", "dataset_id": "test-dataset",
                "evaluation_ids": ["test-evaluation"], "protocol_id": "test-protocol",
                "sdk_protocol_id": "mfass-v2", "runner_code_sha256": sdk._code_digest(),
                "artifacts": [{"id": "test-table", "role": "table", "sha256": file_sha(path),
                               "format": "json", "uri": None}], "table_artifact_id": "test-table",
                "semantics": {"target": "binary", "outcome": "disruption", "unit": "variant",
                              "score_direction": "higher", "join_key": "id", "independent_unit": "exon",
                              "subgroup_fields": ["region", "distance"], "exposed": True, "split": "test"},
                "expected_metrics": {m: metric_values(table["rows"], m, "mfass") for m in ("baseline", "candidate")},
                "metric_tolerance": 1e-9, "verification": {"verified_at": "2026-09-23T00:00:00Z",
                                                          "checks": [], "limitations": []},
                "local_recipes": []}
    return manifest, table, {"test-table": str(path)}


def test_replay_coverage_and_paired_population(evidence):
    manifest, table, _ = evidence
    table["rows"][0]["scores"]["candidate"] = None
    result = replay(table, manifest, ["baseline", "candidate"])
    assert result["coverage"]["candidate"] == {"scored": 23, "denominator": 24, "missing": 1}
    assert paired(table, ["baseline", "candidate"])["pairs"][0]["common_n"] == 23


def test_undefined_correlation_and_shifted_ndcg():
    from sklearn.metrics import ndcg_score
    y = [400., 500., 600., 650.]
    rows = [{"y": v, "scores": {"m": 3.}, "features": {}} for v in y]
    metrics = metric_values(rows, "m", "regression")
    assert metrics["spearman"] is None and metrics["pearson"] is None
    assert metrics["ndcg"] == pytest.approx(ndcg_score([[0., 100., 200., 250.]], [[3.]*4]))
    assert metrics["ndcg"] != pytest.approx(ndcg_score([y], [[3.]*4]))


def test_table_and_artifact_guards(evidence, tmp_path):
    manifest, table, resolver = evidence
    duplicate = copy.deepcopy(table)
    duplicate["rows"][1]["id"] = duplicate["rows"][0]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_table(duplicate, manifest)
    changed = copy.deepcopy(table)
    changed["rows"][0]["split"] = "train"
    with pytest.raises(ValueError, match="test rows"):
        validate_table(changed, manifest)
    changed["rows"][0]["split"] = "test"
    changed["rows"][0]["y"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_table(changed, manifest)
    Path(resolver["test-table"]).write_text("{}")
    with pytest.raises(BlockedOperation, match="checksum"):
        execute_operation(manifest, resolver, operation("verify", "verify"), tmp_path/"out")
    manifest["sdk_protocol_id"] = "mfass-v1"
    with pytest.raises(ValueError, match="archived"):
        validate_manifest(manifest)


def test_registered_operations_and_independence(evidence):
    manifest, table, _ = evidence
    with pytest.raises(ValueError):
        validate_operation({**operation("malicious", "replay"), "command": "anything"}, manifest)
    with pytest.raises(BlockedOperation, match="Unknown"):
        validate_operation(operation("invalid", "replay", methods=["unregistered"]), manifest)
    manifest["semantics"]["independent_unit"] = None
    with pytest.raises(BlockedOperation, match="independence"):
        bootstrap(table, manifest, ["baseline", "candidate"], "auroc")
    manifest["semantics"]["independent_unit"] = "exon"
    result = bootstrap(table, manifest, ["baseline", "candidate"], "auroc", draws=20)
    assert result["bootstrap"][0]["independent_groups"] == 6
    groups, bins = subgroup_bins(table["rows"], "distance")
    assert len(groups) == 4 and bins["edges"] == [5.75, 11.5, 17.25]


def proposed_plan(tests):
    return {"hypotheses": [{"id": "evidence-and-method", "explanation": "A testable alternative.", "tests": tests}],
            "multiple_testing": "descriptive_only", "stopping_rule": "Complete the additional registered tests.",
            "requested_tools": []}


def proposed_model_plan(tests):
    value = proposed_plan(tests)
    value["hypotheses"][0]["blocker"] = None
    return value


def question_design(*, blocked=False):
    """A falsifiable comparison whose necessary test the supervisor must preserve."""
    candidate = {
        "id": "region-performance",
        "question": "Does prediction performance differ between exon and intron variants?",
        "population": "Variants in the pinned test table, grouped by registered region.",
        "comparison": "Candidate and baseline performance in exon and intron variants.",
        "outcome": "Registered AUROC and scored counts within each region.",
        "hypothesis": "The candidate's performance deficit is concentrated in intron variants.",
        "alternative_explanation": "Regional outcome prevalence or scoring coverage differs.",
        "supporting_result": "A candidate deficit is present in introns but absent in exons.",
        "contradicting_result": "Candidate and baseline AUROC agree in both regions.",
        "confounders": ["Outcome prevalence", "Missing predictions", "Shared exon groups"],
        "why_interesting": "Localizing a deficit can distinguish a context issue from a global failure.",
        "missing_evidence": ["Independent genes and assay replication"],
        "validation_needed": "Repeat the fixed comparison on previously unexposed genes.",
        "decisive_test": operation("compare-regions", "subgroups", field="region", metric="auroc",
                                   methods=["baseline", "candidate"]),
        "blocker": None,
    }
    if blocked:
        candidate["decisive_test"] = None
        candidate["blocker"] = "Required cellular context is absent from the registered evidence."
        candidate["missing_evidence"].append("Cellular context for every variant")
    return {"candidates": [candidate],
            "selected_candidate_id": None if blocked else candidate["id"],
            "selection_reason": "No candidate is testable with registered evidence." if blocked else
                "The region comparison is executable and distinguishes two competing explanations.",
            "novelty_status": "unverified"}


def test_hypotheses_only_select_tests_available_in_the_registered_evidence(evidence):
    manifest, _, _ = evidence
    design = question_design()
    validate_question_design(design, manifest)
    invalid_tests = [
        operation("repeat-verify", "verify"),
        operation("repeat-replay", "replay"),
        operation("not-registered", "subgroups", field="cell-type"),
        operation("not-a-method", "paired", methods=["baseline", "unregistered"]),
        operation("invalid-field-argument", "coverage", field="region"),
        operation("invalid-metric", "subgroups", field="region", metric="invented"),
    ]
    manifest["local_recipes"] = ["sdk:train-mean-v1", "sdk:seeded-random-v1"]
    invalid_tests.append(operation("repeat-automatic-mean", "local_recipe", recipe="sdk:train-mean-v1"))
    for invalid in invalid_tests:
        changed = copy.deepcopy(design)
        changed["candidates"][0]["decisive_test"] = invalid
        with pytest.raises(ValueError):
            validate(changed, hypothesis_schema(manifest))
        with pytest.raises(ValueError):
            validate_question_design(changed, manifest)
    allowed = copy.deepcopy(design)
    allowed["candidates"][0]["decisive_test"] = operation(
        "random-control", "local_recipe", recipe="sdk:seeded-random-v1")
    validate_question_design(allowed, manifest)
    manifest["semantics"]["independent_unit"] = None
    no_independence = copy.deepcopy(design)
    no_independence["candidates"][0]["decisive_test"] = operation(
        "bootstrap", "bootstrap", methods=["baseline", "candidate"], metric="auroc")
    with pytest.raises(ValueError):
        validate_question_design(no_independence, manifest)


@pytest.mark.parametrize("select_duplicate", [False, True])
def test_all_candidates_reject_duplicate_methods_even_when_not_selected(evidence, select_duplicate):
    manifest, _, _ = evidence
    design = question_design()
    invalid = copy.deepcopy(design["candidates"][0])
    invalid["id"] = "duplicate-comparison"
    invalid["decisive_test"]["methods"] = ["baseline", "baseline"]
    design["candidates"].append(invalid)
    if select_duplicate:
        design["selected_candidate_id"] = invalid["id"]
    # The interchange contract and manifest-aware execution contract must agree.
    for registered_manifest in (None, manifest):
        with pytest.raises(ValueError):
            validate_question_design(design, registered_manifest)


@pytest.mark.parametrize("mistake", [
    "empty", "too-many", "duplicate-id", "unknown-selection", "unselected-testable",
    "selected-blocked", "blocked-with-test", "no-test-no-blocker", "asserted-novelty",
])
def test_question_selection_cannot_hide_an_untestable_or_ambiguous_design(evidence, mistake):
    manifest, _, _ = evidence
    design = question_design()
    if mistake == "empty":
        design["candidates"] = []
    elif mistake == "too-many":
        design["candidates"] = [dict(design["candidates"][0], id=f"candidate-{i}") for i in range(4)]
        design["selected_candidate_id"] = "candidate-0"
    elif mistake == "duplicate-id":
        design["candidates"].append(copy.deepcopy(design["candidates"][0]))
    elif mistake == "unknown-selection":
        design["selected_candidate_id"] = "never-proposed"
    elif mistake == "unselected-testable":
        design["selected_candidate_id"] = None
    elif mistake == "selected-blocked":
        blocked = question_design(blocked=True)["candidates"][0]
        blocked["id"] = "needs-cell-context"
        design["candidates"].append(blocked)
        design["selected_candidate_id"] = blocked["id"]
    elif mistake == "blocked-with-test":
        design["candidates"][0]["blocker"] = "Missing cellular context."
    elif mistake == "no-test-no-blocker":
        design["candidates"][0]["decisive_test"] = None
    else:
        design["novelty_status"] = "novel"
    with pytest.raises(ValueError):
        validate_question_design(design, manifest)


def test_frozen_question_keeps_its_decisive_test_and_can_only_be_designed_initially():
    from rewirebench.research import contracts

    fixture = read_json(Path(contracts.__file__).parent/"contract-fixtures.json")
    spec = copy.deepcopy(fixture["valid"]["specification"])
    design = question_design()
    selected = design["candidates"][0]
    decisive = copy.deepcopy(selected["decisive_test"])
    decisive["id"] = "supervisor-namespaced-test"
    spec["round"] = 0
    spec["question"] = selected["question"]
    spec["question_design"] = design
    spec["plan"] = proposed_plan([decisive])
    validate_spec(spec)
    for changed in (
        {**spec, "round": 1},
        {**spec, "question": "An unrelated research question."},
        {**spec, "plan": proposed_plan([operation("unrelated-test", "coverage")])},
    ):
        with pytest.raises(ValueError):
            validate_spec(changed)
    wrong_population = copy.deepcopy(spec)
    wrong_population["plan"]["hypotheses"][0]["tests"][0]["methods"] = ["candidate"]
    with pytest.raises(ValueError):
        validate_spec(wrong_population)


def test_live_planner_regression_excludes_core_and_invalid_operation_arguments(evidence):
    manifest, _, _ = evidence
    schema = planner_schema(manifest, initial=True)
    # The real first response repeated these two already supervised tests.
    repeated = proposed_model_plan([operation("verify-evidence", "verify"), operation("replay-metrics", "replay")])
    with pytest.raises(ValueError):
        validate(repeated, schema)
    # It also tried coverage(field=mutation_class), a nonexistent signature.
    invalid = proposed_model_plan([operation("mutation-class-coverage", "coverage", field="region")])
    with pytest.raises(ValueError):
        validate(invalid, schema)
    valid = proposed_model_plan([operation("class-coverage", "subgroups", field="region"),
                           operation("overall-coverage", "coverage")])
    validate(valid, schema)
    too_many = copy.deepcopy(valid)
    too_many["hypotheses"].append({"id": "unbounded-extra", "explanation": "Another diagnostic.", "blocker": None,
                                   "tests": [operation("extra", "sensitivity")]})
    with pytest.raises(ValueError):
        validate(too_many, planner_schema(manifest, initial=True, question_design=question_design()))
    with pytest.raises(ValueError):
        validate(proposed_model_plan([operation("wrong-field", "subgroups", field="not-an-annotation")]), schema)
    with pytest.raises(ValueError):
        validate(proposed_model_plan([operation("wrong-method", "coverage", methods=["not-a-method"])]), schema)
    manifest["semantics"]["independent_unit"] = None
    with pytest.raises(ValueError):
        validate(proposed_model_plan([operation("bootstrap", "bootstrap", methods=["baseline", "candidate"], metric="auroc")]),
                 planner_schema(manifest, initial=True))


def test_core_duplicate_merging_and_id_namespacing_preserve_valid_tests(evidence):
    manifest, _, _ = evidence
    verify = operation("verify-evidence", "verify", expected="The model's explicit additional check expectation.")
    replay_test = operation("replay-metrics", "replay", expected="Replay must agree at the declared precision.")
    source = proposed_plan([verify, replay_test])
    additional = [operation("verify-evidence", "coverage"),
                  operation("verify-evidence", "subgroups", field="region", metric="auroc")]
    source["hypotheses"].append({"id": "evidence-and-method", "explanation": "Coverage may vary by class.", "tests": additional})
    merged = merge_plan(manifest, source, initial=True)
    tests = [t for h in merged["hypotheses"] for t in h["tests"]]
    assert len(tests) == len({t["id"] for t in tests}) == 4
    assert verify["expected_observation"] in tests[0]["expected_observation"]
    assert replay_test["expected_observation"] in tests[1]["expected_observation"]
    assert len({h["id"] for h in merged["hypotheses"]}) == 2
    assert [{k: v for k, v in t.items() if k != "id"} for t in tests[2:]] == [
        {k: v for k, v in t.items() if k != "id"} for t in additional]
    assert source["hypotheses"][0]["tests"][0]["id"] == "verify-evidence"


class FakeReasoner:
    def __init__(self, *, fail=False, fail_role=None, follow_up=False, design=None,
                 requested_tools=None, empty_plan=False):
        self.calls, self.fail, self.follow_up = [], fail, follow_up
        self.fail_role, self.empty_plan = fail_role, empty_plan
        self.design = question_design() if design is None else design
        self.requested_tools = requested_tools or []
        self.payloads = []

    def run(self, *, role, payload, **kwargs):
        self.calls.append(role)
        self.payloads.append(copy.deepcopy(payload))
        if self.fail or role == self.fail_role:
            raise ReasoningFailure("Codex account limit reached; campaign stopped")
        if role == "hypothesizer":
            value = copy.deepcopy(self.design)
        elif role == "planner":
            value = {"hypotheses": [{"id": "score-direction", "explanation": "Score direction may explain disagreement.",
                     "blocker": None,
                     "tests": [operation("test-direction", "sensitivity")]}],
                     "multiple_testing": "descriptive_only", "stopping_rule": "Stop after the registered diagnostic.",
                     "requested_tools": self.requested_tools}
            if self.empty_plan:
                value["hypotheses"] = []
        else:
            follow_up = self.follow_up and payload["followups_remaining"] > 0
            value = {"findings": ["Metric replay supports the recorded scores."],
                     "limitations": ["No independent validation."], "outcome": "inconclusive",
                     "follow_up": follow_up, "next_question": "Does distance alter the comparison?" if follow_up else None,
                     "next_test": operation("followup-distance", "subgroups", methods=["baseline", "candidate"],
                                            field="distance", metric="auroc") if follow_up else None}
        return value, {"role": role, "model": "test-model", "cli_version": "test", "usage": {"input_tokens": 10}}


def test_campaign_freezes_deduplicates_exports_and_resumes(evidence, tmp_path):
    manifest, _, resolver = evidence
    workspace = tmp_path/"campaign"
    create_campaign(workspace, [manifest], resolver, limits={"followup_rounds": 1})
    reasoner = FakeReasoner(follow_up=True)
    state = run_campaign(workspace, reasoner=reasoner)
    assert state["state"] == "completed" and state["codex_calls_used"] == 5
    assert state["attempts"] == [{"state": "completed", "count": 5}]
    run_campaign(workspace, reasoner=reasoner)
    assert reasoner.calls == ["hypothesizer", "planner", "critic", "planner", "critic"]
    out = tmp_path/"export"
    export_campaign(workspace, out)
    bundle = read_json(next(p for p in out.glob("*.json") if p.name != "index.json"))
    validate_bundle(bundle)
    assert len(bundle["specs"]) == 2 and len(bundle["attempts"]) == 5
    assert bundle["specs"][0]["question_design"] == reasoner.design
    assert bundle["question_design_artifact"] == {"design": reasoner.design, "sha256": digest(reasoner.design)}
    assert "question_design" not in bundle["specs"][1]
    assert bundle["claim_level"] == "exploratory" and bundle["review"]["status"] == "pending"
    assert str(tmp_path) not in json.dumps(bundle)
    bad = copy.deepcopy(bundle)
    bad["attempts"][1]["operation"]["methods"] = ["candidate"]
    with pytest.raises(ValueError, match="absent"):
        validate_bundle(bad)
    bad = copy.deepcopy(bundle)
    bad["attempts"][0]["receipt"]["operation_id"] = "invented"
    bad["attempts"][0]["receipt_sha256"] = digest(bad["attempts"][0]["receipt"])
    with pytest.raises(ValueError, match="lineage"):
        validate_bundle(bad)
    bad = copy.deepcopy(bundle)
    bad["question_design_artifact"]["sha256"] = "0"*64
    with pytest.raises(ValueError):
        validate_bundle(bad)
    bad = copy.deepcopy(bundle)
    bad["question_design_artifact"]["design"]["selection_reason"] = "A different reason than the frozen selection."
    with pytest.raises(ValueError):
        validate_bundle(bad)
    # Recomputing the artifact hash cannot disguise disagreement with the frozen design.
    bad["question_design_artifact"]["sha256"] = digest(bad["question_design_artifact"]["design"])
    with pytest.raises(ValueError):
        validate_bundle(bad)


def saved_bundle(workspace, manifest):
    store = Store(workspace)
    try:
        return json.loads(store.case(manifest["id"])["bundle"])
    finally:
        store.close()


def test_selected_question_needs_no_artificial_complementary_planner_test(evidence, tmp_path):
    manifest, _, resolver = evidence
    design = question_design()
    empty = {**proposed_plan([]), "hypotheses": []}
    validate(empty, planner_schema(manifest, initial=True, question_design=design))
    merged = merge_plan(manifest, empty, initial=True, question_design=design)
    assert [t["kind"] for h in merged["hypotheses"] for t in h["tests"]] == [
        "verify", "replay", "subgroups"]
    workspace = tmp_path/"campaign"
    create_campaign(workspace, [manifest], resolver, limits={"followup_rounds": 0})
    reasoner = FakeReasoner(design=design, empty_plan=True)
    state = run_campaign(workspace, reasoner=reasoner)
    assert state["cases"][0]["state"] == "completed"
    assert reasoner.calls == ["hypothesizer", "planner", "critic"]
    bundle = saved_bundle(workspace, manifest)
    validate_bundle(bundle)
    assert [a["operation"]["kind"] for a in bundle["attempts"]] == ["verify", "replay", "subgroups"]
    assert all(a["status"] == "completed" for a in bundle["attempts"])
    assert len(bundle["specs"][0]["plan"]["hypotheses"]) == 2


def test_selected_test_runs_even_when_planner_only_mentions_it_as_a_requested_tool(evidence, tmp_path):
    manifest, _, resolver = evidence
    workspace = tmp_path/"campaign"
    create_campaign(workspace, [manifest], resolver, limits={"followup_rounds": 0})
    reasoner = FakeReasoner(requested_tools=["Compare performance by the registered region annotation."])
    state = run_campaign(workspace, reasoner=reasoner)
    assert state["cases"][0]["state"] == "completed"
    assert reasoner.calls == ["hypothesizer", "planner", "critic"]
    planner_payload = reasoner.payloads[1]
    assert planner_payload["question_design"] == reasoner.design
    assert planner_payload["question"] == reasoner.design["candidates"][0]["question"]
    bundle = saved_bundle(workspace, manifest)
    validate_bundle(bundle)
    assert bundle["specs"][0]["question"] == planner_payload["question"]
    selected_attempts = [a for a in bundle["attempts"] if a["operation"]["kind"] == "subgroups"]
    assert len(selected_attempts) == 1
    selected = selected_attempts[0]
    assert selected["status"] == "completed"
    assert selected["operation"]["field"] == "region"
    assert selected["operation"]["methods"] == ["baseline", "candidate"]
    groups = selected["receipt"]["numerical"]["subgroups"]
    assert [(g["label"], g["n"]) for g in groups] == [("exon", 12), ("intron", 12)]
    # Provenance covers both the question choice and numerical evidence.
    assert bundle["execution"]["codex_calls"] == 3
    assert [r["role"] for r in bundle["execution"]["reasoning"]] == reasoner.calls


def test_followup_decisive_test_runs_when_planner_omits_every_additional_test(evidence, tmp_path):
    manifest, _, resolver = evidence
    workspace = tmp_path/"followup"
    create_campaign(workspace, [manifest], resolver, limits={"followup_rounds": 1})
    reasoner = FakeReasoner(follow_up=True, empty_plan=True)
    state = run_campaign(workspace, reasoner=reasoner)
    assert state["state"] == "completed" and state["codex_calls_used"] == 5
    bundle = saved_bundle(workspace, manifest)
    validate_bundle(bundle)
    spec = bundle["specs"][1]
    assert "question_design" not in spec
    assert spec["followup_test"]["field"] == "distance"
    tests = [t for h in spec["plan"]["hypotheses"] for t in h["tests"]]
    assert len(tests) == 1 and tests[0]["id"] == "followup-decisive-test"
    assert tests[0]["methods"] == ["baseline", "candidate"]
    assert any(a["operation"] == tests[0] and a["status"] == "completed" for a in bundle["attempts"])
    assert len(bundle["attempts"]) == 4  # No forced complementary or dummy operation.
    payload = reasoner.payloads[3]
    assert payload["question_design"] is None
    assert payload["initial_question_design"] == reasoner.design
    assert payload["supervisor_tests"] == tests
    assert reasoner.payloads[-1]["question_design"] is None
    assert reasoner.payloads[-1]["specification"]["followup_test"] == spec["followup_test"]


def test_followup_signature_is_frozen_and_cannot_change_population(evidence, tmp_path):
    manifest, _, resolver = evidence
    workspace = tmp_path/"followup"
    create_campaign(workspace, [manifest], resolver, limits={"followup_rounds": 1})
    run_campaign(workspace, reasoner=FakeReasoner(follow_up=True, empty_plan=True))
    spec = saved_bundle(workspace, manifest)["specs"][1]
    for edit in (
        lambda value: value.update(round=0),
        lambda value: value["plan"]["hypotheses"][0]["tests"][0].update(field="region"),
        lambda value: value["plan"]["hypotheses"][0]["tests"][0].update(methods=["candidate"]),
        lambda value: value.update(followup_test=operation("wrong", "replay")),
    ):
        changed = copy.deepcopy(spec)
        edit(changed)
        with pytest.raises(ValueError):
            validate_spec(changed)
    # Absence is permitted only at the historical interchange boundary.
    old = copy.deepcopy(spec)
    del old["followup_test"]
    validate_spec(old)


def test_live_followup_omission_regression_with_only_extras_and_a_blocked_question(evidence, tmp_path):
    manifest, _, resolver = evidence

    class ExtrasAndBlocker(FakeReasoner):
        def run(self, *, role, payload, **kwargs):
            value, metadata = super().run(role=role, payload=payload, **kwargs)
            if role == "planner" and payload["round"] == 1:
                # The live failure omitted the central comparison, assuming the
                # supervisor supplied it, and gave an unrelated test to a blocker.
                value["hypotheses"].append({
                    "id": "unavailable-group-influence", "explanation": "Can one group explain the deficit?",
                    "blocker": "A registered group-deletion influence operation is unavailable.", "tests": []})
            return value, metadata

    workspace = tmp_path/"live-regression"
    create_campaign(workspace, [manifest], resolver, limits={"followup_rounds": 1})
    state = run_campaign(workspace, reasoner=ExtrasAndBlocker(follow_up=True))
    assert state["state"] == "completed"
    bundle = saved_bundle(workspace, manifest)
    spec = bundle["specs"][1]
    tests = [t for h in spec["plan"]["hypotheses"] for t in h["tests"]]
    assert [(t["kind"], t["field"]) for t in tests] == [("subgroups", "distance"), ("sensitivity", None)]
    assert tests[0]["methods"] == ["baseline", "candidate"]
    assert len(spec["plan"]["requested_tools"]) == 1
    assert any(a["operation"] == tests[0] and a["status"] == "completed" for a in bundle["attempts"])
    assert len(bundle["attempts"]) == 5  # The repeat sensitivity test is reused; no dummy job exists.


def test_blocked_planner_questions_become_requests_without_dummy_execution(evidence, tmp_path):
    manifest, _, resolver = evidence
    blocked = proposed_model_plan([])
    blocked["hypotheses"][0]["blocker"] = "No registered group-deletion influence operation exists."
    validate_proposed_plan(blocked, manifest, initial=True, question_design=question_design())
    merged = merge_plan(manifest, blocked, initial=True, question_design=question_design())
    assert len(merged["requested_tools"]) == 1
    assert "group-deletion influence" in merged["requested_tools"][0]
    assert [t["kind"] for h in merged["hypotheses"] for t in h["tests"]] == ["verify", "replay", "subgroups"]
    assert all("blocker" not in h for h in merged["hypotheses"])
    bad = copy.deepcopy(blocked)
    bad["hypotheses"][0]["tests"] = [operation(
        "dummy", "subgroups", field="region", expected="This test is not executed; it cannot answer the question.")]
    with pytest.raises(ValueError, match="explicit blocker"):
        validate_proposed_plan(bad, manifest, initial=True, question_design=question_design())
    with pytest.raises(ValueError, match="blocked hypothesis"):
        merge_plan(manifest, bad, initial=True, question_design=question_design())
    # Empty executable hypotheses are also rejected; the planner can use [] at top level.
    with pytest.raises(ValueError, match="explicit blocker"):
        validate_proposed_plan(proposed_model_plan([]), manifest, initial=True, question_design=question_design())

    class BlockedPlanner(FakeReasoner):
        def run(self, *, role, **kwargs):
            value, metadata = super().run(role=role, **kwargs)
            return (copy.deepcopy(blocked) if role == "planner" else value), metadata

    workspace = tmp_path/"blocked-extra"
    create_campaign(workspace, [manifest], resolver, limits={"followup_rounds": 0})
    run_campaign(workspace, reasoner=BlockedPlanner())
    bundle = saved_bundle(workspace, manifest)
    assert len(bundle["attempts"]) == 3
    assert len(bundle["specs"][0]["plan"]["requested_tools"]) == 1


def critic_followup(test=None):
    return {"findings": [], "limitations": [], "outcome": "inconclusive", "follow_up": True,
            "next_question": "Does distance alter the comparison?", "next_test": test or operation(
                "next-distance", "subgroups", methods=["baseline", "candidate"], field="distance", metric="auroc")}


def test_blocked_conversion_reserves_request_slots_and_preserves_long_prose(evidence):
    manifest, _, _ = evidence
    test = critic_followup()["next_test"]
    schema = planner_schema(manifest, initial=False, followup_test=test)
    hypothesis_limit = schema["properties"]["hypotheses"]["maxItems"]
    request_limit = schema["properties"]["requested_tools"]["maxItems"]
    assert hypothesis_limit + request_limit == 8
    proposed = {**proposed_plan([]), "hypotheses": [
        {"id": "h"+str(index)+"x"*198, "explanation": "e"*5500, "blocker": "b"*5500, "tests": []}
        for index in range(hypothesis_limit)], "requested_tools": ["r"*12000]*request_limit}
    validate_proposed_plan(proposed, manifest, initial=False, followup_test=test)
    merged = merge_plan(manifest, proposed, initial=False, followup_test=test)
    assert len(merged["requested_tools"]) == 8
    assert all(len(request) <= 12000 for request in merged["requested_tools"])
    for hypothesis, request in zip(proposed["hypotheses"], merged["requested_tools"][request_limit:], strict=True):
        assert hypothesis["id"] in request
        assert hypothesis["explanation"] in request and hypothesis["blocker"] in request
    proposed["requested_tools"].append("One too many.")
    with pytest.raises(ValueError):
        validate_proposed_plan(proposed, manifest, initial=False, followup_test=test)


@pytest.mark.parametrize("follow_up,question,test", [
    (True, None, True), (True, "A question", False), (True, None, False),
    (False, "A question", True), (False, "A question", False), (False, None, True),
])
def test_critic_requires_one_complete_followup_request(evidence, follow_up, question, test):
    manifest, _, _ = evidence
    value = critic_followup()
    value.update(follow_up=follow_up, next_question=question, next_test=value["next_test"] if test else None)
    with pytest.raises(ValueError, match="both a precise"):
        validate_critic(value, manifest, followups_remaining=1)


def test_critic_followup_is_manifest_bound_new_and_within_budget(evidence):
    manifest, _, _ = evidence
    value = critic_followup()
    validate_critic(value, manifest, followups_remaining=1)
    validate(value, critic_schema(manifest, followups_remaining=1))
    for replacement in (
        {"field": "not-an-annotation"}, {"methods": ["unknown"]}, {"metric": "unknown"},
        {"methods": ["baseline", "baseline"]}, {"kind": "coverage"},
    ):
        changed = copy.deepcopy(value)
        changed["next_test"].update(replacement)
        with pytest.raises(ValueError):
            validate_critic(changed, manifest, followups_remaining=1)
    with pytest.raises(ValueError):
        validate_critic(value, manifest, followups_remaining=0)
    for status in ("completed", "failed", "blocked", "interrupted"):
        with pytest.raises(ValueError, match="new operation"):
            validate_critic(value, manifest, followups_remaining=1,
                            attempts=[{"operation": value["next_test"], "status": status}])
    shared = copy.deepcopy(manifest)
    del shared["expected_metrics"]["baseline"]["auroc"]
    with pytest.raises(ValueError):
        validate_critic(critic_followup(operation("candidate-only", "subgroups", methods=["candidate"],
                                                 metric="auroc", field="distance")), shared, followups_remaining=1)


def test_legacy_critic_checkpoint_is_preserved_without_guessing_followup(evidence, tmp_path):
    manifest, _, resolver = evidence
    workspace = tmp_path/"legacy"
    create_campaign(workspace, [manifest], resolver, limits={"followup_rounds": 1})
    reasoner = FakeReasoner()
    run_campaign(workspace, reasoner=reasoner)
    legacy = critic_followup()
    del legacy["next_test"]
    legacy["next_question"] = "Use subgroups(field=distance) next; infer the operation from this text."
    with pytest.raises(ValueError):
        validate_critic(legacy, manifest, followups_remaining=1)
    store = Store(workspace)
    try:
        with store.db:
            store.db.execute("UPDATE reasoning SET payload=? WHERE role='critic'", (json.dumps(legacy),))
        store.set_case(manifest["id"], "pending")
        store.put("state", "stopped")
    finally:
        store.close()
    state = run_campaign(workspace, reasoner=reasoner)
    assert state["state"] == "completed" and state["codex_calls_used"] == 3
    assert reasoner.calls == ["hypothesizer", "planner", "critic"]
    bundle = saved_bundle(workspace, manifest)
    assert len(bundle["specs"]) == 1 and len(bundle["attempts"]) == 4
    assert any("Legacy critic" in text for text in bundle["limitations"])
    store = Store(workspace)
    try:
        saved = json.loads(store.db.execute("SELECT payload FROM reasoning WHERE role='critic'").fetchone()[0])
        assert saved == legacy  # No checkpoint rewrite or invented next_test.
    finally:
        store.close()


def test_resume_does_not_expose_later_attempts_to_an_earlier_critic(evidence, tmp_path):
    manifest, _, resolver = evidence
    workspace = tmp_path/"resume-followup"
    create_campaign(workspace, [manifest], resolver, limits={"followup_rounds": 1})
    reasoner = FakeReasoner(follow_up=True)
    run_campaign(workspace, reasoner=reasoner)
    original = saved_bundle(workspace, manifest)
    store = Store(workspace)
    try:
        store.set_case(manifest["id"], "pending")
        store.put("state", "stopped")
    finally:
        store.close()
    state = run_campaign(workspace, reasoner=reasoner)
    assert state["state"] == "completed" and state["codex_calls_used"] == 5
    resumed = saved_bundle(workspace, manifest)
    assert resumed["specs"] == original["specs"] and resumed["attempts"] == original["attempts"]
    assert len(reasoner.calls) == 5


@pytest.mark.parametrize("call_limit,expected_specs,expected_attempts,limitation", [
    (3, 1, 4, "was not executed"), (4, 2, 5, "without a validated follow-up critique"),
])
def test_budget_stop_preserves_latest_critic_and_unfinished_followup(evidence, tmp_path, call_limit,
                                                                   expected_specs, expected_attempts, limitation):
    manifest, _, resolver = evidence
    workspace = tmp_path/"budget-followup"
    create_campaign(workspace, [manifest], resolver, limits={"followup_rounds": 1, "codex_calls": call_limit})
    state = run_campaign(workspace, reasoner=FakeReasoner(follow_up=True))
    assert state["state"] == "stopped" and state["codex_calls_used"] == call_limit
    bundle = saved_bundle(workspace, manifest)
    validate_bundle(bundle)
    assert len(bundle["specs"]) == expected_specs and len(bundle["attempts"]) == expected_attempts
    assert bundle["findings"] == ["Metric replay supports the recorded scores."]
    assert bundle["outcome"] == "inconclusive" and bundle["status"] == "stopped"
    assert any(limitation in text for text in bundle["limitations"])
    assert "Does distance alter the comparison?" in bundle["limitations"]
    store = Store(workspace)
    try:
        checkpoint = json.loads(store.db.execute("SELECT payload FROM reasoning WHERE role='critic'").fetchone()[0])
        assert checkpoint["follow_up"] and checkpoint["next_test"]["field"] == "distance"
    finally:
        store.close()


def test_no_testable_question_is_retained_and_blocks_before_planning_or_jobs(evidence, tmp_path):
    manifest, _, resolver = evidence
    workspace = tmp_path/"campaign"
    create_campaign(workspace, [manifest], resolver)
    reasoner = FakeReasoner(design=question_design(blocked=True))
    state = run_campaign(workspace, reasoner=reasoner)
    assert state["cases"][0]["state"] == "blocked"
    assert state["codex_calls_used"] == 1 and state["attempts"] == []
    assert reasoner.calls == ["hypothesizer"]
    assert not (workspace/"jobs").exists()
    bundle = saved_bundle(workspace, manifest)
    validate_bundle(bundle)
    assert bundle["outcome"] == "blocked" and len(bundle["specs"]) == 1
    assert bundle["specs"][0]["question_design"] == reasoner.design
    assert {t["kind"] for h in bundle["specs"][0]["plan"]["hypotheses"]
            for t in h["tests"]} == {"verify", "replay"}
    no_spec = copy.deepcopy(bundle)
    no_spec["specs"], no_spec["plan_sha256"] = [], digest([])
    validate_bundle(no_spec)
    no_spec["status"] = "completed"
    with pytest.raises(ValueError):
        validate_bundle(no_spec)
    run_campaign(workspace, reasoner=reasoner)
    assert reasoner.calls == ["hypothesizer"]


def test_replay_only_skips_question_design_and_all_reasoning(evidence, tmp_path):
    manifest, _, resolver = evidence
    workspace = tmp_path/"campaign"
    create_campaign(workspace, [manifest], resolver, mode="replay-only")
    reasoner = FakeReasoner(fail=True)
    state = run_campaign(workspace, reasoner=reasoner)
    assert state["cases"][0]["state"] == "completed"
    assert state["codex_calls_used"] == 0 and reasoner.calls == []
    bundle = saved_bundle(workspace, manifest)
    assert len(bundle["specs"]) == 1 and "question_design" not in bundle["specs"][0]
    assert [a["operation"]["kind"] for a in bundle["attempts"]] == ["verify", "replay"]


def test_completed_hypothesis_checkpoint_survives_interruption_without_another_call(
    evidence, tmp_path, monkeypatch,
):
    from rewirebench.research import campaign

    manifest, _, resolver = evidence
    workspace = tmp_path/"campaign"
    create_campaign(workspace, [manifest], resolver, limits={"followup_rounds": 0})
    reasoner = FakeReasoner()
    original_reason = campaign._reason
    interrupted = False

    def interrupt_after_question_checkpoint(store, reasoner, manifest, role, *args):
        nonlocal interrupted
        value = original_reason(store, reasoner, manifest, role, *args)
        if role == "hypothesizer" and not interrupted:
            interrupted = True
            raise ResourceStop("stop_requested")
        return value

    monkeypatch.setattr(campaign, "_reason", interrupt_after_question_checkpoint)
    first = run_campaign(workspace, reasoner=reasoner)
    assert first["state"] == "stopped" and first["codex_calls_used"] == 1
    assert first["attempts"] == [] and reasoner.calls == ["hypothesizer"]
    resumed = run_campaign(workspace, reasoner=reasoner)
    assert resumed["cases"][0]["state"] == "completed"
    assert resumed["codex_calls_used"] == 3
    assert reasoner.calls == ["hypothesizer", "planner", "critic"]
    assert saved_bundle(workspace, manifest)["specs"][0]["question_design"] == reasoner.design


@pytest.mark.parametrize("call_limit", [0, 1])
def test_question_design_obeys_the_existing_campaign_call_budget(evidence, tmp_path, call_limit):
    manifest, _, resolver = evidence
    workspace = tmp_path/"campaign"
    create_campaign(workspace, [manifest], resolver, limits={"codex_calls": call_limit})
    reasoner = FakeReasoner()
    state = run_campaign(workspace, reasoner=reasoner)
    assert state["state"] == "stopped" and state["error"] == "codex_call_budget"
    assert state["codex_calls_used"] == call_limit and state["attempts"] == []
    assert reasoner.calls == (["hypothesizer"] if call_limit else [])
    out = tmp_path/"export"
    export_campaign(workspace, out)
    bundle = read_json(next(p for p in out.glob("*.json") if p.name != "index.json"))
    validate_bundle(bundle)
    assert bundle["specs"] == []
    if call_limit:
        assert bundle["question_design_artifact"] == {"design": reasoner.design, "sha256": digest(reasoner.design)}
        tampered = copy.deepcopy(bundle)
        tampered["question_design_artifact"]["design"]["selection_reason"] = "Changed after checkpoint."
        with pytest.raises(ValueError):
            validate_bundle(tampered)
    else:
        assert "question_design_artifact" not in bundle
    run_campaign(workspace, reasoner=reasoner)
    assert len(reasoner.calls) == call_limit


def test_planner_failure_exports_completed_question_design_without_a_frozen_plan(evidence, tmp_path):
    manifest, _, resolver = evidence
    workspace = tmp_path/"campaign"
    create_campaign(workspace, [manifest], resolver)
    reasoner = FakeReasoner(fail_role="planner")
    state = run_campaign(workspace, reasoner=reasoner)
    assert state["cases"][0]["state"] == "blocked"
    assert state["codex_calls_used"] == 2 and state["attempts"] == []
    assert reasoner.calls == ["hypothesizer", "planner"]
    out = tmp_path/"export"
    export_campaign(workspace, out)
    bundle = read_json(next(p for p in out.glob("*.json") if p.name != "index.json"))
    validate_bundle(bundle)
    assert bundle["specs"] == []
    assert bundle["question_design_artifact"] == {"design": reasoner.design, "sha256": digest(reasoner.design)}
    run_campaign(workspace, reasoner=reasoner)
    assert reasoner.calls == ["hypothesizer", "planner"]


def test_account_failure_consumes_budget_without_retry(evidence, tmp_path):
    manifest, _, resolver = evidence
    workspace = tmp_path/"campaign"
    create_campaign(workspace, [manifest], resolver)
    reasoner = FakeReasoner(fail=True)
    state = run_campaign(workspace, reasoner=reasoner)
    assert state["state"] == "blocked" and state["codex_calls_used"] == 1
    run_campaign(workspace, reasoner=reasoner)
    assert len(reasoner.calls) == 1


def test_crash_recovery_never_silently_restarts_attempt(evidence, tmp_path):
    manifest, _, resolver = evidence
    workspace = tmp_path/"campaign"
    create_campaign(workspace, [manifest], resolver, mode="replay-only")
    run_campaign(workspace)
    store = Store(workspace)
    store.put("state", "stopped")
    with store.db:
        store.db.execute("UPDATE cases SET state='running'")
        store.db.execute("UPDATE attempts SET state='running' WHERE rowid=(SELECT MIN(rowid) FROM attempts)")
    store.close()
    status = run_campaign(workspace)
    assert status["cases"][0]["state"] == "blocked"
    assert sum(a["count"] for a in status["attempts"]) == 2
    assert {a["state"] for a in status["attempts"]} == {"completed", "interrupted"}


def test_public_export_rejects_hashed_private_prose(evidence, tmp_path):
    manifest, _, resolver = evidence
    manifest["question"] = "Read /var/folders/private-file first"
    workspace = tmp_path/"campaign"
    create_campaign(workspace, [manifest], resolver, mode="replay-only")
    run_campaign(workspace)
    with pytest.raises(ValueError, match="Private machine"):
        export_campaign(workspace, tmp_path/"export")


def test_failed_replay_blocks_later_tests(evidence, tmp_path):
    manifest, _, resolver = evidence
    manifest["expected_metrics"]["candidate"]["auroc"] = .1
    workspace = tmp_path/"campaign"
    create_campaign(workspace, [manifest], resolver)
    reasoner = FakeReasoner()
    state = run_campaign(workspace, reasoner=reasoner)
    assert state["cases"][0]["state"] == "blocked"
    store = Store(workspace)
    try:
        tests = [json.loads(r[0])["operation"]["kind"] for r in store.db.execute("SELECT record FROM attempts")]
        assert tests == ["verify", "replay"]
    finally:
        store.close()


def test_missing_campaign_is_not_created(tmp_path):
    with pytest.raises(ValueError, match="No existing"):
        Store(tmp_path)
    assert not (tmp_path/"campaign.sqlite3").exists()


def guarded(tmp_path, code, **overrides):
    settings = {"seconds": 2, "memory_bytes": 1024**3, "workspace_bytes": 1024**3,
                "deadline": time.time()+5}
    settings.update(overrides)
    return run_guarded([sys.executable, "-c", code], cwd=tmp_path, workspace=tmp_path,
                       stdout=tmp_path/"out.log", stderr=tmp_path/"err.log", **settings)


def test_supervisor_timeout_and_stdin_nonreader(tmp_path):
    start = time.monotonic()
    with pytest.raises(ResourceStop, match="timeout"):
        guarded(tmp_path, "import time;time.sleep(20)", seconds=.2, input_text="x"*2_000_000)
    assert time.monotonic()-start < 3


def test_supervisor_stops_memory_storage_and_expired_campaign(tmp_path):
    with pytest.raises(ResourceStop, match="memory"):
        guarded(tmp_path, "import time; a=bytearray(30_000_000);time.sleep(3)", memory_bytes=10_000_000)
    with pytest.raises(ResourceStop, match="workspace"):
        guarded(tmp_path, "from pathlib import Path;import time;Path('large').write_bytes(b'x'*50000);time.sleep(3)", workspace_bytes=10000)
    (tmp_path/"large").unlink()
    with pytest.raises(ResourceStop, match="deadline"):
        guarded(tmp_path, "pass", deadline=time.time()-1)


def test_supervisor_kills_orphans_on_success(tmp_path):
    child_code = "import time;time.sleep(20)"
    code = f"import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);Path('child').write_text(str(p.pid))"
    guarded(tmp_path, code)
    pid = int((tmp_path/"child").read_text())
    time.sleep(.15)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_supervisor_kills_stubborn_orphan_after_leader_exit(tmp_path):
    child_code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(20)"
    code = (f"import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c',{child_code!r}],stdout=subprocess.PIPE);"
            "p.stdout.readline();Path('child').write_text(str(p.pid))")
    guarded(tmp_path, code)
    pid = int((tmp_path/"child").read_text())
    time.sleep(.15)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_schema_json_stays_in_sync():
    root = Path(__import__("rewirebench.research.contracts", fromlist=["__file__"]).__file__).parent
    assert read_json(root/"contract-schema.json") == {
        "manifest": MANIFEST_SCHEMA, "specification": SPEC_SCHEMA, "bundle": BUNDLE_SCHEMA}


def test_scan_reports_wrinkles(evidence):
    manifest, _, resolver = evidence
    result = scan([manifest], resolver)[0]
    assert result["eligible"] and result["independent_validation"] is False
    assert all(c["status"] == "passed" for c in result["summary"]["checks"])
