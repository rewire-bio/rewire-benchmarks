"""Versioned interchange contracts and strict, shared semantic validation.

Public JSON never carries a local resolver. Hashes of JSON values use RFC 8785
canonical JSON (not file bytes); this preserves Python/JavaScript parity.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from urllib.parse import urlparse

import rfc8785
from jsonschema import Draft202012Validator, ValidationError

VERSION = "1.0"
OPERATIONS = ("verify", "replay", "coverage", "paired", "subgroups", "bootstrap",
              "sensitivity", "local_recipe")
RECIPES = ("sdk:train-mean-v1", "sdk:sequence-composition-v1", "sdk:mfass-kmer-v2",
           "sdk:seeded-random-v1", "sdk:esm2-8m-v1")
OUTCOMES = ("data_or_method_explanation", "exploratory_biological_hypothesis",
            "inconclusive", "blocked")


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


def arr(items, **kwargs):
    return {"type": "array", "items": items, **kwargs}


STR = {"type": "string", "minLength": 1, "maxLength": 12000}
NULL_STR = {"anyOf": [STR, {"type": "null"}]}
SHA = {"type": "string", "pattern": "^[a-f0-9]{64}$"}
ID = {"type": "string", "pattern": "^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,199}$"}
RELEASE = {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}-[a-f0-9]{12}$"}
PUBLIC_ID = {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]{0,254}$"}
STRINGS = arr(STR, maxItems=100)
NUM = {"type": ["number", "null"]}
ARTIFACT_SCHEMA = obj({"id": ID, "role": STR, "sha256": SHA, "format": STR,
                       "uri": {"type": ["string", "null"]}, "semantic_sha256": SHA},
                      ["id", "role", "sha256", "format", "uri"])
NULL = {"type": "null"}


def operation_schema(kind, *, method_schema=STR, field_schema=STR,
                     metric_schema=NULL_STR, recipe_schema=None):
    """Each signature has only meaningful argument slots; unused slots are null."""
    return obj({
        "id": ID, "kind": {"type": "string", "const": kind},
        "methods": arr(method_schema, minItems=2 if kind in {"paired", "bootstrap"} else 0,
                       maxItems=0 if kind in {"verify", "local_recipe"} else 8),
        "metric": metric_schema if kind in {"paired", "bootstrap", "subgroups"} else NULL,
        "field": field_schema if kind == "subgroups" else NULL,
        "recipe": (recipe_schema or {"type": "string", "enum": list(RECIPES)})
                  if kind == "local_recipe" else NULL,
        "expected_observation": STR,
    })


OPERATION_SCHEMA = {"anyOf": [operation_schema(kind, metric_schema=STR if kind == "bootstrap" else NULL_STR)
                               for kind in OPERATIONS]}
QUESTION_DESIGN_SCHEMA = obj({
    "candidates": arr(obj({
        "id": ID, "question": STR, "population": STR, "comparison": STR, "outcome": STR,
        "hypothesis": STR, "alternative_explanation": STR,
        "supporting_result": STR, "contradicting_result": STR,
        "confounders": STRINGS, "why_interesting": STR, "missing_evidence": STRINGS,
        "validation_needed": STR,
        "decisive_test": {"anyOf": [OPERATION_SCHEMA, NULL]}, "blocker": NULL_STR,
    }), minItems=1, maxItems=3),
    "selected_candidate_id": {"anyOf": [ID, NULL]}, "selection_reason": STR,
    "novelty_status": {"type": "string", "const": "unverified"},
})
PLAN_SCHEMA = obj({
    "hypotheses": arr(obj({"id": ID, "explanation": STR,
                            "tests": arr(OPERATION_SCHEMA, minItems=1, maxItems=4)}),
                      minItems=1, maxItems=3),
    "multiple_testing": {"type": "string", "const": "descriptive_only"},
    "stopping_rule": STR, "requested_tools": arr(STR, maxItems=8),
})
CRITIC_SCHEMA = obj({"findings": STRINGS, "limitations": STRINGS,
                     "outcome": {"type": "string", "enum": list(OUTCOMES)}, "follow_up": {"type": "boolean"},
                     "next_question": NULL_STR, "next_test": {"anyOf": [OPERATION_SCHEMA, NULL]}})


REGISTRY_HELP = {
    "coverage": "Counts scored and missing predictions on the whole test population; methods selects methods or [] means all. metric, field and recipe must be null. For coverage within a registered annotation category use subgroups, which reports per-method scored counts in each category.",
    "paired": "Compare at least two explicitly named methods on identical common scored rows. metric can select one registered metric, or null reports registered metrics. field and recipe are null. Effects are candidate minus reference; these are descriptive comparisons.",
    "subgroups": "field must name one registered annotation. Returns group counts, outcome means, per-method scored counts and metrics. Numeric fields use a fixed annotation-only quartile rule; existing categorical labels are retained. methods selects methods or [] means all; metric may select a registered metric or be null. recipe is null. No p-values or independent validation.",
    "bootstrap": "Requires at least two explicitly named methods, a named common metric, documented independent_unit and complete group labels. Resamples whole groups with fixed seed and 500 draws. Undefined/single-class draws are tracked; more than 5% blocks interpretation. field and recipe are null. Intervals are exploratory and unadjusted, never confirmatory.",
    "sensitivity": "For selected methods or [] for all, compare unchanged scores with diagnostic sign reversal and a constant-ranking control; also report unique score counts. metric, field and recipe must be null. This does not change declared score direction and never selects the better direction after seeing outcomes.",
    "local_recipe": "Run exactly one manifest-authorized, pinned local SDK recipe. methods must be []; metric and field must be null. SDK fitting only receives training/allowed validation labels. Missing prepared snapshots, source pins, matching environment or weights block execution; no package installation or arbitrary code is available.",
}


def planner_schema(manifest, *, initial, question_design=None, followup_test=None):
    """The model can only propose executable additional tests for this case."""
    methods = list(manifest["expected_metrics"])
    method_schema = {"type": "string", "enum": methods}
    shared = sorted(set.intersection(*(set(v) for v in manifest["expected_metrics"].values())))
    metric_schema = {"type": ["string", "null"], "enum": [None, *shared]}
    fields = manifest["semantics"]["subgroup_fields"]
    recipes = [r for r in manifest["local_recipes"]
               if not (initial and r == "sdk:train-mean-v1")]
    kinds = ["coverage", "sensitivity"]
    if len(methods) >= 2:
        kinds.append("paired")
    if fields:
        kinds.append("subgroups")
    if len(methods) >= 2 and shared and manifest["semantics"]["independent_unit"]:
        kinds.append("bootstrap")
    if recipes:
        kinds.append("local_recipe")
    variants = [operation_schema(
        kind, method_schema=method_schema,
        field_schema={"type": "string", "enum": fields} if fields else NULL,
        metric_schema={"type": "string", "enum": shared} if kind == "bootstrap" else metric_schema,
        recipe_schema={"type": "string", "enum": recipes} if recipes else NULL,
    ) for kind in kinds]
    schema = copy.deepcopy(PLAN_SCHEMA)
    hypotheses = schema["properties"]["hypotheses"]
    # Reserve slots for every test the supervisor actually supplies this round.
    hypotheses["maxItems"] = (1 if question_design else 2) if initial else (2 if followup_test else 3)
    if (initial and question_design) or followup_test:
        hypotheses["minItems"] = 0
    # This is a model-only shape. Blocked questions are converted to review items
    # before freezing, so historical public plans need no new fields.
    hypothesis = hypotheses["items"]
    # Reserve lossless public request capacity even if every proposed hypothesis
    # is blocked. Combined ID, explanation and blocker remain below STR's limit.
    schema["properties"]["requested_tools"]["maxItems"] = 8-hypotheses["maxItems"]
    hypothesis["properties"]["explanation"] = {**STR, "maxLength": 5500}
    hypothesis["properties"]["blocker"] = {"anyOf": [{**STR, "maxLength": 5500}, NULL]}
    hypothesis["required"].append("blocker")
    tests = hypotheses["items"]["properties"]["tests"]
    tests["minItems"] = 0
    tests["maxItems"] = 2
    tests["items"] = {"anyOf": variants}
    return validate_model_schema(schema)


def critic_schema(manifest, *, followups_remaining):
    """A follow-up must name a real operation, never an instruction in prose."""
    schema = copy.deepcopy(CRITIC_SCHEMA)
    if followups_remaining <= 0:
        schema["properties"]["follow_up"] = {"type": "boolean", "const": False}
        schema["properties"]["next_question"] = NULL
        schema["properties"]["next_test"] = NULL
    else:
        operations = planner_schema(manifest, initial=False)["properties"]["hypotheses"]["items"]["properties"]["tests"]["items"]
        schema["properties"]["next_test"] = {"anyOf": [*operations["anyOf"], NULL]}
    return validate_model_schema(schema)


def hypothesis_schema(manifest):
    """Question design can nominate only a case's executable initial diagnostics."""
    schema = copy.deepcopy(QUESTION_DESIGN_SCHEMA)
    operations = planner_schema(manifest, initial=True)["properties"]["hypotheses"]["items"]["properties"]["tests"]["items"]
    schema["properties"]["candidates"]["items"]["properties"]["decisive_test"] = {
        "anyOf": [*operations["anyOf"], NULL]}
    return validate_model_schema(schema)


MANIFEST_SCHEMA = obj({
    "schema_version": {"type": "string", "const": VERSION}, "id": PUBLIC_ID, "title": STR, "question": STR,
    "catalogue_release_id": RELEASE, "dataset_id": STR, "evaluation_ids": STRINGS,
    "protocol_id": STR, "sdk_protocol_id": STR, "runner_code_sha256": SHA,
    "artifacts": arr(ARTIFACT_SCHEMA, minItems=1),
    "table_artifact_id": ID,
    "semantics": obj({"target": {"type": "string", "enum": ["binary", "continuous"]}, "outcome": STR,
                      "unit": STR, "score_direction": {"type": "string", "const": "higher"},
                      "join_key": {"type": "string", "const": "id"}, "independent_unit": NULL_STR,
                      "subgroup_fields": STRINGS, "exposed": {"type": "boolean"},
                      "split": {"type": "string", "const": "test"}}),
    "expected_metrics": {"type": "object", "minProperties": 1,
                         "additionalProperties": {"type": "object", "minProperties": 1,
                                                  "additionalProperties": NUM}},
    "metric_tolerance": {"type": "number", "minimum": 0, "maximum": .01},
    "verification": obj({"verified_at": STR, "checks": arr(obj({"check": STR,
                         "status": {"type": "string", "enum": ["passed", "failed", "missing"]},
                         "detail": STR})), "limitations": STRINGS}),
    "local_recipes": arr({"type": "string", "enum": list(RECIPES)}, uniqueItems=True),
}, ["schema_version", "id", "title", "question", "catalogue_release_id", "dataset_id",
    "evaluation_ids", "protocol_id", "artifacts", "table_artifact_id", "semantics",
    "expected_metrics", "metric_tolerance", "verification", "local_recipes"])
LIMITS_SCHEMA = obj({
    "campaign_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 28800},
    "experiment_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 3600},
    "codex_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 900},
    "codex_calls": {"type": "integer", "minimum": 0, "maximum": 24},
    "memory_bytes": {"type": "integer", "minimum": 1, "maximum": 8 * 1024**3},
    "workspace_bytes": {"type": "integer", "minimum": 1, "maximum": 20 * 1024**3},
    "followup_rounds": {"type": "integer", "minimum": 0, "maximum": 2},
})
DEFAULT_LIMITS = {"campaign_seconds": 28800, "experiment_seconds": 3600,
                  "codex_seconds": 900, "codex_calls": 24, "memory_bytes": 8 * 1024**3,
                  "workspace_bytes": 20 * 1024**3, "followup_rounds": 2}
SPEC_SCHEMA = obj({
    "schema_version": {"type": "string", "const": VERSION}, "id": ID, "manifest_id": ID,
    "manifest_sha256": SHA, "catalogue_release_id": RELEASE, "question": STR,
    "created_at": STR, "round": {"type": "integer", "minimum": 0, "maximum": 2},
    "evidence": arr(ARTIFACT_SCHEMA, minItems=1), "plan": PLAN_SCHEMA,
    "permitted_actions": arr({"type": "string", "enum": list(OPERATIONS)}),
    "exposure": obj({"previously_exposed": {"type": "boolean"},
                      "usage": {"type": "string", "const": "exploration"}, "independent_validation": {"type": "boolean", "const": False}}),
    "budget": LIMITS_SCHEMA,
})
# Optional only at the interchange boundary: historical frozen specs stay intact.
SPEC_SCHEMA["properties"]["question_design"] = QUESTION_DESIGN_SCHEMA
SPEC_SCHEMA["properties"]["followup_test"] = OPERATION_SCHEMA
RECEIPT_SCHEMA = obj({"operation_id": ID, "kind": {"type": "string", "enum": list(OPERATIONS)},
                      "manifest_sha256": SHA, "table_sha256": SHA, "code_sha256": SHA,
                      "numerical": {"type": "object"}, "limitations": STRINGS})
ATTEMPT_SCHEMA = obj({
    "id": ID, "plan_sha256": SHA, "operation": OPERATION_SCHEMA,
    "status": {"type": "string", "enum": ["completed", "failed", "blocked", "interrupted"]},
    "started_at": STR, "finished_at": STR,
    "receipt_sha256": {"anyOf": [SHA, {"type": "null"}]},
    "receipt": {"anyOf": [RECEIPT_SCHEMA, {"type": "null"}]}, "error": NULL_STR,
})
BUNDLE_SCHEMA = obj({
    "schema_version": {"type": "string", "const": VERSION}, "id": PUBLIC_ID, "catalogue_release_id": RELEASE,
    "manifest_id": ID, "title": STR, "question": STR,
    "status": {"type": "string", "enum": ["completed", "blocked", "failed", "stopped"]},
    "claim_level": {"type": "string", "const": "exploratory"}, "outcome": {"type": "string", "enum": list(OUTCOMES)},
    "created_at": STR, "plan_sha256": SHA, "attempts": arr(ATTEMPT_SCHEMA),
    "findings": STRINGS, "limitations": STRINGS,
    "review": obj({"status": {"type": "string", "const": "pending"}, "method": {"type": "string", "const": "ai_assisted"}}),
    "artifacts": arr(ARTIFACT_SCHEMA), "specs": arr(SPEC_SCHEMA),
    "execution": obj({"campaign_id": ID, "code_sha256": SHA,
                      "codex_calls": {"type": "integer", "minimum": 0},
                      "reasoning": arr(obj({"role": {"type": "string", "enum": ["hypothesizer", "planner", "critic"]},
                                            "model": STR, "cli_version": STR,
                                            "usage": {"type": "object"}}))}),
}, ["schema_version", "id", "catalogue_release_id", "manifest_id", "title", "question",
    "status", "claim_level", "outcome", "created_at", "plan_sha256", "attempts", "findings",
    "limitations", "review", "artifacts", "specs"])
BUNDLE_SCHEMA["properties"]["question_design_artifact"] = obj({
    "design": QUESTION_DESIGN_SCHEMA, "sha256": SHA})


def canonical(value):
    return rfc8785.dumps(value).decode()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique)


def validate(value, schema):
    try:
        canonical(value)  # Reject nonfinite values even in open numerical receipt trees.
        Draft202012Validator(schema).validate(value)
    except ValidationError as exc:
        location = ".".join(map(str, exc.absolute_path)) or "root"
        raise ValueError(f"Invalid research contract at {location}: {exc.validator} constraint failed") from exc
    except Exception as exc:
        raise ValueError(f"Invalid research contract: {exc}") from exc
    return value


def validate_model_schema(schema):
    """Check the stricter structured-output subset before making a Codex call.

    JSON Schema permits an untyped enum/const; the model response-format API
    requires an explicit type. Every output object also needs closed properties
    and every declared property must be required (nullable for optional values).
    """
    def visit(node, location):
        if not isinstance(node, dict) or ("type" not in node and "anyOf" not in node):
            raise ValueError(f"Model-output schema requires an explicit type at {location}")
        if ("const" in node or "enum" in node) and "type" not in node:
            raise ValueError(f"Model-output enum/const requires an explicit type at {location}")
        if node.get("type") == "object":
            properties = node.get("properties", {})
            if node.get("additionalProperties") is not False or set(node.get("required", [])) != set(properties):
                raise ValueError(f"Model-output objects must be closed with every field required at {location}")
            for key, value in properties.items():
                visit(value, f"{location}.{key}")
        if node.get("type") == "array":
            visit(node.get("items"), f"{location}[]")
        for index, child in enumerate(node.get("anyOf", [])):
            visit(child, f"{location}.anyOf[{index}]")
    visit(schema, "response")
    return schema


def _artifacts(values):
    ids = [v["id"] for v in values]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate artifact ID")
    for artifact in values:
        uri = artifact["uri"]
        if uri is not None:
            parsed = urlparse(uri)
            if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                raise ValueError("Artifact URI must be a public HTTPS location or null")


def validate_manifest(value):
    validate(value, MANIFEST_SCHEMA)
    _artifacts(value["artifacts"])
    if value["table_artifact_id"] not in {a["id"] for a in value["artifacts"]}:
        raise ValueError("Table artifact is absent from manifest")
    if value.get("sdk_protocol_id", value["protocol_id"]) == "mfass-v1":
        raise ValueError("The archived MFASS v1 cohort is forbidden for new investigations")
    if len(set(value["evaluation_ids"])) != len(value["evaluation_ids"]):
        raise ValueError("Duplicate evaluation reference")
    if len(set(value["semantics"]["subgroup_fields"])) != len(value["semantics"]["subgroup_fields"]):
        raise ValueError("Duplicate subgroup field")
    return value


def validate_plan(value):
    validate(value, PLAN_SCHEMA)
    tests = [t for h in value["hypotheses"] for t in h["tests"]]
    ids = [t["id"] for t in tests]
    if len(tests) > 8 or len(ids) != len(set(ids)):
        raise ValueError("At most eight uniquely identified tests are allowed per plan")
    return value


def operation_signature(test):
    """Execution identity excludes bookkeeping IDs and explanatory prose."""
    return {k: test[k] for k in ("kind", "methods", "metric", "field", "recipe")}


def validate_proposed_plan(value, manifest, *, initial, question_design=None, followup_test=None):
    validate(value, planner_schema(manifest, initial=initial, question_design=question_design,
                                   followup_test=followup_test))
    for hypothesis in value["hypotheses"]:
        if (hypothesis["blocker"] is None) != bool(hypothesis["tests"]):
            raise ValueError("A proposed hypothesis needs executable tests or an explicit blocker with no tests")
    return value


def validate_followup_test(test, manifest=None):
    schema = (planner_schema(manifest, initial=False)["properties"]["hypotheses"]["items"]["properties"]["tests"]["items"]
              if manifest is not None else OPERATION_SCHEMA)
    validate(test, schema)
    if test["kind"] in {"verify", "replay"}:
        raise ValueError("A follow-up requires an additional test beyond supervised verification and replay")
    if len(test["methods"]) != len(set(test["methods"])):
        raise ValueError("A follow-up test cannot repeat method names")
    return test


def validate_critic(value, manifest, *, followups_remaining, attempts=(), allow_legacy=False):
    if allow_legacy and "next_test" not in value:
        legacy = copy.deepcopy(CRITIC_SCHEMA)
        del legacy["properties"]["next_test"]
        legacy["required"].remove("next_test")
        validate(value, legacy)
        value = copy.deepcopy(value)
        if value["follow_up"]:
            value["limitations"].append(
                "Legacy critic requested a follow-up without a structured decisive test; "
                "no follow-up was scheduled and no operation was inferred from prose.")
        value.update(follow_up=False, next_question=None, next_test=None)
    validate(value, critic_schema(manifest, followups_remaining=followups_remaining))
    has_question, has_test = value["next_question"] is not None, value["next_test"] is not None
    if value["follow_up"] != (has_question and has_test) or has_question != has_test:
        raise ValueError("A follow-up needs both a precise next question and a decisive next test")
    if value["next_test"] is not None:
        validate_followup_test(value["next_test"], manifest)
        if any(operation_signature(attempt["operation"]) == operation_signature(value["next_test"])
               for attempt in attempts):
            raise ValueError("A follow-up must add a new operation; recorded attempts are never automatically rerun")
    return value


def validate_question_design(value, manifest=None):
    validate(value, hypothesis_schema(manifest) if manifest is not None else QUESTION_DESIGN_SCHEMA)
    candidates = value["candidates"]
    ids = [candidate["id"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Question candidate IDs must be unique")
    for candidate in candidates:
        if (candidate["decisive_test"] is None) != (candidate["blocker"] is not None):
            raise ValueError("A question needs a decisive test or an explicit blocker, not both")
        test = candidate["decisive_test"]
        if test and len(test["methods"]) != len(set(test["methods"])):
            raise ValueError("A decisive test cannot repeat method names")
    testable = {c["id"] for c in candidates if c["decisive_test"] is not None}
    selected = value["selected_candidate_id"]
    if selected is None:
        if testable:
            raise ValueError("Select a testable question when one is available")
    elif selected not in testable:
        raise ValueError("Selected question must identify a testable candidate")
    return value


def selected_question(design):
    return next((c for c in design["candidates"]
                 if c["id"] == design["selected_candidate_id"]), None)


def validate_spec(value):
    validate(value, SPEC_SCHEMA)
    validate_plan(value["plan"])
    _artifacts(value["evidence"])
    if "followup_test" in value:
        test = validate_followup_test(value["followup_test"])
        if value["round"] == 0:
            raise ValueError("A follow-up test belongs to a later round only")
        if not any(operation_signature(t) == operation_signature(test)
                   for h in value["plan"]["hypotheses"] for t in h["tests"]):
            raise ValueError("Follow-up question's decisive test is absent from the frozen plan")
    if "question_design" in value:
        design = validate_question_design(value["question_design"])
        if value["round"] != 0:
            raise ValueError("Question design belongs to the initial round only")
        selected = selected_question(design)
        if selected:
            if value["question"] != selected["question"]:
                raise ValueError("Initial specification must use the selected precise question")
            tests = [t for h in value["plan"]["hypotheses"] for t in h["tests"]]
            if not any(operation_signature(t) == operation_signature(selected["decisive_test"])
                       for t in tests):
                raise ValueError("Selected question's decisive test is absent from the frozen plan")
    return value


def validate_bundle(value):
    validate(value, BUNDLE_SCHEMA)
    _artifacts(value["artifacts"])
    design_artifact = value.get("question_design_artifact")
    if design_artifact:
        validate_question_design(design_artifact["design"])
        if digest(design_artifact["design"]) != design_artifact["sha256"]:
            raise ValueError("Question design artifact checksum mismatch")
    plans = {digest(validate_spec(s)): s for s in value["specs"]}
    if len(plans) != len(value["specs"]):
        raise ValueError("Duplicate frozen specification")
    for spec in plans.values():
        if (spec["manifest_id"] != value["manifest_id"] or
                spec["catalogue_release_id"] != value["catalogue_release_id"] or
                spec["evidence"] != value["artifacts"]):
            raise ValueError("Specification lineage differs from investigation")
        if (design_artifact and "question_design" in spec and
                spec["question_design"] != design_artifact["design"]):
            raise ValueError("Frozen question design differs from its artifact")
    if value["plan_sha256"] != digest(value["specs"]):
        raise ValueError("Bundle plan hash does not match frozen specifications")
    designs = [s["question_design"] for s in value["specs"] if "question_design" in s]
    if design_artifact:
        designs.append(design_artifact["design"])
    if (any(d["selected_candidate_id"] is None for d in designs) and
            (value["status"] == "completed" or value["attempts"])):
        raise ValueError("An investigation without a testable question cannot execute tests or complete")
    attempt_ids = set()
    for attempt in value["attempts"]:
        if attempt["id"] in attempt_ids or attempt["plan_sha256"] not in plans:
            raise ValueError("Duplicate attempt or absent frozen plan")
        attempt_ids.add(attempt["id"])
        spec = plans[attempt["plan_sha256"]]
        if attempt["operation"] not in [t for h in spec["plan"]["hypotheses"] for t in h["tests"]]:
            raise ValueError("Attempt operation is absent from its frozen specification")
        receipt = attempt["receipt"]
        if (receipt is None) != (attempt["receipt_sha256"] is None):
            raise ValueError("Receipt and hash must both be present or absent")
        if receipt and digest(receipt) != attempt["receipt_sha256"]:
            raise ValueError("Numerical receipt checksum mismatch")
        if receipt and (receipt["operation_id"] != attempt["operation"]["id"] or
                        receipt["kind"] != attempt["operation"]["kind"] or
                        receipt["manifest_sha256"] != spec["manifest_sha256"] or
                        receipt["table_sha256"] not in {a["sha256"] for a in spec["evidence"]
                                                      if a["role"] in {"table", "normalized_table"}}):
            raise ValueError("Numerical receipt lineage does not match the frozen operation")
        if receipt and "execution" in value and receipt["code_sha256"] != value["execution"]["code_sha256"]:
            raise ValueError("Numerical receipt came from a different implementation")
        if attempt["status"] == "completed" and receipt is None:
            raise ValueError("A completed attempt requires numerical evidence")
    return value


def validate_table(table, manifest):
    if set(table) != {"schema_version", "metric_kind", "rows"} or table["schema_version"] != VERSION:
        raise ValueError("Invalid normalized table contract")
    if table["metric_kind"] not in {"mfass", "proteingym", "regression"} or not table["rows"]:
        raise ValueError("Unknown metric kind or empty table")
    seen, methods = set(), set(manifest["expected_metrics"])
    for row in table["rows"]:
        if set(row) != {"id", "y", "group", "split", "scores", "features"}:
            raise ValueError("Unexpected normalized row fields")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError("Duplicate or empty normalized row identifier")
        seen.add(row["id"])
        if row["split"] != "test":
            raise ValueError("Investigation tables contain test rows only; training uses SDK artifacts")
        if not isinstance(row["features"], dict) or not isinstance(row["scores"], dict):
            raise ValueError("Scores and features must be mappings")  # noqa: TRY004
        if set(row["scores"]) != methods:
            raise ValueError("Every row must reconcile every method, with null for missing scores")
        if row["group"] is not None and not isinstance(row["group"], str):
            raise ValueError("Group identifiers must be strings or null")
        for v in [row["y"], *[s for s in row["scores"].values() if s is not None]]:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError("Outcomes and predictions must be finite numbers")
        if manifest["semantics"]["target"] == "binary" and row["y"] not in {0, 1}:
            raise ValueError("Binary target must be zero or one")
        for k, v in row["features"].items():
            if not isinstance(k, str) or (v is not None and not isinstance(v, (str, int, float))):
                raise ValueError("Feature values must be strings, finite numbers or null")
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError("Nonfinite annotation")
    return table


def redact_public(value):
    """Remove accidental machine paths from errors and model-written prose."""
    if isinstance(value, dict):
        return {k: redact_public(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_public(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"(?:/Users/|/home/|/private/|/tmp/|/var/folders/|[A-Za-z]:\\)[^\"'\n]+",
                      "[local path]", value)
    return value


def assert_public(value):
    """Fail closed when unhashed prose sanitization cannot safely remove a path."""
    if isinstance(value, str) and re.search(
        r"(?:file://|/(?:Users|home|private|tmp|var/folders)/|[A-Z]:\\)", value
    ):
        raise ValueError("Private machine path cannot enter a research export")
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in {"local_path", "resolver", "stdout", "stderr", "raw_events", "environment", "env"}:
                raise ValueError("Private execution fields cannot enter a research export")
            assert_public(child)
    if isinstance(value, list):
        for child in value:
            assert_public(child)
