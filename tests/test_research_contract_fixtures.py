"""The same portable examples are consumed by the database's TypeScript tests."""
from pathlib import Path

import pytest
from rewirebench.research import contracts


def test_portable_contract_examples_and_hashes():
    fixture = contracts.read_json(Path(contracts.__file__).parent/"contract-fixtures.json")
    validators = {"manifest": contracts.validate_manifest, "specification": contracts.validate_spec,
                  "bundle": contracts.validate_bundle}
    for kind, value in fixture["valid"].items():
        validators[kind](value)
        assert contracts.digest(value) == fixture["expected_hashes"][kind]
    for example in fixture["invalid"]:
        with pytest.raises(ValueError):
            validators[example["contract"]](example["value"])


def test_public_strings_cannot_disclose_local_paths_or_private_fields():
    for value in ({"url": "file:///private/example"}, {"text": "/var/folders/path with spaces/data"},
                  {"local_path": "redacted"}, {"nested": {"env": {"key": "value"}}}):
        with pytest.raises(ValueError):
            contracts.assert_public(value)


def test_model_output_schemas_use_explicit_types_and_closed_required_objects():
    import copy
    for schema in (contracts.QUESTION_DESIGN_SCHEMA, contracts.PLAN_SCHEMA, contracts.CRITIC_SCHEMA):
        contracts.validate_model_schema(schema)
    untyped = copy.deepcopy(contracts.PLAN_SCHEMA)
    del untyped["properties"]["multiple_testing"]["type"]
    with pytest.raises(ValueError, match="explicit type"):
        contracts.validate_model_schema(untyped)
    optional = copy.deepcopy(contracts.CRITIC_SCHEMA)
    optional["required"].remove("next_question")
    with pytest.raises(ValueError, match="every field required"):
        contracts.validate_model_schema(optional)
    untyped_enum = copy.deepcopy(contracts.PLAN_SCHEMA)
    del untyped_enum["properties"]["hypotheses"]["items"]["properties"]["tests"]["items"]["anyOf"][-1]["properties"]["recipe"]["type"]
    with pytest.raises(ValueError, match="explicit type"):
        contracts.validate_model_schema(untyped_enum)
