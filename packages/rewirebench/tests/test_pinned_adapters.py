"""Adapters for pinned Hugging Face checkpoints, tested with no weights and no network.

The model and tokenizer are fakes with known outputs, so the pooling and scoring
arithmetic is checked without torch. The digest check runs on small files written here.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import json
import tomllib
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from rewirebench import models, recipes
from rewirebench.adapters import pinned
from rewirebench.adapters.dnabert_s import DNABERTS
from rewirebench.adapters.esm1b import ESM1b
from rewirebench.adapters.mistral_dna import MistralDNA
from rewirebench.adapters.plantrna_fm import PlantRNAFM
from rewirebench.adapters.protbert import ProtBert
from rewirebench.adapters.protgpt2 import ProtGPT2, card_text, sequence_log_likelihood

ROOT = Path(__file__).resolve().parents[3]
ZERO = "0" * 64
FAKE_TORCH = SimpleNamespace(inference_mode=contextlib.nullcontext)

# Model ID, adapter class, extra, recipe path.
ADAPTERS = [
    ("mistral-dna-138m-frozen", MistralDNA, "mistral-dna",
     "mfass/mfass-v2-mistral-dna-frozen-smoke.yaml"),
    ("dnabert-s-frozen", DNABERTS, "dnabert-s", "mfass/mfass-v2-dnabert-s-frozen-smoke.yaml"),
    ("plantrna-fm-embeddings", PlantRNAFM, "plantrna-fm",
     "mrnabench/mrnabench-plantrna-fm-smoke.yaml"),
    ("protbert-embeddings", ProtBert, "protbert", "flip2/flip2-protbert-synthetic-smoke.yaml"),
    ("esm1b-masked-marginal", ESM1b, "esm1b",
     "proteingym/proteingym-v1-3-esm1b-synthetic-smoke.yaml"),
    ("protgpt2-log-likelihood", ProtGPT2, "protgpt2",
     "proteingym/proteingym-v1-3-protgpt2-synthetic-smoke.yaml"),
]
IDS = [entry[0] for entry in ADAPTERS]


def module_of(adapter: type) -> Any:
    return importlib.import_module(adapter.__module__)


# Registration and options


@pytest.mark.parametrize(("model_id", "adapter", "extra", "recipe"), ADAPTERS, ids=IDS)
def test_each_adapter_is_registered_with_a_checkpoint_folder_and_a_device(
    model_id: str, adapter: type, extra: str, recipe: str
):
    spec = models.get_model(model_id)
    assert spec.adapter == f"{adapter.__module__}:{adapter.__name__}"
    assert spec.pretrained and spec.extra == extra and spec.recipe_allowed
    assert set(spec.options or {}) == {"checkpoint", "device"}
    assert spec.required_options == ("checkpoint",)
    assert models.check_options(spec, {"checkpoint": "/x", "device": "cpu"})
    with pytest.raises(ValueError, match="needs the option 'checkpoint'"):
        models.check_options(spec, {"device": "cpu"})
    with pytest.raises(ValueError, match="has no option 'trust_remote_code'"):
        models.check_options(spec, {"checkpoint": "/x", "trust_remote_code": True})
    with pytest.raises(ValueError, match="must be text"):
        models.check_options(spec, {"checkpoint": 3})
    with pytest.raises(ValueError, match="device must be cpu"):
        models.check_cpu(spec, {"checkpoint": "/x", "device": "cuda"})


@pytest.mark.parametrize(("model_id", "adapter", "extra", "recipe"), ADAPTERS, ids=IDS)
def test_each_extra_pins_the_transformers_version_its_adapter_checks(
    model_id: str, adapter: type, extra: str, recipe: str
):
    project = tomllib.loads((ROOT / "packages/rewirebench/pyproject.toml").read_text())
    requirements = project["project"]["optional-dependencies"][extra]
    assert all("==" in requirement for requirement in requirements), requirements
    assert f"transformers=={module_of(adapter).TRANSFORMERS_VERSION}" in requirements


@pytest.mark.parametrize(("model_id", "adapter", "extra", "recipe"), ADAPTERS, ids=IDS)
def test_the_registry_note_says_whether_digests_are_pending(
    model_id: str, adapter: type, extra: str, recipe: str
):
    note = models.get_model(model_id).note or ""
    assert ("PENDING" in note) == bool(adapter.CHECKPOINT.pending)


# The digest check


def pinned_folder(tmp_path: Path, names: tuple[str, ...]) -> tuple[Path, dict[str, str]]:
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    digests = {}
    for name in names:
        (folder / name).write_bytes(name.encode() * 3)
        digests[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    return folder, digests


def checkpoint(files: dict[str, str], *, remote_code: bool = False) -> pinned.Checkpoint:
    return pinned.Checkpoint(repository="example/model", revision="a" * 40,
                             files=files, record="test-record", remote_code=remote_code)


def test_a_pending_digest_is_refused_before_any_file_is_read(tmp_path: Path):
    pending = checkpoint({"config.json": ZERO.replace("0", "1"), "model.bin": pinned.PENDING})
    with pytest.raises(ValueError, match="no SHA-256 is pinned yet for model.bin"):
        pending.verify(tmp_path / "absent")


def test_a_missing_or_changed_file_is_refused(tmp_path: Path):
    folder, digests = pinned_folder(tmp_path, ("config.json", "model.safetensors"))
    assert checkpoint(digests).verify(folder) == folder.resolve()
    (folder / "model.safetensors").write_bytes(b"other weights")
    with pytest.raises(ValueError, match="mismatched pinned example/model file: model.safetensors"):
        checkpoint(digests).verify(folder)
    (folder / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="model.safetensors"):
        checkpoint(digests).verify(folder)


def test_a_pin_needs_a_commit_and_hex_digests():
    with pytest.raises(ValueError, match="40-hex"):
        pinned.Checkpoint(repository="r", revision="main", files={}, record="x")
    with pytest.raises(ValueError, match="SHA-256 or PENDING"):
        checkpoint({"config.json": "TODO"})


@pytest.mark.parametrize(("model_id", "adapter", "extra", "recipe"), ADAPTERS, ids=IDS)
def test_each_adapter_refuses_to_load_while_a_digest_is_pending(
    model_id: str, adapter: type, extra: str, recipe: str,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    files = dict(adapter.CHECKPOINT.files, **{"config.json": pinned.PENDING})
    monkeypatch.setattr(adapter, "CHECKPOINT", checkpoint(files))
    with pytest.raises(ValueError, match="no SHA-256 is pinned yet for config.json"):
        adapter(str(tmp_path))
    with pytest.raises(ValueError, match="no SHA-256 is pinned yet"):
        models.load(models.get_model(model_id), {"checkpoint": str(tmp_path)})


@pytest.mark.parametrize(("model_id", "adapter", "extra", "recipe"), ADAPTERS, ids=IDS)
def test_each_pin_matches_the_saved_file_listing(
    model_id: str, adapter: type, extra: str, recipe: str
):
    """Names come from the listing at the pinned revision, and an LFS file's digest is the
    listing's lfs.oid. A small file has no SHA-256 in the listing, so only its name is
    checked here; its digest was computed from the file fetched at that revision."""
    tree = files("rewirebench").joinpath(f"resources/models/hf-trees/{extra}.json")
    listing = {item["path"]: item for item in json.loads(tree.read_text())}
    for name, digest in adapter.CHECKPOINT.files.items():
        assert name in listing, (model_id, name)
        lfs = listing[name].get("lfs")
        if lfs:
            assert digest == lfs["oid"], (model_id, name)


@pytest.mark.parametrize(("model_id", "adapter", "extra", "recipe"), ADAPTERS, ids=IDS)
def test_each_adapter_checks_files_before_it_imports_a_framework(
    model_id: str, adapter: type, extra: str, recipe: str,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    names = tuple(adapter.CHECKPOINT.files)
    folder, digests = pinned_folder(tmp_path, names)
    monkeypatch.setattr(adapter, "CHECKPOINT", checkpoint(digests))

    def imported() -> None:
        raise RuntimeError("framework import reached")

    monkeypatch.setattr(module_of(adapter), "offline_frameworks", imported)
    (folder / names[-1]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match=f"mismatched pinned example/model file: {names[-1]}"):
        adapter(str(folder))
    (folder / names[-1]).write_bytes(names[-1].encode() * 3)
    with pytest.raises(RuntimeError, match="framework import reached"):
        adapter(str(folder))


def test_remote_code_is_declared_only_where_the_config_needs_it():
    remote = {model_id for model_id, adapter, _, _ in ADAPTERS if adapter.CHECKPOINT.remote_code}
    assert remote == {"dnabert-s-frozen"}
    code = [name for name in DNABERTS.CHECKPOINT.files if name.endswith(".py")]
    assert sorted(code) == ["bert_layers.py", "bert_padding.py", "configuration_bert.py",
                            "flash_attn_triton.py"]
    for _, adapter, _, _ in ADAPTERS:
        tree = ast.parse(Path(module_of(adapter).__file__).read_text())
        trusted = any(isinstance(node, ast.keyword) and node.arg == "trust_remote_code"
                      and isinstance(node.value, ast.Constant) and node.value.value is True
                      for node in ast.walk(tree))
        assert trusted == adapter.CHECKPOINT.remote_code, adapter


# Pooling on a fake encoder


class FakeTokenizer:
    """One token per character between a start and an end token, IDs 1, 2, 3, ..."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str, return_tensors: str, return_special_tokens_mask: bool = False,
                 **options: Any) -> dict[str, Any]:
        self.texts.append(text)
        tokens = text.split(" ") if " " in text else list(text)
        count = len(tokens) + 2
        encoded = {"input_ids": np.arange(1, count + 1)[None, :],
                   "attention_mask": np.ones((1, count), dtype=int)}
        if return_special_tokens_mask:
            encoded["special_tokens_mask"] = np.array([[1] + [0] * len(tokens) + [1]])
        return encoded


class FakeEncoder:
    """Token t has state [t, -t]: the first and last tokens are the special ones."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray]:
        self.calls.append((args, kwargs))
        ids = args[0] if args else kwargs["input_ids"]
        states = np.stack([ids, -ids], axis=-1).astype(float)
        return (states,)


def fake(adapter: type) -> Any:
    instance = object.__new__(adapter)
    instance.tokenizer, instance.model = FakeTokenizer(), FakeEncoder()
    instance.torch, instance.device = FAKE_TORCH, "cpu"
    return instance


PAIR = {"id": "v1", "reference_sequence": "ACGT", "mutant_sequence": "ACGA",
        "rel_position": "4", "region": "exon"}


def test_mistral_dna_takes_the_maximum_over_every_token_from_input_ids_alone():
    adapter = fake(MistralDNA)
    # Six tokens with IDs 1..6: the maximum of [t, -t] is [6, -1].
    assert adapter.embed([PAIR]) == {"v1": {"reference": [6.0, -1.0], "mutant": [6.0, -1.0]}}
    args, kwargs = adapter.model.calls[0]
    assert len(args) == 1 and not kwargs
    assert adapter.tokenizer.texts == ["ACGT", "ACGA"]


def test_dnabert_s_takes_the_mean_over_every_token_including_special_ones():
    adapter = fake(DNABERTS)
    assert adapter.embed([PAIR]) == {"v1": {"reference": [3.5, -3.5], "mutant": [3.5, -3.5]}}
    assert not adapter.model.calls[0][1]


def test_plantrna_fm_reads_t_as_u_and_leaves_out_special_tokens():
    adapter = fake(PlantRNAFM)
    # Tokens 2..5 are the sequence; their mean is 3.5.
    assert adapter.embed([{"id": "r1", "sequence": "acgt"}]) == {"r1": [3.5, -3.5]}
    assert adapter.tokenizer.texts == ["ACGU"]
    args, kwargs = adapter.model.calls[0]
    assert not args and set(kwargs) == {"input_ids", "attention_mask"}


def test_protbert_spaces_residues_rewrites_rare_ones_and_leaves_out_special_tokens():
    adapter = fake(ProtBert)
    assert adapter.embed([{"id": "p1", "sequence": "MKUZ"}]) == {"p1": [3.5, -3.5]}
    assert adapter.tokenizer.texts == ["M K X X"]


@pytest.mark.parametrize("adapter", [MistralDNA, DNABERTS])
def test_paired_inputs_are_checked_before_the_model_runs(adapter: type):
    instance = fake(adapter)
    for rows in ([{**PAIR, "target": 1}], [{**PAIR, "mutant_sequence": "ACGN"}],
                 [{"id": "v1", "reference_sequence": "ACGT"}], [PAIR, dict(PAIR)]):
        with pytest.raises(ValueError):
            instance.embed(rows)
    assert instance.model.calls == []


@pytest.mark.parametrize("adapter", [PlantRNAFM, ProtBert])
def test_sequence_inputs_allow_only_an_id_and_a_sequence(adapter: type):
    instance = fake(adapter)
    for rows in ([{"id": "a", "sequence": "ACG", "target": 1.0}], [{"id": "a", "sequence": ""}],
                 [{"id": "a", "sequence": "AC?"}], [{"id": "", "sequence": "ACG"}]):
        with pytest.raises(ValueError):
            instance.embed(rows)
    assert instance.model.calls == []


def test_a_sequence_longer_than_the_context_is_refused_not_cropped():
    adapter = fake(DNABERTS)
    long = {**PAIR, "reference_sequence": "A" * 600, "mutant_sequence": "C" * 600}
    with pytest.raises(ValueError, match="at most 512 tokens"):
        adapter.embed([long])
    assert adapter.model.calls == []


def test_pool_rejects_an_empty_selection_and_an_unknown_method():
    with pytest.raises(ValueError, match="at least one token"):
        pinned.pool(np.ones((2, 3)), "mean", [False, False])
    with pytest.raises(ValueError, match="Unknown pooling"):
        pinned.pool(np.ones((2, 3)), "sum")


# Scoring on fake language models

AMINO = "ACDEFGHIKLMNPQRSTVWY"
WILD = "MKTAYI"


class ResidueTokenizer:
    """ESM-style: <cls>=0, <eos>=1, <mask>=2, residues from 3."""

    mask_token_id = 2

    def __call__(self, text: str, return_tensors: str) -> dict[str, Any]:
        ids = [0] + [self.convert_tokens_to_ids(r) for r in text] + [1]
        return {"input_ids": np.array([ids]), "attention_mask": np.ones((1, len(ids)), int)}

    def convert_tokens_to_ids(self, token: str) -> int:
        return 3 + AMINO.index(token)


class MaskedLM:
    """Logits at a masked position favour residue index (position mod 20)."""

    def __init__(self) -> None:
        self.masked: list[int] = []

    def __call__(self, input_ids: np.ndarray, attention_mask: Any = None) -> tuple[np.ndarray]:
        positions = np.flatnonzero(input_ids[0] == ResidueTokenizer.mask_token_id)
        assert len(positions) == 1
        self.masked.append(int(positions[0]))
        logits = np.zeros((1, input_ids.shape[1], 3 + len(AMINO)))
        for position in range(input_ids.shape[1]):
            logits[0, position, 3:] = np.arange(len(AMINO)) * 0.1 * position
        return (logits,)


def masked_esm() -> ESM1b:
    adapter = object.__new__(ESM1b)
    adapter.tokenizer, adapter.model = ResidueTokenizer(), MaskedLM()
    adapter.torch, adapter.device, adapter._cache = FAKE_TORCH, "cpu", {}
    return adapter


def expected_log_odds(index: int, wild: str, mutant: str) -> float:
    row = np.zeros(3 + len(AMINO))
    row[3:] = np.arange(len(AMINO)) * 0.1 * (index + 1)
    log_p = pinned.log_softmax(row)
    return float(log_p[3 + AMINO.index(mutant)] - log_p[3 + AMINO.index(wild)])


def test_esm1b_sums_masked_marginal_log_odds_at_each_mutated_position():
    adapter = masked_esm()
    rows = [{"id": "one", "assay_id": "a", "wild_type_sequence": WILD, "mutant": "K2W"},
            {"id": "two", "assay_id": "a", "wild_type_sequence": WILD, "mutant": "K2A:Y5W",
             "mutated_sequence": "MATAWI"}]
    scores = adapter.predict(rows)
    assert scores["one"] == pytest.approx(expected_log_odds(1, "K", "W"))
    assert scores["two"] == pytest.approx(expected_log_odds(1, "K", "A")
                                          + expected_log_odds(4, "Y", "W"))
    # Position 2 is masked once and reused: the cache is keyed by sequence and position.
    assert adapter.model.masked == [2, 5]


def test_esm1b_refuses_labels_and_a_mutated_sequence_that_disagrees():
    adapter = masked_esm()
    base = {"id": "x", "assay_id": "a", "wild_type_sequence": WILD, "mutant": "K2W"}
    for row in ({**base, "DMS_score": 1.0}, {**base, "mutated_sequence": WILD},
                {**base, "mutant": "A2W"}):
        with pytest.raises(ValueError):
            adapter.predict([row])
    assert adapter.model.masked == []


def test_esm1b_leaves_a_protein_beyond_its_context_unscored():
    adapter = masked_esm()
    row = {"id": "long", "wild_type_sequence": "M" + "A" * 1022, "mutant": "A2W"}
    assert adapter.predict([row])["long"]["score"] is None


def test_protgpt2_formats_sequences_as_its_model_card_does():
    sequence = "A" * 61 + "C"
    assert card_text(sequence) == "<|endoftext|>\n" + "A" * 60 + "\nAC\n<|endoftext|>"


def test_the_sequence_log_likelihood_sums_next_token_log_probabilities():
    logits = np.log(np.array([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1], [1 / 3, 1 / 3, 1 / 3]]))
    ids = np.array([[0, 1, 2]])
    assert sequence_log_likelihood(logits, ids) == pytest.approx(np.log(0.25) + np.log(0.1))
    with pytest.raises(ValueError, match="align"):
        sequence_log_likelihood(logits[:2], ids)


class CharacterTokenizer:
    def __call__(self, text: str, return_tensors: str) -> dict[str, Any]:
        return {"input_ids": np.array([[ord(c) % 7 for c in text]])}


class CausalLM:
    """Next-token logits that depend on the current token only."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, input_ids: np.ndarray) -> tuple[np.ndarray]:
        self.calls += 1
        table = np.arange(49, dtype=float).reshape(7, 7) / 10
        return (table[input_ids],)


def test_protgpt2_scores_the_mutant_minus_the_wild_type_log_likelihood():
    adapter = object.__new__(ProtGPT2)
    adapter.tokenizer, adapter.model = CharacterTokenizer(), CausalLM()
    adapter.torch, adapter.device, adapter._cache = FAKE_TORCH, "cpu", {}
    table = np.arange(49, dtype=float).reshape(7, 7) / 10

    def log_likelihood(sequence: str) -> float:
        ids = np.array([[ord(c) % 7 for c in card_text(sequence)]])
        return sequence_log_likelihood(table[ids][0], ids)

    rows = [{"id": "a", "wild_type_sequence": WILD, "mutant": "K2W"},
            {"id": "b", "wild_type_sequence": WILD, "mutant": "T3A"}]
    scores = adapter.predict(rows)
    assert scores["a"] == pytest.approx(log_likelihood("MWTAYI") - log_likelihood(WILD))
    assert scores["b"] == pytest.approx(log_likelihood("MKAAYI") - log_likelihood(WILD))
    # The wild type is scored once and reused.
    assert adapter.model.calls == 3


def test_protgpt2_leaves_a_protein_beyond_its_context_unscored():
    adapter = object.__new__(ProtGPT2)
    adapter.tokenizer, adapter.model = CharacterTokenizer(), CausalLM()
    adapter.torch, adapter.device, adapter._cache = FAKE_TORCH, "cpu", {}
    row = {"id": "long", "wild_type_sequence": "M" + "A" * 1100, "mutant": "A2W"}
    assert adapter.predict([row])["long"]["score"] is None
    assert adapter.model.calls == 0


# Recipes


@pytest.mark.parametrize(("model_id", "adapter", "extra", "recipe"), ADAPTERS, ids=IDS)
def test_each_recipe_validates_and_pins_the_files_its_adapter_checks(
    model_id: str, adapter: type, extra: str, recipe: str
):
    path = ROOT / "recipes" / recipe
    loaded = recipes.load_recipe(path).recipe
    assert loaded.id == path.stem and loaded.backends.names() == ["local"]
    assert loaded.rewirebench_protocol in models.get_model(model_id).protocols
    run = next(command.argv for command in loaded.commands if command.argv[1] == "run")
    assert run[run.index("--model") + 1] == model_id
    options = json.loads(run[run.index("--adapter-options") + 1])
    assert options == {"checkpoint": "{inputs}", "device": "cpu"}
    pinned_files = adapter.CHECKPOINT
    hosted = {item.name: item for item in loaded.inputs if item.url and "huggingface.co/" in item.url
              and "/datasets/" not in item.url}
    assert set(hosted) == set(pinned_files.files)
    for name, item in hosted.items():
        assert item.url == pinned_files.url(name)
        digest = pinned_files.files[name]
        assert item.sha256 == (ZERO if digest == pinned.PENDING else digest)
    limitations = " ".join(loaded.limitations)
    assert all(name in limitations for name in pinned_files.pending)
    if pinned_files.pending:
        assert "weights_pin" in loaded.missing and loaded.weights_pin is None
    else:
        (pin,) = loaded.weights_pin
        assert (pin.host, pin.repository, pin.revision) == (
            "huggingface", pinned_files.repository, pinned_files.revision)
        assert {item.path: item.sha256 for item in pin.files} == dict(pinned_files.files)


def test_the_recipe_zero_digest_is_quoted_text():
    for _, _, _, recipe in ADAPTERS:
        data = yaml.safe_load((ROOT / "recipes" / recipe).read_text())
        assert all(isinstance(item["sha256"], str) for item in data["inputs"])
