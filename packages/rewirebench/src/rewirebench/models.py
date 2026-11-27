"""The model adapter interface and the registry of adapters that ship with rewirebench.

An adapter is a Python object the SDK calls with protocol-approved inputs. The existing
protocols need three methods and no more: ``predict`` (one score per input), ``embed``
(one vector per input, for a protocol-owned probe) and an optional ``fit`` on the
training split. ``score_variants`` and a free ``predict(inputs)`` outside a protocol
(issue #58) are not defined here because no protocol calls them yet.

The registry is also an allowlist. ``rewirebench run --model ID`` and a recipe ``run``
step can only name a registered model, so neither can import an arbitrary callable the
way ``--adapter module:Class`` does.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

Interface = Literal["predict", "embed", "fit"]
OptionType = Literal["text", "integer", "number", "boolean"]
ModelInput = dict[str, Any]


@runtime_checkable
class ScoreModel(Protocol):
    """Returns one finite score per input ID."""

    def predict(self, inputs: list[ModelInput]) -> Mapping[str, Any]: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Returns one vector per input ID. The protocol owns the head fitted on top."""

    def embed(self, inputs: list[ModelInput]) -> Mapping[str, Any]: ...


@runtime_checkable
class FittedModel(Protocol):
    """A score model the SDK fits on the training split before it predicts."""

    def fit(self, inputs: list[ModelInput], targets: Sequence[Any]) -> Any: ...

    def predict(self, inputs: list[ModelInput]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ModelSpec:
    """One registered adapter.

    ``options`` names every constructor option a caller may set, and ``option_types``
    the JSON type of each one that is not text. ``runs_program`` marks an adapter that
    starts an executable named by an option; such a model cannot be used from a recipe
    or through the MCP server, neither of which may start another program.

    ``weights_licence`` is the licence of the pretrained weights the adapter loads, and
    ``commercial_use`` whether that licence permits commercial use: None where no licence
    is stated or the model loads no weights. Nothing is redistributed: a recipe fetches
    the weights from the developer's own pinned address.
    """

    id: str
    title: str
    adapter: str
    interfaces: tuple[Interface, ...]
    protocols: tuple[str, ...]
    pretrained: bool
    extra: str | None = None
    options: Mapping[str, str] | None = None
    required_options: tuple[str, ...] = ()
    option_types: Mapping[str, OptionType] | None = None
    runs_program: bool = False
    note: str | None = None
    weights_licence: str | None = None
    commercial_use: bool | None = None

    @property
    def prediction_type(self) -> Literal["scalar", "embedding"]:
        return "scalar" if "predict" in self.interfaces else "embedding"

    @property
    def recipe_allowed(self) -> bool:
        return not self.runs_program

    def option_type(self, name: str) -> OptionType:
        return (self.option_types or {}).get(name, "text")

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "adapter": self.adapter,
            "interfaces": list(self.interfaces),
            "prediction_type": self.prediction_type,
            "protocols": list(self.protocols),
            "pretrained": self.pretrained,
            "install_extra": self.extra,
            "options": dict(self.options or {}),
            "option_types": {name: self.option_type(name) for name in self.options or {}},
            "required_options": list(self.required_options),
            "runs_program": self.runs_program,
            "recipe_allowed": self.recipe_allowed,
            "weights_licence": self.weights_licence,
            "commercial_use": self.commercial_use,
            "note": self.note,
        }


_CHECKPOINT = "Path of the pinned checkpoint file. Its SHA-256 is checked before loading."
_FOLDER = ("Folder holding the pinned Hugging Face files. Each is checked by SHA-256 "
           "before loading.")
_DEVICE = "Torch device. The local backend hides CUDA devices, so use cpu there."
_PINNED = {"checkpoint": _FOLDER, "device": _DEVICE}

_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="train-mean", title="Training target mean",
        adapter="rewirebench.adapters.sequence:TrainMean",
        interfaces=("fit", "predict"), pretrained=False,
        protocols=("flip2-fitness-v1", "mrnabench-sample-mrl-v1", "tdc-admet-group-v1"),
        note="Constant regression control.",
    ),
    ModelSpec(
        id="seeded-random", title="Seeded random ranking control",
        adapter="rewirebench.adapters.sequence:SeededRandomScore",
        interfaces=("predict",), pretrained=False,
        protocols=("proteingym-v1.3-dms-substitutions", "dart-eval-task1-zero-shot-v1", "clingen-pp3-bp4-v1"),
        options={"seed": "Integer seed, default 0."}, option_types={"seed": "integer"},
    ),
    ModelSpec(
        id="sequence-composition", title="Sequence length and composition features",
        adapter="rewirebench.adapters.sequence:SequenceComposition",
        interfaces=("embed",), pretrained=False,
        protocols=("flip2-fitness-v1", "mrnabench-sample-mrl-v1"),
        options={"alphabet": "Unique symbols to count, default ACGT."},
    ),
    ModelSpec(
        id="training-prior", title="Training class prior or majority",
        adapter="rewirebench.adapters.baselines:TrainingPrior",
        interfaces=("fit", "predict"), pretrained=False,
        protocols=("mfass-v2", "genomic-benchmarks-v2", "tdc-admet-group-v1"),
        options={"multiclass": "true for more than two classes."},
        option_types={"multiclass": "boolean"},
    ),
    ModelSpec(
        id="string-ngram-linear", title="Character 1 to 3-gram counts with a linear head",
        adapter="rewirebench.adapters.baselines:StringNgramReference",
        interfaces=("fit", "predict"), pretrained=False,
        protocols=("genomic-benchmarks-v2", "tdc-admet-group-v1"),
        options={"field": "sequence or smiles.",
                 "classification": "false for a regression task.",
                 "multiclass": "true for more than two classes."},
        option_types={"classification": "boolean", "multiclass": "boolean"},
    ),
    ModelSpec(
        id="paired-composition", title="Paired sequence composition features",
        adapter="rewirebench.adapters.baselines:PairedComposition",
        interfaces=("embed",), pretrained=False,
        protocols=("mfass-v2-frozen-encoder",),
        options={"constant": "true for the constant-feature control."},
        option_types={"constant": "boolean"},
    ),
    ModelSpec(
        id="mfass-kmer-position", title="MFASS corrected k-mer and position baseline",
        adapter="rewirebench.adapters.mfass:KmerBaseline",
        interfaces=("fit", "predict"), pretrained=False, protocols=("mfass-v2",),
    ),
    ModelSpec(
        id="dnabert2-frozen", title="DNABERT-2 frozen encoder",
        adapter="rewirebench.adapters.mfass:DNABERT2",
        interfaces=("embed",), pretrained=True, extra="dnabert2",
        protocols=("mfass-v2-frozen-encoder",),
        options={"checkpoint": "Folder holding the pinned DNABERT-2 files. Each is "
                               "checked by SHA-256 before loading.",
                 "batch_size": "Encoder batch size, default 2."},
        option_types={"batch_size": "integer"},
        required_options=("checkpoint",),
        weights_licence=("Not stated: the model card at revision b5ae377f declares no "
                         "licence"),
        commercial_use=None,
    ),
    ModelSpec(
        id="esm2-t6-8m-masked-marginal", title="ESM-2 8M masked-marginal substitution score",
        adapter="rewirebench.adapters.esm:ESM2Adapter",
        interfaces=("predict",), pretrained=True, extra="esm",
        protocols=("proteingym-v1.3-dms-substitutions",),
        options={"checkpoint": _CHECKPOINT, "device": _DEVICE},
        required_options=("checkpoint",),
        weights_licence=("MIT, the licence of the facebookresearch/esm repository "
                         "that distributes the weights"),
        commercial_use=True,
    ),
    ModelSpec(
        id="esm2-embeddings", title="ESM-2 mean-pooled embeddings (8M or 35M)",
        adapter="rewirebench.adapters.esm_embeddings:ESM2Embeddings",
        interfaces=("embed",), pretrained=True, extra="esm",
        protocols=("flip2-fitness-v1",),
        options={"checkpoint": _CHECKPOINT, "device": _DEVICE,
                 "model_name": "esm2_t6_8M_UR50D (default) or esm2_t12_35M_UR50D."},
        required_options=("checkpoint",),
        weights_licence=("MIT, the licence of the facebookresearch/esm repository "
                         "that distributes the weights"),
        commercial_use=True,
    ),
    ModelSpec(
        id="mistral-dna-138m-frozen", title="Mistral-DNA-v1-138M-hg38 frozen encoder",
        adapter="rewirebench.adapters.mistral_dna:MistralDNA",
        interfaces=("embed",), pretrained=True, extra="mistral-dna",
        protocols=("mfass-v2-frozen-encoder",),
        options=_PINNED, required_options=("checkpoint",),
        note="Max-pooled final states, as on the model card.",
        weights_licence="Apache-2.0",
        commercial_use=True,
    ),
    ModelSpec(
        id="dnabert-s-frozen", title="DNABERT-S frozen encoder",
        adapter="rewirebench.adapters.dnabert_s:DNABERTS",
        interfaces=("embed",), pretrained=True, extra="dnabert-s",
        protocols=("mfass-v2-frozen-encoder",),
        options=_PINNED, required_options=("checkpoint",),
        note="Mean-pooled final states, as in the README. Loads the checkpoint's own "
             "model code (trust_remote_code), pinned by SHA-256 like the weights.",
        weights_licence="Apache-2.0",
        commercial_use=True,
    ),
    ModelSpec(
        id="plantrna-fm-embeddings", title="PlantRNA-FM mean-pooled embeddings",
        adapter="rewirebench.adapters.plantrna_fm:PlantRNAFM",
        interfaces=("embed",), pretrained=True, extra="plantrna-fm",
        protocols=("mrnabench-sample-mrl-v1",),
        options=_PINNED, required_options=("checkpoint",),
        note="Plant RNA model; T is read as U.",
        weights_licence="MIT",
        commercial_use=True,
    ),
    ModelSpec(
        id="protbert-embeddings", title="ProtBert mean-pooled embeddings",
        adapter="rewirebench.adapters.protbert:ProtBert",
        interfaces=("embed",), pretrained=True, extra="protbert",
        protocols=("flip2-fitness-v1",),
        options=_PINNED, required_options=("checkpoint",),
        note="Weights under AFL-3.0.",
        weights_licence="AFL-3.0",
        commercial_use=True,
    ),
    ModelSpec(
        id="esm1b-masked-marginal", title="ESM-1b masked-marginal substitution score",
        adapter="rewirebench.adapters.esm1b:ESM1b",
        interfaces=("predict",), pretrained=True, extra="esm1b",
        protocols=("proteingym-v1.3-dms-substitutions",),
        options=_PINNED, required_options=("checkpoint",),
        note="Transformers port of ESM-1b at a pinned revision.",
        weights_licence="MIT",
        commercial_use=True,
    ),
    ModelSpec(
        id="protgpt2-log-likelihood", title="ProtGPT2 log-likelihood ratio",
        adapter="rewirebench.adapters.protgpt2:ProtGPT2",
        interfaces=("predict",), pretrained=True, extra="protgpt2",
        protocols=("proteingym-v1.3-dms-substitutions",),
        options=_PINNED, required_options=("checkpoint",),
        note="Mutant minus wild-type summed log-likelihood.",
        weights_licence="MIT in the LICENSE file; apache-2.0 in the card metadata",
        commercial_use=True,
    ),
    ModelSpec(
        id="biofm-265m-frozen", title="BioFM-265M frozen encoder",
        adapter="rewirebench.adapters.biofm:BioFM",
        interfaces=("embed",), pretrained=True, extra="transformers-5-16",
        protocols=("mfass-v2-frozen-encoder",),
        options=_PINNED, required_options=("checkpoint",),
        note="Last-layer state at the last base, as the biofm-eval embedder takes it.",
        weights_licence="CC-BY-NC-4.0", commercial_use=False,
    ),
    ModelSpec(
        id="grover-frozen", title="GROVER frozen encoder",
        adapter="rewirebench.adapters.grover:GROVER",
        interfaces=("embed",), pretrained=True, extra="transformers-4-41",
        protocols=("mfass-v2-frozen-encoder",),
        options=_PINNED, required_options=("checkpoint",),
        note="Mean-pooled final states over the sequence's own tokens.",
        weights_licence=("CC-BY-4.0, from the authors' Zenodo deposit of the same "
                         "model; the Hugging Face card states none"),
        commercial_use=True,
    ),
    ModelSpec(
        id="hyenadna-large-1m-log-likelihood", title="HyenaDNA large-1m log-likelihood",
        adapter="rewirebench.adapters.hyenadna:HyenaDNA",
        interfaces=("predict",), pretrained=True, extra="transformers-4-41",
        protocols=("dart-eval-task1-zero-shot-v1",),
        options=_PINNED, required_options=("checkpoint",),
        note="Summed base log-likelihood. Loads the checkpoint's own model and tokenizer "
             "code (trust_remote_code), pinned by SHA-256 like the weights.",
        weights_licence="BSD-3-Clause", commercial_use=True,
    ),
    ModelSpec(
        id="mp-rna-embeddings", title="MP-RNA mean-pooled embeddings",
        adapter="rewirebench.adapters.mp_rna:MPRNA",
        interfaces=("embed",), pretrained=True, extra="transformers-4-41",
        protocols=("mrnabench-sample-mrl-v1",),
        options=_PINNED, required_options=("checkpoint",),
        note="T is read as U. Loads the checkpoint's own model code (trust_remote_code), "
             "pinned by SHA-256 like the weights.",
        weights_licence="Apache-2.0", commercial_use=True,
    ),
    ModelSpec(
        id="ankh-base-embeddings", title="Ankh Base mean-pooled embeddings",
        adapter="rewirebench.adapters.ankh:AnkhBase",
        interfaces=("embed",), pretrained=True, extra="transformers-4-41",
        protocols=("flip2-fitness-v1",),
        options=_PINNED, required_options=("checkpoint",),
        note="Encoder of the T5 checkpoint, mean over residues.",
        weights_licence="CC-BY-NC-SA-4.0", commercial_use=False,
    ),
    ModelSpec(
        id="prot-t5-xl-encoder-embeddings", title="ProtT5-XL-U50 encoder mean-pooled embeddings",
        adapter="rewirebench.adapters.prot_t5:ProtT5Encoder",
        interfaces=("embed",), pretrained=True, extra="transformers-4-41",
        protocols=("flip2-fitness-v1",),
        options=_PINNED, required_options=("checkpoint",),
        note="Half-precision weights run in float32, as the ProtTrans README says for a CPU.",
        weights_licence="AFL-3.0", commercial_use=True,
    ),
    ModelSpec(
        id="glm2-150m-embeddings", title="gLM2-150M mean-pooled protein embeddings",
        adapter="rewirebench.adapters.glm2:GLM2",
        interfaces=("embed",), pretrained=True, extra="transformers-4-41",
        protocols=("flip2-fitness-v1",),
        options=_PINNED, required_options=("checkpoint",),
        note="A protein is one coding element, <+> then the residues. Loads the "
             "checkpoint's own model and tokenizer code (trust_remote_code), pinned by "
             "SHA-256 like the weights.",
        weights_licence="Apache-2.0", commercial_use=True,
    ),
    ModelSpec(
        id="esmc-600m-masked-marginal", title="ESM C 600M masked-marginal substitution score",
        adapter="rewirebench.adapters.esmc:ESMC",
        interfaces=("predict",), pretrained=True, extra="transformers-5-16",
        protocols=("proteingym-v1.3-dms-substitutions",),
        options=_PINNED, required_options=("checkpoint",),
        note="The transformers port of ESM C, no remote code.",
        weights_licence="MIT plus other, under the developers' acceptable use policy",
        commercial_use=True,
    ),
    ModelSpec(
        id="evcouplings-independent", title="EVcouplings independent (site) model",
        adapter="rewirebench.adapters.evcouplings_independent:EVCouplingsIndependent",
        interfaces=("predict",), pretrained=True,
        protocols=("proteingym-v1.3-dms-substitutions",),
        options={"artifact": "Artifact JSON from prepare-baseline-artifact.",
                 "artifact_sha256": "SHA-256 of that artifact."},
        required_options=("artifact", "artifact_sha256"),
        note="Fitted on alignments, not on assay labels.",
    ),
    ModelSpec(
        id="fimo-h12core-hit-count", title="FIMO hit count over HOCOMOCO v12 H12CORE motifs",
        adapter="rewirebench.adapters.fimo_motifs:H12CoreFimoHitCount",
        interfaces=("predict",), pretrained=False,
        protocols=("dart-eval-task1-zero-shot-v1",),
        options={"fimo": "Path of a FIMO 5.5.9 executable.",
                 "motifs": "Path of the H12CORE MEME file.",
                 "fimo_sha256": "SHA-256 of that executable.",
                 "timeout": "Seconds allowed per FIMO call."},
        required_options=("fimo", "motifs", "fimo_sha256"), option_types={"timeout": "number"},
        runs_program=True,
        note="Starts the FIMO executable named in the options, so it runs from the command "
             "line only: a recipe and the MCP server both refuse it.",
    ),
    ModelSpec(
        id="clingen-score-table",
        title="SHA-256 verified precomputed ClinGen variant score table",
        adapter="rewirebench.adapters.clingen:ScoreTableLookup",
        interfaces=("predict",),
        pretrained=False,
        protocols=("clingen-pp3-bp4-v1",),
        options={
            "artifact": "Path to a TSV holding chrom, pos, ref, alt and score columns.",
            "artifact_sha256": "Expected SHA-256 hex digest of that TSV.",
            "column": "Score column name in the TSV, default score.",
        },
        required_options=("artifact", "artifact_sha256"),
        note="Looks up precomputed model or clinical baseline scores by genomic variant key.",
    ),
)

REGISTRY: dict[str, ModelSpec] = {spec.id: spec for spec in _MODELS}


def get_model(identifier: str) -> ModelSpec:
    if identifier not in REGISTRY:
        raise ValueError(f"Unknown model: {identifier}. Choose one of {', '.join(REGISTRY)}")
    return REGISTRY[identifier]


def list_models(protocol: str | None = None) -> list[dict[str, Any]]:
    """Describe the registered models, or those a protocol accepts. Nothing is imported."""
    if protocol is not None:
        from rewirebench import sdk

        if protocol not in sdk.PROTOCOLS:
            raise ValueError(f"Unknown protocol: {protocol}")
    return [spec.describe() for spec in _MODELS if protocol is None or protocol in spec.protocols]


def _has_type(value: Any, wanted: OptionType) -> bool:
    """JSON types, strictly: true is not an integer and "3" is not a number."""
    if wanted == "boolean":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if wanted == "integer":
        return isinstance(value, int)
    if wanted == "number":
        return isinstance(value, (int, float)) and math.isfinite(value)
    return isinstance(value, str) and bool(value)


def check_protocol(spec: ModelSpec, protocol: Any) -> None:
    """Refuse prepared data of a protocol the model is not registered for."""
    if protocol not in spec.protocols:
        raise ValueError(
            f"Model {spec.id} is not registered for protocol {protocol}. "
            f"Its protocols are: {', '.join(spec.protocols)}"
        )


def check_cpu(spec: ModelSpec, options: Mapping[str, Any]) -> None:
    """Refuse a device other than cpu, where the caller may not use an accelerator."""
    if options.get("device", "cpu") != "cpu":
        raise ValueError(f"Model {spec.id}: the local backend is CPU only, so device must be cpu")


def check_options(spec: ModelSpec, options: Any) -> dict[str, Any]:
    """Refuse an option the model does not declare, one of the wrong JSON type, and a
    missing required one."""
    if not isinstance(options, dict):
        raise ValueError("Adapter options are a JSON object")  # noqa: TRY004
    allowed = set(spec.options or {})
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(
            f"Model {spec.id} has no option {unknown[0]!r}. "
            f"Its options are: {', '.join(sorted(allowed)) or 'none'}"
        )
    for name, value in options.items():
        wanted = spec.option_type(name)
        if not _has_type(value, wanted):
            raise ValueError(f"Model {spec.id}: option {name!r} must be {wanted}, "
                             f"not {value!r}")
    absent = [name for name in spec.required_options if name not in options]
    if absent:
        raise ValueError(f"Model {spec.id} needs the option {absent[0]!r}")
    return options


def load(spec: ModelSpec, options: dict[str, Any] | None = None) -> Any:
    """Import and construct a registered adapter. This is where model code first runs."""
    checked = check_options(spec, options or {})
    module_name, _, class_name = spec.adapter.partition(":")
    if not module_name.startswith("rewirebench.adapters."):
        raise ValueError(f"Model {spec.id} is not a rewirebench adapter")
    try:
        factory = getattr(importlib.import_module(module_name), class_name)
    except ImportError as error:
        hint = f" Install rewirebench[{spec.extra}]." if spec.extra else ""
        raise ValueError(f"Model {spec.id} cannot be imported: {error}.{hint}") from error
    return factory(**checked)
