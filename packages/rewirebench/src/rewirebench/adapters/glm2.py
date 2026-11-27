"""Frozen gLM2-150M per-protein embeddings for the FLIP2 probe.

Source: data repository record ``bacformer-20261115-model-glm2``, run guide step "Extract
embeddings for a contig", taken from the model card (spec
``data/omics/model-pages-20261127-revision1/specs/glm2.json``). The card loads
``tattabio/gLM2_650M``; this adapter loads the 150M checkpoint the card and README also
name, at a pinned revision, because it fits the CPU runner. gLM2 reads a contig as
strand-prefixed elements, amino acids in upper case. A FLIP2 protein is one coding
element, so it is passed as ``<+>`` followed by the residues. Loading follows the card:
``AutoTokenizer`` and ``AutoModel`` with ``trust_remote_code``, and only ``input_ids``
passed to the model, then ``last_hidden_state``. The tokenizer also returns
``token_type_ids``, which ``gLM2Model.forward`` does not accept. The card gives no pooling: the adapter takes the mean over residue
tokens and leaves out the strand token. The card loads bfloat16 on a GPU; this adapter
uses float32 on the CPU.

The model cannot load without remote code: ``config.json`` maps the Auto classes to
``modeling_glm2.py`` and the tokenizer configuration to ``glm_tokenizer.py``. All three
Python files are pinned here like the weights. ``modeling_glm2.py`` imports ``einops``.

Weights licence: Apache-2.0 (card metadata and Hugging Face API).

Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/glm2.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.
"""

from __future__ import annotations

from typing import Any

from rewirebench.adapters.pinned import Checkpoint, FrozenEncoder, offline_frameworks

TRANSFORMERS_VERSION = "4.41.1"
CHECKPOINT = Checkpoint(
    repository="tattabio/gLM2_150M",
    revision="93c88529115476b27bdd85da14311779114a8a64",
    record="bacformer-20261115-model-glm2",
    remote_code=True,
    files={
        "config.json": "01b99a415d0be6f5976a4b1f52a93faee2ddf8b777ef75ace0643f26ac9b4d66",
        "configuration_glm2.py": "987df0584a9df63d6027ff75a0f1f9dbdbf4eda1a81ef56643cf40367efa083a",
        "glm_tokenizer.py": "ea4c47e392f51a1924e93eeedd4396ec2102dc52ba87c468ba90bc6dab214f31",
        "modeling_glm2.py": "503e5ec0bfd45957b9ecd92e37ae6414bd3574aa78887b14f8769a397e4b27f2",
        "special_tokens_map.json": "6abc3f2fe29ea77edacdb356f151ae93898d31aec82b1be4636485d7023e2ccc",
        "tokenizer.json": "e803e69825b468900a18989d8e89b53c4ef34630a77cfaba7a213f710859516e",
        "tokenizer_config.json": "beadc8517f5e82bad0657789c83e70a7b84087640dbfa92afbd73ba48f1f66b5",
        "model.safetensors": "047facd39472afe082c985058fa042f43a7b10d4ffb2ab51b9a3e6c63e9f3834",
    },
)


class GLM2(FrozenEncoder):
    """Mean of the final hidden states over residue tokens of one coding element."""

    CHECKPOINT = CHECKPOINT
    POOLING = "mean"
    MAX_TOKENS = 4096
    EXCLUDE_SPECIAL_TOKENS = True
    IDS_ONLY = True
    ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO")

    def __init__(self, checkpoint: str, *, device: str = "cpu"):
        folder = self.CHECKPOINT.verify(checkpoint)
        torch, transformers = offline_frameworks()
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise ValueError(f"This adapter is reviewed for transformers=={TRANSFORMERS_VERSION}")
        self.torch, self.device = torch, device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=True)
        self.model = transformers.AutoModel.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=True,
            use_safetensors=True, torch_dtype=torch.float32,
        ).eval().to(device)

    def text(self, sequence: str) -> str:
        return "<+>" + sequence

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": "gLM2-150M frozen mean-pooled residue embeddings",
            "transformers_version": TRANSFORMERS_VERSION,
            "device": self.device, "dtype": "float32", "pretrained": True,
            "weights_licence": "Apache-2.0",
            "training_overlap": "pretrained on the OMG metagenomic corpus; overlap with "
                                "benchmark proteins is unreported",
            "published_result_reproduction": False,
        }
