"""Frozen Mistral-DNA-v1-138M-hg38 pair embeddings for the MFASS frozen-encoder protocol.

Source: data repository record ``lit-genomics-20261011-guanine-model-mistral-dna``, run
guide step "Load the model and embed a sequence", taken from the model card saved at
``data/omics/model-pages-20261126-batch3/artifacts/mistral-dna/hf-card.md`` (revision
4c3c8c7b). Loading and pooling follow the card: ``AutoTokenizer`` and ``AutoModel``, one
sequence per forward pass with only ``input_ids``, then the maximum over tokens of the
first output.

Differences from the card, each deliberate:

- The card passes ``trust_remote_code=True``. The pinned ``config.json`` has no
  ``auto_map`` and names ``MixtralForCausalLM``, which transformers ships, so the flag
  loads nothing extra. It is set to False here.
- The card's checkpoint is stored in bfloat16; it is loaded in float32, which is what
  ``from_pretrained`` does by default in the card's snippet too.

``config.json`` is pinned from the saved copy, whose SHA-256 the data repository
records. The other file names are the usual layout of a transformers checkpoint with
safetensors weights; no file listing was saved, so names and digests are both PENDING
until someone reads the listing at the pinned revision.
"""

from __future__ import annotations

from typing import Any

from rewirebench.adapters.pinned import PENDING, Checkpoint, FrozenEncoder, offline_frameworks

TRANSFORMERS_VERSION = "4.41.1"
CHECKPOINT = Checkpoint(
    repository="RaphaelMourad/Mistral-DNA-v1-138M-hg38",
    revision="4c3c8c7b44fae10e63b5902d27c8699feeca7a56",
    record="lit-genomics-20261011-guanine-model-mistral-dna",
    files={
        "config.json": "651be00955c11f3b25779a20afd041eb7560a314a1b6afd74bf2684113e4c62d",
        "model.safetensors": PENDING,
        "tokenizer.json": PENDING,
        "tokenizer_config.json": PENDING,
        "special_tokens_map.json": PENDING,
    },
)


class MistralDNA(FrozenEncoder):
    """Max-pooled final hidden states of the reference and the mutant 170-base window."""

    CHECKPOINT = CHECKPOINT
    POOLING = "max"
    MAX_TOKENS = 512
    IDS_ONLY = True
    PAIRED = True

    def __init__(self, checkpoint: str, *, device: str = "cpu"):
        folder = self.CHECKPOINT.verify(checkpoint)
        torch, transformers = offline_frameworks()
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise ValueError(f"This adapter is reviewed for transformers=={TRANSFORMERS_VERSION}")
        self.torch, self.device = torch, device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False)
        self.model = transformers.AutoModel.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.float32,
        ).eval().to(device)

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": "Mistral-DNA frozen max-pooled pair embeddings",
            "transformers_version": TRANSFORMERS_VERSION,
            "device": self.device, "dtype": "float32", "pretrained": True,
            "training_overlap": "pretrained on hg38, which contains the MFASS reference windows",
            "published_result_reproduction": False,
        }
