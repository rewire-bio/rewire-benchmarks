"""Frozen Ankh Base per-protein embeddings for the FLIP2 probe.

Source: data repository record ``hyplm-flash-20261106-model-ankh`` (depth tier 2), run
guide step "Get embeddings from Ankh Large", read from the Ankh README (spec
``data/omics/model-pages-20261126-batch5/specs/ankh.json``). The README loads Ankh
through the ``ankh`` package, whose ``load_ankh_base`` returns the T5 encoder and the
tokenizer of ``ElnaggarLab/ankh-base``; this adapter loads the same two with
``T5EncoderModel`` and ``AutoTokenizer`` from transformers. As in the README, a protein
is passed as a list of residues with ``is_split_into_words`` and the special tokens
added. The README stops at ``last_hidden_state``: the adapter takes the mean over residue
tokens and leaves out the end-of-sequence token. It uses Ankh Base, not the README's Ankh
Large, whose 7.5 GB checkpoint is too large for the CPU runner.

Weights licence: CC BY-NC-SA 4.0, non-commercial use only (card metadata and README).
Nothing is redistributed and the weights are not modified: the recipe fetches the files
from the developer's repository at the pinned revision. The full encoder and decoder
checkpoint is fetched; only the encoder weights are loaded.

Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/ankh-base.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rewirebench.adapters.pinned import Checkpoint, FrozenEncoder, offline_frameworks

TRANSFORMERS_VERSION = "4.41.1"
CHECKPOINT = Checkpoint(
    repository="ElnaggarLab/ankh-base",
    revision="d99cb6b966530dfc2ae96bc69d9255c2a07308b0",
    record="hyplm-flash-20261106-model-ankh",
    files={
        "config.json": "6cc1a8c7056b14591f7b23a7c01a64a3b62be7eab7a898aa306128926a84ce84",
        "special_tokens_map.json": "c8995d2f8037fe3a8cfdef30475365e1c314c417880b75abaf8296e3c05d42d6",
        "tokenizer.json": "b4533f607d9fd665f2d9d94b0cf71870a6fc2fc2ae7cbd516d0e43a9efb406fd",
        "tokenizer_config.json": "e51582e899d935a3e03cc0a99c0d7996a68bd6670dda6970ba60b7d9f022fa90",
        "pytorch_model.bin": "9b2a886374f0ff4a893f4e7a989deed76bb2458c8998bd5202ea8e97d92ddcc3",
    },
)


class AnkhBase(FrozenEncoder):
    """Mean of the encoder's final hidden states over residue tokens."""

    CHECKPOINT = CHECKPOINT
    POOLING = "mean"
    MAX_TOKENS = 2048
    EXCLUDE_SPECIAL_TOKENS = True
    ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO")
    TOKENIZER_OPTIONS: ClassVar[dict[str, Any]] = {"is_split_into_words": True, "add_special_tokens": True}

    def __init__(self, checkpoint: str, *, device: str = "cpu"):
        folder = self.CHECKPOINT.verify(checkpoint)
        torch, transformers = offline_frameworks()
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise ValueError(f"This adapter is reviewed for transformers=={TRANSFORMERS_VERSION}")
        self.torch, self.device = torch, device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False)
        self.model = transformers.T5EncoderModel.from_pretrained(
            str(folder), local_files_only=True, torch_dtype=torch.float32,
        ).eval().to(device)

    def text(self, sequence: str) -> list[str]:
        return list(sequence)

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": "Ankh Base frozen mean-pooled residue embeddings",
            "transformers_version": TRANSFORMERS_VERSION,
            "device": self.device, "dtype": "float32", "pretrained": True,
            "weights_licence": "CC-BY-NC-SA-4.0",
            "training_overlap": "unreported; UniRef50 pretraining may overlap benchmark proteins",
            "published_result_reproduction": False,
        }
