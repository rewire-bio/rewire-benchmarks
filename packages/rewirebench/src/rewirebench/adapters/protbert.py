"""Frozen ProtBert per-protein embeddings for the FLIP2 probe.

Source: data repository record ``lit-protein-20261011-deepprotein-model-protbert``
(depth tier 2), run guide step "Get features for one sequence", taken from the model
card at revision 7a894481 and the ProtTrans README at commit 27c02f07
(``data/omics/model-pages-20261126-batch5/artifacts/protbert/prottrans-readme.md``).
Loading follows the card: ``BertTokenizer`` with ``do_lower_case=False`` and
``BertModel``, residues separated by spaces and U, Z, O and B rewritten as X, one
sequence per forward pass with the full tokenizer output.

The card gives no per-protein pooling. This adapter takes the mean over residue tokens
and leaves out [CLS] and [SEP], which is how the ProtTrans README pools ProtT5.

Weights licence: Academic Free License v3.0, from the ProtTrans README; the code is
MIT. The card's metadata declares no licence. ``config.json`` is pinned from the saved
copy. No file listing was saved: the other names are the usual layout of a BERT
checkpoint, and their names and digests are PENDING until someone reads the listing at
the pinned revision. The weights are a pickle (the API response lists no safetensors);
transformers unpickles them after the digest check.
"""

from __future__ import annotations

import re
from typing import Any

from rewirebench.adapters.pinned import PENDING, Checkpoint, FrozenEncoder, offline_frameworks

TRANSFORMERS_VERSION = "4.41.1"
CHECKPOINT = Checkpoint(
    repository="Rostlab/prot_bert",
    revision="7a894481acdc12202f0a415dd567f6cfdb698908",
    record="lit-protein-20261011-deepprotein-model-protbert",
    files={
        "config.json": "44c1e1229d520841e5b29f1dd2292595219061daf0c5f5dacb4f68d1272d69a3",
        "pytorch_model.bin": PENDING,
        "vocab.txt": PENDING,
        "tokenizer_config.json": PENDING,
        "special_tokens_map.json": PENDING,
    },
)


class ProtBert(FrozenEncoder):
    """Mean of the final hidden states over residue tokens."""

    CHECKPOINT = CHECKPOINT
    POOLING = "mean"
    MAX_TOKENS = 2048
    EXCLUDE_SPECIAL_TOKENS = True
    ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO")

    def __init__(self, checkpoint: str, *, device: str = "cpu"):
        folder = self.CHECKPOINT.verify(checkpoint)
        torch, transformers = offline_frameworks()
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise ValueError(f"This adapter is reviewed for transformers=={TRANSFORMERS_VERSION}")
        self.torch, self.device = torch, device
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            str(folder), do_lower_case=False, local_files_only=True)
        self.model = transformers.BertModel.from_pretrained(
            str(folder), local_files_only=True, torch_dtype=torch.float32,
        ).eval().to(device)

    def text(self, sequence: str) -> str:
        return re.sub(r"[UZOB]", "X", " ".join(sequence))

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": "ProtBert frozen mean-pooled residue embeddings",
            "transformers_version": TRANSFORMERS_VERSION,
            "device": self.device, "dtype": "float32", "pretrained": True,
            "weights_licence": "AFL-3.0",
            "training_overlap": "unreported; UniRef100 pretraining may overlap benchmark proteins",
            "published_result_reproduction": False,
        }
