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
MIT. The card's metadata declares no licence. The weights are ``pytorch_model.bin``, a
pickle, unpickled after the digest check.

Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/protbert.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.
"""

from __future__ import annotations

import re
from typing import Any

from rewirebench.adapters.pinned import Checkpoint, FrozenEncoder, offline_frameworks

TRANSFORMERS_VERSION = "4.41.1"
CHECKPOINT = Checkpoint(
    repository="Rostlab/prot_bert",
    revision="7a894481acdc12202f0a415dd567f6cfdb698908",
    record="lit-protein-20261011-deepprotein-model-protbert",
    files={
        "config.json": "44c1e1229d520841e5b29f1dd2292595219061daf0c5f5dacb4f68d1272d69a3",
        "pytorch_model.bin": "6ea3edd26cfefc3111176100ee2a027ed510f9c86b63e5b53a2a050b59f2af9d",
        "vocab.txt": "7ea18f4fe3a796c1146f9830d50334a251f368db96bdd0606e78846ca3a30607",
        "tokenizer_config.json": "73f5e97ab29042bac49dc96f7b8a7a790994509bbd2fe3e9cd4f59affc46c2ce",
        "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
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
