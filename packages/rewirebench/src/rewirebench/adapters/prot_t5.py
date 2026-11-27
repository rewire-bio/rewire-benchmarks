"""Frozen ProtT5-XL-U50 encoder per-protein embeddings for the FLIP2 probe.

Source: data repository record ``lit-protein-20261011-deepprotein-model-prot-t5-xl``
(depth tier 2), run guide step "Extract embeddings", taken from the ProtTrans README
(spec ``data/omics/model-pages-20261126-batch5/specs/prot-t5-xl.json``). Loading and
pooling follow the README: ``T5Tokenizer`` with ``do_lower_case=False`` and
``T5EncoderModel`` from ``Rostlab/prot_t5_xl_half_uniref50-enc``, U, Z, O and B
rewritten as X, residues separated by spaces, special tokens added, and the per-protein
vector as the mean over the residues, leaving out the end-of-sequence token. As the
README says for a CPU, the half-precision weights are run in float32. One sequence per
forward pass.

Weights licence: Academic Free License v3.0 (ProtTrans README); the Hugging Face card
declares none.

Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/prot-t5.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from rewirebench.adapters.pinned import Checkpoint, FrozenEncoder, offline_frameworks

TRANSFORMERS_VERSION = "4.41.1"
CHECKPOINT = Checkpoint(
    repository="Rostlab/prot_t5_xl_half_uniref50-enc",
    revision="94a6abc029ae13029317b140b7424e012bf8dfbf",
    record="lit-protein-20261011-deepprotein-model-prot-t5-xl",
    files={
        "config.json": "c93cc8c31433fb062cad1825a9c16cde962a2015c8a3b10cfdc5f76a85128db3",
        "special_tokens_map.json": "a4b8dfd3d9a75b0f6f9ddbcb52b147c7377f08aa9310e182829cda2a55684334",
        "spiece.model": "74da7b4afcde53faa570114b530c726135bdfcdb813dec3abfb27f9d44db7324",
        "tokenizer_config.json": "d3adf2fb21d1242b5c0b72602f02ce5b0c43f767c0f957e3b57a67b2f93bf4f2",
        "pytorch_model.bin": "7f51ba885541c7dc569d46b796af57cc7a2ba7945107dced4f19d1b5ec091157",
    },
)


class ProtT5Encoder(FrozenEncoder):
    """Mean of the encoder's final hidden states over residue tokens."""

    CHECKPOINT = CHECKPOINT
    POOLING = "mean"
    MAX_TOKENS = 2048
    EXCLUDE_SPECIAL_TOKENS = True
    ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO")
    TOKENIZER_OPTIONS: ClassVar[dict[str, Any]] = {"add_special_tokens": True}

    def __init__(self, checkpoint: str, *, device: str = "cpu"):
        folder = self.CHECKPOINT.verify(checkpoint)
        torch, transformers = offline_frameworks()
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise ValueError(f"This adapter is reviewed for transformers=={TRANSFORMERS_VERSION}")
        self.torch, self.device = torch, device
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(
            str(folder), do_lower_case=False, local_files_only=True)
        self.model = transformers.T5EncoderModel.from_pretrained(
            str(folder), local_files_only=True,
        ).to(torch.float32).eval().to(device)

    def text(self, sequence: str) -> str:
        return " ".join(re.sub(r"[UZOB]", "X", sequence))

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": "ProtT5-XL-U50 encoder frozen mean-pooled residue embeddings",
            "transformers_version": TRANSFORMERS_VERSION,
            "device": self.device, "dtype": "float32", "pretrained": True,
            "weights_licence": "AFL-3.0",
            "training_overlap": "unreported; UniRef50 pretraining may overlap benchmark proteins",
            "published_result_reproduction": False,
        }
