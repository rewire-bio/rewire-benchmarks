"""Frozen DNABERT-S pair embeddings for the MFASS frozen-encoder protocol.

Source: data repository record ``gfm2-20261011-buzkova2026-model-dnabert-s`` (its twin
``lit-genomics-20261011-geneb-model-dnabert-s`` carries the same run guide), step
"Compute an embedding", read from the README at commit ac01b55e. The spec is
``data/omics/model-pages-20261127-batch10/specs/dnabert-s.json``. Loading and pooling
follow the README: ``AutoTokenizer`` and ``AutoModel`` with ``trust_remote_code``, one
sequence per forward pass with only ``input_ids``, then the mean over every token of the
first output, special tokens included.

The model cannot load without remote code: the pinned ``config.json`` maps ``AutoModel``
to ``bert_layers.BertModel``, a file in the checkpoint. Its four Python files are pinned
here like the weights, so the code transformers imports is checked before it runs.
Their SHA-256 values are PENDING: the saved file listing
(``artifacts/dnabert-s/hf-tree.json`` at revision 00e47f96) gives git SHA-1 object IDs
for them, not SHA-256. The weights digest is the listing's ``lfs.oid``, and
``config.json`` is the saved copy, whose git object ID matches the listing.

The weights are ``pytorch_model.bin``, a pickle: transformers 4.27 unpickles it after
the digest check, as the ESM adapters do with their checkpoints.
"""

from __future__ import annotations

from typing import Any

from rewirebench.adapters.pinned import PENDING, Checkpoint, FrozenEncoder, offline_frameworks

TRANSFORMERS_VERSION = "4.27.0"
CHECKPOINT = Checkpoint(
    repository="zhihan1996/DNABERT-S",
    revision="00e47f96cdea35e4b6f5df89e5419cbe47d490c6",
    record="gfm2-20261011-buzkova2026-model-dnabert-s",
    remote_code=True,
    files={
        "config.json": "d54c142114563aa34ef7c2b54ccc921d08e50cd7c3cd69c439d5e0d2bbcfcdee",
        "pytorch_model.bin": "f3cfc3d0541859df64759e758cc4bc40fe1d5760799a7d098a9f9f09501c1e5f",
        "tokenizer.json": PENDING,
        "tokenizer_config.json": PENDING,
        "special_tokens_map.json": PENDING,
        "bert_layers.py": PENDING,
        "bert_padding.py": PENDING,
        "configuration_bert.py": PENDING,
        "flash_attn_triton.py": PENDING,
    },
)


class DNABERTS(FrozenEncoder):
    """Mean-pooled final hidden states of the reference and the mutant 170-base window."""

    CHECKPOINT = CHECKPOINT
    POOLING = "mean"
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
            str(folder), local_files_only=True, trust_remote_code=True)
        self.model = transformers.AutoModel.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.float32,
        ).eval().to(device)
        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden != 768:
            raise ValueError("Pinned DNABERT-S must have 768-dimensional states")

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": "DNABERT-S frozen mean-pooled pair embeddings",
            "transformers_version": TRANSFORMERS_VERSION,
            "device": self.device, "dtype": "float32", "pretrained": True,
            "training_overlap": "exact MFASS sequence overlap with pretraining is unreported",
            "published_result_reproduction": False,
        }
