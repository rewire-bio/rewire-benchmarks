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
Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/dnabert-s.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.

The transformers pin is the one in ``requirements.txt`` at commit ac01b55e
(``transformers==4.27``; SHA-256 af018b27, as the data repository records).

The weights are ``pytorch_model.bin``, a pickle: transformers 4.27 unpickles it after
the digest check, as the ESM adapters do with their checkpoints.
"""

from __future__ import annotations

from typing import Any

from rewirebench.adapters.pinned import Checkpoint, FrozenEncoder, offline_frameworks

TRANSFORMERS_VERSION = "4.27.0"
CHECKPOINT = Checkpoint(
    repository="zhihan1996/DNABERT-S",
    revision="00e47f96cdea35e4b6f5df89e5419cbe47d490c6",
    record="gfm2-20261011-buzkova2026-model-dnabert-s",
    remote_code=True,
    files={
        "config.json": "d54c142114563aa34ef7c2b54ccc921d08e50cd7c3cd69c439d5e0d2bbcfcdee",
        "pytorch_model.bin": "f3cfc3d0541859df64759e758cc4bc40fe1d5760799a7d098a9f9f09501c1e5f",
        "tokenizer.json": "667f5a0e6e9630f7052b004784dda7c7aeba395cc4ca3c01b11b02e6aa29d7b1",
        "tokenizer_config.json": "f13dbe9968de35aff25b06965f18e30449ec37ed94811bd0c7f97990f3b18ebb",
        "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
        "bert_layers.py": "cd2fe0efbb90db624fe4e25aa96bd049513e81740ba3f980163c385e894df561",
        "bert_padding.py": "44d1c68afb1f585fdc66c150d4c60f1ed44a89c006abc57d50531d71940d7421",
        "configuration_bert.py": "d030166bd279ca81e9d2304eaff9419171f2f8c69ab419e7cd37873b72f47a22",
        "flash_attn_triton.py": "568d1ac3beca0b5e1df528a1f136aa19b6489a616fcf3784f33336a50bb1de81",
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
