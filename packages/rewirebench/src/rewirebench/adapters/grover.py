"""Frozen GROVER pair embeddings for the MFASS frozen-encoder protocol.

Source: data repository record ``lit-genomics-20261011-geneb-model-grover``, run guide step
"Load the tokenizer and the pretrained model", taken from the model card at revision
6b223110 (spec ``data/omics/model-pages-20261127-revision1/specs/grover.json``). The card
loads ``AutoTokenizer`` and ``AutoModelForMaskedLM``; for embeddings this adapter loads
the same weights with ``AutoModel``, the BERT encoder without the masked-language head.
The card gives no pooling: the adapter takes the mean of the final hidden states over the
sequence's own tokens, leaving out ``[CLS]`` and ``[SEP]``. One sequence per forward pass.

Weights licence: the Hugging Face card declares none. The authors deposited the same
model on Zenodo under CC BY 4.0, and this adapter relies on that statement.

Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/grover.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.

The repository also holds tokenised chromosomes (``tokenized_chromosomes/``, several GB);
they are not needed to embed a sequence and are not pinned or fetched.
"""

from __future__ import annotations

from typing import Any

from rewirebench.adapters.pinned import Checkpoint, FrozenEncoder, offline_frameworks

TRANSFORMERS_VERSION = "4.41.1"
CHECKPOINT = Checkpoint(
    repository="PoetschLab/GROVER",
    revision="6b223110f0d6963e849f55bc2a2f3cff0e38c7a4",
    record="lit-genomics-20261011-geneb-model-grover",
    files={
        "config.json": "de60db6f5cb476d9f38630517abec87803e32a87672ace7946b5cf82f68ac5de",
        "special_tokens_map.json": "894589f317dfaaa10cbb6e467380b89bb848fe99f60ff6863ce120d81d71eff4",
        "tokenizer.json": "86b51bd8ecfc52cdc04f9f6787a2bd64fc5784709e7fd988b82d2e97348f9afd",
        "tokenizer_config.json": "8c6c05d4a32b1235387c4d2a546bce640d7889f74d4b8cc6c8da549641e97517",
        "vocab.txt": "affab9041e04e027bbcd359648e0fbc8af1610e58a16dc7954a81fd4b28f4e76",
        "pytorch_model.bin": "81fa4ee056e65ff1c06c32512589f3ba3bcb9d054960f35f634b9cf6da7aadda",
    },
)


class GROVER(FrozenEncoder):
    """Mean of the final hidden states over the reference and the mutant window."""

    CHECKPOINT = CHECKPOINT
    POOLING = "mean"
    MAX_TOKENS = 512
    EXCLUDE_SPECIAL_TOKENS = True
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
            torch_dtype=torch.float32,
        ).eval().to(device)

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": "GROVER frozen mean-pooled pair embeddings",
            "transformers_version": TRANSFORMERS_VERSION,
            "device": self.device, "dtype": "float32", "pretrained": True,
            "weights_licence": "CC-BY-4.0",
            "training_overlap": "pretrained on hg19, which contains the MFASS reference windows",
            "published_result_reproduction": False,
        }
