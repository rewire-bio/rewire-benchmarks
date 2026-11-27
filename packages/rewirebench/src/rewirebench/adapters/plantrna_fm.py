"""Frozen PlantRNA-FM sequence embeddings for the mRNABench probe.

Source: data repository record ``lit-rna-20261011-rnalm-zeroshot-model-plantrna-fm``,
run guide step "Load the model and embed a sequence", read from the README at commit
8c9359d5 (``data/omics/model-pages-20261126-batch2/artifacts/plantrna-fm/readme.md``)
and the model card at revision 5e7baf4e. Loading follows the README: ``AutoModel`` and
``AutoTokenizer`` with no remote code, one sequence per forward pass with the full
tokenizer output, then ``last_hidden_state``.

The README stops at the per-token states and gives no pooling. This adapter takes the
mean over the sequence's own tokens and leaves out the special tokens, as the official
ESM example does for the ESM architecture this checkpoint uses. The README's example is
written in U, so T is rewritten as U before tokenisation.

The README lists ``omnigenome`` in its requirements, but its loading snippet does not
import it, so the extra leaves it out. The pinned ``tokenizer_config.json`` names
``EsmTokenizer``, which transformers ships, so no remote code is needed for it either.

Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/plantrna-fm.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.
"""

from __future__ import annotations

from typing import Any

from rewirebench.adapters.pinned import Checkpoint, FrozenEncoder, offline_frameworks

TRANSFORMERS_VERSION = "4.41.1"
CHECKPOINT = Checkpoint(
    repository="yangheng/PlantRNA-FM",
    revision="5e7baf4e9d884f7a3ff7385b6a90dd2f63c7b1e5",
    record="lit-rna-20261011-rnalm-zeroshot-model-plantrna-fm",
    files={
        "config.json": "b520579b517e6b5c9c3ed993086a06c0d5af452a98c4bf8d8633eae4a93b3166",
        "model.safetensors": "01f66172a243d589945b0b270cde3c5567c09d4ba537fcd710c8295e0d60d8c4",
        "vocab.txt": "5c6fecdc35423dc779aa46529b81d5981b4eb77676a57eda961deba46a0f05df",
        "tokenizer_config.json": "2f96b43c7f15f31aaa8ae3fab569590c6c6909dee9ed80b57b790f22c24042a4",
        "special_tokens_map.json": "01e416f1aa5cb48b1d2c763ed8f2ef6e1344a5fc351657c1e8f2c73db82443d9",
    },
)


class PlantRNAFM(FrozenEncoder):
    """Mean of the final hidden states over the sequence's own tokens."""

    CHECKPOINT = CHECKPOINT
    POOLING = "mean"
    MAX_TOKENS = 1026
    EXCLUDE_SPECIAL_TOKENS = True
    ALPHABET = frozenset("ACGTUN")

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

    def text(self, sequence: str) -> str:
        return sequence.replace("T", "U")

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": "PlantRNA-FM frozen mean-pooled sequence embeddings",
            "transformers_version": TRANSFORMERS_VERSION,
            "device": self.device, "dtype": "float32", "pretrained": True,
            "training_overlap": "pretrained on plant transcriptomes (1KP); overlap with "
                                "mRNABench sequences is unreported",
            "published_result_reproduction": False,
        }
