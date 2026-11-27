"""Frozen MP-RNA sequence embeddings for the mRNABench probe.

Source: data repository record ``lit-rna-20261011-rnalm-zeroshot-model-mp-rna`` (twin
``lit-microbial-20261011-virobench-model-mp-rna``), run guide step "Load the model and run
one sequence", taken from the model card (spec
``data/omics/model-pages-20261127-batch10/specs/mp-rna.json``). Loading follows the card:
``AutoTokenizer`` and ``AutoModel``, one sequence per forward pass with the full tokenizer
output. The card stops at the model output and gives no pooling: the adapter takes the
mean of the last hidden states over the sequence's own tokens and leaves out the special
tokens. The card's example is written in U, so T is rewritten as U.

Difference from the card: it loads without ``trust_remote_code``, but the pinned
``config.json`` maps every Auto class to ``modeling_omnigenome.py`` in the checkpoint, so
the model cannot load without remote code. Its two Python files are pinned here like the
weights. The tokenizer is ``EsmTokenizer``, which transformers ships.

Weights licence: Apache-2.0 (model card metadata).

Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/mp-rna.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.
"""

from __future__ import annotations

from typing import Any

from rewirebench.adapters.pinned import Checkpoint, FrozenEncoder, offline_frameworks

TRANSFORMERS_VERSION = "4.41.1"
CHECKPOINT = Checkpoint(
    repository="yangheng/MP-RNA",
    revision="0a727ebb92df41949266b234ef70b1621bfd5742",
    record="lit-rna-20261011-rnalm-zeroshot-model-mp-rna",
    remote_code=True,
    files={
        "config.json": "ebc021d327a6ef556fa277dd3c842ae219a0ff619d55f939e0d0dca3b19b9b16",
        "configuration_omnigenome.py": "aabbeaa27857544e060e47af9293baef1d5b4b95789f717f56cfa3e9be57a78d",
        "modeling_omnigenome.py": "bfbb1ef7b8726def69cd22f968ce48dc83c1c7c5ed76846d491d2541cb68676e",
        "special_tokens_map.json": "01e416f1aa5cb48b1d2c763ed8f2ef6e1344a5fc351657c1e8f2c73db82443d9",
        "tokenizer_config.json": "2f96b43c7f15f31aaa8ae3fab569590c6c6909dee9ed80b57b790f22c24042a4",
        "vocab.txt": "5c6fecdc35423dc779aa46529b81d5981b4eb77676a57eda961deba46a0f05df",
        "model.safetensors": "f66b44233c66723a359a3a7864af93e42f2cadb43835440ca9c318fe17383923",
    },
)


class MPRNA(FrozenEncoder):
    """Mean of the last hidden states over the sequence's own tokens."""

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
            str(folder), local_files_only=True, trust_remote_code=True,
            use_safetensors=True, torch_dtype=torch.float32,
        ).eval().to(device)

    def text(self, sequence: str) -> str:
        return sequence.replace("T", "U")

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": "MP-RNA frozen mean-pooled sequence embeddings",
            "transformers_version": TRANSFORMERS_VERSION,
            "device": self.device, "dtype": "float32", "pretrained": True,
            "weights_licence": "Apache-2.0",
            "training_overlap": "overlap with mRNABench sequences is unreported",
            "published_result_reproduction": False,
        }
