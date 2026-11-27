"""Frozen BioFM-265M pair embeddings for the MFASS frozen-encoder protocol.

Source: data repository record ``lit-genomics-20261011-geneb-model-biofm``, run guide step
"Extract sequence embeddings", read from the biofm-eval README at commit fcb86e1c (spec
``data/omics/model-pages-20261127-revision1/specs/biofm.json``). The run guide loads the
weights through the ``biofm_eval`` package. Its model class is ``MistralForCausalLM``
unchanged, and its tokenizer is a fixed character map, so this adapter reads both
without installing the package:

- The vocabulary is the six characters in the pinned ``tokenizer_config.json``
  (``char_ords``, in order D, N, A, T, G, C) as IDs 0 to 5, then ``[CLS]`` 6 and ``[SEP]``
  7, as ``AnnotationTokenizer`` defines them in ``biofm_eval/tokenizers/base.py``.
- A sequence is ``[CLS]``, one token per base, ``[SEP]``.
- ``Embedder.get_sequence_embeddings`` (``biofm_eval/embedder.py``) passes only
  ``input_ids`` and returns the last layer's hidden state at the last base, the token
  before ``[SEP]``. This adapter does the same for the reference and the mutant window.

The package was read at commit fcb86e1c as text and not installed or run. It requires
transformers 4.46.1 or later; this adapter uses 5.16.0, the version the ESM C adapter
needs, so the two share one extra. The run guide loads the weights in bfloat16; this
adapter loads them in float32 for the CPU.

Weights licence: CC BY-NC 4.0, non-commercial use only (model card metadata and
README). Nothing is redistributed: the recipe fetches the files from the developer's
repository at the pinned revision.

Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/biofm.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rewirebench.adapters.pinned import (
    Checkpoint,
    FrozenEncoder,
    as_array,
    offline_frameworks,
    to_device,
)

TRANSFORMERS_VERSION = "5.16.0"
CLS, SEP = 6, 7
CHECKPOINT = Checkpoint(
    repository="m42-health/BioFM-265M",
    revision="52218bbdd383c123f6ced585c9cf2f62ae1fbb17",
    record="lit-genomics-20261011-geneb-model-biofm",
    files={
        "config.json": "68656f74065d92cd077bc1e1b54ecdb3061dfc74a67c67b64cf103ec07b82620",
        "generation_config.json": "5a037a70ac785795f6258505c78482ead2f4ce77b7214c386a43d0d65417cf71",
        "tokenizer_config.json": "8429c2b5485bd51a010c0980d5871c11c6386e770886e9f87ab0d03dcd7700bb",
        "model.safetensors": "ff6ea18ca716b5686ad3e0f025e0d3c86257a88766b6aa0063db5cb6493ae108",
    },
)


class BioFM(FrozenEncoder):
    """Last-layer state at the last base of the reference and the mutant window."""

    CHECKPOINT = CHECKPOINT
    MAX_TOKENS = 6144
    PAIRED = True
    ALPHABET = frozenset("ACGTN")

    def __init__(self, checkpoint: str, *, device: str = "cpu"):
        folder = self.CHECKPOINT.verify(checkpoint)
        torch, transformers = offline_frameworks()
        self.vocabulary = vocabulary(folder / "tokenizer_config.json")
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise ValueError(f"This adapter is reviewed for transformers=={TRANSFORMERS_VERSION}")
        self.torch, self.device = torch, device
        self.model = transformers.MistralForCausalLM.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32,
        ).eval().to(device)

    def token_ids(self, sequence: str) -> list[int]:
        return [CLS, *(self.vocabulary[base] for base in sequence), SEP]

    def vector(self, sequence: str) -> list[float]:
        ids = self.token_ids(sequence)
        if len(ids) > self.MAX_TOKENS:
            raise ValueError(f"BioFM accepts at most {self.MAX_TOKENS} tokens and this "
                             f"sequence has {len(ids)}; nothing is cropped")
        with self.torch.inference_mode():
            output = self.model(input_ids=to_device(self.torch.tensor([ids]), self.device),
                                output_hidden_states=True)
        return as_array(output.hidden_states[-1])[0][len(ids) - 2].tolist()

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": "BioFM frozen last-base pair embeddings",
            "transformers_version": TRANSFORMERS_VERSION,
            "device": self.device, "dtype": "float32", "pretrained": True,
            "weights_licence": "CC-BY-NC-4.0",
            "training_overlap": "pretrained on human genomes, which contain the MFASS "
                                "reference windows",
            "published_result_reproduction": False,
        }


def vocabulary(path: Path) -> dict[str, int]:
    """Base to token ID, from the ``char_ords`` of the pinned tokenizer configuration."""
    characters = [chr(code) for code in json.loads(path.read_text())["char_ords"]]
    return {character: index for index, character in enumerate(characters)}
