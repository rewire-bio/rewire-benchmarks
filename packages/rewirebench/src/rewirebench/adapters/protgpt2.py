"""ProtGPT2 log-likelihood ratio substitution scores for ProteinGym.

Source: data repository record ``uc20260930-model-protgpt2`` (depth tier 2), spec
``data/omics/model-pages-20261127-batch7/specs/protgpt2.json``. The run guide's only
inference step is generation. This adapter instead follows the card's section "How to
select the best sequences" (``artifacts/protgpt2/hf-card.md`` at revision ff981fc9): the
sequence is written as ``<|endoftext|>``, a newline, the residues with a newline every
60, a newline and ``<|endoftext|>``, and scored by a causal-LM forward pass over its
token IDs. The card turns the mean loss into perplexity; this adapter keeps the summed
log-probability of every token after the first, because a substitution can change the
number of tokens, and scores a variant as log P(mutant) minus log P(wild type).

Weights licence: the repository's LICENSE file says MIT and its card metadata says
apache-2.0; both are open. The weights are ``pytorch_model.bin``, a pickle, unpickled
after the digest check.

Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/protgpt2.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rewirebench.adapters.esm1b import check_record
from rewirebench.adapters.pinned import (
    Checkpoint,
    as_array,
    log_softmax,
    offline_frameworks,
    to_device,
    unique_ids,
)

TRANSFORMERS_VERSION = "4.41.1"
MAX_TOKENS = 1024
LINE = 60
CHECKPOINT = Checkpoint(
    repository="nferruz/ProtGPT2",
    revision="ff981fc96771d6f6d3b6453f94772d3cb6c7b90b",
    record="uc20260930-model-protgpt2",
    files={
        "config.json": "b1eb50b2e360c8c9e433f234eeec53e809c99df254271ace5454b89244818195",
        "pytorch_model.bin": "95ee34898a635ad046aa3c53a08f50ca169ff67443db50c0ba328320d8ed1f04",
        "tokenizer.json": "3e550dd50b13bc05ebbb681b8caa6d3fd9f04879e1f9c5bf3087c3f8517fad3e",
        "vocab.json": "e89bc0dccbaf4ffbf62771d04e72aa7217c2d7c16bc8e5a8c520597880de3c20",
        "merges.txt": "b8d3b38f3e97efa2483bedf198cdb757b988ed17fb080f3cba94869f189eddf6",
        "special_tokens_map.json": "a00e1f660c842693e7f667898759c24eb538c2dbb09b91c22dbde1b45d4cdee8",
    },
)


def card_text(sequence: str) -> str:
    """The card's input format for one protein."""
    lines = [sequence[start:start + LINE] for start in range(0, len(sequence), LINE)]
    return "<|endoftext|>\n" + "\n".join(lines) + "\n<|endoftext|>"


def sequence_log_likelihood(logits: Any, ids: Any) -> float:
    """Sum of log P(token t | tokens before t) for every token after the first."""
    values = log_softmax(as_array(logits))
    tokens = np.asarray(as_array(ids), dtype=np.int64).reshape(-1)
    if values.shape[0] != tokens.shape[0] or tokens.shape[0] < 2:
        raise ValueError("Logits and token IDs must align and hold at least two tokens")
    return float(values[np.arange(tokens.shape[0] - 1), tokens[1:]].sum())


class ProtGPT2:
    """Pinned local checkpoint only. Construct once, then call predict in batches."""

    CHECKPOINT = CHECKPOINT

    def __init__(self, checkpoint: str, *, device: str = "cpu"):
        folder = self.CHECKPOINT.verify(checkpoint)
        torch, transformers = offline_frameworks()
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise ValueError(f"This adapter is reviewed for transformers=={TRANSFORMERS_VERSION}")
        self.torch, self.device = torch, device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32,
        ).eval().to(device)
        self._cache: dict[str, float | None] = {}

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": "ProtGPT2 summed log-likelihood, mutant minus wild type",
            "transformers_version": TRANSFORMERS_VERSION,
            "maximum_tokens": MAX_TOKENS, "device": self.device,
            "training_overlap": "unreported; UniRef50 pretraining may overlap benchmark proteins",
            "published_result_reproduction": False,
        }

    def log_likelihood(self, sequence: str) -> float | None:
        """None when the formatted sequence is longer than the model's context."""
        if sequence not in self._cache:
            ids = self.tokenizer(card_text(sequence), return_tensors="pt")["input_ids"]
            if as_array(ids).shape[-1] > MAX_TOKENS:
                self._cache[sequence] = None
            else:
                with self.torch.inference_mode():
                    logits = self.model(to_device(ids, self.device))[0]
                self._cache[sequence] = sequence_log_likelihood(as_array(logits)[0], ids)
        return self._cache[sequence]

    def predict(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        unique_ids(inputs)
        checked = [check_record(record) for record in inputs]
        results: dict[str, Any] = {}
        for identifier, sequence, substitutions in checked:
            mutated = list(sequence)
            for index, _, alt in substitutions:
                mutated[index] = alt
            wild, mutant = self.log_likelihood(sequence), self.log_likelihood("".join(mutated))
            if wild is None or mutant is None:
                results[identifier] = {
                    "score": None,
                    "reason": f"ProtGPT2 reads at most {MAX_TOKENS} tokens; "
                              "windowed inference is not implemented",
                }
                continue
            results[identifier] = mutant - wild
        return results
