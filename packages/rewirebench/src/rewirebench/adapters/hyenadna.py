"""HyenaDNA sequence log-likelihood for DART-Eval task 1 zero-shot.

Source: data repository record ``regulatory-variant-20261009-model-hyenadna``, spec
``data/omics/model-pages-20261126-batch1/specs/hyenadna.json``, and the model card and
``huggingface.py`` saved beside it (``artifacts/hyenadna/``). The run guide runs the
repository's ``huggingface.py``, which clones the weights with git lfs and builds Flash
Attention. The README and card also publish the transformers port
``LongSafari/hyenadna-large-1m-seqlen-hf``, which this adapter loads instead because it
has a pinned revision and needs no compiled kernels.

The port cannot load without remote code: ``config.json`` maps the Auto classes to
``modeling_hyena.py`` and the tokenizer to ``tokenization_hyena.py``, files in the
checkpoint. All three Python files are pinned here like the weights.

Scoring: the card shows fine-tuning only. DART-Eval task 1 compares a regulatory element
with its shuffled control, and for a causal model the natural zero-shot score is the
log-likelihood. This adapter loads ``AutoModelForCausalLM`` and scores a sequence as the
summed log-probability of each base given the bases before it. The tokenizer adds no
start token, so the first base has no context and is not scored, and the end token the
tokenizer appends is left out.

Weights licence: BSD-3-Clause (model card metadata).

Pins: the file names and the weights digest come from the Hugging Face file listing
at the pinned revision, saved as ``resources/models/hf-trees/hyenadna.json``. The other
digests are of the small files fetched once at that revision; see ``pinned``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rewirebench.adapters.pinned import (
    Checkpoint,
    as_array,
    log_softmax,
    offline_frameworks,
    to_device,
    unique_ids,
)

TRANSFORMERS_VERSION = "4.41.1"
MAX_TOKENS = 1_000_002
ALPHABET = frozenset("ACGTN")
CHECKPOINT = Checkpoint(
    repository="LongSafari/hyenadna-large-1m-seqlen-hf",
    revision="0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446",
    record="regulatory-variant-20261009-model-hyenadna",
    remote_code=True,
    files={
        "config.json": "e63e0dcb3bc0ff6f5207b3e0dbad48f6f0933dde42a47872cf1ab86e4b43f038",
        "configuration_hyena.py": "29cb6cae67f5010e279d67349e688663043f0bd6791a09c36d96da8cac3668a6",
        "modeling_hyena.py": "d78029cacea2c9259f1293fdb50b60f90872089469185e27d5c769086c475607",
        "special_tokens_map.json": "026831cec9f2ac5e501aa9d3a8b3a93c021aebe55813574c493e40b49ad28582",
        "tokenization_hyena.py": "d424da2f794958b0360e033bbf5edc9ee4e3e3126e883093621348f07761e408",
        "tokenizer_config.json": "7ceea744a19e155f056818b05e80195093ade809614964d8016ce2505c1485d4",
        "model.safetensors": "deafb53209bfafb314d14bd81108546a34d473c07ed7ab7a355674376b56ad12",
    },
)


def base_log_likelihood(logits: Any, ids: Any, special: Any) -> float:
    """Sum of log P(token t | tokens before t) over targets that are not special tokens."""
    values = log_softmax(as_array(logits))
    tokens = np.asarray(as_array(ids), dtype=np.int64).reshape(-1)
    targets = ~np.asarray(as_array(special), dtype=bool).reshape(-1)
    if values.shape[0] != tokens.shape[0] or tokens.shape[0] != targets.shape[0]:
        raise ValueError("Logits, token IDs and the special-token mask must align")
    positions = np.flatnonzero(targets[1:])
    if not positions.size:
        raise ValueError("A sequence needs at least one base after the first token")
    return float(values[positions, tokens[positions + 1]].sum())


class HyenaDNA:
    """Pinned local checkpoint only. Construct once, then call predict in batches."""

    CHECKPOINT = CHECKPOINT

    def __init__(self, checkpoint: str, *, device: str = "cpu"):
        folder = self.CHECKPOINT.verify(checkpoint)
        torch, transformers = offline_frameworks()
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise ValueError(f"This adapter is reviewed for transformers=={TRANSFORMERS_VERSION}")
        self.torch, self.device = torch, device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=True)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            str(folder), local_files_only=True, trust_remote_code=True,
            use_safetensors=True, torch_dtype=torch.float32,
        ).eval().to(device)

    def provenance(self) -> dict[str, Any]:
        return {
            **self.CHECKPOINT.provenance(),
            "method": "HyenaDNA summed base log-likelihood",
            "transformers_version": TRANSFORMERS_VERSION, "device": self.device,
            "weights_licence": "BSD-3-Clause",
            "training_overlap": "pretrained on hg38, which contains the DART-Eval elements",
            "published_result_reproduction": False,
        }

    def predict(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        unique_ids(inputs)
        for row in inputs:
            sequence = row.get("sequence")
            if set(row) != {"id", "sequence"}:
                raise ValueError("Inputs allow only id and sequence, never labels")
            if not isinstance(sequence, str) or not sequence or set(sequence.upper()) - ALPHABET:
                raise ValueError("Sequences must be nonempty and use ACGTN")
        results: dict[str, Any] = {}
        for row in inputs:
            encoded = self.tokenizer(row["sequence"].upper(), return_tensors="pt",
                                     return_special_tokens_mask=True)
            ids = encoded["input_ids"]
            if as_array(ids).shape[-1] > MAX_TOKENS:
                results[row["id"]] = {"score": None,
                                      "reason": f"HyenaDNA reads at most {MAX_TOKENS} tokens"}
                continue
            with self.torch.inference_mode():
                logits = self.model(to_device(ids, self.device))[0]
            results[row["id"]] = base_log_likelihood(
                as_array(logits)[0], ids, encoded["special_tokens_mask"])
        return results
