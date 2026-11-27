"""Shared rules for adapters that load a pinned Hugging Face checkpoint from a local folder.

The checkpoint folder holds the files of one repository at one revision. Every file the
adapter loads is listed with its SHA-256 and checked before any framework is imported.
A file whose digest is not known yet is listed as ``PENDING``, and the adapter refuses
to load until it is filled in. Loading never reaches the network: the Hugging Face
offline switches are set and every ``from_pretrained`` call is ``local_files_only``.

Where the digests come from: the repository's file listing at the pinned revision
(``/api/models/<repo>/tree/<revision>?recursive=1``) is saved under
``resources/models/hf-trees``. A weights file is stored with Git LFS, and its ``lfs.oid``
is its SHA-256. A small file has only a git SHA-1 object ID in the listing, so it was
fetched once at the revision, its git object ID checked against the listing, and hashed.
Nothing was loaded or run, and no weights were downloaded.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PENDING = "pending"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Checkpoint:
    """One Hugging Face repository at one revision, and the files an adapter loads."""

    repository: str
    revision: str
    files: Mapping[str, str]
    record: str
    remote_code: bool = False

    def __post_init__(self) -> None:
        if not _REVISION.fullmatch(self.revision):
            raise ValueError(f"{self.repository}: the revision must be a 40-hex commit")
        for name, digest in self.files.items():
            if digest != PENDING and not _SHA256.fullmatch(digest):
                raise ValueError(f"{self.repository}: {name} needs a SHA-256 or PENDING")

    @property
    def pending(self) -> list[str]:
        return sorted(name for name, digest in self.files.items() if digest == PENDING)

    def url(self, name: str) -> str:
        return f"https://huggingface.co/{self.repository}/resolve/{self.revision}/{name}"

    def verify(self, folder: str | Path) -> Path:
        """Refuse a pending digest, then a missing or changed file. Returns the folder."""
        if self.pending:
            raise ValueError(
                f"{self.repository}: no SHA-256 is pinned yet for {', '.join(self.pending)}. "
                f"Fill them in from the Hugging Face file listing at revision {self.revision} "
                "before this adapter can load."
            )
        root = Path(folder).resolve()
        for name, expected in self.files.items():
            path = root / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Missing or mismatched pinned {self.repository} file: {name}")
        return root

    def provenance(self) -> dict[str, Any]:
        return {
            "model_id": self.repository,
            "checkpoint_revision": self.revision,
            "artifact_sha256": dict(self.files),
            "record": self.record,
            "remote_code": self.remote_code,
        }


def offline_frameworks() -> tuple[Any, Any]:
    """Import torch and transformers with the Hugging Face offline switches on."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers

    return torch, transformers


def as_array(value: Any) -> np.ndarray:
    """A float64 NumPy copy of a torch tensor or array-like."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float64)


def to_device(value: Any, device: str) -> Any:
    return value.to(device) if hasattr(value, "to") else value


def model_inputs(model: Any, encoded: Mapping[str, Any], device: str) -> dict[str, Any]:
    """The tokenizer outputs the model's forward accepts, on the device.

    A tokenizer can return fields its model does not take (``token_type_ids`` for a model
    without segment embeddings), and an unexpected keyword stops the forward pass.
    """
    forward = getattr(model, "forward", model)
    parameters = inspect.signature(forward).parameters.values()
    if any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters):
        names = set(encoded)
    else:
        names = {parameter.name for parameter in parameters}
    return {key: to_device(value, device) for key, value in encoded.items() if key in names}


def pool(hidden: Any, how: str, keep: Any = None) -> list[float]:
    """Pool one sequence's token states (tokens x features) into one vector.

    ``keep`` marks the tokens to pool; without it every token is pooled.
    """
    if how not in ("mean", "max"):
        raise ValueError(f"Unknown pooling: {how}")
    states = as_array(hidden)
    if states.ndim != 2 or not states.shape[0]:
        raise ValueError("Token states must be a nonempty tokens x features matrix")
    if keep is not None:
        mask = np.asarray(keep, dtype=bool).reshape(-1)
        if mask.shape[0] != states.shape[0] or not mask.any():
            raise ValueError("The pooling mask must select at least one token")
        states = states[mask]
    vector = states.mean(axis=0) if how == "mean" else states.max(axis=0)
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite embedding")
    return vector.tolist()


def log_softmax(logits: Any) -> np.ndarray:
    """Log-probabilities over the last axis, computed in float64."""
    values = as_array(logits)
    shifted = values - values.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def unique_ids(inputs: list[dict[str, Any]]) -> None:
    ids = [row.get("id") for row in inputs]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Inputs require unique nonempty IDs")


class FrozenEncoder:
    """A pinned encoder run on one sequence at a time, with no padding, as the model
    cards show. Subclasses set the checkpoint, the pooling and how a sequence becomes
    tokenizer text; ``embed`` checks the inputs before the model is touched."""

    CHECKPOINT: Checkpoint
    POOLING = "mean"
    MAX_TOKENS = 512
    EXCLUDE_SPECIAL_TOKENS = False
    IDS_ONLY = False
    PAIRED = False
    ALPHABET = frozenset("ACGT")
    TOKENIZER_OPTIONS: Mapping[str, Any] = {}

    torch: Any
    tokenizer: Any
    model: Any
    device: str

    def text(self, sequence: str) -> str:
        return sequence

    def vector(self, sequence: str) -> list[float]:
        encoded = dict(self.tokenizer(
            self.text(sequence), return_tensors="pt",
            return_special_tokens_mask=self.EXCLUDE_SPECIAL_TOKENS, **self.TOKENIZER_OPTIONS,
        ))
        special = encoded.pop("special_tokens_mask", None)
        count = as_array(encoded["input_ids"]).shape[-1]
        if count > self.MAX_TOKENS:
            raise ValueError(f"{self.CHECKPOINT.repository} accepts at most {self.MAX_TOKENS} "
                             f"tokens and this sequence has {count}; nothing is cropped")
        with self.torch.inference_mode():
            if self.IDS_ONLY:
                output = self.model(to_device(encoded["input_ids"], self.device))
            else:
                output = self.model(**model_inputs(self.model, encoded, self.device))
        keep = None if special is None else as_array(special)[0] == 0
        return pool(output[0][0], self.POOLING, keep)

    def check(self, row: dict[str, Any]) -> list[str]:
        """The sequences of one input row, after refusing fields a model must not see."""
        keys = ("reference_sequence", "mutant_sequence") if self.PAIRED else ("sequence",)
        if not self.PAIRED and set(row) != {"id", "sequence"}:
            raise ValueError("Embedding inputs allow only id and sequence, never labels")
        if self.PAIRED and ("target" in row or not set(keys) <= set(row)):
            raise ValueError("Paired inputs need reference_sequence and mutant_sequence, "
                             "never labels")
        sequences = [row[key] for key in keys]
        for sequence in sequences:
            if (not isinstance(sequence, str) or not sequence
                    or set(sequence.upper()) - self.ALPHABET):
                raise ValueError(f"Sequences must be nonempty and use {''.join(sorted(self.ALPHABET))}")
        return [sequence.upper() for sequence in sequences]

    def embed(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        unique_ids(inputs)
        checked = [(row["id"], self.check(row)) for row in inputs]
        output: dict[str, Any] = {}
        for identifier, sequences in checked:
            vectors = [self.vector(sequence) for sequence in sequences]
            output[identifier] = ({"reference": vectors[0], "mutant": vectors[1]}
                                  if self.PAIRED else vectors[0])
        return output
