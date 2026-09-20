"""Small CPU controls, not implementations of published neural-model baselines."""
from __future__ import annotations

import hashlib
import math


class SequenceComposition:
    """Untrained length/composition features for a protocol-owned supervised head.

    No labels, learned vocabulary, model download or sequence truncation. For
    paired components encoded with separators, those separators contribute to
    the explicit unknown-character fraction rather than being silently removed.
    """

    def __init__(self, alphabet="ACGT"):
        if not isinstance(alphabet, str) or not alphabet or len(set(alphabet.upper())) != len(alphabet):
            raise ValueError("Supply an alphabet of unique symbols")
        self.alphabet = alphabet.upper()

    def embed(self, inputs):
        out = {}
        for row in inputs:
            sequence = row["sequence"].upper()
            if not sequence:
                raise ValueError("Empty sequence")
            counts = [sequence.count(base) for base in self.alphabet]
            out[row["id"]] = [math.log1p(len(sequence))] + [
                count / len(sequence) for count in counts
            ] + [(len(sequence) - sum(counts)) / len(sequence)]
        return out


class TrainMean:
    """A constant regression control fitted exclusively on the supplied training labels."""

    def fit(self, inputs, targets):
        if not targets or len(inputs) != len(targets):
            raise ValueError("Nonempty aligned training data required")
        self.mean = sum(targets) / len(targets)

    def predict(self, inputs):
        if not hasattr(self, "mean"):
            raise ValueError("TrainMean must be fitted through a supervised protocol")
        return {row["id"]: self.mean for row in inputs}


class SeededRandomScore:
    """Deterministic random-ranking control tied to opaque prepared IDs, never targets."""

    def __init__(self, seed=0):
        self.seed = int(seed)

    def predict(self, inputs):
        return {
            row["id"]: int.from_bytes(hashlib.sha256(
                f"{self.seed}:{row['id']}".encode()
            ).digest()[:8], "big") / 2**64
            for row in inputs
        }
