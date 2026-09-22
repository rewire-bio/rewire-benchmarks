"""Small local reference methods; no weights, downloads or evaluation labels.

These are Rewire controls, not claims to reproduce a paper's baseline. The SDK
owns train/test access and supplies only permitted training labels to ``fit``.
"""
from __future__ import annotations

import numpy as np


class TrainingPrior:
    """Binary training prevalence or multiclass training majority (ties: lowest class)."""

    def __init__(self, *, multiclass=False):
        self.multiclass = multiclass

    def fit(self, inputs, targets):
        if not targets or len(inputs) != len(targets):
            raise ValueError("Nonempty aligned training data required")
        labels = np.asarray(targets)
        if not np.isfinite(labels).all() or not np.equal(labels, labels.astype(int)).all():
            raise ValueError("Classification labels must be finite integers")
        if self.multiclass:
            classes, counts = np.unique(labels, return_counts=True)
            self.value = float(classes[np.argmax(counts)])
        else:
            if not set(labels).issubset({0, 1}):
                raise ValueError("Binary labels required")
            self.value = float(labels.mean())

    def predict(self, inputs):
        if not hasattr(self, "value"):
            raise ValueError("TrainingPrior requires fitting")
        return {row["id"]: self.value for row in inputs}


class StringNgramReference:
    """Train-only character 1–3-gram counts with a fixed regularised linear head.

    DNA and SMILES text are accepted through distinct input fields. SMILES counts
    are a representation-dependent string reference, not chemical descriptors or
    an invariant molecular fingerprint. No canonicalisation or chemistry toolkit
    is silently applied. Inputs are uppercased only for DNA, never for SMILES.
    """

    def __init__(self, *, field="sequence", classification=True, multiclass=False):
        if field not in {"sequence", "smiles"}:
            raise ValueError("Unsupported reference input field")
        self.field = field
        self.classification = classification
        self.multiclass = multiclass

    def _strings(self, inputs):
        values = [row[self.field] for row in inputs]
        if any(not isinstance(v, str) or not v for v in values):
            raise ValueError("Nonempty string inputs required")
        return [v.upper() for v in values] if self.field == "sequence" else values

    def fit(self, inputs, targets):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import Normalizer

        if not targets or len(inputs) != len(targets):
            raise ValueError("Nonempty aligned training data required")
        self.vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False)
        self.normalizer = Normalizer(norm="l2")
        features = self.normalizer.fit_transform(self.vectorizer.fit_transform(self._strings(inputs)))
        if self.classification:
            labels = np.asarray(targets)
            if len(set(targets)) < 2 or not np.equal(labels, labels.astype(int)).all():
                raise ValueError("Classification fitting requires at least two integer classes")
            self.head = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=0)
        else:
            self.head = Ridge(alpha=1.0, solver="lsqr", tol=1e-8, max_iter=10000)
        self.head.fit(features, targets)

    def predict(self, inputs):
        if not hasattr(self, "head"):
            raise ValueError("StringNgramReference requires fitting")
        features = self.normalizer.transform(self.vectorizer.transform(self._strings(inputs)))
        if self.classification and not self.multiclass:
            if list(self.head.classes_) != [0, 1]:
                raise ValueError("Binary classification requires labels 0 and 1")
            values = self.head.predict_proba(features)[:, 1]
        else:
            values = self.head.predict(features)
        return {r["id"]: float(v) for r, v in zip(inputs, values, strict=True)}


class PairedComposition:
    """Untrained composition with the existing MFASS fixed train-only probe.

    Constant vectors test the probe without sequence information; because that
    probe uses balanced class weights, this is NOT training prevalence.
    """

    def __init__(self, *, constant=False):
        self.constant = constant

    def embed(self, inputs):
        from rewirebench.adapters.sequence import SequenceComposition

        if self.constant:
            return {row["id"]: {"reference": [0.0], "mutant": [0.0]} for row in inputs}
        encoder = SequenceComposition("ACGT")
        reference = encoder.embed([
            {"id": r["id"], "sequence": r["reference_sequence"]} for r in inputs
        ])
        mutant = encoder.embed([
            {"id": r["id"], "sequence": r["mutant_sequence"]} for r in inputs
        ])
        return {r["id"]: {"reference": reference[r["id"]], "mutant": mutant[r["id"]]}
                for r in inputs}
